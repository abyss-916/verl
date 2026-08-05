#!/usr/bin/env bash
# GRPO → merge → eval 一键自动接续：启动一次，后台按顺序跑完三步，无需手动接续。
#   ① GRPO(verl_grpo 环境，从 SFT-merged 起、叠 LoRA r32) → ② merge(GRPO-LoRA 折进 SFT-merged 底模)
#   → ③ eval(verl 环境，held-out OlymMATH-hard、n=4，与 base/SFT 同环境，结果可比)
# 每步失败即 fail-loud 终止后续，避免在无效产物上继续评测。
#
# 用法（先 git pull 到最新；在任一已初始化 conda 的 shell 里）：
#   cd /data/liujiachen/verl/qwen3_4b_distill
#   setsid bash run/grpo.sh > /data/liujiachen/logs/run/grpo.log 2>&1 < /dev/null &
#   tail -f /data/liujiachen/logs/run/grpo.log        # 看三步总进度（banner）
#   # GRPO 细节同在该 log；eval 逐卡进度在 $LOGS/run/eval_omni_standard_s{0,1}.log
# 可覆盖：METHOD / FROM / EPOCHS / GM / RESP / N / TBS / EVAL_N / EVAL_GM / GRPO_ENV / EVAL_ENV
#   例：显存更紧 → GM=0.68 setsid bash run/grpo.sh ...
set -o pipefail   # 不用 -e/-u：脚本内 conda activate（-u 会被 conda 内部脚本误触发）、且每步失败需自定义处理
source "$(dirname "$0")/env.sh"

# ---- conda 环境切换：GRPO=verl_grpo；eval=verl（与 base/SFT eval 同环境，保证可比）----
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "!! conda 未就绪（conda info --base 失败）——请在已初始化 conda 的 shell 里启动本脚本"; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
GRPO_ENV=${GRPO_ENV:-verl_grpo}
EVAL_ENV=${EVAL_ENV:-verl}

# ---- 参数（默认= standard 一法、EPOCHS=1 短 PoC）----
METHOD=${METHOD:-omni_standard}                 # 接在前缀后的方法名（与 SFT-merged 命名对齐）
FROM=${FROM:-sft_omni_standard_cot_merged}      # GRPO 起点 = SFT-merged 底模
EXP=grpo_${METHOD}                              # ckpt 目录名 = grpo_omni_standard
SFT_BASE="$CKPT/$FROM"
EPOCHS=${EPOCHS:-1}; GM=${GM:-0.70}; RESP=${RESP:-16384}; N=${N:-5}; TBS=${TBS:-32}
TP=${TP:-2}    # rollout 张量并行。2=每卡半权重省显存(summon 可过)，但无 NVLink 时逐 token 跨卡 all-reduce 较慢；
               #   1=每卡满 8G，update_weights(summon)会 OOM，此硬件只能用 TP=2
STEPS=${STEPS:-5}   # 默认 5=报告所用短 PoC（无 NVLink TP=2 每步~3h）；STEPS=0 则按 EPOCHS 跑满(~15步~47h)
EVAL_N=${EVAL_N:-4}; EVAL_GM=${EVAL_GM:-0.8}
MERGED="$CKPT/${EXP}_merged"
RESULT="$LOGS/eval/olymmath_${EXP}/summary.json"

echo "############################################################"
echo "# [chain] $EXP  |  GRPO: EPOCHS=$EPOCHS STEPS=$STEPS TP=$TP GM=$GM RESP=$RESP N=$N TBS=$TBS  |  eval: n=$EVAL_N gm=$EVAL_GM"
echo "#   起点 SFT-merged = $SFT_BASE"
echo "#   env: GRPO=$GRPO_ENV  eval=$EVAL_ENV   起始 $(date '+%F %T')"
echo "############################################################"
[ -d "$SFT_BASE" ] || { echo "!! 缺 SFT 底模 $SFT_BASE，终止"; exit 1; }

# ===================== ① GRPO =====================
GSTEP_EXIST=$(ls -d "$CKPT/$EXP"/global_step_* 2>/dev/null | sort -V | tail -1)
if [ "${RESUME:-0}" = "1" ] && [ -n "$GSTEP_EXIST" ]; then
  # 复用已训好的 ckpt（如 merge/eval 失败需重来时），不重复运行数小时的 GRPO
  GSTEP="$GSTEP_EXIST"
  echo ">>>>> [1/3] RESUME=1：复用已存 GRPO ckpt=$GSTEP，跳过训练 $(date '+%F %T')"
else
  echo ">>>>> [1/3] GRPO 训练开始 $(date '+%F %T')  (env=$GRPO_ENV)"
  conda activate "$GRPO_ENV" || { echo "!! 无法 conda activate $GRPO_ENV"; exit 1; }
  rm -rf "$CKPT/$EXP"    # 干净重启（短 PoC，不从中断点续训，避免半写 ckpt）
  EXP="$EXP" MODEL_PATH="$SFT_BASE" TP="$TP" GM="$GM" RESP="$RESP" N="$N" EPOCHS="$EPOCHS" TBS="$TBS" STEPS="$STEPS" \
    REWARD="$REWARD_FN" RM="$REWARD_MGR" TRAIN_DIR="$ABILITY_SEED_DIR" VAL_DIR="$ABILITY_EVAL_DIR" \
    bash "$PROJ/train/grpo.sh"
  GRC=$?    # 退出码仅作诊断，不据此判成败：verl/Ray teardown 常在训练成功后仍返回非零码，
            #  若据退出码终止，会出现“训练成功却不接 merge”的情况
  timeout 60 ray stop >/dev/null 2>&1 || true    # 停掉本次的 Ray worker，把两卡完整交给 eval（防孤儿进程占卡致 eval OOM）
  # 成败以“最终 ckpt 是否产出”为准：save_freq>0 时 is_last_step 必存；EPOCHS=1 仅末步存，有 ckpt 即训练已跑完
  GSTEP=$(ls -d "$CKPT/$EXP"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$GSTEP" ]; then
    echo "!! [1/3] 训练未产出 ckpt(退出码=$GRC)。若 step1 显存尖峰 OOM → 降 GM=0.68 重跑本脚本。链终止"; exit 1
  fi
  [ "$GRC" -ne 0 ] && echo "   (注:grpo.sh 退出码=$GRC,但最终 ckpt 已产出=训练成功,照常接续;非零多为 Ray 关闭噪声)"
  echo ">>>>> [1/3] GRPO 完成 $(date '+%F %T')  ckpt=$GSTEP"
fi

# ===================== ② merge =====================
echo ">>>>> [2/3] merge（GRPO-LoRA 折进 SFT-merged 底模）开始 $(date '+%F %T')"
conda activate "$GRPO_ENV" || { echo "!! 无法 conda activate $GRPO_ENV"; exit 1; }   # RESUME 模式下前面未激活，此处显式保证（model_merger+peft 与写 ckpt 用同环境最稳）
PREFIX=grpo_ METHODS="$METHOD" BASE="$SFT_BASE" bash "$PROJ/run/merge.sh"
[ -d "$MERGED" ] || { echo "!! [2/3] 未产出 $MERGED（见上方 merge 日志），链终止"; exit 1; }
echo ">>>>> [2/3] merge 完成 $(date '+%F %T')  merged=$MERGED"
echo "     注：自检 rel 偏小属正常——短 GRPO PoC 权重移动本就远小于 SFT，有效性看下一步 pass@1"

# ===================== ③ eval =====================
echo ">>>>> [3/3] eval（held-out OlymMATH-hard, n=$EVAL_N）开始 $(date '+%F %T')  (env=$EVAL_ENV)"
conda activate "$EVAL_ENV" || { echo "!! 无法 conda activate $EVAL_ENV"; exit 1; }
PREFIX=grpo_ METHODS="$METHOD" N="$EVAL_N" GM="$EVAL_GM" bash "$PROJ/run/eval.sh"
if [ -f "$RESULT" ]; then
  echo "############################################################"
  echo "# [chain] 全部完成 $(date '+%F %T') → $RESULT"
  echo "#   对照 base@4  pass@1/pass@4/cons@4 = 15.75 / 32.16 / 17.46"
  echo "#         SFT-standard              = 15.25 / 34.0  / 20.0"
  echo "############################################################"
  cat "$RESULT"
else
  echo "!! [3/3] 未见 $RESULT（看 $LOGS/run/eval_${METHOD}_s{0,1}.log），eval 可能失败"; exit 1
fi
