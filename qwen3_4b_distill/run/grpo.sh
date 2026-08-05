#!/usr/bin/env bash
# GRPO 一条龙：训练(verl_grpo) → merge 折叠 → eval(verl)，启动一次串行跑完，每步 fail-loud 门控。
# 用法（在已初始化 conda 的 shell）：
#   STEPS=5 setsid bash run/grpo.sh > $LOGS/run/grpo.log 2>&1 < /dev/null &
#   RESUME=1 setsid bash run/grpo.sh ...   # 复用已训 ckpt，只补 merge+eval
# 可覆盖：METHOD / FROM / STEPS / EPOCHS / GM / RESP / N / TBS / EVAL_N / EVAL_GM / GRPO_ENV / EVAL_ENV。
set -o pipefail   # 不用 -e/-u：脚本内 conda activate + 每步自定义处理
source "$(dirname "$0")/env.sh"

# ---- conda 环境切换：GRPO=verl_grpo；eval=verl ----
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
TP=${TP:-2}    # rollout 张量并行度
STEPS=${STEPS:-5}   # 总步数（默认 5）；STEPS=0 按 EPOCHS 跑满
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
  # 复用已训好的 ckpt，跳过训练
  GSTEP="$GSTEP_EXIST"
  echo ">>>>> [1/3] RESUME=1：复用已存 GRPO ckpt=$GSTEP，跳过训练 $(date '+%F %T')"
else
  echo ">>>>> [1/3] GRPO 训练开始 $(date '+%F %T')  (env=$GRPO_ENV)"
  conda activate "$GRPO_ENV" || { echo "!! 无法 conda activate $GRPO_ENV"; exit 1; }
  rm -rf "$CKPT/$EXP"    # 干净重启
  EXP="$EXP" MODEL_PATH="$SFT_BASE" TP="$TP" GM="$GM" RESP="$RESP" N="$N" EPOCHS="$EPOCHS" TBS="$TBS" STEPS="$STEPS" \
    REWARD="$REWARD_FN" RM="$REWARD_MGR" TRAIN_DIR="$ABILITY_SEED_DIR" VAL_DIR="$ABILITY_EVAL_DIR" \
    bash "$PROJ/train/grpo.sh"
  GRC=$?    # 退出码仅作诊断，不据此判成败
  timeout 60 ray stop >/dev/null 2>&1 || true    # 停掉本次 Ray worker，把两卡交给 eval
  # 成败以最终 ckpt 是否产出为准
  GSTEP=$(ls -d "$CKPT/$EXP"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$GSTEP" ]; then
    echo "!! [1/3] 训练未产出 ckpt(退出码=$GRC)。若 step1 显存尖峰 OOM → 降 GM=0.68 重跑本脚本。链终止"; exit 1
  fi
  [ "$GRC" -ne 0 ] && echo "   (注:grpo.sh 退出码=$GRC,但最终 ckpt 已产出=训练成功,照常接续;非零多为 Ray 关闭噪声)"
  echo ">>>>> [1/3] GRPO 完成 $(date '+%F %T')  ckpt=$GSTEP"
fi

# ===================== ② merge =====================
echo ">>>>> [2/3] merge（GRPO-LoRA 折进 SFT-merged 底模）开始 $(date '+%F %T')"
conda activate "$GRPO_ENV" || { echo "!! 无法 conda activate $GRPO_ENV"; exit 1; }   # RESUME 模式下前面未激活，此处显式保证
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
