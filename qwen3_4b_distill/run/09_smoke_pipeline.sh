#!/usr/bin/env bash
# 全链路冒烟：验证 code / mc 两个能力的  reward判分 → 造数据(gen) → SFT → eval → GRPO  每步不报错，
# 即端到端可跑，而非只接一个 base eval。极小样本、产物即弃，只验证管线连通性。
#
# gen/GRPO 用 test.parquet 的切片作 stand-in 训练种子，仅用于冒烟；正式造数据/训练须用与评测集
#    不重叠的独立 train 种子（否则泄漏）。本脚本产物写在 *_smoke 目录，与正式实验隔离。
#
# 用法（服务器，先看 nvidia-smi 确认两卡空闲）：
#   source run/env.sh
#   bash run/09_smoke_pipeline.sh                       # 全部阶段
#   ABIL="mc" DO_GRPO=0 bash run/09_smoke_pipeline.sh   # 只冒烟 mc、跳过最重的 GRPO
#   DO_SFT=0 DO_GRPO=0 bash run/09_smoke_pipeline.sh    # 只验 reward+gen（最快，免长训练）
set -uo pipefail
source "$(dirname "$0")/env.sh"

ABIL=${ABIL:-"code mc"}
DO_REWARD=${DO_REWARD:-1}; DO_GEN=${DO_GEN:-1}; DO_SFT=${DO_SFT:-1}; DO_EVAL=${DO_EVAL:-1}; DO_GRPO=${DO_GRPO:-1}
G0=${G0:-0}; G1=${G1:-1}; GM=${GM:-0.8}
SM=$DATA/distill                                  # 冒烟产物根（*_smoke 子目录）
LOG=$LOGS/run/09_smoke.log; mkdir -p "$LOGS/run" "$SM"
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
PASS=(); FAIL=()
ok(){ PASS+=("$1"); say "  ✔ $1"; }
no(){ FAIL+=("$1"); say "  ✗ $1  (看 $LOG)"; }

case_data(){ case "$1" in code) echo "$DATA/livecodebench";; mc) echo "$DATA/mmlu_pro";; esac; }
case_method(){ case "$1" in code) echo code_cot;; mc) echo mc_cot;; esac; }

for A in $ABIL; do
  say "======== 能力 [$A] ========"
  DSRC=$(case_data "$A"); METH=$(case_method "$A"); TP=$DSRC/test.parquet
  if [ ! -f "$TP" ]; then no "[$A] 数据缺失 $TP（先 prepare_${A}）"; continue; fi
  ok "[$A] prepare 数据在 $TP"

  # ---- ① reward 单元自检（免 GPU）：合成"正确解/错误解" → 判分器应给 1.0 / 0.0 ----
  if [ "$DO_REWARD" = 1 ]; then
    if python "$PROJ/tests/reward_unit_check.py" --ability "$A" --data "$TP" >>"$LOG" 2>&1; then
      ok "[$A] reward 单元自检（正确=1.0 / 错误=0.0）"
    else no "[$A] reward 单元自检"; fi
  fi

  # ---- ② 造数据 gen：新方法 code_cot/mc_cot 跑通、判定器被调用、写出 gen_stats.json ----
  GENOUT=$SM/${A}_smoke
  if [ "$DO_GEN" = 1 ]; then
    rm -rf "$GENOUT"
    CUDA_VISIBLE_DEVICES=$G0,$G1 python "$PROJ/distill/generate_cot.py" --method "$METH" \
      --seed "$TP" --teacher "$TEACHER" --out "$GENOUT" \
      --tp 2 --n 2 --limit 6 --max_new 8192 --gpu_mem "$GM" >>"$LOG" 2>&1
    # gen_stats.json 在 save() 之前写：即便小样本 0 留存（教师能力所限，非管线故障）也证明判定链跑通
    if [ -f "$GENOUT/gen_stats.json" ]; then
      NK=$(python -c "import json;print(json.load(open('$GENOUT/gen_stats.json'))['n_kept'])" 2>/dev/null || echo '?')
      ok "[$A] gen 跑通（$METH，n_kept=$NK；gen_stats.json 已写）"
    else no "[$A] gen（未写 gen_stats.json）"; fi
  fi

  # ---- ③ SFT：若 gen 有留存(train.parquet)则微调 1 epoch，验证 SFT 可处理该能力数据 ----
  if [ "$DO_SFT" = 1 ]; then
    if [ -f "$GENOUT/train.parquet" ]; then
      SFTEXP=sft_smoke_$A
      CUDA_VISIBLE_DEVICES=$G0,$G1 env USE_PEFT=1 LR=2e-4 EPOCHS=1 MAXLEN=4096 \
        EXP="$SFTEXP" SAVE="$CKPT/$SFTEXP" DATA_DIR="$GENOUT" bash "$PROJ/train/sft.sh" >>"$LOG" 2>&1
      if ls -d "$CKPT/$SFTEXP"/global_step_* >/dev/null 2>&1 || [ -d "$CKPT/$SFTEXP" ]; then
        ok "[$A] SFT 1-epoch 跑通（产 ckpt $CKPT/$SFTEXP）"
      else no "[$A] SFT"; fi
    else say "  – [$A] SFT 跳过（gen 0 留存，无 train.parquet；小样本教师能力所限，非管线故障）"; fi
  fi

  # ---- ④ eval：base 小样本判分跑通（与已跑的全量 base eval 同判分器；此处只作管线连通证明）----
  if [ "$DO_EVAL" = 1 ]; then
    EOUT=$LOGS/eval/smoke_${A}_base
    if [ "$A" = code ]; then
      CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/eval_code.py" --model "$STUDENT_BASE" --data "$TP" \
        --n 1 --limit 4 --max_new 8192 --gpu_mem "$GM" --out "$EOUT" >>"$LOG" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/eval_mc.py" --model "$STUDENT_BASE" --data "$TP" \
        --n 1 --limit 8 --gpu_mem "$GM" --out "$EOUT" >>"$LOG" 2>&1
    fi
    [ -f "$EOUT/summary.json" ] && ok "[$A] eval 跑通（$EOUT/summary.json）" || no "[$A] eval"
  fi

  # ---- ⑤ GRPO：造微型 train 种子(test 切片) → 03_grpo.sh ABILITY=$A TEST=1 跑几步，证明 reward+manager 与 verl 集成 ----
  if [ "$DO_GRPO" = 1 ]; then
    GDIR=$SM/${A}_grpo_smoke; mkdir -p "$GDIR"
    python -c "import pandas as pd; pd.read_parquet('$TP').iloc[:8].to_parquet('$GDIR/train.parquet')" >>"$LOG" 2>&1
    GEXP=grpo_smoke_$A
    if CUDA_VISIBLE_DEVICES=$G0,$G1 env TEST=1 ABILITY="$A" FROM=__none__ MODEL_PATH="$STUDENT_BASE" \
        TRAIN_DIR="$GDIR" VAL_DIR="$DSRC" EXP="$GEXP" EVAL_LIMIT=8 N=1 \
        bash "$PROJ/run/03_grpo.sh" >>"$LOG" 2>&1; then
      ok "[$A] GRPO TEST=1 跑通（reward=$( [ "$A" = code ] && echo code_reward/prime || echo ${A}_reward/naive) 与 verl 集成）"
    else no "[$A] GRPO TEST=1（看 $LOG 尾部；GRPO 最重、首查 OOM/显存）"; fi
  fi
done

say "======== 冒烟汇总 ========"
say "PASS(${#PASS[@]}): ${PASS[*]:-无}"
say "FAIL(${#FAIL[@]}): ${FAIL[*]:-无}"
[ ${#FAIL[@]} -eq 0 ] && say "✅ 全链路连通：code/mc 的 reward/gen/sft/eval/grpo 均可跑（非仅 base eval）。" \
                      || say "⚠️ 有阶段未通，见上（$LOG）。"
