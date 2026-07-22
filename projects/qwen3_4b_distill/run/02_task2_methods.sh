#!/usr/bin/env bash
# 任务二：三种蒸馏方法 → 造数据 + 数据度量 + SFT + sft_eval + 三法对比。
# 公平对比：同 teacher / 同 student / 同格式 / 同预算（LIMIT 控种子数）。
# 首轮建议：LIMIT=200 TEST=1 bash run/02_task2_methods.sh  先跑通全链路。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"

SEED="$DATA/olymmath/train.parquet"
METHODS=${METHODS:-"standard_cot reverse question_aug"}

for M in $METHODS; do
  echo "===== 方法: $M ====="
  # 1) teacher 造数据
  python "$PROJ/distill/generate_cot.py" --method "$M" \
    --seed "$SEED" --teacher "$TEACHER" --out "$DATA/distill/$M" \
    --tp "${TP:-2}" --n "${GEN_N:-1}" ${LIMIT:+--limit $LIMIT}

  # 2) 数据度量（student 视角 PPL/IFD）
  python "$PROJ/metrics/data_metrics.py" \
    --data "$DATA/distill/$M/train.parquet" --model "$STUDENT_BASE" \
    --limit "${MLIM:-300}" --out "$LOGS/metrics_$M.json"

  # 3) SFT（首轮 TEST=1）
  EXP="sft_$M" DATA_DIR="$DATA/distill/$M" bash "$PROJ/train/sft.sh"

  # 4) sft_eval
  python "$PROJ/eval/eval_math.py" \
    --model "$CKPT/sft_$M" --data "$DATA/olymmath/test.parquet" \
    --n "${N:-8}" --out "$LOGS/eval/olymmath_sft_$M"
done

# 5) 三法数据对比表
python "$PROJ/metrics/compare_methods.py" --out "$LOGS/compare_methods.md" \
  --in standard_cot="$LOGS/metrics_standard_cot.json" \
       reverse="$LOGS/metrics_reverse.json" \
       question_aug="$LOGS/metrics_question_aug.json" || true

echo "任务二完成：对比表 $LOGS/compare_methods.md，各法 sft_eval 见 $LOGS/eval/"
echo "注：SFT ckpt 若非 HF 目录，eval 前需指到具体 checkpoint 子目录或做 model_merger（见 RUNBOOK.md）"
