#!/usr/bin/env bash
# 任务一：数据接入 + Qwen3-4B base 评测（与论文锚点对齐：OlymMATH HARD-EN≈13.9）
set -xeuo pipefail
source "$(dirname "$0")/env.sh"

# 1) OlymMATH → RL parquet
python "$PROJ/data_preprocess/prepare_math.py" \
  --hf "$MATH_HF" --subset "$MATH_SUBSET" \
  --out "$DATA/olymmath" --data_source olymmath

# 2) base eval（thinking，avg@N）
python "$PROJ/eval/eval_math.py" \
  --model "$STUDENT_BASE" --data "$DATA/olymmath/test.parquet" \
  --n "${N:-8}" --out "$LOGS/eval/olymmath_base"

# 3)（可选）LiveCodeBench 数据；code 评测需 sandbox / 官方 harness（见 train/grpo.sh、RUNBOOK.md）
# python "$PROJ/data_preprocess/prepare_code.py" --version release_v5 --out "$DATA/livecodebench"

echo "任务一完成：base 分见 $LOGS/eval/olymmath_base/summary.json，对齐 doc/Qwen3报告_精读笔记.md 锚点"
