#!/usr/bin/env bash
# 任务一：数据接入 + Qwen3-4B base 评测（与论文锚点对齐：OlymMATH HARD-EN≈13.9）
set -xeuo pipefail
source "$(dirname "$0")/env.sh"

# 1) SEED（训练/蒸馏种子，MATH train）→ RL parquet（train + test）
python "$PROJ/data_preprocess/prepare_math.py" \
  --hf "$SEED_HF" --subset "$SEED_SUBSET" --out "$SEED_DIR" --data_source math_seed

# 2) EVAL（held-out，OlymMATH）→ RL parquet（仅评测用）
python "$PROJ/data_preprocess/prepare_math.py" \
  --hf "$EVAL_HF" --subset "$EVAL_SUBSET" --out "$EVAL_DIR" --data_source olymmath

# 3) base eval 在 held-out（thinking，avg@N），对齐锚点（OlymMATH HARD-EN≈13.9）
python "$PROJ/eval/eval_math.py" \
  --model "$STUDENT_BASE" --data "$EVAL_DIR/test.parquet" \
  --n "${N:-8}" --out "$LOGS/eval/olymmath_base"

# 4)（可选）LiveCodeBench 数据；code 评测需 sandbox / 官方 harness（见 train/grpo.sh、项目 doc/RUNBOOK.md）
# python "$PROJ/data_preprocess/prepare_code.py" --version release_v5 --out "$DATA/livecodebench"

echo "任务一完成：SEED=$SEED_DIR（train），EVAL=$EVAL_DIR（held-out）；base 分见 $LOGS/eval/olymmath_base/summary.json"
