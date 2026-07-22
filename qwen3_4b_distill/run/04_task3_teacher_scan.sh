#!/usr/bin/env bash
# 任务三：teacher 差异研究（off-policy 强度扫描）。同 student(4B) × 不同 teacher × standard_cot。
# 本地能跑 Qwen3-8B；更强 teacher(Qwen3-32B-4bit / API) 按需加进 TEACHERS。
# on-policy 对照见 train/opd.sh（stretch）。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"

SEED="$SEED_DIR/train.parquet"     # 蒸馏种子 = MATH train
TEACHERS=${TEACHERS:-"Qwen3-8B"}   # 例：TEACHERS="Qwen3-8B Qwen3-32B-4bit"

for T in $TEACHERS; do
  echo "===== teacher: $T ====="
  python "$PROJ/distill/generate_cot.py" --method standard_cot \
    --seed "$SEED" --teacher "$MODELS/$T" --out "$DATA/distill/teacher_$T" \
    --tp "${TP:-2}" ${LIMIT:+--limit $LIMIT}

  python "$PROJ/metrics/data_metrics.py" \
    --data "$DATA/distill/teacher_$T/train.parquet" --model "$STUDENT_BASE" \
    --limit "${MLIM:-300}" --out "$LOGS/metrics_teacher_$T.json"

  EXP="sft_teacher_$T" DATA_DIR="$DATA/distill/teacher_$T" bash "$PROJ/train/sft.sh"

  python "$PROJ/eval/eval_math.py" \
    --model "$(latest_hf "$CKPT/sft_teacher_$T")" --data "$EVAL_DIR/test.parquet" \
    --n "${N:-8}" --out "$LOGS/eval/teacher_$T"
done

echo "任务三(off-policy)完成：比较不同 teacher 的 metrics_teacher_*.json 与 eval/teacher_*，验 H2（更强≠更适合）"
echo "on-policy 对照： EXP=opd_4b_from_8b DATA_DIR=$DATA/olymmath bash train/opd.sh  （stretch，可能 OOM）"

# ── 双轴扩展：API teacher（本地跑不了 32B/235B，借 API 补强度轴/家族轴，仅 off-policy）──
# 强度轴（Qwen 同家族大模型，DashScope）：
#   export DASHSCOPE_API_KEY=...
#   python "$PROJ/distill/generate_cot.py" --method standard_cot --seed "$SEED_DIR/train.parquet" \
#     --teacher_type api --api_base "$QWEN_API_BASE" --api_model qwen3-235b-a22b --api_key_env DASHSCOPE_API_KEY \
#     --out "$DATA/distill/teacher_qwen235b" --workers 16
# 家族/风格轴（DeepSeek-V4 reasoning teacher；v4-flash 带 thinking 便宜，或 v4-pro 最强）：
#   export DEEPSEEK_API_KEY=...
#   python "$PROJ/distill/generate_cot.py" --method standard_cot --seed "$SEED_DIR/train.parquet" \
#     --teacher_type api --api_base "$DEEPSEEK_API_BASE" --api_model deepseek-v4-flash --api_key_env DEEPSEEK_API_KEY \
#     --out "$DATA/distill/teacher_dsv4" --workers 8
# 之后对 teacher_qwen235b / teacher_r1 照样 sft + eval + metrics，与本地 8B 对比。
