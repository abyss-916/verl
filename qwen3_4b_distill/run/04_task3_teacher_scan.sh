#!/usr/bin/env bash
# 任务三：teacher 差异研究（off-policy 强度扫描）。同 student(4B) × 不同 teacher × standard_cot。
# 本地能跑 Qwen3-8B；更强 teacher(Qwen3-32B-4bit / API) 按需加进 TEACHERS。
# on-policy 对照见 train/opd.sh（stretch）。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/run"; exec > >(tee -a "$LOGS/run/$(basename "$0" .sh).log") 2>&1  # 全部输出落 $LOGS/run/

SEED="$SEED_DIR/train.parquet"     # 蒸馏种子 = MATH train
# teacher 均本地免费（名字=$MODELS 下权重目录）：强度轴 8B/14B/32B-AWQ + 专精 Math-7B。
# TP：8B/7B=1，14B(bf16)=2，32B 用 AWQ/int4 量化版单卡。按需下到 $MODELS（见 doc/下载部署清单.md）。
TEACHERS=${TEACHERS:-"Qwen3-8B Qwen3-14B Qwen3-32B-AWQ Qwen2.5-Math-7B-Instruct"}

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

# ── API teacher（仅 off-policy）：强度轴主走本地(上面 TEACHERS)，API 只补两处 ──
# (可选) 强度轴顶点 Qwen3-235B —— 免费额度：阿里百炼(新用户2000万token) 或 魔搭(2000次/天,大模型~200/天)，故 --limit 小：
#   export DASHSCOPE_API_KEY=...
#   python "$PROJ/distill/generate_cot.py" --method standard_cot --seed "$SEED_DIR/train.parquet" --limit 500 \
#     --teacher_type api --api_base "$QWEN_API_BASE" --api_model qwen3-235b-a22b --api_key_env DASHSCOPE_API_KEY \
#     --out "$DATA/distill/teacher_qwen235b" --workers 8
# reasoning 轴 DeepSeek-V4（不同家族，**唯一付费**，便宜；v4-flash 带 thinking）：
#   export DEEPSEEK_API_KEY=...
#   python "$PROJ/distill/generate_cot.py" --method standard_cot --seed "$SEED_DIR/train.parquet" --limit 500 \
#     --teacher_type api --api_base "$DEEPSEEK_API_BASE" --api_model deepseek-v4-flash --api_key_env DEEPSEEK_API_KEY \
#     --out "$DATA/distill/teacher_dsv4" --workers 8
# 之后对 teacher_qwen235b / teacher_dsv4 照样 sft + eval + metrics + slice_eval，与本地 teacher 对比。
