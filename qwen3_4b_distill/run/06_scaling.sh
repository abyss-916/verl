#!/usr/bin/env bash
# 数据 scaling 研究（优秀杠杆，性价比最高）：standard_cot 在不同种子规模各训一次，
# 看 accuracy–数据量 + Pass@1(稳定性)/pass@k(能力) 曲线，检验 Xwin-Math"扩数据主要抬稳定性"。
# 全 SFT、便宜，2×3090 友好。首轮可先 SIZES="200 500" TEST=1 跑通。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"

SIZES=${SIZES:-"500 2000 7500"}
for N in $SIZES; do
  echo "===== scaling: $N 种子 ====="
  python "$PROJ/distill/generate_cot.py" --method standard_cot \
    --seed "$SEED_DIR/train.parquet" --teacher "$TEACHER" \
    --out "$DATA/distill/scale_$N" --tp "${TP:-2}" --limit "$N"

  EXP="sft_scale_$N" DATA_DIR="$DATA/distill/scale_$N" bash "$PROJ/train/sft.sh"

  python "$PROJ/eval/eval_math.py" \
    --model "$(latest_hf "$CKPT/sft_scale_$N")" --data "$EVAL_DIR/test.parquet" \
    --n "${N_EVAL:-8}" --out "$LOGS/eval/scale_$N"
done

echo "scaling 完成：汇总各 $LOGS/eval/scale_*/summary.json 的 pass@1 与 pass@k 画曲线"
