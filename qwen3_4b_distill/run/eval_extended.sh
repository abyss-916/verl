#!/usr/bin/env bash
# 扩展 benchmark base eval（held-out）：code=LiveCodeBench / mc=MMLU-Pro。
# 用法：bash run/eval_extended.sh；用 LIMIT 控采样量，可覆盖 MODEL/CODE_LIMIT。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/run"; exec > >(tee -a "$LOGS/run/$(basename "$0" .sh).log") 2>&1  # 全部输出落 $LOGS/run/
M=${MODEL:-$STUDENT_BASE}
LIM=${LIMIT:-200}

# ── code：LiveCodeBench ──
python "$PROJ/data_preprocess/prepare_code.py" --version "$CODE_VERSION" --out "$DATA/livecodebench" || true
python "$PROJ/eval/eval_code.py" --model "$M" --data "$DATA/livecodebench/test.parquet" \
  --n 1 --limit "${CODE_LIMIT:-50}" --out "$LOGS/eval/lcb_base" || true

# ── mc：MMLU-Pro ──
python "$PROJ/data_preprocess/prepare_mc.py" --hf "$MMLU_PRO_HF" --subset default \
  --out "$DATA/mmlu_pro" --data_source mmlu_pro
python "$PROJ/eval/eval_mc.py" --model "$M" --data "$DATA/mmlu_pro/test.parquet" \
  --n 1 --limit "$LIM" --out "$LOGS/eval/mmlu_pro_base"

echo "扩展 benchmark base eval 完成，见 $LOGS/eval/{lcb,mmlu_pro}_base"
