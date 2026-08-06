#!/usr/bin/env bash
# 扩展 benchmark base eval（held-out）：code=LiveCodeBench / mc=MMLU-Pro。
# 用法：bash run/eval_extended.sh；可覆盖 MODEL/LIMIT/CODE_LIMIT/GPU_MEM（共卡调低）。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/run"; exec > >(tee -a "$LOGS/run/$(basename "$0" .sh).log") 2>&1  # 全部输出落 $LOGS/run/
M=${MODEL:-$STUDENT_BASE}
LIM=${LIMIT:-200}
GPU_MEM=${GPU_MEM:-0.8}   # vLLM 显存占比；与他人共卡时调低

# ── code：LiveCodeBench ──
python "$PROJ/data_preprocess/prepare_code.py" --version "$CODE_VERSION" --out "$DATA/livecodebench" || true
python "$PROJ/eval/eval_code.py" --model "$M" --data "$DATA/livecodebench/test.parquet" \
  --n 1 --limit "${CODE_LIMIT:-50}" --gpu_mem "$GPU_MEM" --out "$LOGS/eval/lcb_base" || true

# ── mc：MMLU-Pro ──
python "$PROJ/data_preprocess/prepare_mc.py" --hf "$MMLU_PRO_HF" --subset default \
  --out "$DATA/mmlu_pro" --data_source mmlu_pro
python "$PROJ/eval/eval_mc.py" --model "$M" --data "$DATA/mmlu_pro/test.parquet" \
  --n 1 --limit "$LIM" --gpu_mem "$GPU_MEM" --out "$LOGS/eval/mmlu_pro_base"

echo "扩展 benchmark base eval 完成，见 $LOGS/eval/{lcb,mmlu_pro}_base"
