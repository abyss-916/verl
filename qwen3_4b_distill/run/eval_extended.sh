#!/usr/bin/env bash
# 扩展 benchmark base eval（全部 held-out）：code=LiveCodeBench / mc=MMLU-Pro+SuperGPQA / math=AIME。
# 用法：bash run/eval_extended.sh；用 LIMIT 控采样量，可覆盖 MODEL/CODE_LIMIT/AIME_N。
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

# ── mc：SuperGPQA ──
python "$PROJ/data_preprocess/prepare_mc.py" --hf "$SUPERGPQA_HF" --subset default \
  --out "$DATA/supergpqa" --data_source supergpqa
python "$PROJ/eval/eval_mc.py" --model "$M" --data "$DATA/supergpqa/test.parquet" \
  --n 1 --limit "$LIM" --out "$LOGS/eval/supergpqa_base"

# ── math：AIME（设了 AIME_HF 才跑）──
if [ -n "${AIME_HF:-}" ]; then
  python "$PROJ/data_preprocess/prepare_math.py" --hf "$AIME_HF" --subset "${AIME_SUBSET:-}" \
    --out "$DATA/aime" --data_source aime
  python "$PROJ/eval/eval_math.py" --model "$M" --data "$DATA/aime/test.parquet" \
    --n "${AIME_N:-32}" --out "$LOGS/eval/aime_base"
fi

echo "扩展 benchmark base eval 完成，见 $LOGS/eval/{lcb,mmlu_pro,supergpqa,aime}_base"
