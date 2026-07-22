#!/usr/bin/env bash
# 扩展 benchmark base eval（加分，全部 held-out）：
#   code = LiveCodeBench（需测试用例格式核对，见 reward/code_reward.py）
#   mc   = MMLU-Pro / SuperGPQA（科学推理，开放替代 GPQA）
#   math = AIME（须 avg@k；设了 $AIME_HF 才跑）
# 规模大，用 LIMIT 控采样。首轮先 base，训练后再评 SFT/GRPO ckpt。
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

# ── math：AIME（设了 AIME_HF 才跑；30 题须 avg@k）──
if [ -n "${AIME_HF:-}" ]; then
  python "$PROJ/data_preprocess/prepare_math.py" --hf "$AIME_HF" --subset "${AIME_SUBSET:-}" \
    --out "$DATA/aime" --data_source aime
  python "$PROJ/eval/eval_math.py" --model "$M" --data "$DATA/aime/test.parquet" \
    --n "${AIME_N:-32}" --out "$LOGS/eval/aime_base"
fi

echo "扩展 benchmark base eval 完成，见 $LOGS/eval/{lcb,mmlu_pro,supergpqa,aime}_base"
