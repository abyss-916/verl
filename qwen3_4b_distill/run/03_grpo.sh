#!/usr/bin/env bash
# GRPO 后训练 + grpo_eval（从 SFT ckpt 起）。2×3090 上单次量级数天——tmux/nohup 后台跑，早启动。
# 首次务必： TEST=1 bash run/03_grpo.sh   验证不 OOM 再正式跑。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/run"; exec > >(tee -a "$LOGS/run/$(basename "$0" .sh).log") 2>&1  # 全部输出落 $LOGS/run/

FROM=${FROM:-sft_standard_cot}                    # 起点 SFT 实验名
EXP=${EXP:-grpo_from_${FROM}}
MODEL_PATH=${MODEL_PATH:-$(latest_hf "$CKPT/$FROM")}   # SFT 后的 HF 权重目录

# GRPO：prompt 用 SEED(MATH train)，reward=math-verify；held-out 评测另做
MODEL_PATH="$MODEL_PATH" DATA_DIR="$SEED_DIR" EXP="$EXP" bash "$PROJ/train/grpo.sh"

# grpo_eval 在 held-out（OlymMATH）+ 与 base/SFT 三方对照
python "$PROJ/eval/eval_math.py" \
  --model "$(latest_hf "$CKPT/$EXP")" --data "$EVAL_DIR/test.parquet" \
  --n "${N:-8}" --out "$LOGS/eval/olymmath_$EXP"

echo "GRPO 完成：$LOGS/eval/olymmath_$EXP/summary.json（对比 base / sft_$FROM 得三方结果）"
