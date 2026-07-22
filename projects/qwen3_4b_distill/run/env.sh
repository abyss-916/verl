#!/usr/bin/env bash
# 公共路径与默认（被 run/*.sh source）。按服务器实际改。
export PROJ=${PROJ:-/data/liujiachen/verl/projects/qwen3_4b_distill}
export MODELS=${MODELS:-/data/liujiachen/models}
export DATA=${DATA:-/data/liujiachen/datasets}
export CKPT=${CKPT:-/data/liujiachen/checkpoints}
export LOGS=${LOGS:-/data/liujiachen/logs}

export STUDENT_BASE=${STUDENT_BASE:-$MODELS/Qwen3-4B-Base}
export TEACHER=${TEACHER:-$MODELS/Qwen3-8B}

# math 主 benchmark（任务一选型：OlymMATH）
export MATH_HF=${MATH_HF:-RUC-AIBOX/OlymMATH}
export MATH_SUBSET=${MATH_SUBSET:-en-hard}   # OlymMATH config（小写）：en-hard/en-easy/zh-hard/zh-easy/lean

# 取某训练输出目录下"最新 global_step 的 HF 权重目录"（供 vLLM eval / 从 SFT 起 GRPO 加载）
latest_hf() { ls -d "$1"/global_step_*/huggingface 2>/dev/null | sort -V | tail -1; }
export -f latest_hf

mkdir -p "$DATA" "$CKPT" "$LOGS/eval" 2>/dev/null || true
echo "[env] PROJ=$PROJ STUDENT=$STUDENT_BASE TEACHER=$TEACHER"
