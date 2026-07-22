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
export MATH_SUBSET=${MATH_SUBSET:-EN-HARD}

mkdir -p "$DATA" "$CKPT" "$LOGS/eval" 2>/dev/null || true
echo "[env] PROJ=$PROJ STUDENT=$STUDENT_BASE TEACHER=$TEACHER"
