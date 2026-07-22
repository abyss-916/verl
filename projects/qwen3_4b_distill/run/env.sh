#!/usr/bin/env bash
# 公共路径与默认（被 run/*.sh source）。按服务器实际改。
export PROJ=${PROJ:-/data/liujiachen/verl/projects/qwen3_4b_distill}
export MODELS=${MODELS:-/data/liujiachen/models}
export DATA=${DATA:-/data/liujiachen/datasets}
export CKPT=${CKPT:-/data/liujiachen/checkpoints}
export LOGS=${LOGS:-/data/liujiachen/logs}

# ⚠️ 系统盘 / 仅 25G：所有缓存/临时/日志强制重定向到 /data，别写 ~/.cache 或 /tmp
export HF_HOME=${HF_HOME:-/data/liujiachen/hf}                       # HF 模型/数据/token 缓存
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/data/liujiachen/modelscope}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data/liujiachen/.cache}     # torch/triton/vllm 默认走这
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}
export TMPDIR=${TMPDIR:-/data/liujiachen/tmp}                        # ray/临时文件，避开系统盘 /tmp
export RAY_TMPDIR=${RAY_TMPDIR:-/data/liujiachen/tmp/ray}
export WANDB_DIR=${WANDB_DIR:-$LOGS/wandb}
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$RAY_TMPDIR" "$WANDB_DIR" 2>/dev/null || true

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
