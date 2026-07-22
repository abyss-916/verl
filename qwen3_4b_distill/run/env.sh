#!/usr/bin/env bash
# 公共路径与默认（被 run/*.sh source）。按服务器实际改。
export PROJ=${PROJ:-/data/liujiachen/verl/qwen3_4b_distill}
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
export WANDB_MODE=${WANDB_MODE:-offline}   # 默认离线（免登录、不卡）；想上传设 WANDB_MODE=online 并先 wandb login
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$RAY_TMPDIR" "$WANDB_DIR" 2>/dev/null || true

export STUDENT_BASE=${STUDENT_BASE:-$MODELS/Qwen3-4B-Base}
export TEACHER=${TEACHER:-$MODELS/Qwen3-8B}

# ── 数据角色（严格分离，服务高质量课题）──
# SEED：训练/蒸馏种子 + GRPO prompt（大数学训练集）。MATH-lighteval train ~7500，服务 task2 scaling。
export SEED_HF=${SEED_HF:-DigitalLearningGmbH/MATH-lighteval}
export SEED_SUBSET=${SEED_SUBSET:-default}
export SEED_DIR=${SEED_DIR:-$DATA/math_seed}
# EVAL：held-out 评测（任务一选型 OlymMATH），绝不进训练。
export EVAL_HF=${EVAL_HF:-RUC-AIBOX/OlymMATH}
export EVAL_SUBSET=${EVAL_SUBSET:-en-hard}
export EVAL_DIR=${EVAL_DIR:-$DATA/olymmath}

# 取某训练输出目录下"最新 global_step 的 HF 权重目录"（供 vLLM eval / 从 SFT 起 GRPO 加载）
latest_hf() { ls -d "$1"/global_step_*/huggingface 2>/dev/null | sort -V | tail -1; }
export -f latest_hf

mkdir -p "$DATA" "$CKPT" "$LOGS/eval" 2>/dev/null || true
echo "[env] PROJ=$PROJ STUDENT=$STUDENT_BASE TEACHER=$TEACHER"
