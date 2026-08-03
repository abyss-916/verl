#!/usr/bin/env bash
# TRL GRPO 启动器（单进程，绕开 verl 的 Ray/colocation 段错）。详见 trl_grpo.py 顶部。
# 需要：pip 已装 trl；reward 用 verl 的 math-verify（verl 可 import）。
#
# ① 冒烟（HF 生成，不需 vLLM server；~2-3min 只验管线：数据/reward/LoRA/训练循环跑通）：
#     CUDA_VISIBLE_DEVICES=0 USE_VLLM=0 MAXSTEPS=3 RESP=512 PBS=5 GRAD_ACCUM=2 N=5 \
#       MODEL_PATH=$CKPT/sft_omni_standard_cot_merged OUT=$CKPT/grpo_trl_smoke \
#       bash train/trl_grpo.sh
#
# ② 正式-A（vLLM server，快、可长；另开终端占 GPU1 起 server）：
#     CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model $CKPT/sft_omni_standard_cot_merged \
#       --gpu_memory_utilization 0.85 --max_model_len 17408 --enforce_eager True --port 8000
#   训练（GPU0，连 server）：
#     CUDA_VISIBLE_DEVICES=0 USE_VLLM=1 RESP=16384 N=5 LR=1e-5 EPOCHS=2 \
#       MODEL_PATH=$CKPT/sft_omni_standard_cot_merged OUT=$CKPT/grpo_trl_omni_standard \
#       bash train/trl_grpo.sh
#
# ② 正式-B（若 vLLM server 因版本不兼容起不来 → HF 生成，压短 RESP 保可行）：
#     CUDA_VISIBLE_DEVICES=0 USE_VLLM=0 RESP=4096 N=5 LR=1e-5 EPOCHS=2 \
#       MODEL_PATH=$CKPT/sft_omni_standard_cot_merged OUT=$CKPT/grpo_trl_omni_standard \
#       bash train/trl_grpo.sh
set -euo pipefail
source "$(dirname "$0")/../run/env.sh"
export MODEL_PATH=${MODEL_PATH:-$CKPT/sft_omni_standard_cot_merged}
python "$(dirname "$0")/trl_grpo.py"
