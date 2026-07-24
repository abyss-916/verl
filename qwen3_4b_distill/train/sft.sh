#!/usr/bin/env bash
# off-policy 序列蒸馏 SFT | Qwen3-4B | 2×3090 | verl sft_trainer
# 改编自 verl/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh
# 用法（服务器）：
#   EXP=sft_standard_cot DATA_DIR=/data/liujiachen/datasets/distill/standard_cot bash train/sft.sh
#   TEST=1 EXP=... DATA_DIR=... bash train/sft.sh      # 极小配置先验证不 OOM / 不缺库
#
# ── 加速：本机 glibc 2.31 装不上 flash-attn 预编译轮子，故全程不依赖它 ──
#   · attn 用 sdpa（Ampere+bf16 下 sdpa 本就调 FlashAttention-2 内核，注意力不慢）；
#   · use_remove_padding=false + pad_mode=right：变长打包在 CUDA 上硬依赖 flash_attn.bert_padding，
#     没有纯 torch 兜底，故关掉走 padding（按长度分桶把浪费压小；我们数据 p50≈2.5K/p95≈6K 本就不长）；
#   · 提速改走 Triton 融合算子（不碰 glibc）：use_liger（SwiGLU/RMSNorm/RoPE）+ use_fused_kernels
#     （融合 linear cross-entropy，Qwen3 词表 15 万，省下 [seq×150k] logits 的几 G 显存 —— 比打包更实用）。
#   · 这些都不改数值结果，纯提速；想单独关：USE_LIGER=false / USE_FUSED=false。
#   ⚠️ 不装 flash-attn 时 ulysses SP 也别开（SP_SIZE 保持 1），序列并行同样依赖变长路径。
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/liujiachen/models/Qwen3-4B}
DATA_DIR=${DATA_DIR:-/data/liujiachen/datasets/distill/standard_cot}
EXP=${EXP:-sft_standard_cot}
SAVE=${SAVE:-/data/liujiachen/checkpoints/$EXP}
NPROC=${NPROC:-2}
SP_SIZE=${SP_SIZE:-1}       # 2×3090 先不开序列并行；长 CoT 显存紧可设 2
USE_PEFT=${USE_PEFT:-0}     # 默认全参 SFT；显存不够改 1 走 LoRA
LR=${LR:-1e-5}

if [ "${TEST:-0}" = "1" ]; then
  MB=1; MAXLEN=1024; EPOCHS=1
else
  MB=${MB:-2}; MAXLEN=${MAXLEN:-4096}; EPOCHS=${EPOCHS:-3}
fi

extra=()
if [ "$USE_PEFT" = "1" ]; then
  extra+=(model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear)
fi

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
  -m verl.trainer.sft_trainer \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/val.parquet \
  data.messages_key=messages \
  data.ignore_input_ids_mismatch=True \
  data.micro_batch_size_per_gpu=$MB \
  data.max_length=$MAXLEN \
  data.truncation=right \
  data.pad_mode=right \
  optim.lr=$LR \
  engine=fsdp \
  engine.ulysses_sequence_parallel_size=$SP_SIZE \
  model.path=$MODEL_PATH \
  model.use_remove_padding=false \
  model.override_config.attn_implementation=sdpa \
  model.use_liger=${USE_LIGER:-true} \
  model.use_fused_kernels=${USE_FUSED:-true} \
  trainer.default_local_dir=$SAVE \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.project_name=qwen3-4b-distill \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  trainer.total_epochs=$EPOCHS \
  "${extra[@]}" "$@"

# Liger 需先装：pip install liger-kernel（纯 Triton 轮子，不编译、不碰 glibc）。装前先 TEST=1 冒烟。
# 三法（standard_cot / reverse / question_aug）分别改 EXP 与 DATA_DIR，其余保持一致以公平对比。
