#!/usr/bin/env bash
# off-policy 序列蒸馏 SFT | Qwen3-4B | 2×3090 | verl sft_trainer
# 改编自 verl/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh
# 用法（服务器）：
#   EXP=sft_standard_cot DATA_DIR=/data/liujiachen/datasets/distill/standard_cot bash train/sft.sh
#   TEST=1 EXP=... DATA_DIR=... bash train/sft.sh      # 极小配置先验证不 OOM / 不缺库
#
# ── 加速（本机 glibc 2.31）──
#   flash-attn 已用预编译轮子装上（mjun0812 flash_attn 2.8.3+cu128torch2.9-cp312，manylinux_2_28，
#   glibc≥2.28 兼容，绕开官方轮子的 GLIBC_2.32 坑）。故默认走 flash-attn 最快路径：
#   · USE_FLASH=1（默认）：attn=flash_attention_2 + use_remove_padding=true + pad_mode=no_padding
#     （变长打包，省 padding，长短不一时 1.5–2×）。
#   · USE_FLASH=0（回退）：attn=sdpa + use_remove_padding=false + pad_mode=right（不依赖 flash-attn；
#     若 flash-attn 算子在本机跑挂就用这个。sdpa 在 Ampere 本就是 FlashAttention-2 内核，注意力不慢）。
#   · 叠加的 Triton 融合算子（不碰 flash-attn）：use_liger（SwiGLU/RMSNorm/RoPE）+ use_fused_kernels
#     （融合 linear cross-entropy，Qwen3 词表 15 万，省 [seq×150k] logits 的几 G 显存）。两档都开。
#   · 以上都不改数值结果，纯提速；单独关：USE_LIGER=false / USE_FUSED=false。
#   ⚠️ ulysses SP（SP_SIZE>1）依赖变长路径，仅在 USE_FLASH=1 时可开。
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
  # ⚠️ MAXLEN 必须覆盖蒸馏数据长度，否则 data.truncation=right 会把长 CoT 右截断——
  #    保留开头、丢掉结尾 \boxed 答案，变成"没有答案的半截 CoT"坏样本(比不训还糟)。
  #    正式训前按该方法 gen_stats.json：MAXLEN 尽量设 ≥ tok_max(而非仅 p99)；若 tok_max 太大装不下，
  #    宁可造数据后【预删】超长行，也别让它们被截断进训练。显存不够降 MB(micro-batch)，别降 MAXLEN。
  #    16384 覆盖实测 p99≈11.6K，仅作缺省；见 gen_stats.json 再定。
  MB=${MB:-2}; MAXLEN=${MAXLEN:-16384}; EPOCHS=${EPOCHS:-3}
fi

extra=()
if [ "$USE_PEFT" = "1" ]; then
  extra+=(model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear)
fi

# 加速开关：flash-attn 变长打包(默认) vs sdpa+padding 回退；Triton 融合算子两档都开
accel=(model.use_liger=${USE_LIGER:-true} model.use_fused_kernels=${USE_FUSED:-true})
if [ "${USE_FLASH:-1}" = "1" ]; then
  accel+=(data.pad_mode=no_padding model.use_remove_padding=true)   # attn 默认即 flash_attention_2
else
  accel+=(data.pad_mode=right model.use_remove_padding=false model.override_config.attn_implementation=sdpa)
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
  optim.lr=$LR \
  engine=fsdp \
  engine.ulysses_sequence_parallel_size=$SP_SIZE \
  model.path=$MODEL_PATH \
  trainer.default_local_dir=$SAVE \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.project_name=qwen3-4b-distill \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  trainer.total_epochs=$EPOCHS \
  "${accel[@]}" "${extra[@]}" "$@"

# Liger 需先装：pip install liger-kernel（纯 Triton 轮子，不编译、不碰 glibc）。首跑先 TEST=1 冒烟，
# 同时验证 flash-attn 算子在本机能跑；若冒烟报 flash-attn 相关错，用 USE_FLASH=0 回退再跑。
# 三法（standard_cot / reverse / question_aug）分别改 EXP 与 DATA_DIR，其余保持一致以公平对比。
