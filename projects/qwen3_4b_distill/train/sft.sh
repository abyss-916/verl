#!/usr/bin/env bash
# off-policy 序列蒸馏 SFT | Qwen3-4B-Base | 2×3090 | verl sft_trainer
# 改编自 verl/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh
# 用法（服务器）：
#   EXP=sft_standard_cot DATA_DIR=/data/liujiachen/datasets/distill/standard_cot bash train/sft.sh
#   TEST=1 EXP=... DATA_DIR=... bash train/sft.sh      # 极小配置先验证不 OOM
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/liujiachen/models/Qwen3-4B-Base}
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
  optim.lr=$LR \
  engine=fsdp \
  engine.ulysses_sequence_parallel_size=$SP_SIZE \
  model.path=$MODEL_PATH \
  model.use_remove_padding=true \
  trainer.default_local_dir=$SAVE \
  trainer.project_name=qwen3-4b-distill \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  trainer.total_epochs=$EPOCHS \
  "${extra[@]}" "$@"

# 省显存可选：追加 model.use_liger=true（Liger kernel）。
# 三法（standard_cot / reverse / question_aug）分别改 EXP 与 DATA_DIR，其余保持一致以公平对比。
