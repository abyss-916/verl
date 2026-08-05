#!/usr/bin/env bash
# off-policy 序列蒸馏 SFT | Qwen3-4B | 2×3090 | 改编自 verl/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh
# 用法：EXP=sft_standard_cot DATA_DIR=$DATA/distill/standard_cot bash train/sft.sh
#   TEST=1 EXP=... DATA_DIR=... bash train/sft.sh    # 小配置先验证不 OOM
# 加速开关：USE_FLASH（flash-attn 变长打包，默认 1；本机不可用则设 0 回退 sdpa）、USE_LIGER、USE_FUSED。
set -xeuo pipefail
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"   # 抗显存碎片

MODEL_PATH=${MODEL_PATH:-$MODELS/Qwen3-4B}
DATA_DIR=${DATA_DIR:-$DATA/distill/standard_cot}
EXP=${EXP:-sft_standard_cot}
SAVE=${SAVE:-$CKPT/$EXP}   # CKPT 由 env.sh 继承（export）
NPROC=${NPROC:-2}
SP_SIZE=${SP_SIZE:-1}      # 序列并行度（长 CoT 显存紧可设 2）
USE_PEFT=${USE_PEFT:-1}    # LoRA（大显存机全参改 0）
LR=${LR:-2e-4}            # LoRA 学习率

if [ "${TEST:-0}" = "1" ]; then
  MB=1; MAXLEN=1024; EPOCHS=1; TRUNC=right
else
  # MAXLEN 须覆盖蒸馏数据长度；truncation=error 令超长行直接报错，不静默截断
  MB=${MB:-2}; MAXLEN=${MAXLEN:-40960}; EPOCHS=${EPOCHS:-5}; TRUNC=error
fi
MAX_TOKENS=${MAX_TOKENS:-$MAXLEN}   # 动态批每卡 token 预算（须 ≥ 最长样本）
TBS=${TBS:-32}                      # 全局 batch（三法一致以公平对比）

# torchao 8-bit 优化器（省显存；大显存机 OPT8BIT=0 走 fp32）
OPT8BIT=${OPT8BIT:-1}
OPT_NAME=${OPT_NAME:-AdamW8bit}
OPT_IMPL=${OPT_IMPL:-torchao.optim}
opt_args=()
[ "$OPT8BIT" = "1" ] && opt_args=(optim.optimizer="$OPT_NAME" optim.optimizer_impl="$OPT_IMPL")

# bf16 主权重 + 激活 offload（2×3090 装下 4B 的主要手段；大显存机 MODEL_DTYPE=fp32 ACT_OFFLOAD=false）
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
ACT_OFFLOAD=${ACT_OFFLOAD:-true}
mem_args=(engine.model_dtype="$MODEL_DTYPE" model.enable_activation_offload="$ACT_OFFLOAD")

# 整段渲染保 Qwen3 <think>（CUSTOM_DS=0 退回 verl 原生逐轮分词）
CUSTOM_DS=${CUSTOM_DS:-1}
CUSTOM_DS_PATH=${CUSTOM_DS_PATH:-$PROJ/train/whole_conv_sft_dataset.py}
ds_args=()
[ "$CUSTOM_DS" = "1" ] && ds_args=(data.custom_cls.path="$CUSTOM_DS_PATH" data.custom_cls.name=WholeConvSFTDataset)

extra=()
if [ "$USE_PEFT" = "1" ]; then
  extra+=(model.lora_rank=${LORA_RANK:-32} model.lora_alpha=${LORA_ALPHA:-32} model.target_modules=all-linear)
fi

accel=(model.use_liger=${USE_LIGER:-true} model.use_fused_kernels=${USE_FUSED:-true})
if [ "${USE_FLASH:-1}" = "1" ]; then
  accel+=(data.pad_mode=no_padding model.use_remove_padding=true)   # attn 默认即 flash_attention_2
else
  accel+=(data.pad_mode=right model.use_remove_padding=false model.override_config.attn_implementation=sdpa)
fi

# 训练前预检：确认无样本 > MAXLEN
if [ "${TEST:-0}" != "1" ]; then
  python - "$MODEL_PATH" "$DATA_DIR/train.parquet" "$DATA_DIR/val.parquet" "$MAXLEN" <<'PY'
import sys, pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
maxlen, mx, n_over = int(sys.argv[4]), 0, 0
for pq in sys.argv[2:4]:
    for msgs in pd.read_parquet(pq)["messages"]:
        L = len(tok.apply_chat_template(list(msgs), tokenize=True, add_generation_prompt=False))
        mx = max(mx, L); n_over += L > maxlen
print(f"[sft] 预检：最长样本 {mx} tokens；> MAXLEN({maxlen}) 的行数：{n_over}")
if n_over:
    sys.exit(f"[sft] X {n_over} 行超 MAXLEN={maxlen}：请预删超长行或调大 MAXLEN/MAX_TOKENS 再训（否则中途崩=白跑）")
PY
fi

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
  -m verl.trainer.sft_trainer \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/val.parquet \
  data.messages_key=messages \
  data.ignore_input_ids_mismatch=True \
  data.micro_batch_size_per_gpu=$MB \
  data.train_batch_size=$TBS \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=$MAX_TOKENS \
  data.max_length=$MAXLEN \
  data.truncation=$TRUNC \
  optim.lr=$LR \
  optim.lr_warmup_steps_ratio=${WARMUP:-0.05} \
  engine=fsdp \
  engine.ulysses_sequence_parallel_size=$SP_SIZE \
  model.path=$MODEL_PATH \
  trainer.default_local_dir=$SAVE \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.project_name=qwen3-4b-distill \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  trainer.total_epochs=$EPOCHS \
  trainer.test_freq=${TEST_FREQ:-10} \
  "${accel[@]}" "${extra[@]}" ${opt_args[@]+"${opt_args[@]}"} "${mem_args[@]}" ${ds_args[@]+"${ds_args[@]}"} "$@"
