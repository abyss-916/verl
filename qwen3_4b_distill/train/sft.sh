#!/usr/bin/env bash
# off-policy 序列蒸馏 SFT | Qwen3-4B | 2×3090 | verl sft_trainer
# 改编自 verl/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh
# 用法（服务器）：
#   EXP=sft_standard_cot DATA_DIR=/data/liujiachen/datasets/distill/standard_cot bash train/sft.sh
#   TEST=1 EXP=... DATA_DIR=... bash train/sft.sh      # 极小配置先验证不 OOM、依赖齐全
#
# ── 加速（本机 glibc 2.31）──
#   flash-attn 用 mjun0812 预编译轮子安装（flash_attn 2.8.3+cu128torch2.9-cp312，manylinux_2_28，
#   glibc≥2.28 兼容，避开官方轮子要求 GLIBC_2.32 的限制）。默认走 flash-attn 路径：
#   · USE_FLASH=1（默认）：attn=flash_attention_2 + use_remove_padding=true + pad_mode=no_padding
#     （变长打包，省 padding，长短不一时约 1.5–2× 加速）。
#   · USE_FLASH=0（回退）：attn=sdpa + use_remove_padding=false + pad_mode=right（不依赖 flash-attn；
#     flash-attn 算子在本机不可用时用此档。sdpa 在 Ampere 即 FlashAttention-2 内核，注意力不慢）。
#   · 叠加的 Triton 融合算子（不涉及 flash-attn）：use_liger（SwiGLU/RMSNorm/RoPE）+ use_fused_kernels
#     （融合 linear cross-entropy，Qwen3 词表 15 万，省 [seq×150k] logits 的数 G 显存）。两档都开。
#   · 以上均不改数值结果，仅提速；单独关闭：USE_LIGER=false / USE_FUSED=false。
#   ulysses SP（SP_SIZE>1）依赖变长路径，仅在 USE_FLASH=1 时可开。
set -xeuo pipefail
# 抗碎片：长序列训练易产生显存碎片，expandable_segments 减少碎片型 OOM（PYTORCH_ALLOC_CONF 为当前名，替代已弃用的 PYTORCH_CUDA_ALLOC_CONF）
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH=${MODEL_PATH:-/data/liujiachen/models/Qwen3-4B}
DATA_DIR=${DATA_DIR:-/data/liujiachen/datasets/distill/standard_cot}
EXP=${EXP:-sft_standard_cot}
SAVE=${SAVE:-/data/liujiachen/checkpoints/$EXP}
NPROC=${NPROC:-2}
SP_SIZE=${SP_SIZE:-1}       # 2×3090 先不开序列并行；长 CoT 显存紧可设 2
USE_PEFT=${USE_PEFT:-1}     # 默认 LoRA（本项目实跑值；2×3090 上 4B 全参装不下）。大显存机走全参改 0
LR=${LR:-2e-4}             # LoRA 主配置实跑值（1e-5 为全参量级、对 LoRA 欠拟合致 merged≈base，见报告 §6.3）

if [ "${TEST:-0}" = "1" ]; then
  MB=1; MAXLEN=1024; EPOCHS=1; TRUNC=right     # 冒烟仅验证跑通、依赖齐全、小配置不 OOM，允许右截断
else
  # MAXLEN 须覆盖蒸馏数据长度。data.truncation=error 让超长行直接报错而非静默右截断，
  #    避免"丢掉结尾 \boxed 的半截 CoT"坏样本进入训练；配合造数据后预删超长行 + 训练前长度预检使用。
  #    正式训练前依 gen_stats.json 定 MAXLEN：尽量 ≥ tok_max；tok_max 过大装不下则预删超长行。
  #    40960=Qwen3 位置上限，覆盖长 CoT（实测 p99≈11.6K）；具体可依 gen_stats.json 调。
  #    显存不足时：动态批(use_dynamic_bsz)下 MB 基本无效，省显存靠 ①SP_SIZE=2(序列并行，跨两卡切分
  #       长序列、不丢样本，需 USE_FLASH=1) ②activation offload / 梯度检查点。不建议降 MAXLEN/MAX_TOKENS，否则须丢弃长 CoT。
  MB=${MB:-2}; MAXLEN=${MAXLEN:-40960}; EPOCHS=${EPOCHS:-5}; TRUNC=error   # 实跑值：LoRA / lr2e-4 / 40960 / 5 epoch / batch32
fi
# 动态批每卡 token 预算须 >= 最长样本；verl 缺省 8192，长样本会触发 seqlen_balancing 的 assert 报错
MAX_TOKENS=${MAX_TOKENS:-$MAXLEN}
# 全局(优化器)batch；verl 缺省 256 时 ~1000 种子(过滤后~700)一个 epoch 仅 ~3 步、3 epoch ~9 步易欠拟合。显式设 32(~22 步/epoch)；三法须一致以保证公平对比
TBS=${TBS:-32}

# 优化器：2×3090(24G/卡)装不下 4B 全参 fp32 AdamW —— 优化器状态(fp32 双动量)≈16G/卡，单步峰值~23.7G > 24G。
#   offload 对此无效：verl 强制关闭原生 CPUOffload(与梯度累积并用会算错)，手动 offload 仅在阶段间生效，
#      optimizer.step() 那一步 params+grads+状态仍须全在 GPU；SP_SIZE/MAX_TOKENS 只降激活、不降此峰值。
#   默认用 torchao 8-bit 优化器：状态量化到~4G，单步峰值~12G，近乎无损、全参不变(需 pip install torchao)。
#     大显存机需 fp32 全精度：设 OPT8BIT=0。该 torchao 版本若无 AdamW8bit：改 OPT_NAME=Adam8bit/AdamW4bit（可用名见报错）。
OPT8BIT=${OPT8BIT:-1}
OPT_NAME=${OPT_NAME:-AdamW8bit}
OPT_IMPL=${OPT_IMPL:-torchao.optim}
opt_args=()
[ "$OPT8BIT" = "1" ] && opt_args=(optim.optimizer="$OPT_NAME" optim.optimizer_impl="$OPT_IMPL")

# 显存大头(2×3090)：model_dtype 默认 fp32 时 FSDP 主权重 fp32(~8G/卡) + fp32 梯度(~8G/卡)=~16G/卡，装不下。
#   默认 bf16 主权重(减半到~4G) + 激活 offload 到 CPU，是装下 4B 全参 SFT 的主要手段(优化器/SP 为次要)。
#   实测(standard_cot)：bf16 主权重 + activation_offload + 8-bit 优化器 → max_reserved~20.4G，GPU0(另有 2.7G 占用)可装下。
#   精度：bf16 主权重 + 8-bit 优化器，SFT 数百步精度损失很小；大显存机需全精度：MODEL_DTYPE=fp32 ACT_OFFLOAD=false。
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
ACT_OFFLOAD=${ACT_OFFLOAD:-true}
mem_args=(engine.model_dtype="$MODEL_DTYPE" model.enable_activation_offload="$ACT_OFFLOAD")

# Qwen3 <think> 保真：verl 默认逐条 assistant 套模板，Qwen3 会把单条 assistant 的 <think>…</think>
#   当作"历史思考"剥除，使"<think>推理</think>解答"被训成"只有解答"，模型学成不思考、难题上退化(实测 pass@1 约 2%)。
#   默认用整段渲染的自定义数据集(train/whole_conv_sft_dataset.py)保留 <think>。CUSTOM_DS=0 退回 verl 原生实现。
CUSTOM_DS=${CUSTOM_DS:-1}
CUSTOM_DS_PATH=${CUSTOM_DS_PATH:-/data/liujiachen/verl/qwen3_4b_distill/train/whole_conv_sft_dataset.py}
ds_args=()
[ "$CUSTOM_DS" = "1" ] && ds_args=(data.custom_cls.path="$CUSTOM_DS_PATH" data.custom_cls.name=WholeConvSFTDataset)

extra=()
if [ "$USE_PEFT" = "1" ]; then
  # alpha 默认 32（§6.3 订正后、三法/教师轴所用值）；可用 LORA_ALPHA= 覆盖
  extra+=(model.lora_rank=${LORA_RANK:-32} model.lora_alpha=${LORA_ALPHA:-32} model.target_modules=all-linear)
fi

# 加速开关：flash-attn 变长打包(默认) vs sdpa+padding 回退；Triton 融合算子两档都开
accel=(model.use_liger=${USE_LIGER:-true} model.use_fused_kernels=${USE_FUSED:-true})
if [ "${USE_FLASH:-1}" = "1" ]; then
  accel+=(data.pad_mode=no_padding model.use_remove_padding=true)   # attn 默认即 flash_attention_2
else
  accel+=(data.pad_mode=right model.use_remove_padding=false model.override_config.attn_implementation=sdpa)
fi

# 预检(非冒烟)：训练前确认无样本 > MAXLEN，否则 truncation=error 会在训练中途才报错，浪费数小时
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

# Liger 需先安装：pip install liger-kernel（纯 Triton 轮子，不编译、不依赖特定 glibc）。首次先 TEST=1 冒烟，
# 同时验证 flash-attn 算子在本机可用；若冒烟报 flash-attn 相关错误，用 USE_FLASH=0 回退再跑。
# 三法（standard_cot / reverse / question_aug）分别改 EXP 与 DATA_DIR，其余保持一致以保证公平对比。
