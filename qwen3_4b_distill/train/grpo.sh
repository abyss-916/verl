#!/usr/bin/env bash
# GRPO(LoRA) 后训练 | Qwen3-4B | 2×3090(无 NVLink) | 改编自 verl/examples/tuning/lora/run_qwen3_8b_fsdp.sh
# 与 SFT 一致走 LoRA 策略：可训参/梯度/优化器极小 → 免全参 offload、更快，且 RESP 可顶大
#   （避免难题 rollout 被截断而丢 reward 信号——这是 Omni d4–5 上出效果的关键）。
# 用法：
#   EXP=... MODEL_PATH=<sft_merged> TRAIN_DIR=<omni_seed> VAL_DIR=<olymmath> RESP=16384 EPOCHS=5 bash train/grpo.sh
#   TEST=1 ... bash train/grpo.sh    # 先 1~2 step 验不崩 + 第一步验显存，过了再放大
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/liujiachen/models/Qwen3-4B}   # 建议 = SFT 后 merged ckpt（LoRA GRPO 在其上再加一层 LoRA）
# ⚠️ 训练 prompt 用难度匹配的 Omni d4–5(omni_seed)——4B 有头部空间(探针 68.7%)才有对/错方差=reward 信号；
#    MATH 种子太简单已饱和(§5.3.1)、GRPO 也学不到。held-out(olymmath) 只放 VAL_DIR 监控，绝不进训练=防泄漏。
TRAIN_DIR=${TRAIN_DIR:-/data/liujiachen/datasets/omni_seed}
VAL_DIR=${VAL_DIR:-/data/liujiachen/datasets/olymmath}
EXP=${EXP:-grpo_lora}
REWARD=${REWARD:-/data/liujiachen/verl/qwen3_4b_distill/reward/math_reward.py}
RM=${RM:-naive}                                   # reward_manager：math/mc=naive；code=prime
CKPT=${CKPT:-/data/liujiachen/checkpoints}
SAVE=${SAVE:-$CKPT/$EXP}
LORA_RANK=${LORA_RANK:-32}; LORA_ALPHA=${LORA_ALPHA:-32}   # 与 SFT 一致（rank32/alpha32/all-linear）
GM=${GM:-0.6}                                     # vLLM 显存占比：LoRA 免全参 optimizer 常驻，可给 vLLM 多点(rollout 更快)

if [ "${TEST:-0}" = "1" ]; then
  TBS=8; MINI=8; RESP=256; N=4; EPOCHS=1; LR=${LR:-3e-6}
else
  # RESP：LoRA 免全参优化器显存 → 可顶到覆盖 Omni d4–5 学生解(~10–20K)、避免截断丢 reward。
  #   默认 16384(覆盖大多数)；正式跑第一步验显存：够则保留、OOM 就退 12288/8192；想更高(32768)可加激活 offload。
  TBS=${TBS:-32}; MINI=${MINI:-16}; RESP=${RESP:-16384}; N=${N:-5}; EPOCHS=${EPOCHS:-5}; LR=${LR:-3e-6}
fi
TOTLEN=$(( 1024 + RESP ))   # 单序列最长 = prompt(1024)+response(RESP)；micro-batch token 预算须 ≥ 它

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$TRAIN_DIR/train.parquet \
  data.val_files=$VAL_DIR/test.parquet \
  data.train_batch_size=$TBS \
  data.max_prompt_length=1024 \
  data.max_response_length=$RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward.custom_reward_function.path=$REWARD \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=$RM \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.lora_rank=$LORA_RANK \
  actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOTLEN \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GM \
  actor_rollout_ref.rollout.max_model_len=$TOTLEN \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOTLEN \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$TOTLEN \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.use_v1=False \
  trainer.default_local_dir=$SAVE \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=$EPOCHS \
  trainer.val_before_train=False \
  trainer.save_freq=20 \
  trainer.test_freq=-1 \
  trainer.project_name=qwen3-4b-grpo \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  "$@"

# ── LoRA GRPO 关键点(与全参的区别) ──
# LoRA(rank32/alpha32/all-linear) → 可训参/梯度/优化器极小：
#   · fsdp param_offload=False / optimizer_offload=False（LoRA 不需要全参那套 offload，更快）
#   · rollout 必须 load_format=safetensors + layered_summon=True（vLLM 加载 base + 逐层 summon LoRA 权重同步）
#   · use_kl_loss=True(coef 0.001, low_var_kl)：ref=冻结的 base(LoRA 关)，不额外占显存，稳训练（DeepSeekMath 式）
#   · RESP 可顶大(默认 16384)，覆盖 Omni d4–5 学生解、避免截断丢 reward；第一步验显存,OOM 就 RESP=12288/8192
#   · test_freq=-1 / val_before_train=False：不在训练中途跑昂贵的 held-out eval(最终 eval 由 03_grpo.sh 单独做)
# ── code / mc ──（同前）reward 换 code_reward.py(RM=prime) / mc_reward.py(RM=naive)。
