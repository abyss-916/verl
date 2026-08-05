#!/usr/bin/env bash
# GRPO(LoRA) 后训练 | Qwen3-4B | 2×3090 | 改编自 verl/examples/tuning/lora/run_qwen3_8b_fsdp.sh
# 用法：EXP=... MODEL_PATH=<sft_merged> TRAIN_DIR=<omni_seed> VAL_DIR=<olymmath> bash train/grpo.sh
#   TEST=1 ... bash train/grpo.sh    # 小配置先验证不报错/不 OOM
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-$MODELS/Qwen3-4B}   # = SFT merged ckpt（其上再加一层 LoRA）
TRAIN_DIR=${TRAIN_DIR:-$DATA/omni_seed}      # GRPO prompt 池（难度匹配的 Omni d4–5）
VAL_DIR=${VAL_DIR:-$DATA/olymmath}           # held-out 监控，不进训练
EXP=${EXP:-grpo_lora}
REWARD=${REWARD:-$PROJ/reward/math_reward.py}
RM=${RM:-naive}                              # reward_manager：math/mc=naive；code=prime
CKPT=${CKPT:-/data/checkpoints}              # 通常由 env.sh 继承（export）
SAVE=${SAVE:-$CKPT/$EXP}
LORA_RANK=${LORA_RANK:-32}; LORA_ALPHA=${LORA_ALPHA:-32}
GM=${GM:-0.70}                               # vLLM 显存占比（colocate）
TP=${TP:-2}                                  # 张量并行度
FUSED=${FUSED:-True}                         # 融合 LM-head+CE（省 log_prob 的 logits 显存）

if [ "${TEST:-0}" = "1" ]; then
  TBS=8; MINI=8; RESP=256; N=4; EPOCHS=1; LR=${LR:-1e-5}
else
  TBS=${TBS:-32}; MINI=${MINI:-16}; RESP=${RESP:-16384}; N=${N:-5}; EPOCHS=${EPOCHS:-5}; LR=${LR:-1e-5}
fi
TOTLEN=$(( 1024 + RESP ))                    # 单序列最长 = prompt(1024)+response(RESP)

STEPS=${STEPS:-0}                            # >0 时限定总步数（短 PoC，覆盖 EPOCHS）
STEP_CAP=""
[ "$STEPS" -gt 0 ] 2>/dev/null && STEP_CAP="trainer.total_training_steps=$STEPS"

# 缺训练种子则提前退出
[ -f "$TRAIN_DIR/train.parquet" ] || { echo "!! 缺 $TRAIN_DIR/train.parquet —— code/mc 需自备与评测集不重叠的 train 种子（设 TRAIN_DIR= 指向含 train.parquet 的目录）"; exit 1; }

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
  actor_rollout_ref.model.use_fused_kernels=$FUSED \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOTLEN \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GM \
  actor_rollout_ref.rollout.enforce_eager=True \
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
  $STEP_CAP \
  "$@"
