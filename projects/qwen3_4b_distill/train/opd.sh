#!/usr/bin/env bash
# On-Policy Distillation（加分/stretch）| student Qwen3-4B-Base ← teacher Qwen3-8B | 2×3090
# 改编自 verl/examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
# ⚠️ 2×3090(48GB) 上 teacher(8B)+student(4B)+vLLM 显存极紧，很可能 OOM。
#    必须 TEST=1 起；OOM 就退回 train/sft.sh（off-policy 序列蒸馏），把 OPD 作"尝试+分析"写进报告。
set -xeuo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-/data/liujiachen/models/Qwen3-4B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-/data/liujiachen/models/Qwen3-8B}
DATA_DIR=${DATA_DIR:-/data/liujiachen/datasets/olymmath}
EXP=${EXP:-opd_4b_from_8b}

NGPUS=${NGPUS:-2}                       # student/trainer 资源
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}   # teacher 独立推理池（尽量压 1 卡）
LOSS_MODE=${LOSS_MODE:-forward_kl_topk} # 纯蒸馏(GKD-OPD)；带 PG 用 k1/k3 + USE_PG=True
USE_PG=${USE_PG:-False}
TOPK=${TOPK:-64}

if [ "${TEST:-0}" = "1" ]; then TBS=8; MINI=8; RESP=512; EPOCHS=1; else
  TBS=${TBS:-32}; MINI=${MINI:-16}; RESP=${RESP:-1024}; EPOCHS=${EPOCHS:-5}; fi
MAXTOK=$(( 1024 + RESP + 1 ))

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=$TBS \
  data.max_prompt_length=1024 \
  data.max_response_length=$RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$STUDENT_MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_model_len=$MAXTOK \
  actor_rollout_ref.rollout.free_cache_engine=True \
  distillation.enabled=True \
  distillation.n_gpus_per_node=$TEACHER_WORLD_SIZE \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.3 \
  distillation.teacher_models.teacher_model.inference.max_model_len=$MAXTOK \
  distillation.distillation_loss.loss_mode=$LOSS_MODE \
  distillation.distillation_loss.topk=$TOPK \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=$USE_PG \
  trainer.balance_batch=True \
  trainer.n_gpus_per_node=$NGPUS \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.total_epochs=$EPOCHS \
  trainer.save_freq=200 \
  trainer.test_freq=5 \
  trainer.project_name=qwen3-4b-opd \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  "$@"

# 硬约束：teacher 与 student 必须同 tokenizer/词表（Qwen3-4B + Qwen3-8B 满足）。
# forward_kl_topk = 纯 on-policy 蒸馏（对齐 teacher top-k 分布）；想要 PG-OPD 设 LOSS_MODE=k1 USE_PG=True。
