#!/usr/bin/env bash
# GRPO 后训练 | Qwen3-4B | 2×3090(无NVLink) | 改编自 verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh
# 用法（服务器，tmux/nohup 后台）：
#   EXP=grpo_olymmath MODEL_PATH=/data/liujiachen/checkpoints/sft_standard_cot \
#     DATA_DIR=/data/liujiachen/datasets/olymmath bash train/grpo.sh
#   TEST=1 ... bash train/grpo.sh      # 先跑通 1~2 step 不 OOM 再放大
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/liujiachen/models/Qwen3-4B-Base}   # 建议用 SFT 后 ckpt
DATA_DIR=${DATA_DIR:-/data/liujiachen/datasets/olymmath}          # RL parquet（含 ground_truth）
EXP=${EXP:-grpo_olymmath}
REWARD=${REWARD:-/data/liujiachen/verl/qwen3_4b_distill/reward/math_reward.py}
CKPT=${CKPT:-/data/liujiachen/checkpoints}
SAVE=${SAVE:-$CKPT/$EXP}

if [ "${TEST:-0}" = "1" ]; then
  TBS=8; MINI=8; RESP=256; N=4; EPOCHS=1
else
  TBS=${TBS:-32}; MINI=${MINI:-16}; RESP=${RESP:-1024}; N=${N:-5}; EPOCHS=${EPOCHS:-5}
fi

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
  reward.custom_reward_function.path=$REWARD \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.default_local_dir=$SAVE \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=$EPOCHS \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.project_name=qwen3-4b-grpo \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","wandb"]' \
  "$@"

# ── 2×3090 关键点 ──
# use_kl_loss=False + use_kl_in_reward=False → 去 ref model，省一份 4B 权重（GRPO 仍成立）。
# param/optimizer offload=True → 必须。 rollout TP=1（DP=2）+ gpu_mem_util=0.4 + free_cache_engine=True。
#
# ── code / LiveCodeBench ──
# 用我们的 code_reward.py（复用 verl prime_code 本地执行单测）：
#   reward.custom_reward_function.path=/data/liujiachen/verl/qwen3_4b_distill/reward/code_reward.py \
#   reward.custom_reward_function.name=compute_score \
#   reward.reward_manager.name=prime
# 想用云沙箱则设环境变量 SANDBOX_FUSION_URL，或直接：
#   reward.sandbox_fusion.url=http://127.0.0.1:PORT/run_code   reward.sandbox_fusion.max_concurrent=64
# 注意 code 测试用例格式（prime_code 期望 {inputs,outputs}，见 reward/code_reward.py 的转换 TODO）。
