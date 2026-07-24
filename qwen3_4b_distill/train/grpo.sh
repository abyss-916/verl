#!/usr/bin/env bash
# GRPO 后训练 | Qwen3-4B | 2×3090(无NVLink) | 改编自 verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh
# 用法（服务器，tmux/nohup 后台）：
#   EXP=grpo_olymmath MODEL_PATH=/data/liujiachen/checkpoints/sft_standard_cot \
#     DATA_DIR=/data/liujiachen/datasets/olymmath bash train/grpo.sh
#   TEST=1 ... bash train/grpo.sh      # 先跑通 1~2 step 不 OOM 再放大
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/liujiachen/models/Qwen3-4B}   # 建议用 SFT 后 ckpt（如 $CKPT/sft_standard_cot）
# ⚠️ 训练 prompt 必须用 MATH 种子(math_seed)，绝不能用 olymmath——那是 held-out 评测集，
#    拿它训 GRPO = 在评测题上训 = 数据泄漏，最终分数作废。held-out 只放 VAL_DIR 做监控(不更新权重)。
TRAIN_DIR=${TRAIN_DIR:-/data/liujiachen/datasets/math_seed}   # GRPO 训练 prompt = MATH(含 ground_truth)
VAL_DIR=${VAL_DIR:-/data/liujiachen/datasets/olymmath}        # 仅监控 held-out 泛化，不参与训练/模型选择
EXP=${EXP:-grpo_math}
REWARD=${REWARD:-/data/liujiachen/verl/qwen3_4b_distill/reward/math_reward.py}
CKPT=${CKPT:-/data/liujiachen/checkpoints}
SAVE=${SAVE:-$CKPT/$EXP}

if [ "${TEST:-0}" = "1" ]; then
  TBS=8; MINI=8; RESP=256; N=4; EPOCHS=1
else
  # RESP(max_response_length)：数学题要产出完整 CoT 到 \boxed 才有 reward。教师 MATH 解实测
  #   p50≈2.5K / p97≈8K token → 8192 覆盖大多数、给足学习信号；1024 会让多数题截断、reward 恒 0 学不到。
  #   以课题质量为先，默认 8192，不为省显存砍。显存实在装不下时退 4096(覆盖~p85)，别退回 1024。
  TBS=${TBS:-32}; MINI=${MINI:-16}; RESP=${RESP:-8192}; N=${N:-5}; EPOCHS=${EPOCHS:-5}
fi
# 单条序列最长 = prompt(1024) + response(RESP)；训练 micro-batch 与 rollout KV 都必须能装下它
TOTLEN=$(( 1024 + RESP ))

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
  reward.reward_manager.name=naive \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_liger=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOTLEN \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=$TOTLEN \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.use_v1=False \
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
# param/optimizer offload=True → 必须。 rollout TP=1（DP=2）+ gpu_mem_util=0.5 + free_cache_engine=True。
#
# ── code / LiveCodeBench ──
# 用我们的 code_reward.py（复用 verl prime_code 本地执行单测）：
#   reward.custom_reward_function.path=/data/liujiachen/verl/qwen3_4b_distill/reward/code_reward.py \
#   reward.custom_reward_function.name=compute_score \
#   reward.reward_manager.name=prime
# 想用云沙箱则设环境变量 SANDBOX_FUSION_URL，或直接：
#   reward.sandbox_fusion.url=http://127.0.0.1:PORT/run_code   reward.sandbox_fusion.max_concurrent=64
# 注意 code 测试用例格式：prepare_code 存 LCB 原始 input/output 串，prime_code 按 fn_name 自解析（已打通，合成解验证 1.0/0）。
