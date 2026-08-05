#!/usr/bin/env bash
# GRPO 后训练（从 SFT ckpt 起，仅训练；LoRA GRPO 的 eval 须先 merge 折叠，见文末与 run/10_grpo_chain.sh）。
#   定位为短 PoC（EPOCHS=1，约几百步、一晚）；tmux/nohup 后台运行。
# 首次运行：TEST=1 bash run/03_grpo.sh，验证不 OOM 再正式跑。
# 实跑用 run/10_grpo_chain.sh（GRPO→merge→eval 一键；从 SFT-merged 起、GM=0.70）；本脚本保留作单步 GRPO。
#
# 能力可切 ABILITY=math|code|mc，切换 reward 判分器 / 训练·评测数据 / reward_manager / eval 脚本：
#   math(默认): reward=math_reward,  train=math_seed,  eval=olymmath(held-out), manager=naive
#   code      : reward=code_reward(prime_code 跑单测), eval=eval_code, manager=prime
#   mc        : reward=mc_reward(抽字母比对),           eval=eval_mc,   manager=naive
# code/mc 无独立 train 种子：正式跑须用与评测集不重叠的 train 种子（设 TRAIN_DIR= 指向含 train.parquet 的目录），
#    否则拿评测集当训练即数据泄漏。math 侧 train=math_seed / eval=olymmath 无重叠。冒烟见 run/09_smoke_pipeline.sh。
set -xeuo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/run"; exec > >(tee -a "$LOGS/run/$(basename "$0" .sh).log") 2>&1  # 全部输出落 $LOGS/run/

ABILITY=${ABILITY:-math}
case "$ABILITY" in
  math) REWARD=$PROJ/reward/math_reward.py; RM=naive; TRAIN=${TRAIN_DIR:-$SEED_DIR};          VAL=${VAL_DIR:-$EVAL_DIR};            EVALPY=eval_math.py; EVALARGS="--n ${N:-8}";;
  code) REWARD=$PROJ/reward/code_reward.py; RM=prime; TRAIN=${TRAIN_DIR:-$DATA/livecodebench}; VAL=${VAL_DIR:-$DATA/livecodebench};   EVALPY=eval_code.py; EVALARGS="--n ${N:-4}";;
  mc)   REWARD=$PROJ/reward/mc_reward.py;   RM=naive; TRAIN=${TRAIN_DIR:-$DATA/mmlu_pro};      VAL=${VAL_DIR:-$DATA/mmlu_pro};       EVALPY=eval_mc.py;   EVALARGS="--n ${N:-1} --limit ${EVAL_LIMIT:-500}";;
  *) echo "未知 ABILITY=$ABILITY（math|code|mc）"; exit 1;;
esac

# train/grpo.sh 用 $TRAIN_DIR/train.parquet 作 GRPO 训练 prompt；code/mc 数据集只有 test.parquet，须自备 train 种子
if [ ! -f "$TRAIN/train.parquet" ]; then
  echo "!! [$ABILITY] 缺 $TRAIN/train.parquet —— code/mc 需自备与评测集不重叠的 train 种子（设 TRAIN_DIR=），或用 run/09_smoke_pipeline.sh 造冒烟种子"; exit 1
fi

FROM=${FROM:-sft_standard_cot}                    # 起点 SFT 实验名
EXP=${EXP:-grpo_${ABILITY}_from_${FROM}}
MODEL_PATH=${MODEL_PATH:-$(latest_hf "$CKPT/$FROM")}   # SFT 后的 HF 权重目录
[ -z "$MODEL_PATH" ] && { MODEL_PATH=$STUDENT_BASE; echo "⚠️ 未找到 $CKPT/$FROM 的 HF 权重，回退 base=$STUDENT_BASE（PoC/冒烟可，正式应从 SFT ckpt 起）"; }

# GRPO：REWARD/RM/TRAIN_DIR/VAL_DIR 传给 train/grpo.sh（reward_manager 由 RM 切换：code=prime）
MODEL_PATH="$MODEL_PATH" REWARD="$REWARD" RM="$RM" TRAIN_DIR="$TRAIN" VAL_DIR="$VAL" EXP="$EXP" \
  bash "$PROJ/train/grpo.sh"

# 已移除自动 eval：本项目 GRPO 走 LoRA，raw ckpt 的 huggingface/(hf_model) 不是折叠后的完整模型——
#    verl 把 PEFT 模型 state_dict(带 base_layer/lora_A/lora_B 键)存入 base 架构、未 merge_and_unload
#    （fsdp_checkpoint_manager.py:386→423，model_config=底模架构），直接评测得到的不是 GRPO 真权重，不可信。
#    正确评测：先折叠 LoRA 为完整模型，再评（math 用 olymmath held-out）：
#      PREFIX=grpo_ METHODS=<法> BASE=$CKPT/sft_<法>_merged bash run/07_merge.sh
#      PREFIX=grpo_ METHODS=<法>                            bash run/06_sft_eval_all.sh
#    以上两步由 run/10_grpo_chain.sh 串成一键（GRPO→merge→eval，启动一次）。
echo "GRPO[$ABILITY] 训练完成：ckpt=$(ls -d "$CKPT/$EXP"/global_step_* 2>/dev/null | sort -V | tail -1)"
echo "  → 评测请走 merge+eval（run/10_grpo_chain.sh）；勿直接评 raw ckpt 的 huggingface/（=未折叠 LoRA）"
