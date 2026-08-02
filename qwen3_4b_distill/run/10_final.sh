#!/usr/bin/env bash
# 10_final.sh —— 最后一批（3 个实验），全部复用现成脚本、仅编排 + 幂等守卫 + GRPO 先 TEST=1：
#   ① SFT + eval  omni_shortest_cot   （第4方法 Concise-CoT 的下游）
#   ② SFT + eval  omni_prompt_ps       （prompt 轴 Plan-and-Solve 的下游）
#   ③ GRPO(from sft_omni_standard_cot) + eval  （on-policy RL PoC，测 H2 后半"on-policy 挽回"）
#
# 配方与三法 SFT 逐字节一致：USE_PEFT=1 LR=2e-4 EPOCHS=5 MAXLEN=40960 + hydra 覆盖 lora_alpha=32 / warmup=0.05 / test_freq=10
#   （三法/教师轴 SFT 正是这么跑的——sft.sh 内 alpha=16 被末尾 hydra 覆盖为 32；rank32 / NPROC2 / TBS32 由 sft.sh）。
# GRPO：MATH 种子作 RL prompt（奖励密、解短适配 RESP=8192；Omni d4–5 解长~12K 会大量撞 8192 截断→奖励稀疏），
#       起点 = sft_omni_standard_cot 的 merged 全权重，EPOCHS=1 短 PoC，eval n=4（与 SFT 对照一致）。
#
# ⚠️ 串行、每步占两卡；粗估 SFT×2(~6h)+eval×2(~14h)+GRPO(~4h)+GRPOeval(~7h) ≈ 30h ≈ 1.5–2 晚。
# ⚠️ 幂等：各步产物已存在则跳过，中断后重跑自动续（不重跑已完成的 SFT/eval/GRPO）。
# 用法（服务器）：source run/env.sh; setsid bash run/10_final.sh > $LOGS/run/10_final.log 2>&1 < /dev/null &
set -uo pipefail
source "$(dirname "$0")/env.sh"
say(){ echo "[$(date '+%F %T')] $*"; }
say "===== 10 最后一批启动（shortest / prompt_ps SFT+eval + GRPO PoC）====="

# ───────── 阶段1：两个新数据集 SFT（配方同三法）─────────
for M in omni_shortest_cot omni_prompt_ps; do
  if ls -d "$CKPT/sft_$M"/global_step_* >/dev/null 2>&1; then say "阶段1: ↷ SFT $M 已存在，跳过"; continue; fi
  [ -f "$DATA/distill/$M/train.parquet" ] || { say "阶段1: ✗ SFT $M 跳过（缺 $DATA/distill/$M/train.parquet）"; continue; }
  say "阶段1: SFT $M （USE_PEFT=1 LR=2e-4 EPOCHS=5 MAXLEN=40960，两卡）"
  USE_PEFT=1 LR=2e-4 EPOCHS=5 MAXLEN=40960 EXP="sft_$M" SAVE="$CKPT/sft_$M" \
    DATA_DIR="$DATA/distill/$M" bash "$PROJ/train/sft.sh" \
    model.lora_alpha=32 optim.lr_warmup_steps_ratio=0.05 trainer.test_freq=10 > "$LOGS/run/sft_$M.log" 2>&1
  ls -d "$CKPT/sft_$M"/global_step_* >/dev/null 2>&1 && say "  ✔ SFT $M" || say "  ✗ SFT $M（查 $LOGS/run/sft_$M.log）"
done

# ───────── 阶段2：合并 LoRA（CPU，07_merge 自带 rel 自检）─────────
say "阶段2: merge LoRA → *_merged"
METHODS="omni_shortest_cot omni_prompt_ps" bash "$PROJ/run/07_merge.sh" > "$LOGS/run/merge_final.log" 2>&1
for M in omni_shortest_cot omni_prompt_ps; do
  [ -d "$CKPT/sft_${M}_merged" ] && say "  ✔ merged $M" || say "  ✗ merged $M（查 $LOGS/run/merge_final.log）"
done

# ───────── 阶段3：两法下游 eval（OlymMATH-hard，n=4，两卡分片，复用 06）─────────
if [ -f "$LOGS/eval/olymmath_sft_omni_shortest_cot/summary.json" ] && [ -f "$LOGS/eval/olymmath_sft_omni_prompt_ps/summary.json" ]; then
  say "阶段3: ↷ 两法 eval 已存在，跳过"
else
  say "阶段3: eval（omni_shortest_cot / omni_prompt_ps，各两卡分片、串行，~7h/法）"
  METHODS="omni_shortest_cot omni_prompt_ps" bash "$PROJ/run/06_sft_eval_all.sh" > "$LOGS/run/eval_final.log" 2>&1
  for M in omni_shortest_cot omni_prompt_ps; do
    [ -f "$LOGS/eval/olymmath_sft_$M/summary.json" ] && say "  ✔ eval $M" || say "  ✗ eval $M（查 eval_final.log 及 eval_${M}_s{0,1}.log）"
  done
fi

# ───────── 阶段4：GRPO from sft_omni_standard_cot（先 TEST=1 验不 OOM，再正式 EPOCHS=1）+ eval ─────────
FROM=sft_omni_standard_cot; MM="$CKPT/${FROM}_merged"
if [ ! -d "$MM" ]; then say "阶段4: 缺 $MM，先补 merge $FROM"; METHODS="omni_standard_cot" bash "$PROJ/run/07_merge.sh" >> "$LOGS/run/merge_final.log" 2>&1; fi

if [ -f "$LOGS/eval/math_grpo_omni_standard/summary.json" ]; then
  say "阶段4: ↷ GRPO+eval 已存在，跳过"
elif [ ! -d "$MM" ]; then
  say "阶段4: ✗ 无 $MM，GRPO 跳过"
else
  say "阶段4a: GRPO TEST=1 冒烟（仅训练 1–2 步验不 OOM，不评测）"
  if TEST=1 MODEL_PATH="$MM" REWARD="$PROJ/reward/math_reward.py" RM=naive \
       TRAIN_DIR="$SEED_DIR" VAL_DIR="$EVAL_DIR" EXP=grpo_smoke \
       bash "$PROJ/train/grpo.sh" > "$LOGS/run/grpo_test.log" 2>&1; then
    say "  ✔ GRPO TEST=1 通过 → 起正式"
    say "阶段4b: GRPO 正式（EPOCHS=1，MATH 种子，from $FROM）+ eval(n=4)"
    EPOCHS=1 N=4 ABILITY=math FROM="$FROM" MODEL_PATH="$MM" EXP=grpo_omni_standard \
      bash "$PROJ/run/03_grpo.sh" > "$LOGS/run/grpo_final.log" 2>&1
    [ -f "$LOGS/eval/math_grpo_omni_standard/summary.json" ] \
      && say "  ✔ GRPO+eval → $LOGS/eval/math_grpo_omni_standard/summary.json" \
      || say "  ✗ GRPO+eval（查 $LOGS/run/grpo_final.log）"
  else
    say "  ✗ GRPO TEST=1 未过（查 $LOGS/run/grpo_test.log，多半 OOM）——跳过正式 GRPO；SFT/eval 结果不受影响"
  fi
fi

say "===== 10 最后一批 全部结束 ====="
