#!/usr/bin/env bash
# 三法下游 SFT→eval 对比 | OlymMATH-hard(held-out) 100题 | n=4 | 每法两卡分片、三法串行
#   对比对象：base(M1) vs standard_cot / reverse / question_aug。
#   模型用 peft 合并后的 $CKPT/sft_<法>_merged（verl model_merger 不折叠 LoRA，须用 _merged）。
#
# 用法（服务器，先 conda activate verl；先看 nvidia-smi 确认两卡空闲）：
#   setsid bash /data/liujiachen/verl/qwen3_4b_distill/run/06_sft_eval_all.sh \
#       > /data/liujiachen/logs/run/06_sft_eval_all.log 2>&1 < /dev/null &
#   # 只跑某几法：      METHODS="reverse question_aug" setsid bash …/06_sft_eval_all.sh …
#   # 换卡(先看 smi)：  G0=0 G1=1  （分片0→G0，分片1→G1）
#   # 实时看进度：      tail -f $LOGS/run/eval_standard_cot_s0.log $LOGS/run/eval_standard_cot_s1.log
#
# 每法两卡各跑一片、占满两张卡，故三法串行；单法约 7h、三法约 22h（n=4、思考约 22K token/条）。
# 用 setsid 而非 nohup：vLLM/torchrun 的孙进程自带信号处理器，关终端的 SIGHUP 会穿透 nohup。
set -uo pipefail   # 不加 -e：某一法/某片失败时跳过，不影响其余两法
source "$(dirname "$0")/env.sh"

DATA_PARQUET=${DATA_PARQUET:-$EVAL_DIR/test.parquet}   # held-out OlymMATH，不用训练集
N=${N:-4}
GM=${GM:-0.8}          # 共卡时 0.8 稳定；0.85 会在 sampler warmup 阶段 OOM
G0=${G0:-0}            # 分片0用的物理卡
G1=${G1:-1}            # 分片1用的物理卡
PREFIX=${PREFIX-sft_}    # 模型目录前缀（用 - 而非 :- 使空串也生效）：SFT=sft_（默认）；GRPO 传 PREFIX=grpo_ + METHODS=omni_<法>（或 PREFIX= 空 + METHODS=grpo_<法>）
METHODS=${METHODS:-"standard_cot reverse question_aug"}

echo "==== eval 开始 | n=$N gm=$GM 卡=($G0,$G1) 前缀=[$PREFIX] 方法=[$METHODS] data=$DATA_PARQUET ===="
for M in $METHODS; do
  MODEL=$CKPT/${PREFIX}${M}_merged
  S0=$LOGS/eval/olymmath_${PREFIX}${M}_s0
  S1=$LOGS/eval/olymmath_${PREFIX}${M}_s1
  OUT=$LOGS/eval/olymmath_${PREFIX}${M}
  if [ ! -d "$MODEL" ]; then echo "!! [$M] 缺模型目录 $MODEL —— 跳过"; continue; fi

  echo "---- [$M] 开始 model=$MODEL ----"
  rm -rf "$S0" "$S1" "$OUT"      # 清除旧分片/合并结果，避免脏数据混入

  CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/eval_math.py" \
    --model "$MODEL" --data "$DATA_PARQUET" --n "$N" --gpu_mem "$GM" \
    --num_shards 2 --shard 0 --out "$S0" > "$LOGS/run/eval_${M}_s0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=$G1 python "$PROJ/eval/eval_math.py" \
    --model "$MODEL" --data "$DATA_PARQUET" --n "$N" --gpu_mem "$GM" \
    --num_shards 2 --shard 1 --out "$S1" > "$LOGS/run/eval_${M}_s1.log" 2>&1 &
  P1=$!
  wait $P0; R0=$?
  wait $P1; R1=$?
  if [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ]; then
    echo "!! [$M] 分片失败(R0=$R0 R1=$R1)；看 $LOGS/run/eval_${M}_s{0,1}.log。跳过合并，继续下一法"
    continue
  fi
  python "$PROJ/eval/merge_shards.py" --shards "$S0" "$S1" --out "$OUT" \
    | tee "$LOGS/run/eval_${M}_merged.log"
  echo "---- [$M] 完成 → $OUT/summary.json ----"
done
echo "==== 三法 eval 全部结束 ===="
