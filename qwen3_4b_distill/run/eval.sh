#!/usr/bin/env bash
# held-out 评测（OlymMATH-hard 100 题）：base 或已 merge 的 SFT/GRPO 模型；每法两卡分片、多法串行。
#   模型：METHODS=base → $STUDENT_BASE；否则 $CKPT/<PREFIX><法>_merged（LoRA 须先经 merge.sh 折叠）。
# 用法（先 conda activate verl，确认两卡空闲）：
#   N=8 METHODS=base setsid bash run/eval.sh > $LOGS/run/eval.log 2>&1 < /dev/null &
#   METHODS="standard_cot reverse question_aug" setsid bash run/eval.sh ...    # SFT 三法
#   PREFIX=grpo_ METHODS=omni_standard setsid bash run/eval.sh ...             # GRPO
#   换卡 G0=0 G1=1；进度 tail -f $LOGS/run/eval_<法>_s{0,1}.log
set -uo pipefail   # 不加 -e：某法/某片失败时跳过
source "$(dirname "$0")/env.sh"

DATA_PARQUET=${DATA_PARQUET:-${ABILITY_EVAL_DIR:-$EVAL_DIR}/test.parquet}   # held-out OlymMATH
N=${N:-4}
GM=${GM:-0.8}          # vLLM 显存占比
G0=${G0:-0}            # 分片0用的物理卡
G1=${G1:-1}            # 分片1用的物理卡
PREFIX=${PREFIX-sft_}    # 模型目录前缀：SFT=sft_（默认）；GRPO 传 PREFIX=grpo_
METHODS=${METHODS:-"standard_cot reverse question_aug"}
[ -f "$DATA_PARQUET" ] || { echo "!! 缺评测集 $DATA_PARQUET，终止"; exit 1; }
done_n=0

echo "==== eval 开始 | n=$N gm=$GM 卡=($G0,$G1) 前缀=[$PREFIX] 方法=[$METHODS] data=$DATA_PARQUET ===="
for M in $METHODS; do
  if [ "$M" = "base" ]; then MODEL=$STUDENT_BASE; TAG=base; else MODEL=$CKPT/${PREFIX}${M}_merged; TAG=${PREFIX}${M}; fi
  S0=$LOGS/eval/olymmath_${TAG}_s0
  S1=$LOGS/eval/olymmath_${TAG}_s1
  OUT=$LOGS/eval/olymmath_${TAG}
  if [ ! -d "$MODEL" ]; then echo "!! [$M] 缺模型目录 $MODEL —— 跳过"; continue; fi

  echo "---- [$M] 开始 model=$MODEL ----"
  rm -rf "$S0" "$S1" "$OUT"      # 清除旧分片/合并结果

  CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/${EVAL_PY:-eval_math.py}" \
    --model "$MODEL" --data "$DATA_PARQUET" --n "$N" --gpu_mem "$GM" \
    --num_shards 2 --shard 0 --out "$S0" > "$LOGS/run/eval_${M}_s0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=$G1 python "$PROJ/eval/${EVAL_PY:-eval_math.py}" \
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
  echo "---- [$M] 完成 → $OUT/summary.json ----"; done_n=$((done_n+1))
done
[ "$done_n" -gt 0 ] || { echo "!! eval 未评测任何模型（模型目录都不存在或分片全部失败），终止"; exit 1; }
echo "==== eval 结束：$done_n 个模型已评 ===="
