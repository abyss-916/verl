#!/usr/bin/env bash
# 扩展 benchmark base eval（held-out）：code=LiveCodeBench / mc=MMLU-Pro；每基准两卡分片再合并（与 eval.sh 一致）。
# 用法：bash run/eval_extended.sh；可覆盖 MODEL/LIMIT/CODE_LIMIT/N/GM/G0/G1（共卡调低 GM）。
set -uo pipefail   # 不加 -e：某基准/分片失败时跳过，不影响其余
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/eval" "$LOGS/run"
M=${MODEL:-$STUDENT_BASE}
LIM=${LIMIT:-200}
N=${N:-1}                # 每题采样数（base eval 默认 1；pass@k 时调高）
GM=${GM:-0.8}            # vLLM 显存占比（与 eval.sh 对齐）；与他人共卡时调低
G0=${G0:-0}              # 分片0用的物理卡
G1=${G1:-1}              # 分片1用的物理卡

# 两卡分片评一个基准再合并：$1=eval 脚本 $2=数据 $3=输出目录 $4=limit
sharded_eval() {
  local py=$1 data=$2 out=$3 lim=$4 tag; tag=$(basename "$out")
  rm -rf "${out}_s0" "${out}_s1" "$out"
  CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/$py" --model "$M" --data "$data" --n "$N" \
    --limit "$lim" --gpu_mem "$GM" --num_shards 2 --shard 0 --out "${out}_s0" > "$LOGS/run/${tag}_s0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=$G1 python "$PROJ/eval/$py" --model "$M" --data "$data" --n "$N" \
    --limit "$lim" --gpu_mem "$GM" --num_shards 2 --shard 1 --out "${out}_s1" > "$LOGS/run/${tag}_s1.log" 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  if [ "$r0" -ne 0 ] || [ "$r1" -ne 0 ]; then echo "!! [$tag] 分片失败(r0=$r0 r1=$r1)，看 $LOGS/run/${tag}_s{0,1}.log"; return 1; fi
  python "$PROJ/eval/merge_shards.py" --shards "${out}_s0" "${out}_s1" --out "$out"
  echo "---- [$tag] 完成 → $out/summary.json ----"
}

# ── code：LiveCodeBench ──
python "$PROJ/data_preprocess/prepare_code.py" --version "$CODE_VERSION" --out "$DATA/livecodebench" || true
sharded_eval eval_code.py "$DATA/livecodebench/test.parquet" "$LOGS/eval/lcb_base" "${CODE_LIMIT:-50}" || true

# ── mc：MMLU-Pro ──
python "$PROJ/data_preprocess/prepare_mc.py" --hf "$MMLU_PRO_HF" --subset default --out "$DATA/mmlu_pro" --data_source mmlu_pro || true
sharded_eval eval_mc.py "$DATA/mmlu_pro/test.parquet" "$LOGS/eval/mmlu_pro_base" "$LIM" || true

echo "扩展 benchmark base eval 完成，见 $LOGS/eval/{lcb,mmlu_pro}_base"
