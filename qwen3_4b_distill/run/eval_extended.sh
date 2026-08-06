#!/usr/bin/env bash
# 扩展 benchmark base eval（held-out）；每基准两卡分片再合并（与 eval.sh 一致）。设定对齐报告：
#   code=LiveCodeBench：167 题防污染窗全量、avg@4；mc=MMLU-Pro：2000 题、n=1；均 max_new=38912。
# 用法：bash run/eval_extended.sh；可覆盖 MODEL/CODE_N/CODE_LIMIT/MC_N/MC_LIMIT/GM/G0/G1（共卡调低 GM）。
set -uo pipefail   # 不加 -e：某基准/分片失败时跳过，不影响其余
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/eval" "$LOGS/run"
M=${MODEL:-$STUDENT_BASE}
CODE_N=${CODE_N:-4}          # LiveCodeBench avg@4（报告 §2.2）
CODE_LIMIT=${CODE_LIMIT:-0}  # 0=全量（防污染窗 167 题，无 --limit）
MC_N=${MC_N:-1}              # MMLU-Pro n=1（报告 §5.1）
MC_LIMIT=${MC_LIMIT:-2000}   # MMLU-Pro 评 2000 题（prepare_mc 产全量 ~12k；MC_LIMIT=0 则评全量）
GM=${GM:-0.8}               # vLLM 显存占比；共卡调低
G0=${G0:-0}; G1=${G1:-1}    # 两分片各自的物理卡

# 两卡分片评一个基准再合并：$1=eval脚本 $2=数据 $3=输出 $4=limit(0=全量) $5=n
sharded_eval() {
  local py=$1 data=$2 out=$3 lim=$4 nn=$5 tag; tag=$(basename "$out")
  rm -rf "${out}_s0" "${out}_s1" "$out"
  CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/$py" --model "$M" --data "$data" --n "$nn" \
    --limit "$lim" --gpu_mem "$GM" --num_shards 2 --shard 0 --out "${out}_s0" > "$LOGS/run/${tag}_s0.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=$G1 python "$PROJ/eval/$py" --model "$M" --data "$data" --n "$nn" \
    --limit "$lim" --gpu_mem "$GM" --num_shards 2 --shard 1 --out "${out}_s1" > "$LOGS/run/${tag}_s1.log" 2>&1 &
  local p1=$!
  wait $p0; local r0=$?; wait $p1; local r1=$?
  if [ "$r0" -ne 0 ] || [ "$r1" -ne 0 ]; then echo "!! [$tag] 分片失败(r0=$r0 r1=$r1)，看 $LOGS/run/${tag}_s{0,1}.log"; return 1; fi
  python "$PROJ/eval/merge_shards.py" --shards "${out}_s0" "${out}_s1" --out "$out"
  echo "---- [$tag] 完成 → $out/summary.json ----"
}

# ── code：LiveCodeBench（167 题防污染窗，全量，avg@4）──
python "$PROJ/data_preprocess/prepare_code.py" --version "$CODE_VERSION" --out "$DATA/livecodebench" || true
sharded_eval eval_code.py "$DATA/livecodebench/test.parquet" "$LOGS/eval/lcb_base" "$CODE_LIMIT" "$CODE_N" || true

# ── mc：MMLU-Pro（2000 题，n=1）──
python "$PROJ/data_preprocess/prepare_mc.py" --hf "$MMLU_PRO_HF" --subset default --out "$DATA/mmlu_pro" --data_source mmlu_pro || true
sharded_eval eval_mc.py "$DATA/mmlu_pro/test.parquet" "$LOGS/eval/mmlu_pro_base" "$MC_LIMIT" "$MC_N" || true

echo "扩展 benchmark base eval 完成，见 $LOGS/eval/{lcb,mmlu_pro}_base"
