#!/usr/bin/env bash
# 08_expansion.sh —— 扩展批（全部数据侧 / base eval，无 SFT）：
#   ① 归因分析全套(data_metrics/compare/attribution/slice_eval/manifest，原文"解释为什么变"的正身)
#   ② shortest_cot 造数据(任务二第4方法，Concise-CoT，N=4 选最短正确)
#   ③ prompt 轴造数据(同 8B、standard_cot + Plan-and-Solve[PS] 触发，Wang et al. ACL2023 2305.04091)
#   ④ MMLU-Pro base eval(任务一科学推理，2/5→3/5)
#   ⑤ code 重测(max_new 16384→38912，实测 prompt max=1343 零截断)
# 参数一律与已完成实验一致(见各步注释)。排在 chain2(DeepSeek/14B eval)之后自动启动。
#
# 用法(服务器，先 git pull 拿 shortest_cot)：
#   setsid bash run/08_expansion.sh > /data/liujiachen/logs/run/08_expansion.log 2>&1 < /dev/null &
#   看总进度： tail -f /data/liujiachen/logs/run/08_expansion.log
set +e   # 某步失败要跳过、不拖垮整批(故意不加 -e/-u/pipefail)
source "$(dirname "$0")/env.sh"
say(){ echo "[$(date '+%F %T')] $*"; }
G0=0; G1=1

say "===== 08 扩展批启动 ====="
say "计划: [等 chain2 收尾腾卡] -> ①data_metrics(现有) -> ④MMLU-Pro -> ⑤code重测 -> ②shortest gen -> ③prompt-PS gen -> data_metrics(新) -> compare/attribution/slice_eval/manifest"

# ───────── 阶段0：等 chain2(DeepSeek/14B eval) 整体结束 + 两卡释放 ─────────
PREV="$LOGS/run/chain2_14b_teachers.log"
say "阶段0: 等待 chain2 结束标记(链全部结束)于 $PREV …"
deadline=$(( $(date +%s) + 86400 ))   # 24h 兜底
until grep -q "链全部结束" "$PREV" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && { say "WARN 24h 兜底触发(未见 chain2 结束标记),继续"; break; }
  sleep 120
done
say "阶段0: chain2 结束标记命中。确认两卡显存释放(>=15GB空,最多再等30min)…"
fdl=$(( $(date +%s) + 1800 ))
until
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G0 2>/dev/null | tr -dc 0-9); f0=${f0:-0}
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G1 2>/dev/null | tr -dc 0-9); f1=${f1:-0}
  [ "$f0" -gt 15000 ] && [ "$f1" -gt 15000 ]
do
  [ "$(date +%s)" -ge "$fdl" ] && { say "WARN 30min 后卡仍不空,仍启动"; break; }
  sleep 60
done
say "阶段0: 两卡就绪(free=$f0/$f1 MiB)。"

# ───────── 阶段1a：data_metrics 现有数据(3法 + 教师轴，IFD/PPL/distinct/长度) ─────────
# 与文档一致：--model $STUDENT_BASE(同学生)。数据侧打分，单卡即可。
say "阶段1a: data_metrics(现有 6 个数据集)"
for m in omni_standard_cot omni_reverse omni_question_aug omni_t3_14b omni_t3_dsv4pro omni_t3_math7b; do
  D="$DATA/distill/$m/train.parquet"
  if [ ! -f "$D" ]; then say "  跳过 $m(缺 $D)"; continue; fi
  CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/metrics/data_metrics.py" --data "$D" --model "$STUDENT_BASE" \
    --out "$LOGS/metrics_$m.json" > "$LOGS/run/metrics_$m.log" 2>&1
  [ -f "$LOGS/metrics_$m.json" ] && say "  ✔ $m" || say "  ✗ $m(查 $LOGS/run/metrics_$m.log)"
done

# ───────── 阶段1b：MMLU-Pro base eval(科学推理，任务一 2/5→3/5) ─────────
# 与项目既定一致(05_extended.sh)：eval_mc --n 1 --limit 200；单卡跑(题小)。
say "阶段1b: MMLU-Pro base eval"
LG="$LOGS/run/eval_mmlu_pro_base.log"
{
  if [ -f "$DATA/mmlu_pro/test.parquet" ]; then
    CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/eval/eval_mc.py" --model "$STUDENT_BASE" --data "$DATA/mmlu_pro/test.parquet" \
      --n 1 --limit 200 --gpu_mem 0.8 --out "$LOGS/eval/mmlu_pro_base"
  else
    echo "[mmlu] test.parquet 未预下载,跳过 MMLU-Pro eval(数据须提前 prepare_mc 下好)"
  fi
} > "$LG" 2>&1
[ -f "$LOGS/eval/mmlu_pro_base/summary.json" ] && say "  ✔ MMLU-Pro -> $LOGS/eval/mmlu_pro_base/summary.json" || say "  ✗ MMLU-Pro(查 $LG)"

# ───────── 阶段1c：code 重测(max_new 38912，两卡分片) ─────────
# 与已完成 code eval 一致(n4/全167/两卡分片/gpu_mem0.8)，仅 max_new 16384->38912(实测 prompt max=1343 零截断)。
say "阶段1c: code base 重测 @max_new=38912"
CG="$LOGS/run/eval_lcb_base_mn38912.log"
if [ ! -f "$DATA/livecodebench/test.parquet" ]; then
  say "  ✗ code重测跳过:$DATA/livecodebench/test.parquet 未预下载"
else
  for s in $G0 $G1; do
    CUDA_VISIBLE_DEVICES=$s python "$PROJ/eval/eval_code.py" --model "$STUDENT_BASE" \
      --data "$DATA/livecodebench/test.parquet" --n 4 --max_new 38912 --out "$LOGS/eval/lcb_base_mn38912_s$s" \
      --shard $s --num_shards 2 --gpu_mem 0.8 > "$LOGS/run/eval_lcb_base_mn38912_s$s.log" 2>&1 &
  done; wait
  python "$PROJ/eval/merge_shards.py" --shards "$LOGS/eval/lcb_base_mn38912_s$G0" "$LOGS/eval/lcb_base_mn38912_s$G1" --out "$LOGS/eval/lcb_base_mn38912" > "$CG" 2>&1
  [ -f "$LOGS/eval/lcb_base_mn38912/summary.json" ] && say "  ✔ code重测 -> $LOGS/eval/lcb_base_mn38912/summary.json" || say "  ✗ code重测(查 $CG)"
fi

# ───────── 阶段2a：shortest_cot 造数据(任务二第4方法) ─────────
# 与三法一致：8B / omni_seed / LIMIT500 / TP2 / max_new默认38912。仅 N=4(方法必需，选最短正确)。走 gen_distill.sh(带冒烟门控)。
say "阶段2a: shortest_cot gen (N=4, LIMIT=500, ~16h)"
SL="$LOGS/run/gen_omni_shortest_cot.log"
TEACHER="$MODELS/Qwen3-8B" SEED="$DATA/omni_seed/train.parquet" OUT="$DATA/distill/omni_shortest_cot" \
  METHOD=shortest_cot N=4 LIMIT=500 TP=2 GPU_MEM=0.8 \
  bash "$PROJ/run/gen_distill.sh" > "$SL" 2>&1
[ -f "$DATA/distill/omni_shortest_cot/gen_stats.json" ] && say "  ✔ shortest_cot -> gen_stats.json" || say "  ✗ shortest_cot(查 $SL)"

# ───────── 阶段2b：prompt 轴造数据(同 8B、standard_cot + Plan-and-Solve[PS] 触发) ─────────
# 与 standard_cot 逐字节一致：8B / omni_seed / LIMIT500 / N1 / max_new默认。仅加 system 指令(PS 触发，Wang et al. ACL2023)。
say "阶段2b: prompt-PS gen (standard_cot + PS system, LIMIT=500, ~4h)"
PL="$LOGS/run/gen_omni_prompt_ps.log"
PS_PROMPT="Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step."
TEACHER="$MODELS/Qwen3-8B" SEED="$DATA/omni_seed/train.parquet" OUT="$DATA/distill/omni_prompt_ps" \
  METHOD=standard_cot N=1 LIMIT=500 TP=2 GPU_MEM=0.8 SYS_PROMPT="$PS_PROMPT" \
  bash "$PROJ/run/gen_distill.sh" > "$PL" 2>&1
[ -f "$DATA/distill/omni_prompt_ps/gen_stats.json" ] && say "  ✔ prompt-PS -> gen_stats.json" || say "  ✗ prompt-PS(查 $PL)"

# ───────── 阶段2c：data_metrics 两个新数据集 ─────────
say "阶段2c: data_metrics(新: shortest_cot / prompt_ps)"
for m in omni_shortest_cot omni_prompt_ps; do
  D="$DATA/distill/$m/train.parquet"
  if [ ! -f "$D" ]; then say "  跳过 $m(缺 $D，可能上游 gen 失败)"; continue; fi
  CUDA_VISIBLE_DEVICES=$G0 python "$PROJ/metrics/data_metrics.py" --data "$D" --model "$STUDENT_BASE" \
    --out "$LOGS/metrics_$m.json" > "$LOGS/run/metrics_$m.log" 2>&1
  [ -f "$LOGS/metrics_$m.json" ] && say "  ✔ $m" || say "  ✗ $m"
done

# ───────── 阶段3：CPU 归因(compare / attribution / slice_eval / manifest) ─────────
say "阶段3: CPU 归因分析"
MET(){ echo "$LOGS/metrics_$1.json"; }        # data_metrics json
SUM(){ echo "$LOGS/eval/olymmath_sft_$1/summary.json"; }  # 下游 eval summary
PQ(){ echo "$LOGS/eval/$1/per_question.jsonl"; }

# 3a compare：方法轴(含 shortest) / 教师轴 / prompt 轴
python "$PROJ/metrics/compare_methods.py" --out "$LOGS/compare_methods.md" \
  --in standard=$(MET omni_standard_cot) reverse=$(MET omni_reverse) qaug=$(MET omni_question_aug) shortest=$(MET omni_shortest_cot) \
  > "$LOGS/run/compare_methods.log" 2>&1 && say "  ✔ compare_methods.md" || say "  ✗ compare_methods"
python "$PROJ/metrics/compare_methods.py" --out "$LOGS/compare_teachers.md" \
  --in 8b=$(MET omni_standard_cot) 14b=$(MET omni_t3_14b) dsv4pro=$(MET omni_t3_dsv4pro) math7b=$(MET omni_t3_math7b) \
  > "$LOGS/run/compare_teachers.log" 2>&1 && say "  ✔ compare_teachers.md" || say "  ✗ compare_teachers"
python "$PROJ/metrics/compare_methods.py" --out "$LOGS/compare_prompt.md" \
  --in standard=$(MET omni_standard_cot) ps=$(MET omni_prompt_ps) \
  > "$LOGS/run/compare_prompt.log" 2>&1 && say "  ✔ compare_prompt.md" || say "  ✗ compare_prompt"

# 3b attribution：数据指标 ↔ 下游 pass@1(5 点:standard/reverse/qaug/14b/dsv4pro，均有下游)
python "$PROJ/metrics/attribution.py" --out "$LOGS/attribution.md" \
  --point standard:$(MET omni_standard_cot):$(SUM omni_standard_cot) \
  --point reverse:$(MET omni_reverse):$(SUM omni_reverse) \
  --point qaug:$(MET omni_question_aug):$(SUM omni_question_aug) \
  --point 14b:$(MET omni_t3_14b):$(SUM omni_t3_14b) \
  --point dsv4pro:$(MET omni_t3_dsv4pro):$(SUM omni_t3_dsv4pro) \
  > "$LOGS/run/attribution.log" 2>&1 && say "  ✔ attribution.md" || say "  ✗ attribution"

# 3c slice_eval：base vs 各法逐学科 + McNemar(有 per_question 的 5 个下游)
for m in omni_standard_cot omni_reverse omni_question_aug omni_t3_14b omni_t3_dsv4pro; do
  [ -f "$(PQ olymmath_sft_$m)" ] || { say "  slice 跳过 $m(缺 per_question)"; continue; }
  python "$PROJ/metrics/slice_eval.py" --jsonl "$(PQ olymmath_base)" --vs "$(PQ olymmath_sft_$m)" \
    --name_a base --name_b "$m" --by subject --dump 5 --out "$LOGS/slice_base_vs_$m.md" \
    > "$LOGS/run/slice_$m.log" 2>&1 && say "  ✔ slice base_vs_$m" || say "  ✗ slice $m"
done

# 3d make_manifest：每实验统一记录(§1 要求)
mkdir -p "$LOGS/manifests"
manifest(){ # $1=name $2=distill_dir $3=eval_name  其余=--set
  local name="$1" dd="$2" en="$3"; shift 3
  python "$PROJ/run/make_manifest.py" --name "$name" --out "$LOGS/manifests" \
    --gen_stats "$DATA/distill/$dd/gen_stats.json" --eval "$LOGS/eval/olymmath_sft_$en/summary.json" "$@" \
    >> "$LOGS/run/make_manifest.log" 2>&1
}
manifest sft_omni_standard_cot omni_standard_cot omni_standard_cot --set dataset=OlymMATH-d45 --set student=Qwen3-4B --set teacher=Qwen3-8B --set method=standard_cot --set stage=SFT
manifest sft_omni_reverse       omni_reverse       omni_reverse       --set dataset=OlymMATH-d45 --set student=Qwen3-4B --set teacher=Qwen3-8B --set method=reverse --set stage=SFT
manifest sft_omni_question_aug  omni_question_aug  omni_question_aug  --set dataset=OlymMATH-d45 --set student=Qwen3-4B --set teacher=Qwen3-8B --set method=question_aug --set stage=SFT
manifest sft_omni_t3_14b        omni_t3_14b        omni_t3_14b        --set dataset=OlymMATH-d45 --set student=Qwen3-4B --set teacher=Qwen3-14B --set method=standard_cot --set stage=SFT
manifest sft_omni_t3_dsv4pro    omni_t3_dsv4pro    omni_t3_dsv4pro    --set dataset=OlymMATH-d45 --set student=Qwen3-4B --set teacher=DeepSeek-V4-pro --set method=standard_cot --set stage=SFT
say "  ✔ manifests -> $LOGS/manifests/"

say "===== 08 扩展批全部结束 ====="
