#!/usr/bin/env bash
# 造蒸馏数据：tp=2 满预算(教师 Qwen3-8B, max_new=38912)，两卡全占，冒烟通过再起正式。
# 冒烟(--limit SMOKE)先验：① tp=2 教师能在 2×3090(无 NVLink) 起来；② 数据质量对。冒烟崩则不起正式(set -e)。
#
# 用法（服务器，先 source env.sh，且两卡都空——tp=2 需要两张卡）：
#   source run/env.sh
#   # 先看 nvidia-smi 定 GPU_MEM：两卡都空用 0.9；GPU0 有别人占用则相应调低(如 0.8)
#   GPU_MEM=0.9 METHOD=standard_cot LIMIT=2000 nohup bash run/gen_distill.sh \
#       > "$LOGS/run/gen_standard_cot.log" 2>&1 &
#   # reverse 同法：METHOD=reverse（reverse 每 seed 产 2 条长链，耗时约 2 倍）
#   # question_aug 只做数据分析、不训练 → 小规模即可：METHOD=question_aug LIMIT=500
#   # 任务三 MVP（同种子换教师）：换 TEACHER + OUT，其余不变，事后 data_metrics 对比两份数据
#   #   TEACHER=/data/liujiachen/models/Qwen3-14B OUT="$DATA/distill/t3_teacher14b" \
#   #     METHOD=standard_cot LIMIT=500 GPU_MEM=0.9 nohup bash run/gen_distill.sh > "$LOGS/run/gen_t3_14b.log" 2>&1 &
set -euo pipefail
: "${PROJ:?先 source run/env.sh}"
: "${TEACHER:?先 source run/env.sh}"

METHOD=${METHOD:-standard_cot}          # standard_cot / reverse / question_aug
LIMIT=${LIMIT:-2000}                     # 正式用多少 seed
SMOKE=${SMOKE:-16}                       # 冒烟 seed 数
GPU_MEM=${GPU_MEM:-0.9}                   # 按 nvidia-smi 定：两卡空 0.9；共卡调低
TP=${TP:-2}                              # 8B 教师满预算(40960 KV)单卡放不下 → 必须 tp=2
GPUS=${GPUS:-0,1}                         # 先看 nvidia-smi 再定用哪两张卡（共享机，避免撞别人占用的卡）
export CUDA_VISIBLE_DEVICES=$GPUS
SEED=${SEED:-$SEED_DIR/train.parquet}    # MATH 种子（绝不用 olymmath——那是 held-out 评测集）
OUT=${OUT:-$DATA/distill/$METHOD}
# max_new/max_len 不传 → 用 generate_cot.py 默认(38912/40960)满预算，别为省时改小

echo "[gen] method=$METHOD limit=$LIMIT tp=$TP gpu_mem=$GPU_MEM"
echo "[gen] seed=$SEED  out=$OUT  teacher=$TEACHER"

echo "[gen] === 冒烟 $SMOKE 条（验 tp=2 起得来 + 数据质量），写 ${OUT}_smoke ==="
python "$PROJ/distill/generate_cot.py" --method "$METHOD" --seed "$SEED" --teacher "$TEACHER" \
  --out "${OUT}_smoke" --tp "$TP" --gpu_mem "$GPU_MEM" --limit "$SMOKE"

# 冒烟门控：只看退出码不够——reverse/qaug 可能"跑通但 0 产出"（如 thinking 没关），必须按 n_kept 拦截，
# 否则会放行几小时的正式 run 全程白跑。
python - "$SMOKE" "${OUT}_smoke/gen_stats.json" <<'PY'
import json, sys
smoke, p = int(sys.argv[1]), sys.argv[2]
st = json.load(open(p))
kept = st.get("n_kept", 0)
need = max(1, int(smoke * 0.3))            # 要求 >=30% 种子有产出，防"跑通但全灭"
if kept < need:
    sys.exit(f"[gen] X 冒烟 n_kept={kept} < 阈值 {need}（良率过低/疑似 0 产出），拒绝起正式 run。先查 {p}")
print(f"[gen] OK 冒烟 n_kept={kept} >= {need}，放行正式 run")
PY

echo "[gen] === 冒烟通过，起正式 $LIMIT 条 -> $OUT ==="
python "$PROJ/distill/generate_cot.py" --method "$METHOD" --seed "$SEED" --teacher "$TEACHER" \
  --out "$OUT" --tp "$TP" --gpu_mem "$GPU_MEM" --limit "$LIMIT"

echo "[gen] === 完成。看 $OUT/gen_stats.json（良率/截断率/长度分位）==="
