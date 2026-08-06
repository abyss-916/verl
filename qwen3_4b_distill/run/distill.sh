#!/usr/bin/env bash
# 造蒸馏数据：教师生成 CoT + 可验证过滤 → SFT messages parquet。冒烟（SMOKE 条）通过后再起正式。
# 用法（先 source env.sh；两卡空闲）：METHOD=omni_standard_cot bash run/distill.sh
# 可覆盖：METHOD / SEED / LIMIT / TEACHER / OUT / GPU_MEM / TP / N（shortest_cot 用 4）/ MAX_LEN·MAX_NEW（短上下文教师须设）/ SYS_PROMPT（仅 standard_cot）。
set -euo pipefail
: "${PROJ:?先 source run/env.sh}"
: "${TEACHER:?先 source run/env.sh}"

METHOD=${METHOD:-$DISTILL_METHOD}       # 按 ABILITY 默认；math 另可传 reverse/question_aug/shortest_cot
LIMIT=${LIMIT:-500}                      # 正式用多少 seed（omni_seed 全量 500）
SMOKE=${SMOKE:-16}                       # 冒烟 seed 数
N=${N:-1}                                # 每题采样候选数；shortest_cot 用 >1（如 4），其余用 1
GPU_MEM=${GPU_MEM:-0.75}                  # vLLM 显存占比（8B 教师较吃显存，默认低于 eval；共卡再调低）
TP=${TP:-2}                              # 张量并行度
GPUS=${GPUS:-0,1}                         # 用哪两张卡
export CUDA_VISIBLE_DEVICES=$GPUS
SEED=${SEED:-$ABILITY_SEED_DIR/train.parquet}   # 按 ABILITY 选造数据种子（math=omni_seed）
OUT=${OUT:-$DATA/distill/$METHOD}
# MAX_LEN/MAX_NEW/SYS_PROMPT 未设时 EXTRA 为空，用 generate_cot.py 默认
EXTRA=()
[ -n "${MAX_LEN:-}" ]    && EXTRA+=(--max_len "$MAX_LEN")
[ -n "${MAX_NEW:-}" ]    && EXTRA+=(--max_new "$MAX_NEW")
[ -n "${SYS_PROMPT:-}" ] && EXTRA+=(--sys_prompt "$SYS_PROMPT")

echo "[gen] method=$METHOD limit=$LIMIT n=$N tp=$TP gpu_mem=$GPU_MEM"
echo "[gen] seed=$SEED  out=$OUT  teacher=$TEACHER"
[ -f "$SEED" ] || { echo "!! 缺造数据种子 $SEED（math=omni_seed 须先切分；code/mc=先 prepare 对应数据集）"; exit 1; }

echo "[gen] === 冒烟 $SMOKE 条（验 tp=2 起得来 + 数据质量），写 ${OUT}_smoke ==="
python "$PROJ/distill/generate_cot.py" --method "$METHOD" --seed "$SEED" --teacher "$TEACHER" \
  --out "${OUT}_smoke" --tp "$TP" --gpu_mem "$GPU_MEM" --n "$N" --limit "$SMOKE" ${EXTRA[@]+"${EXTRA[@]}"}

# 冒烟门控：按 n_kept 拦截"跑通但 0 产出"
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
  --out "$OUT" --tp "$TP" --gpu_mem "$GPU_MEM" --n "$N" --limit "$LIMIT" ${EXTRA[@]+"${EXTRA[@]}"}

echo "[gen] === 完成。看 $OUT/gen_stats.json（良率/截断率/长度分位）==="
