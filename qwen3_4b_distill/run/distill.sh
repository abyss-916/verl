#!/usr/bin/env bash
# 造蒸馏数据：教师生成 CoT + 可验证过滤 → SFT messages parquet。tp=2 满预算，冒烟（SMOKE 条）通过后再起正式。
# 用法（先 source env.sh；两卡空闲）：
#   METHOD=omni_standard_cot bash run/distill.sh
# 可覆盖：METHOD / SEED / LIMIT / TEACHER / OUT / GPU_MEM / TP / N（shortest_cot 用 4）/
#   MAX_LEN·MAX_NEW（短上下文教师如 Math-7B ctx=4096 须设，否则 vLLM 起不来）/ SYS_PROMPT（prompt 轴，仅 standard_cot）。
# 各教师轴/方法的具体命令见 material/复现记录。
set -euo pipefail
: "${PROJ:?先 source run/env.sh}"
: "${TEACHER:?先 source run/env.sh}"

METHOD=${METHOD:-$DISTILL_METHOD}       # 按 ABILITY 默认：math=standard_cot / code=code_cot / mc=mc_cot；math 另可传 reverse / question_aug / shortest_cot
LIMIT=${LIMIT:-500}                      # 正式用多少 seed（=omni_seed 全量 500；三法一致=受控对比）
SMOKE=${SMOKE:-16}                       # 冒烟 seed 数
N=${N:-1}                                # 每题采样候选数；shortest_cot 需 >1(如 4)才有"选最短"空间；标准三法/教师轴用 1
GPU_MEM=${GPU_MEM:-0.9}                   # 按 nvidia-smi 定：两卡空 0.9；共卡调低
TP=${TP:-2}                              # 8B 教师满预算(40960 KV)单卡放不下，须用 tp=2
GPUS=${GPUS:-0,1}                         # 先看 nvidia-smi 再定用哪两张卡（共享机，避开已被占用的卡）
export CUDA_VISIBLE_DEVICES=$GPUS
SEED=${SEED:-$ABILITY_SEED_DIR/train.parquet}   # 按 ABILITY：math=omni_seed(500)/code=livecodebench/mc=mmlu_pro；不用评测集=防泄漏
OUT=${OUT:-$DATA/distill/$METHOD}
# max_new/max_len 默认不传，用 generate_cot.py 满预算默认(38912/40960)，不为省时改小。
# 覆盖场景（任务三教师轴）：短上下文教师 Qwen2.5-Math-7B(ctx=4096) 须设 MAX_LEN=4096 MAX_NEW=3584（否则 vLLM 无法启动）；
#   prompt 轴用 SYS_PROMPT="..." 给 standard_cot 换蒸馏风格。三个 env 都未设时 EXTRA 为空，与原行为一致。
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

# 冒烟门控：仅看退出码不够——reverse/qaug 可能"跑通但 0 产出"（如 thinking 未关），须按 n_kept 拦截，
# 否则会放行数小时的正式 run 而无有效产出。
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
