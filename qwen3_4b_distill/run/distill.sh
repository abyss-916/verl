#!/usr/bin/env bash
# 造蒸馏数据：tp=2 满预算(教师 Qwen3-8B, max_new=38912)，占用两卡，冒烟通过后再起正式。
# 冒烟(--limit SMOKE)先验证：① tp=2 教师能在 2×3090(无 NVLink) 启动；② 数据质量正常。冒烟失败则不起正式(set -e)。
#
# 用法（服务器，先 source env.sh，且两卡空闲——tp=2 需两张卡）：
#   source run/env.sh
#   # 先看 nvidia-smi 定 GPU_MEM：两卡空用 0.9；GPU0 被占用则相应调低(如 0.8)
#   # 三法均 LIMIT=1000（同种子预算=受控三向对比；依据 s1K=1000 / LIMO=817 的"少而精"推理蒸馏）
#   GPU_MEM=0.9 METHOD=standard_cot LIMIT=1000 nohup bash run/distill.sh \
#       > "$LOGS/run/gen_standard_cot.log" 2>&1 &
#   # reverse 同法：METHOD=reverse LIMIT=1000（每 seed 2 条长链≈2× 时长；产 3 条/题多目标）
#   # question_aug 同法：METHOD=question_aug LIMIT=1000（无 gold 靠 self-consistency k≥3，≈4×/题最重，排最后）
#   # 任务三强度轴（同种子换教师）：换 TEACHER + OUT，其余不变，事后 data_metrics 对比
#   #   TEACHER=/data/liujiachen/models/Qwen3-14B OUT="$DATA/distill/t3_14b" \
#   #     METHOD=standard_cot LIMIT=500 GPU_MEM=0.9 nohup bash run/distill.sh > "$LOGS/run/gen_t3_14b.log" 2>&1 &
#   # 任务三家族轴（Qwen2.5-Math-7B ctx=4096，须设 MAX_LEN/MAX_NEW，否则 vLLM 无法启动）：
#   #   TEACHER=/data/liujiachen/models/Qwen2.5-Math-7B-Instruct OUT="$DATA/distill/t3_math7b" \
#   #     METHOD=standard_cot LIMIT=500 MAX_LEN=4096 MAX_NEW=3584 GPU_MEM=0.9 nohup bash run/distill.sh > "$LOGS/run/gen_t3_math7b.log" 2>&1 &
#   # 任务三 prompt 轴（同 8B 换蒸馏风格，只 standard_cot 生效）：
#   #   TEACHER=$MODELS/Qwen3-8B OUT="$DATA/distill/t3_prompt2" SYS_PROMPT="Always restate ... verify by an alternative method." \
#   #     METHOD=standard_cot LIMIT=500 GPU_MEM=0.9 nohup bash run/distill.sh > "$LOGS/run/gen_t3_prompt2.log" 2>&1 &
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
