#!/usr/bin/env bash
# 下载课题所需模型到 $MODELS，并接入数据集（→ verl parquet）到 $DATA。自带路径声明（source env.sh）。
# 模型优先 modelscope；重试直到无 .incomplete 残留；已完整则跳过。全部输出落 $LOGS/run/download.log。
# 用法：
#   bash run/download.sh          # T1 必需模型(~40G)；已完整的跳过，只补未下全的
#   bash run/download.sh t3       # 追加强度轴 T3(14B+32B-AWQ,~46G)
#   bash run/download.sh data     # 只接入数据集（SEED 种子 + EVAL held-out → parquet）
#   bash run/download.sh all      # 模型全档 + 数据集
#   HF=1 bash run/download.sh     # 模型改用 HF 镜像(hf-mirror)
set -euo pipefail

# ---- 环境与路径声明（无需自行声明 logs/models 等路径）----
source "$(dirname "$0")/env.sh"                         # 声明 $MODELS/$LOGS/$HF_ENDPOINT 并创建 /data 下各目录
mkdir -p "$LOGS/run" "$MODELS"
exec > >(tee -a "$LOGS/run/download.log") 2>&1         # 全部输出落日志（nohup 后台也能查）

echo "==== download start | MODELS=$MODELS ===="
df -h "$(dirname "$MODELS")" | tail -1 | awk '{print "磁盘可用: "$4"（已用 "$5"）@ "$6}'

# ---- 模型清单 ----
T1=(Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen2.5-Math-7B-Instruct Qwen/Qwen3-Embedding-0.6B)  # student+8B teacher+专精teacher+embedding
T3=(Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ)                                                    # 强度轴 teacher（进行到 T3 时再下）
DL_DATA=0
case "${1:-t1}" in
  t1)   LIST=("${T1[@]}") ;;
  t3)   LIST=("${T3[@]}") ;;
  data) LIST=(); DL_DATA=1 ;;
  all)  LIST=("${T1[@]}" "${T3[@]}"); DL_DATA=1 ;;
  *)    echo "用法: bash run/download.sh [t1|t3|data|all]  # t1/t3=模型档；data=数据集；all=模型全档+数据集"; exit 1 ;;
esac

# ---- 下载器（缺失时自动安装）----
if [ "${HF:-0}" = "1" ]; then
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
else
  command -v modelscope >/dev/null 2>&1 || pip install -U modelscope
fi

# 检查目录中是否有未下完的临时文件
has_incomplete() { find "$1" \( -name '*.incomplete' -o -name '*.tmp' \) 2>/dev/null | grep -q .; }

dl() {  # $1 = repo，如 Qwen/Qwen3-8B —— 重试直到完整（config.json 存在 且 无 .incomplete）
  local repo="$1" dest="$MODELS/${1##*/}" try
  if [ -f "$dest/config.json" ] && ! has_incomplete "$dest"; then echo "跳过(已完整): ${1##*/}"; return; fi
  echo "==== 下载 $repo -> $dest ===="
  mkdir -p "$dest"
  for try in 1 2 3 4 5 6; do
    if [ "${HF:-0}" = "1" ]; then hf download "$repo" --local-dir "$dest" || true
    else modelscope download --model "$repo" --local_dir "$dest" || true; fi
    if [ -f "$dest/config.json" ] && ! has_incomplete "$dest"; then echo "  ✓ ${1##*/} 完整"; return; fi
    echo "  ⚠️ 第 $try 次后仍有 .incomplete，8s 后重试续下..."; sleep 8
  done
  echo "  ❌ ${1##*/} 多次重试仍不完整——网速太差，晚点再跑或换 HF=1"
}

for m in "${LIST[@]}"; do dl "$m"; done

if [ "${#LIST[@]}" -gt 0 ]; then
  echo "==== 模型清单（✓=完整 / ❌=不完整需重跑本脚本续下）===="
  for m in "${LIST[@]}"; do
    d="$MODELS/${m##*/}"
    if [ -f "$d/config.json" ] && ! has_incomplete "$d"; then st="✓ $(du -sh "$d" 2>/dev/null | cut -f1)"; else st="❌ 不完整"; fi
    printf "  %-28s %s\n" "${m##*/}" "$st"
  done
fi

# ---- 数据集接入（下载并转 verl parquet；datasets 内部完成下载，故一并放在 download）----
if [ "$DL_DATA" = "1" ]; then
  echo "==== 数据集接入 → verl parquet ===="
  # SEED（训练/蒸馏种子，默认 MATH-lighteval；主线 Omni d4-5 另行准备，见 doc）
  python "$PROJ/data_preprocess/prepare_math.py" --source "$SEED_SOURCE" --hf "$SEED_HF" --subset "$SEED_SUBSET" --out "$SEED_DIR" --data_source math_seed
  # EVAL（held-out，OlymMATH），仅评测
  python "$PROJ/data_preprocess/prepare_math.py" --source "$EVAL_SOURCE" --hf "$EVAL_HF" --subset "$EVAL_SUBSET" --out "$EVAL_DIR" --data_source olymmath
  # 扩展 benchmark（code / 选择题），需要时取消注释
  # python "$PROJ/data_preprocess/prepare_code.py" --version "$CODE_VERSION" --out "$DATA/livecodebench"
  # python "$PROJ/data_preprocess/prepare_mc.py"   --hf "$MMLU_PRO_HF" --subset default --out "$DATA/mmlu_pro" --data_source mmlu_pro
  echo "数据集接入完成：SEED=$SEED_DIR  EVAL=$EVAL_DIR"
fi
