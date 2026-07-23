#!/usr/bin/env bash
# 下载课题所需模型到 $MODELS。**自带环境/路径声明（source env.sh）——你无需自己 export/mkdir 任何路径**。
# 国内优先 modelscope；**重试直到没有 .incomplete 残留**（网络差会留半截文件，故重试）；已完整则跳过。
# 全部输出自动落 $LOGS/run/download_models.log。
# 用法：
#   nohup bash download_models.sh >/dev/null 2>&1 &   # 后台下 T1 必需 4 个(~40G)；已完整的会跳过，只补没下全的
#   bash download_models.sh t3                        # 追加 T3(14B+32B-AWQ,~46G)
#   bash download_models.sh all                       # 全下
#   HF=1 bash download_models.sh                       # 改用 HF 镜像(hf-mirror)
set -euo pipefail

# ---- 环境与路径声明（你不用自己声明 logs/models 等路径）----
source "$(dirname "$0")/env.sh"                         # 声明 $MODELS/$LOGS/$HF_ENDPOINT 并 mkdir 好 /data 下各目录
mkdir -p "$LOGS/run" "$MODELS"
exec > >(tee -a "$LOGS/run/download_models.log") 2>&1   # 全部输出落日志（nohup 后台也能查）

echo "==== download start | MODELS=$MODELS ===="
df -h "$(dirname "$MODELS")" | tail -1 | awk '{print "磁盘可用: "$4"（已用 "$5"）@ "$6}'

# ---- 模型清单 ----
T1=(Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen2.5-Math-7B-Instruct Qwen/Qwen3-Embedding-0.6B)  # student+8B teacher+专精teacher+embedding
T3=(Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ)                                                    # 强度轴 teacher（做到 T3 再下）
case "${1:-t1}" in
  t1)  LIST=("${T1[@]}") ;;
  t3)  LIST=("${T3[@]}") ;;
  all) LIST=("${T1[@]}" "${T3[@]}") ;;
  *)   echo "用法: bash download_models.sh [t1|t3|all]"; exit 1 ;;
esac

# ---- 下载器（缺了自动装）----
if [ "${HF:-0}" = "1" ]; then
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  command -v hf >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
else
  command -v modelscope >/dev/null 2>&1 || pip install -U modelscope
fi

# 目录里有没有没下完的临时文件
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

echo "==== 结束，清单（✓=完整 / ❌=不完整需重跑本脚本续下）===="
for m in "${LIST[@]}"; do
  d="$MODELS/${m##*/}"
  if [ -f "$d/config.json" ] && ! has_incomplete "$d"; then st="✓ $(du -sh "$d" 2>/dev/null | cut -f1)"; else st="❌ 不完整"; fi
  printf "  %-28s %s\n" "${m##*/}" "$st"
done
