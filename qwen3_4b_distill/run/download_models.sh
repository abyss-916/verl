#!/usr/bin/env bash
# 下载课题所需模型到 $MODELS。**自带环境/路径声明（source env.sh）——你无需自己 export/mkdir 任何路径**。
# 国内优先 modelscope；已存在则跳过（可断点续）。全部输出自动落 $LOGS/run/download_models.log。
# 用法：
#   nohup bash download_models.sh >/dev/null 2>&1 &   # 后台下 T1 必需 4 个(~40G)
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

dl() {  # $1 = repo，如 Qwen/Qwen3-8B
  local repo="$1" dest="$MODELS/${1##*/}"
  if [ -f "$dest/config.json" ]; then echo "跳过(已存在): ${1##*/}"; return; fi
  echo "==== 下载 $repo -> $dest ===="
  mkdir -p "$dest"
  if [ "${HF:-0}" = "1" ]; then
    hf download "$repo" --local-dir "$dest"
  else
    modelscope download --model "$repo" --local_dir "$dest"
  fi
}

for m in "${LIST[@]}"; do dl "$m"; done

echo "==== 全部完成，清单如下（缺失=下载失败，重跑本脚本续下）===="
for m in "${LIST[@]}"; do
  d="$MODELS/${m##*/}"
  if [ -f "$d/config.json" ]; then sz=$(du -sh "$d" 2>/dev/null | cut -f1); else sz="缺失!"; fi
  printf "  %-28s %s\n" "${m##*/}" "$sz"
done
