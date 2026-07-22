#!/usr/bin/env bash
# 下载课题所需模型到 $MODELS（默认 /data/liujiachen/models）。国内优先 modelscope；已存在则跳过（可断点续）。
# 用法：
#   bash download_models.sh          # 只下 T1 必需（student + 2 teacher + embedding，~40G）——先跑这个就能开工
#   bash download_models.sh t3       # 追加 T3 强度轴 teacher（14B + 32B-AWQ，~46G）
#   bash download_models.sh all      # 全下
#   HF=1 bash download_models.sh     # 改用 HuggingFace 镜像(hf-mirror)下
set -euo pipefail
source "$(dirname "$0")/env.sh"   # 取 $MODELS + HF_ENDPOINT

# T1 必需：唯一 student=Qwen3-4B(instruct,带thinking) + 主力teacher 8B + 专精teacher Math-7B + metrics embedding
T1=(Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen2.5-Math-7B-Instruct Qwen/Qwen3-Embedding-0.6B)
# T3 强度轴 teacher（做到 T3 再下）
T3=(Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ)

case "${1:-t1}" in
  t1)  LIST=("${T1[@]}") ;;
  t3)  LIST=("${T3[@]}") ;;
  all) LIST=("${T1[@]}" "${T3[@]}") ;;
  *)   echo "用法: bash download_models.sh [t1|t3|all]"; exit 1 ;;
esac

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
echo "完成。$MODELS 下已就绪：${LIST[*]##*/}"
