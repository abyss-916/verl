#!/usr/bin/env bash
# 折叠 LoRA 为完整 HF 模型（供 vLLM eval）。可复现，CPU 即可。
# 用法：METHODS="omni_standard_cot omni_reverse omni_question_aug" bash run/merge.sh
#   单法 METHODS="omni_standard_cot"；GRPO 传 PREFIX=grpo_ + BASE=$CKPT/sft_<法>_merged
set -uo pipefail   # 不加 -e：某法失败时跳过
source "$(dirname "$0")/env.sh"

BASE=${BASE:-$MODELS/Qwen3-4B}                 # LoRA 的 base；GRPO 叠在 SFT-merged 上时传 BASE=$CKPT/sft_<法>_merged
PREFIX=${PREFIX-sft_}                           # ckpt 目录前缀：SFT=sft_（默认）；GRPO 传 PREFIX=grpo_
METHODS=${METHODS:-"omni_standard_cot omni_reverse omni_question_aug"}

for M in $METHODS; do
  CKPT_DIR="$CKPT/${PREFIX}$M"
  STEP=$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$STEP" ]; then echo "!! [$M] 无 global_step 于 $CKPT_DIR，跳过"; continue; fi
  # GRPO ckpt 在 $STEP/actor/，SFT 直接在 $STEP/
  SRC="$STEP"; [ -d "$STEP/actor" ] && SRC="$STEP/actor"
  VM="$CKPT/${PREFIX}${M}_vmerge"; OUT="$CKPT/${PREFIX}${M}_merged"
  echo "==== [$M] src=$SRC -> $OUT ===="
  rm -rf "$VM" "$OUT"

  # 1) verl 抽 base + lora_adapter（peft 格式）
  python -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$VM" \
    || torchrun --nproc_per_node 1 -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$VM" \
    || { echo "!! [$M] model_merger（python 与 torchrun 兜底均）失败，跳过"; continue; }
  if [ ! -d "$VM/lora_adapter" ]; then echo "!! [$M] 未生成 lora_adapter（可能非 LoRA ckpt），跳过"; continue; fi

  # 2) peft 折叠 adapter → 完整模型
  python - "$VM" "$VM/lora_adapter" "$OUT" "$BASE" <<'PY' || { echo "!! [$M] peft 折叠失败"; continue; }
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
vm, adapter, out, base = sys.argv[1:5]
model = AutoModelForCausalLM.from_pretrained(vm, torch_dtype=torch.bfloat16)   # vm 内即抽出的 base
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
print("[merge] merged ->", out)
PY

  # 3) 自检：merged vs base 逐张量相对差（rel≳1e-3 表示学到，~1e-4 未训动）
  python - "$BASE" "$OUT" <<'PY'
import sys, glob, os, statistics as S
from safetensors import safe_open
def idx(d):
    m={}
    for f in sorted(glob.glob(os.path.join(d,"*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for k in st.keys(): m[k]=f
    return m
b, o = idx(sys.argv[1]), idx(sys.argv[2])
common = [k for k in b if k in o and k.endswith(".weight")][:40]
rels=[]
for k in common:
    with safe_open(b[k], framework="pt") as st: A=st.get_tensor(k).float()
    with safe_open(o[k], framework="pt") as st: B=st.get_tensor(k).float()
    rels.append(((A-B).norm()/(A.norm()+1e-9)).item())
med, mx = S.median(rels), max(rels)
flag = "OK 学到了" if med >= 1e-3 else "⚠ 疑似没训动(rel太小),查 LR/数据"
print(f"[{os.path.basename(sys.argv[2])}] rel median/max(40张量)= {med:.2e} / {mx:.2e}  -> {flag}")
PY

  rm -rf "$VM"
  echo "==== [$M] 完成 ===="
done
echo "==== 合并全部结束 ===="
