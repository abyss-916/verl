#!/usr/bin/env bash
# 折叠 LoRA → 完整 HF 模型（供 vLLM eval）。取代之前手工合并，可复现。
#   verl model_merger 对 LoRA ckpt 只抽出 base + lora_adapter/（peft 格式）、【不折叠】；
#   本脚本第 2 步用 peft merge_and_unload 把 adapter 折进权重，得到 $CKPT/sft_<法>_merged。
#   第 3 步自检：merged 与 base 逐张量相对差 —— rel≥~1e-2 才说明 LoRA 真学到了
#   （2026-07-26 曾因 LR=1e-5 欠拟合，merged≈base、rel~1e-4，eval 假性等于 base；此自检即为拦截该情形）。
#
# 用法（服务器，先 source env.sh；CPU 即可，无需 GPU）：
#   METHODS="omni_standard_cot omni_reverse omni_question_aug" bash run/07_merge.sh
# 单法：METHODS="omni_standard_cot" bash run/07_merge.sh
set -uo pipefail   # 不加 -e：某法失败要跳过、不拖垮其余
source "$(dirname "$0")/env.sh"

BASE=${BASE:-$MODELS/Qwen3-4B}                 # LoRA 冻结 base，故 base=原始 student。⚠ GRPO 叠 SFT-merged 时传 BASE=$CKPT/sft_<法>_merged，末尾自检才量的是 GRPO 增量而非 SFT+GRPO 合量
PREFIX=${PREFIX-sft_}                           # ckpt 目录前缀（去冒号:空串也生效,非仅未设）：SFT=sft_（默认,向后兼容）；GRPO 传 PREFIX=grpo_ + METHODS=omni_<法>（或 PREFIX= 空 + METHODS=grpo_<法>）
METHODS=${METHODS:-"omni_standard_cot omni_reverse omni_question_aug"}

for M in $METHODS; do
  CKPT_DIR="$CKPT/${PREFIX}$M"
  STEP=$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$STEP" ]; then echo "!! [$M] 无 global_step 于 $CKPT_DIR，跳过"; continue; fi
  VM="$CKPT/${PREFIX}${M}_vmerge"; OUT="$CKPT/${PREFIX}${M}_merged"
  echo "==== [$M] step=$STEP -> $OUT ===="
  rm -rf "$VM" "$OUT"

  # 1) verl 抽 base + lora_adapter（peft 格式）。若报分布式相关错，改用：torchrun --nproc_per_node 1 -m verl.model_merger ...
  python -m verl.model_merger merge --backend fsdp --local_dir "$STEP" --target_dir "$VM" \
    || { echo "!! [$M] model_merger 失败，跳过"; continue; }
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

  # 3) 自检：merged vs base 逐张量相对差（rel≳1e-3=学到了；~1e-4=没训动，须查 LR/数据）
  #    阈值订正：rank-32 LoRA 低秩修正下单张量相对变化本在 ~1e-3 量级（Omni 三法实测 4–7e-3、
  #    确有下降+权重移动），故判据为 1e-3 而非初设的 1e-2（后者对 LoRA 偏严、会误报"没训动"）。
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
