#!/usr/bin/env bash
# 环境自检 + verl 入口核对（M0→M1）。装完 env 先跑这个。
set -uo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOGS/run"; exec > >(tee -a "$LOGS/run/$(basename "$0" .sh).log") 2>&1  # 全部输出落 $LOGS/run/

echo "=== 1) 关键包导入 ==="
python - <<'PY'
for m in ["torch","vllm","verl","math_verify","transformers","datasets","ray"]:
    try:
        __import__(m); print(f"  OK  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
import torch; print("  cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
try:
    import verl; print("  verl:", getattr(verl,"__version__","?"))
except Exception: pass
PY

echo; echo "=== 2) verl 入口核对（应含 sft_trainer / main_ppo / main_generation_server / main_eval）==="
python - <<'PY'
import importlib.util as u
for m in ["verl.trainer.sft_trainer","verl.trainer.main_ppo","verl.trainer.main_generation_server","verl.trainer.main_eval"]:
    print(f"  {'OK ' if u.find_spec(m) else 'MISSING'} {m}")
PY

echo; echo "=== 3) 下一步 ==="
echo "  bash run/01_task1_data_and_base_eval.sh   # 数据 + base eval"
echo "  首次训练/生成务必 TEST=1 起（见 train/*.sh、项目 doc/RUNBOOK.md）"
