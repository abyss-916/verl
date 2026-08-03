#!/usr/bin/env python
"""TRL GRPO(单进程,绕开 verl 的 Ray/colocation 段错)| Qwen3-4B + LoRA | Omni d4-5 prompt。

- reward 复用 verl 内置 math-verify 判分(与原方案 1:1 一致:答对 1.0 / 否则 0.0)。
- 生成后端二选一(GRPO 瓶颈在生成):
    USE_VLLM=1 : vLLM server 模式(快、可长)。需先另开进程 `trl vllm-serve ...`(见 train/trl_grpo.sh)。
                 ⚠️ 本机 vLLM 0.12 低于 TRL 建议的 0.17+,server/LoRA 权重同步可能不兼容——不行就退 HF 生成。
    USE_VLLM=0 : HF 生成(transformers generate,单进程最稳,但长补全慢——正式跑需把 RESP 压到可行区间)。
- 冒烟:USE_VLLM=0 MAXSTEPS=3 RESP=512 ...(~2-3min,验管线通,不需 vLLM server)。

用法见 train/trl_grpo.sh。
"""
import os
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
# verl 的 math-verify 判分器(子进程 30s 超时保护;签名 compute_score(solution_str, ground_truth)-> 1.0/0.0）
from verl.utils.reward_score.math_verify import compute_score as _vscore

MODEL = os.environ["MODEL_PATH"]
TRAIN = os.environ.get("TRAIN_DIR", "/data/liujiachen/datasets/omni_seed")
OUT = os.environ.get("OUT", "/data/liujiachen/checkpoints/grpo_trl_omni_standard")
N = int(os.environ.get("N", "5"))                     # 组内采样数(GRPO group size)
RESP = int(os.environ.get("RESP", "8192"))            # max_completion_length
LR = float(os.environ.get("LR", "1e-5"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
PBS = int(os.environ.get("PBS", str(N)))              # per-device batch，须是 N 的倍数
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
BETA = float(os.environ.get("BETA", "0.0"))           # KL 系数(0=不加载 ref，省显存/更快；LoRA 本身有约束)
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"
MAXSTEPS = int(os.environ.get("MAXSTEPS", "-1"))      # >0 时覆盖 epochs（冒烟用）
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "20"))

# ---- 数据：Omni prompt(chat 消息列表) + ground_truth ----
df = pd.read_parquet(f"{TRAIN}/train.parquet")


def _gt(rm):
    return str(rm.get("ground_truth", "")) if isinstance(rm, dict) else str(rm)


ds = Dataset.from_dict({
    "prompt": [list(p) for p in df["prompt"]],                # [{'role':'user','content': '...\\boxed{}'}]
    "ground_truth": [_gt(rm) for rm in df["reward_model"]],
})
print(f"[trl_grpo] dataset={len(ds)} rows  RESP={RESP} N={N} LR={LR} EPOCHS={EPOCHS} "
      f"PBS={PBS} GA={GRAD_ACCUM} BETA={BETA} USE_VLLM={USE_VLLM} MAXSTEPS={MAXSTEPS}")


# ---- reward：复用 verl math-verify（答对 1.0 / 否则 0.0）----
def math_reward(completions=None, ground_truth=None, **kwargs):
    scores = []
    for comp, gt in zip(completions, ground_truth):
        # 会话式：completion 是 [{'role':'assistant','content': ...}]；取最后一段文本
        text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
        try:
            scores.append(float(_vscore(text, str(gt))))
        except Exception:
            scores.append(0.0)
    return scores


vllm_kwargs = {}
if USE_VLLM:
    vllm_kwargs = dict(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=os.environ.get("VLLM_HOST", "localhost"),
        vllm_server_port=int(os.environ.get("VLLM_PORT", "8000")),
        vllm_server_timeout=1200.0,
    )

cfg = GRPOConfig(
    output_dir=OUT,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    max_steps=MAXSTEPS,
    per_device_train_batch_size=PBS,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_generations=N,
    max_completion_length=RESP,
    temperature=1.0,
    beta=BETA,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    log_completions=True,
    num_completions_to_print=1,
    report_to="none",           # 靠 logging_steps=1 的控制台 reward/kl 看信号；避免 wandb 额外故障点
    **vllm_kwargs,
)

peft = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules="all-linear", task_type="CAUSAL_LM")

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=math_reward,
    args=cfg,
    train_dataset=ds,
    peft_config=peft,
)
trainer.train()
trainer.save_model(OUT)
print("TRL_GRPO_DONE ->", OUT)
