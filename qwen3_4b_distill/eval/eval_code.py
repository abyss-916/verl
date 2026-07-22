"""代码评测（LiveCodeBench）：vLLM 生成 → 抽 ```python 代码 → prime_code/sandbox 判分 → pass@1 / pass@k。
用法（服务器）：
  python eval_code.py --model /data/liujiachen/models/Qwen3-4B-Base \
    --data /data/liujiachen/datasets/livecodebench/test.parquet --n 1 --out $LOGS/eval/lcb_base
判分复用 reward/code_reward.py（本地 prime_code，或设 SANDBOX_FUSION_URL 用沙箱）。
⚠️ 依赖测试用例格式正确（见 reward/code_reward.py 的 TODO），首次跑需核对。
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reward.code_reward import compute_score  # noqa: E402


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[-1].strip() if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="RL parquet（prompt + reward_model.ground_truth=测试用例）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(a.data)
    if a.limit > 0:
        df = df.iloc[: a.limit]
    items = [(r["prompt"][0]["content"], r["reward_model"]["ground_truth"]) for _, r in df.iterrows()]

    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=0.85, max_model_len=a.max_new + 2048)
    tok = llm.get_tokenizer()
    ck = {} if not a.no_thinking else {"enable_thinking": False}
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
               for q, _ in items]
    outs = llm.generate(prompts, SamplingParams(temperature=a.temp, top_p=0.95, max_tokens=a.max_new, n=a.n))

    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out = os.path.expanduser(a.out)
    n_pass1, n_passk = 0.0, 0.0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt), o in zip(items, outs):
            scores = [1 if compute_score("livecodebench", extract_code(c.text), gt) >= 1.0 else 0 for c in o.outputs]
            n_pass1 += sum(scores) / len(scores)
            n_passk += 1 if any(scores) else 0
            f.write(json.dumps({"n_pass": sum(scores), "n": len(scores)}, ensure_ascii=False) + "\n")

    N = len(items)
    summary = {"model": a.model, "data": a.data, "n_samples": a.n, "num_questions": N,
               "pass@1 (avg@n)": round(n_pass1 / N, 4) if N else 0,
               f"pass@{a.n}": round(n_passk / N, 4) if N else 0}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
