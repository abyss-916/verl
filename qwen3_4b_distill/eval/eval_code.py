"""代码评测（LiveCodeBench）：vLLM 生成 → 抽 ```python 代码 → prime_code/sandbox 判分 → pass@1 / pass@k。
用法（服务器）：
  python eval_code.py --model /data/liujiachen/models/Qwen3-4B \
    --data /data/liujiachen/datasets/livecodebench/test.parquet --n 1 --out $LOGS/eval/lcb_base
判分复用 reward/code_reward.py（本地 prime_code，或设 SANDBOX_FUSION_URL 用沙箱）。
⚠️ 依赖测试用例格式正确（见 reward/code_reward.py 的 TODO），首次跑需核对。
⚠️ 与 eval_math 一致：thinking 开着时 Qwen3 会先长思考再出代码，故默认 max_new 放大并统计 truncated_rate——
   若截断率高说明预算不足（重演数学 8192 假性归零的雷），据此调大 --max_new。
"""

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reward.code_reward import compute_score  # noqa: E402


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[-1].strip() if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="RL parquet（prompt + reward_model.ground_truth=测试用例）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比；与他人共卡时调低")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new", type=int, default=16384, help="thinking+代码；截断率高就调大")
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()
    k = a.k or a.n

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(a.data)
    if a.limit > 0:
        df = df.iloc[: a.limit]
    if a.num_shards > 1:
        df = df.iloc[a.shard :: a.num_shards]
    items = [(r["prompt"][0]["content"], r["reward_model"]["ground_truth"]) for _, r in df.iterrows()]

    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_new + 2048)
    tok = llm.get_tokenizer()
    ck = {} if not a.no_thinking else {"enable_thinking": False}
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
               for q, _ in items]
    outs = llm.generate(prompts, SamplingParams(temperature=a.temp, top_p=a.top_p, max_tokens=a.max_new, n=a.n))

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    sum_avg, sum_passk, n_trunc, n_tok, n_gen = 0.0, 0.0, 0, 0, 0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt), o in zip(items, outs):
            scores = [1 if compute_score("livecodebench", extract_code(s.text), gt) >= 1.0 else 0 for s in o.outputs]
            trunc = [1 if s.finish_reason == "length" else 0 for s in o.outputs]
            toks = [len(s.token_ids) for s in o.outputs]
            nc, nn = sum(scores), len(scores)
            sum_avg += nc / nn
            sum_passk += pass_at_k(nn, nc, k)
            n_trunc += sum(trunc)
            n_tok += sum(toks)
            n_gen += nn
            f.write(json.dumps({"n_pass": nc, "n": nn, "n_truncated": sum(trunc), "new_tokens": toks},
                               ensure_ascii=False) + "\n")

    N = len(items)
    summary = {"model": a.model, "data": a.data, "n_samples": a.n, "num_questions": N,
               "pass@1 (avg@n)": round(sum_avg / N, 4) if N else 0,
               f"pass@{k}": round(sum_passk / N, 4) if N else 0,
               "max_new": a.max_new,
               "truncated_rate": round(n_trunc / n_gen, 4) if n_gen else 0,
               "mean_new_tokens": round(n_tok / n_gen, 1) if n_gen else 0}
    if a.num_shards > 1:
        summary["shard"] = f"{a.shard}/{a.num_shards}"
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
