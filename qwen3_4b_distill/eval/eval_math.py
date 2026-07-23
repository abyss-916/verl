"""数学评测（base / SFT / GRPO 通用）：pass@1、avg@k、pass@k。
- thinking 模式默认开（Qwen3 thinking 下推理类才不被低估；--no_thinking 关）。
- 判定复用 verl 内置 math-verify。
- 输出逐题 jsonl + 汇总 json，便于任务一"与论文对齐"和后续归因。

用法（服务器）：
  python eval_math.py --model /data/liujiachen/models/Qwen3-4B-Base \
    --data /data/liujiachen/datasets/olymmath/test.parquet \
    --n 8 --out /data/liujiachen/logs/eval/olymmath_base
  # avg@64（AIME 类小集降方差）：--n 64
"""

import argparse
import json
import math
import os

import pandas as pd


def pass_at_k(n, c, k):
    """无偏 pass@k 估计（Chen et al. 2021）：n 个样本里 c 个正确。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="RL parquet（prompt + reward_model.ground_truth）")
    ap.add_argument("--out", required=True, help="输出目录前缀")
    ap.add_argument("--n", type=int, default=8, help="每题采样数（pass@1 用均值，pass@k 用无偏估计）")
    ap.add_argument("--k", type=int, default=0, help="pass@k 的 k；默认=n")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比；与他人共卡时调低(如 0.7)")
    ap.add_argument("--temp", type=float, default=0.6)   # thinking 采样默认
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k = a.k or a.n

    from vllm import LLM, SamplingParams

    from verl.utils.reward_score.math_verify import compute_score

    df = pd.read_parquet(a.data)
    if a.limit > 0:
        df = df.iloc[: a.limit]
    def _meta(r):
        ex = r["extra_info"] if "extra_info" in df.columns and r["extra_info"] is not None else {}
        try:
            ex = dict(ex)
        except Exception:
            ex = {}
        return {k: ex[k] for k in ("level", "type", "subject", "difficulty") if ex.get(k) is not None}

    items = [(r["prompt"][0]["content"], r["reward_model"]["ground_truth"], _meta(r)) for _, r in df.iterrows()]

    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_new + 2048)
    tok = llm.get_tokenizer()
    ck = {} if a.no_thinking is False else {"enable_thinking": False}
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
        for q, _ in items
    ]
    sp = SamplingParams(temperature=a.temp, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_new, n=a.n)
    outs = llm.generate(prompts, sp)

    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out = os.path.expanduser(a.out)
    per_q, sum_avg, sum_passk = [], 0.0, 0.0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt, meta), o in zip(items, outs):
            corr = [1 if compute_score(c.text, str(gt)) >= 1.0 else 0 for c in o.outputs]
            c = sum(corr)
            avg = c / len(corr)
            pk = pass_at_k(len(corr), c, k)
            sum_avg += avg
            sum_passk += pk
            per_q.append({"avg": avg, "pass_at_k": pk, "n_correct": c})
            f.write(json.dumps({"question": q, "gt": str(gt), **meta, "n_correct": c, "n": len(corr),
                                "avg": avg, f"pass@{k}": pk,
                                "samples": [c.text for c in o.outputs]}, ensure_ascii=False) + "\n")

    N = len(items)
    summary = {
        "model": a.model, "data": a.data, "n_samples": a.n, "thinking": not a.no_thinking,
        "num_questions": N,
        "pass@1 (avg@n)": round(sum_avg / N, 4) if N else 0,
        f"pass@{k}": round(sum_passk / N, 4) if N else 0,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
