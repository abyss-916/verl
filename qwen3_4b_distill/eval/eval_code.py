"""代码评测（LiveCodeBench）：vLLM 生成 → 抽取 ```python 代码 → prime_code/sandbox 判分 → pass@1 / pass@k。
判分复用 reward/code_reward.py（本地 prime_code，或设 SANDBOX_FUSION_URL 用沙箱）。
thinking 开启时 Qwen3 会先思考再输出代码，故 max_new 默认放大并统计 truncated_rate。

用法：
  python eval_code.py --model $MODELS/Qwen3-4B \
    --data $DATA/livecodebench/test.parquet --n 1 --out $LOGS/eval/lcb_base
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
    ap.add_argument("--gpu_mem", type=float, default=0.8, help="vLLM 显存占比；与他人共卡时调低(如 0.7)")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=38912, help="满生成预算（thinking+代码）；对齐 eval_math/报告")
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
    def _meta(r):
        ex = r["extra_info"] if "extra_info" in df.columns and r["extra_info"] is not None else {}
        try:
            ex = dict(ex)
        except Exception:
            ex = {}
        return {k: ex[k] for k in ("difficulty", "platform", "testtype", "question_id") if ex.get(k) is not None}

    items = [(r["prompt"][0]["content"], r["reward_model"]["ground_truth"], _meta(r)) for _, r in df.iterrows()]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ck = {} if not a.no_thinking else {"enable_thinking": False}
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
               for q, _, _ in items]
    # max_model_len 按最长题面动态确定，保证每题留足 max_new 生成预算
    max_prompt = max((len(tok(p)["input_ids"]) for p in prompts), default=0)
    max_model_len = min(40960, a.max_new + max_prompt + 256)
    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=max_model_len)
    outs = llm.generate(prompts, SamplingParams(temperature=a.temp, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_new, n=a.n))

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    sum_avg, sum_passk, n_trunc, n_tok, n_gen = 0.0, 0.0, 0, 0, 0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt, meta), o in zip(items, outs):
            scores = [1 if compute_score("livecodebench", extract_code(s.text), gt) >= 1.0 else 0 for s in o.outputs]
            trunc = [1 if s.finish_reason == "length" else 0 for s in o.outputs]
            toks = [len(s.token_ids) for s in o.outputs]
            nc, nn = sum(scores), len(scores)
            avg, pk = nc / nn, pass_at_k(nn, nc, k)
            sum_avg += avg
            sum_passk += pk
            n_trunc += sum(trunc)
            n_tok += sum(toks)
            n_gen += nn
            # 逐题记录：含 question/avg/pass@k 及切片字段，供 slice_eval 与 merge_shards 使用
            f.write(json.dumps({"question": q, **meta, "n_pass": nc, "n": nn,
                                "avg": avg, f"pass@{k}": pk,
                                "n_truncated": sum(trunc), "new_tokens": toks}, ensure_ascii=False) + "\n")

    N = len(items)
    summary = {"model": a.model, "data": a.data, "n_samples": a.n, "thinking": not a.no_thinking,
               "num_questions": N,
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
