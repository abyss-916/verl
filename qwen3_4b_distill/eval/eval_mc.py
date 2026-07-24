"""选择题评测（MMLU-Pro / SuperGPQA）：vLLM 生成 → 抽答案字母 → pass@1(avg@n)/pass@k/cons@n（thinking 模式）。
抽取顺序：\\boxed{X} > "answer is X" > 文中最后一个孤立大写字母。
与 eval_math/eval_code 同构：per_question 带 question/avg/pass@k/切片字段 → slice_eval 归因 + merge_shards 分片合并均可用。
用法（服务器）：
  python eval_mc.py --model /data/liujiachen/models/Qwen3-4B \
    --data /data/liujiachen/datasets/mmlu_pro/test.parquet --n 1 --out $LOGS/eval/mmlu_pro_base
  # MMLU-Pro ~12k 题、thinking 长 → 两卡交错分片：各卡 --shard 0/1 --num_shards 2，再 merge_shards.py 合并
⚠️ thinking 开着时先长思考再出字母，故 max_new 放大并统计 truncated_rate（截断=丢字母=判错，重演数学 8192 之雷）。
"""

import argparse
import json
import math
import os
import re
from collections import Counter


def pass_at_k(n, c, k):
    """无偏 pass@k 估计（Chen et al. 2021）：n 个样本里 c 个正确。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def extract_letter(text):
    m = re.search(r"\\boxed\{\s*([A-P])\s*\}", text)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*(?:is|:)\s*\(?\s*([A-P])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-P])\b", text)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--k", type=int, default=0, help="pass@k 的 k；默认=n")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比;与他人共卡时调低")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new", type=int, default=16384, help="thinking+答案;截断率高就调大")
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    # 多卡分片：df.iloc[shard::num_shards] 交错切开，各卡一片、各自 --out，最后 merge_shards.py 合并
    ap.add_argument("--shard", type=int, default=0, help="分片编号，0-based")
    ap.add_argument("--num_shards", type=int, default=1, help="总分片数；=1 即不分片")
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
        return {kk: ex[kk] for kk in ("category", "discipline", "field", "subfield", "difficulty") if ex.get(kk) is not None}

    items = [(r["prompt"][0]["content"], str(r["reward_model"]["ground_truth"]).strip(), _meta(r)) for _, r in df.iterrows()]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ck = {} if not a.no_thinking else {"enable_thinking": False}
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
               for q, _, _ in items]
    # max_model_len 按最长题面动态定：保证每题都留够 max_new 的真实生成预算，长题面才不会把预算挤小→假性截断(不可比)
    max_prompt = max((len(tok(p)["input_ids"]) for p in prompts), default=0)
    max_model_len = min(40960, a.max_new + max_prompt + 256)
    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=max_model_len)
    outs = llm.generate(prompts, SamplingParams(temperature=a.temp, top_p=a.top_p, max_tokens=a.max_new, n=a.n))

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    sum_avg, sum_passk, sum_cons, n_trunc, n_tok, n_gen = 0.0, 0.0, 0.0, 0, 0, 0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt, meta), o in zip(items, outs):
            preds = [extract_letter(c.text) for c in o.outputs]
            corr = [1 if p == gt else 0 for p in preds]
            trunc = [1 if c.finish_reason == "length" else 0 for c in o.outputs]
            toks = [len(c.token_ids) for c in o.outputs]
            nc, nn = sum(corr), len(corr)
            avg, pk = nc / nn, pass_at_k(nn, nc, k)
            # cons@n（多数投票，对齐 OlymMATH Cons）：取众数字母判其是否正确
            votes = [p for p in preds if p]
            cons = 1 if votes and Counter(votes).most_common(1)[0][0] == gt else 0
            sum_avg += avg
            sum_passk += pk
            sum_cons += cons
            n_trunc += sum(trunc)
            n_tok += sum(toks)
            n_gen += nn
            f.write(json.dumps({"question": q, **meta, "gt": gt, "preds": preds,
                                "n_correct": nc, "n": nn, "avg": avg, f"pass@{k}": pk, "cons": cons,
                                "n_truncated": sum(trunc), "new_tokens": toks}, ensure_ascii=False) + "\n")

    N = len(items)
    summary = {"model": a.model, "data": a.data, "n_samples": a.n, "thinking": not a.no_thinking,
               "num_questions": N,
               "pass@1 (avg@n)": round(sum_avg / N, 4) if N else 0,
               f"pass@{k}": round(sum_passk / N, 4) if N else 0,
               f"cons@{a.n}": round(sum_cons / N, 4) if N else 0,
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
