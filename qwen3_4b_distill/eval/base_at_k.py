"""从一次 eval 的 per_question.jsonl（每题存有 n 个 samples）在任意 k≤n 上重算
pass@1 / pass@k / cons@k，使不同 n 的运行能在同一 k 上公平对比。

- pass@1 (avg@n)：全部样本的均值。
- pass@k：用全部 n 个样本算无偏估计 pass_at_k(n, c, k)。
- cons@k：对每题在 C(n,k) 个 k-子集（至多 --max_subsets 个，确定性均匀取样）上各做一次多数投票、
  判众数答案对错后取平均。

用法：
  python eval/base_at_k.py --pq $LOGS/eval/olymmath_base/per_question.jsonl --k 4
"""

import argparse
import json
import math
from collections import Counter
from itertools import combinations


def pass_at_k(n, c, k):
    """无偏 pass@k：n 个样本里 c 个正确。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def extract_boxed(text):
    """取最后一个 \\boxed{...}，括号配平支持任意层嵌套。"""
    key = "\\boxed{"
    i = text.rfind(key)
    if i == -1:
        return None
    depth, j = 1, i + len(key)
    start = j
    while j < len(text) and depth:
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1
    return text[start:j - 1].strip() if depth == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq", required=True, help="per_question.jsonl（须含 samples/gt）")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_subsets", type=int, default=70,
                    help="cons@k 平均用的最大 k-子集数（C(8,4)=70，默认全取）")
    a = ap.parse_args()

    from verl.utils.reward_score.math_verify import compute_score

    rows = [json.loads(line) for line in open(a.pq) if line.strip()]
    sum_avg = sum_pk = sum_cons = 0.0
    n_tok = n_gen = n_trunc = 0
    N = 0
    for r in rows:
        samples = r["samples"]
        gt = str(r["gt"])
        n = len(samples)
        if n < a.k:
            raise SystemExit(f"❌ 某题只有 {n} 个样本 < k={a.k}，无法重算")

        # 每个样本是否正确
        corr = [1 if compute_score(s, gt) >= 1.0 else 0 for s in samples]
        c = sum(corr)
        # 每个样本的 boxed 答案 + 该答案对错（cons 用）
        boxeds = [extract_boxed(s) for s in samples]
        box_ok = {b: (compute_score("\\boxed{" + b + "}", gt) >= 1.0)
                  for b in {b for b in boxeds if b}}

        avg = c / n
        pk = pass_at_k(n, c, a.k)

        # cons@k：在至多 max_subsets 个 k-子集上多数投票取众数、判对错、平均
        subsets = list(combinations(range(n), a.k))
        if len(subsets) > a.max_subsets:                      # 确定性均匀抽取
            step = len(subsets) / a.max_subsets
            subsets = [subsets[int(i * step)] for i in range(a.max_subsets)]
        hits = 0
        for sub in subsets:
            bs = [boxeds[i] for i in sub if boxeds[i]]
            if not bs:
                continue
            mode = Counter(bs).most_common(1)[0][0]
            hits += 1 if box_ok.get(mode, False) else 0
        cons = hits / len(subsets) if subsets else 0.0

        sum_avg += avg
        sum_pk += pk
        sum_cons += cons
        N += 1
        # 若原文件带有 new_tokens/n_truncated，一并汇报
        if "new_tokens" in r:
            n_tok += sum(r["new_tokens"])
            n_gen += len(r["new_tokens"])
        n_trunc += r.get("n_truncated", 0)

    out = {
        "pq": a.pq, "k": a.k, "num_questions": N,
        "pass@1 (avg@n)": round(sum_avg / N, 4) if N else 0,
        f"pass@{a.k}": round(sum_pk / N, 4) if N else 0,
        f"cons@{a.k}": round(sum_cons / N, 4) if N else 0,
    }
    if n_gen:
        out["truncated_rate"] = round(n_trunc / n_gen, 4)
        out["mean_new_tokens"] = round(n_tok / n_gen, 1)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
