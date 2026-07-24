"""合并 eval_math.py 的多卡分片结果，产出与单卡跑等价的一份 per_question.jsonl + summary.json。

分片只是把同一份题目按 df.iloc[shard::num_shards] 交错切开分给不同 GPU，
每题的采样/判定完全独立，所以合并 = 把逐题记录并起来重新求平均，指标与单卡一致。

用法：
  python merge_shards.py --shards $LOGS/eval/olymmath_base_s0 $LOGS/eval/olymmath_base_s1 \
      --out $LOGS/eval/olymmath_base
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="各分片的 --out 目录")
    ap.add_argument("--out", required=True, help="合并后的输出目录")
    a = ap.parse_args()

    metas, rows, seen = [], [], set()
    for d in a.shards:
        d = os.path.expanduser(d)
        with open(os.path.join(d, "summary.json")) as f:
            metas.append(json.load(f))
        with open(os.path.join(d, "per_question.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["question"] in seen:      # 分片本应互斥；重复只可能来自误配，跳过并告警
                    print(f"⚠️ 跳过重复题目（{d}）")
                    continue
                seen.add(r["question"])
                rows.append(r)

    # 分片间采样配置必须一致，否则合出来的平均没有意义
    for key in ("model", "data", "n_samples", "thinking", "max_new"):
        vals = {json.dumps(m.get(key), ensure_ascii=False) for m in metas}
        if len(vals) > 1:
            raise SystemExit(f"❌ 各分片 {key} 不一致，拒绝合并：{vals}")

    pk_key = next(k for k in rows[0] if k.startswith("pass@"))
    N = len(rows)
    n_gen = sum(r["n"] for r in rows)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    m = metas[0]
    summary = {
        "model": m["model"], "data": m["data"], "n_samples": m["n_samples"],
        "thinking": m["thinking"], "num_questions": N,
        "pass@1 (avg@n)": round(sum(r["avg"] for r in rows) / N, 4),
        pk_key: round(sum(r[pk_key] for r in rows) / N, 4),
        "max_new": m["max_new"],
        "truncated_rate": round(sum(r.get("n_truncated", 0) for r in rows) / n_gen, 4),
        "mean_new_tokens": round(sum(sum(r.get("new_tokens", [])) for r in rows) / n_gen, 1),
        "merged_from": [os.path.expanduser(d) for d in a.shards],
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
