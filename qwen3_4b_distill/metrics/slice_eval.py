"""深度归因：读 eval_math.py 的 per_question.jsonl，按 level/type/subject/difficulty 切片统计准确率，
并 dump 每组的错例题面，供人工做错误类型分析（证据/结构/计算/执行）。
用法：
  python slice_eval.py --jsonl $LOGS/eval/olymmath_sft_standard_cot/per_question.jsonl --by type --dump 5
  # 对比两个模型在同切片上的提升：分别跑再看两张表
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="eval_math per_question.jsonl")
    ap.add_argument("--by", default="type", help="切片字段：level / type / subject / difficulty")
    ap.add_argument("--dump", type=int, default=3, help="每组 dump 几条错例题面")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    acc = defaultdict(lambda: [0.0, 0])  # key -> [avg 累加, 计数]
    wrong = defaultdict(list)
    for r in rows:
        key = r.get(a.by, "NA")
        avg = r.get("avg", 0.0)
        acc[key][0] += avg
        acc[key][1] += 1
        if avg < 1e-9:  # 全采样都错
            wrong[key].append(r.get("question", "")[:180])

    # 按切片准确率排序
    lines = [f"# 切片准确率（by {a.by}）", "", "| 组 | 题数 | 平均准确率(avg@n) |", "|---|---|---|"]
    for key, (s, n) in sorted(acc.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
        lines.append(f"| {key} | {n} | {s / n:.3f} |")

    lines += ["", f"## 错例（每组 ≤{a.dump} 条，供人工归类错误类型）"]
    for key, qs in wrong.items():
        lines.append(f"\n### {key}（全错 {len(qs)} 题）")
        for q in qs[: a.dump]:
            lines.append(f"- {q}")

    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
