"""深度归因：读 eval_math.py 的 per_question.jsonl。
单模型模式：按 level/type/subject/difficulty 切片统计准确率 + dump 错例，供人工做错误类型分析。
配对模式(--vs)：对比两个模型在同一批题上的表现——逐切片 Δ准确率 + 谁修好/谁弄坏的题(McNemar 混淆)。
    这是"只有 2 种方法时"做归因的主力：统计单位是题(n=题数)，扎实；
    不靠跨方法相关系数那种 n=2/3 的伪相关（见 attribution.py，点数<3 相关无意义）。
用法：
  # 单模型切片
  python slice_eval.py --jsonl $LOGS/eval/olymmath_sft_standard/per_question.jsonl --by type --dump 5
  # 配对对比：reverse 相对 standard 在哪类题上提升/退步
  python slice_eval.py --jsonl $LOGS/eval/olymmath_sft_standard/per_question.jsonl \
      --vs $LOGS/eval/olymmath_sft_reverse/per_question.jsonl \
      --name_a standard --name_b reverse --by type --dump 5
"""

import argparse
import json
from collections import defaultdict


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def single(rows, by, dump):
    acc = defaultdict(lambda: [0.0, 0])  # key -> [avg 累加, 计数]
    wrong = defaultdict(list)
    for r in rows:
        key = r.get(by, "NA")
        avg = r.get("avg", 0.0)
        acc[key][0] += avg
        acc[key][1] += 1
        if avg < 1e-9:  # 全采样都错
            wrong[key].append(r.get("question", "")[:180])

    lines = [f"# 切片准确率（by {by}）", "", "| 组 | 题数 | 平均准确率(avg@n) |", "|---|---|---|"]
    for key, (s, n) in sorted(acc.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
        lines.append(f"| {key} | {n} | {s / n:.3f} |")
    lines += ["", f"## 错例（每组 ≤{dump} 条，供人工归类错误类型）"]
    for key, qs in wrong.items():
        lines.append(f"\n### {key}（全错 {len(qs)} 题）")
        for q in qs[:dump]:
            lines.append(f"- {q}")
    return "\n".join(lines)


def paired(rows_a, rows_b, by, dump, name_a, name_b):
    """按题面对齐两个模型的逐题结果 → 逐切片 Δ + McNemar 混淆 + 修好/弄坏错例。"""
    A = {r["question"]: r for r in rows_a}
    B = {r["question"]: r for r in rows_b}
    common = [q for q in A if q in B]
    solved = lambda r: r.get("avg", 0.0) >= 0.5  # 多数采样答对＝"解出"（n=8 时 ≥4/8）

    per = defaultdict(lambda: [0.0, 0.0, 0])  # key -> [sumA, sumB, n]
    fixed, broke = defaultdict(list), defaultdict(list)  # B 修好 / B 弄坏
    both_r = both_w = a_only = b_only = 0
    for q in common:
        ra, rb = A[q], B[q]
        key = ra.get(by, "NA")
        per[key][0] += ra.get("avg", 0.0)
        per[key][1] += rb.get("avg", 0.0)
        per[key][2] += 1
        sa, sb = solved(ra), solved(rb)
        if sa and sb:
            both_r += 1
        elif not sa and not sb:
            both_w += 1
        elif sa and not sb:
            a_only += 1
            broke[key].append(q[:180])
        else:
            b_only += 1
            fixed[key].append(q[:180])

    m = max(1, len(common))
    accA = sum(A[q].get("avg", 0.0) for q in common) / m
    accB = sum(B[q].get("avg", 0.0) for q in common) / m
    lines = [
        f"# 配对对比：{name_b} 相对 {name_a}（{len(common)} 道共同题）", "",
        f"- 总体 avg@n：{name_a}={accA:.3f}  {name_b}={accB:.3f}  Δ={accB - accA:+.3f}",
        f"- McNemar 混淆(阈值 avg≥0.5=解出)：都对 {both_r} | 都错 {both_w} | "
        f"仅 {name_a} 对(被 {name_b} 弄坏) {a_only} | 仅 {name_b} 对({name_b} 修好) {b_only}",
        "", f"## 逐切片 Δ准确率（by {by}，按 Δ 升序，负=退步）",
        f"| 组 | 题数 | {name_a} | {name_b} | Δ |", "|---|---|---|---|---|",
    ]
    for key, (sa, sb, n) in sorted(per.items(), key=lambda kv: (kv[1][1] - kv[1][0]) / max(1, kv[1][2])):
        lines.append(f"| {key} | {n} | {sa / n:.3f} | {sb / n:.3f} | {(sb - sa) / n:+.3f} |")

    lines += ["", f"## {name_b} 修好的题（{name_a} 错 → {name_b} 对，每组 ≤{dump}）"]
    for key, qs in fixed.items():
        if qs:
            lines.append(f"\n### {key}（{len(qs)} 题）")
            lines += [f"- {q}" for q in qs[:dump]]
    lines += ["", f"## {name_b} 弄坏的题（{name_a} 对 → {name_b} 错，每组 ≤{dump}）"]
    for key, qs in broke.items():
        if qs:
            lines.append(f"\n### {key}（{len(qs)} 题）")
            lines += [f"- {q}" for q in qs[:dump]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="eval_math per_question.jsonl（模型 A）")
    ap.add_argument("--vs", default=None, help="第二个模型的 per_question.jsonl；给了就进配对对比模式")
    ap.add_argument("--by", default="type", help="切片字段：level / type / subject / difficulty")
    ap.add_argument("--dump", type=int, default=3, help="每组 dump 几条错例/差异题面")
    ap.add_argument("--name_a", default="A", help="配对模式下模型 A 的名字")
    ap.add_argument("--name_b", default="B", help="配对模式下模型 B 的名字")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load(a.jsonl)
    text = paired(rows, load(a.vs), a.by, a.dump, a.name_a, a.name_b) if a.vs else single(rows, a.by, a.dump)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
