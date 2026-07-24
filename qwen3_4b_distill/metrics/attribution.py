"""归因分析（课题核心"解释 accuracy 为什么变"）：把"数据侧指标"与"训练后表现"关联。
每个实验点 = data_metrics.py 的 json + eval_math/eval_code 的 summary.json。
输出合并表 + 各数据指标与 pass@1 的相关系数，看哪个数据属性最能解释表现变化。

用法：
  python attribution.py --out $LOGS/attribution.md \
    --point standard_cot:$LOGS/metrics_standard_cot.json:$LOGS/eval/olymmath_sft_standard_cot/summary.json \
    --point reverse:$LOGS/metrics_reverse.json:$LOGS/eval/olymmath_sft_reverse/summary.json \
    --point question_aug:$LOGS/metrics_question_aug.json:$LOGS/eval/olymmath_sft_question_aug/summary.json
"""

import argparse
import json

# 数据侧指标（来自 data_metrics.py）
DATA_FIELDS = [
    ("len_mean", lambda d: d.get("answer_len_tokens", {}).get("mean")),
    ("distinct_2", lambda d: d.get("distinct_2")),
    ("ppl", lambda d: d.get("ppl_student_view_mean")),
    ("ifd_mean", lambda d: d.get("ifd_mean")),
    ("ifd_ge1", lambda d: d.get("ifd_ge1_ratio")),
]


def get_perf(ev):
    """从 eval summary 取 pass@1（eval_math/eval_code 键名为 'pass@1 (avg@n)'）。"""
    for k, v in ev.items():
        if k.startswith("pass@1") or k == "accuracy":
            return v
    return None


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    n = len(pts)
    if n < 3:
        return None
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    dx = sum((x - mx) ** 2 for x, _ in pts) ** 0.5
    dy = sum((y - my) ** 2 for _, y in pts) ** 0.5
    return round(num / (dx * dy), 3) if dx * dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", required=True, help="name:metrics.json:eval_summary.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for p in a.point:
        name, mpath, epath = p.split(":", 2)
        with open(mpath) as f:
            m = json.load(f)
        with open(epath) as f:
            e = json.load(f)
        row = {"name": name, "pass@1": get_perf(e)}
        for label, fn in DATA_FIELDS:
            row[label] = fn(m)
        rows.append(row)

    cols = ["name"] + [c for c, _ in DATA_FIELDS] + ["pass@1"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    table = "\n".join(lines)

    # 相关性：每个数据指标 vs pass@1
    perf = [r["pass@1"] for r in rows]
    corr = ["", "## 数据指标 ↔ pass@1 相关系数（点数≥3 才算）"]
    corr.append("| 数据指标 | Pearson r |\n|---|---|")
    for label, _ in DATA_FIELDS:
        corr.append(f"| {label} | {pearson([r[label] for r in rows], perf)} |")
    corr_txt = "\n".join(corr)

    out = f"# 归因分析：数据属性 → 表现\n\n{table}\n{corr_txt}\n"
    print(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
