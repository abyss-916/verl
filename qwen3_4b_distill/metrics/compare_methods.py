"""把多个方法/teacher 的数据度量结果（data_metrics.py 产出的 json）汇成一张对比表（任务二/三用）。

用法（服务器）：
  # 先分别对三法数据跑 data_metrics.py --out standard.json / reverse.json / qaug.json
  python compare_methods.py --in standard_cot=standard.json reverse=reverse.json question_aug=qaug.json \
    --out compare_methods.md
"""

import argparse
import json


FIELDS = [
    ("n_samples", lambda d: d.get("n_samples")),
    ("len_mean", lambda d: round(d.get("answer_len_tokens", {}).get("mean", 0), 1)),
    ("len_p90", lambda d: d.get("answer_len_tokens", {}).get("p90")),
    ("distinct_1", lambda d: d.get("distinct_1")),
    ("distinct_2", lambda d: d.get("distinct_2")),
    ("ppl(student)", lambda d: round(d["ppl_student_view_mean"], 2) if d.get("ppl_student_view_mean") else None),
    ("ifd_mean", lambda d: round(d["ifd_mean"], 3) if d.get("ifd_mean") else None),
    ("ifd>=1 ratio", lambda d: round(d["ifd_ge1_ratio"], 3) if d.get("ifd_ge1_ratio") is not None else None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="name=path.json ...")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = {}
    for kv in a.inputs:
        name, path = kv.split("=", 1)
        with open(path) as f:
            data[name] = json.load(f)

    names = list(data.keys())
    header = "| 指标 | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [header, sep]
    for label, fn in FIELDS:
        row = [str(fn(data[n])) for n in names]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    table = "\n".join(lines)

    print(table)
    if a.out:
        with open(a.out, "w") as f:
            f.write("# 蒸馏方法数据度量对比\n\n" + table + "\n")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
