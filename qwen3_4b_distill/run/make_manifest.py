"""汇总单个实验的完整记录 → manifest.md + manifest.json。
满足课题原文 §1："每个实验至少要记录 dataset/model/teacher/蒸馏方法/采样参数/filter/数据统计/
base-SFT-GRPO 结果/论文对齐"——把散落的 gen_stats.json(造数据侧) + eval summary.json(评测侧) +
手填元数据(数据集/教师/学生/论文锚点/阶段/备注) 聚成一份可追溯记录。

用法（服务器）：
  python run/make_manifest.py --name sft_standard_cot --out "$LOGS/manifests" \
    --gen_stats "$DATA/distill/standard_cot/gen_stats.json" \
    --eval "$LOGS/eval/olymmath_sft_standard/summary.json" \
    --set dataset=OlymMATH-en-hard --set student=Qwen3-4B --set teacher=Qwen3-8B \
    --set method=standard_cot --set stage=SFT --set paper_anchor="OlymMATH Qwen3-4B≈13.9%" \
    --set filter="math-verify 拒绝采样 + 完成度过滤" --set notes="..."
"""

import argparse
import json
import os
from datetime import datetime


def load_json(p):
    p = os.path.expanduser(p) if p else None
    if p and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="实验名（manifest 文件名）")
    ap.add_argument("--out", required=True, help="manifest 输出目录（建议 $LOGS/manifests）")
    ap.add_argument("--gen_stats", default=None, help="造数据侧 gen_stats.json")
    ap.add_argument("--eval", default=None, help="评测侧 summary.json")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="key=value 手填元数据（可多次）：dataset/student/teacher/method/采样/filter/paper_anchor/stage/notes")
    a = ap.parse_args()

    gen = load_json(a.gen_stats)
    ev = load_json(a.eval)
    meta = {}
    for kv in a.sets:
        if "=" not in kv:
            raise SystemExit(f"--set 需 key=value 格式，得到: {kv}")
        k, v = kv.split("=", 1)
        meta[k] = v

    manifest = {
        "experiment": a.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "meta": meta,                                  # dataset/model/teacher/方法/采样/filter/论文对齐
        "data_construction": gen,                      # 方法/采样/过滤/良率/截断率/长度分位（gen_stats）
        "evaluation": ev,                              # pass@1/pass@k/截断率/平均长度（summary）
        "sources": {"gen_stats": a.gen_stats, "eval": a.eval},
    }

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    jpath = os.path.join(out, f"{a.name}.manifest.json")
    with open(jpath, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 人读 md
    lines = [f"# 实验 manifest：{a.name}", "", f"- 生成时间：{manifest['timestamp']}", ""]

    def sec(title, d):
        if not d:
            return
        lines.append(f"## {title}")
        for k, v in d.items():
            lines.append(f"- **{k}**：{v}")
        lines.append("")

    sec("元数据（dataset / student / teacher / 方法 / 采样 / filter / 论文对齐 / 阶段）", meta)
    sec("数据构造侧（gen_stats：良率 / 截断率 / 长度分位）", gen)
    sec("评测侧（summary：pass@1 / pass@k / 截断率 / 平均长度）", ev)
    if not (meta or gen or ev):
        lines.append("_（暂无内容——检查 --gen_stats/--eval 路径或 --set 元数据）_")
    mpath = os.path.join(out, f"{a.name}.manifest.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"-> {jpath}\n-> {mpath}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
