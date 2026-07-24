"""LiveCodeBench → verl RL parquet（供 base eval / GRPO）。
⚠️ 代码判定需执行单测：GRPO/eval 走 verl 的 sandbox_fusion / prime_code（见 train/grpo.sh 注释），
   或用 LiveCodeBench 官方 harness。本脚本只负责把数据整理成 verl parquet（prompt + 测试用例）。

⚠️⚠️ 数据源坑（与 MATH-lighteval 同类）：`livecodebench/code_generation_lite` 是**脚本型数据集**，
   `datasets 5.0` 已禁用脚本加载 → `--source hf` 会报 "Dataset scripts are no longer supported" 直接崩。
   解决：先把 LCB parquet 抓到本地（modelscope 镜像 / 官方 repo release / HF parquet 分支），再 `--source local`
   指向该目录。首次接入前必须先确认能拿到 parquet 版本，否则本脚本跑不通。

用法（服务器）：
  # 推荐：本地 parquet 目录
  python prepare_code.py --source local --hf /data/liujiachen/datasets/_lcb_raw \
    --out /data/liujiachen/datasets/livecodebench --data_source livecodebench
注意：LiveCodeBench 列名随版本变化，脚本会先打印实际列名，按需在 build_row 里调整。
"""

import argparse
import base64
import glob
import json
import os
import pickle
import zlib

import datasets

# 类型化 prompt：stdin(AtCoder/CF)读标准输入；functional(LeetCode)补全 starter_code 里的方法。
STDIN_TMPL = (
    "You are an expert competitive programmer. Solve the problem in Python 3. "
    "Read input from standard input and write the answer to standard output. "
    "Put the complete solution in a single ```python code block.\n\n{q}"
)
FUNC_TMPL = (
    "You are an expert competitive programmer. Complete the Python solution below by implementing "
    "the required method. Put the complete solution in a single ```python code block.\n\n"
    "{q}\n\n### Complete this starter code:\n```python\n{starter}\n```"
)


def _decode_private(raw):
    """LCB private_test_cases 解码：优先直接 json；否则 base64→zlib→pickle→json（LCB 官方编码）。
    ⚠️ pickle.loads 仅用于官方 LCB 数据（可信来源）；本步在数据准备期一次性做，不在评测热路径。"""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def _to_prime(ex):
    """LCB 一题的 public+private 测试 → prime_code 期望格式。
    inputs[i]/outputs[i] 一律用 LCB 的 input/output **原始字符串**——prime_code 内部按类型自解析：
      有 fn_name → call_based（对 input 每行 json.loads 成参数、output json.loads 成期望返回）；
      无 fn_name → standard_input（input 原样作 stdin、output 作期望 stdout）。
    ⚠️ 千万别在这里预解析 functional 的参数：prime_code(testing_util 222-223) 会再 json.loads 一次，
       预解析后它对 list 做 .split('\\n') 会类型错 → 判分全 0（血泪教训）。"""
    pub = json.loads(ex["public_test_cases"]) if ex.get("public_test_cases") else []
    tests = list(pub) + list(_decode_private(ex.get("private_test_cases")))
    functional = any(t.get("testtype") == "functional" for t in tests)
    prime = {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests]}
    if functional:
        meta = ex.get("metadata")
        meta = json.loads(meta) if isinstance(meta, str) and meta else (meta or {})
        if meta.get("func_name"):
            prime["fn_name"] = meta["func_name"]
    return prime


def build_row(ex, idx, split, data_source):
    q = ex.get("question_content") or ex.get("question") or ""
    starter = ex.get("starter_code") or ""
    prime = _to_prime(ex)
    functional = "fn_name" in prime
    content = FUNC_TMPL.format(q=q, starter=starter) if functional else STDIN_TMPL.format(q=q)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": content}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(prime, ensure_ascii=False)},
        "extra_info": {
            "split": split, "index": idx,
            "question_id": ex.get("question_id", idx),
            "difficulty": ex.get("difficulty", ""),
            "platform": ex.get("platform", ""),
            "testtype": "functional" if functional else "stdin",
            "n_tests": len(prime["inputs"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="livecodebench/code_generation_lite", help="hf 名 或 local parquet 目录")
    ap.add_argument("--source", default="hf", choices=["hf", "local"],
                    help="hf=直连(datasets5.0 对脚本型 LCB 会崩) / local=从本地 parquet 目录读(推荐)")
    ap.add_argument("--version", default="release_v5", help="LiveCodeBench version_tag（仅 --source hf）")
    ap.add_argument("--out", default="/data/liujiachen/datasets/livecodebench")
    ap.add_argument("--data_source", default="livecodebench")
    # LCB 防污染时间窗（按 contest_date 过滤）。Qwen3 技术报告评 LCB v5 用 2024-08-01~2025-02-01，
    # 与之对齐才可比 + 防污染。留空则不过滤（全量）。
    ap.add_argument("--date_start", default=None, help="只留 contest_date >= 此 (YYYY-MM-DD)")
    ap.add_argument("--date_end", default=None, help="只留 contest_date <= 此 (YYYY-MM-DD)")
    a = ap.parse_args()

    if a.source == "local":
        root = os.path.expanduser(a.hf)
        pq = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
        jl = sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))
        if pq:
            ds = {"test": datasets.load_dataset("parquet", data_files=pq)["train"]}
        elif jl:  # LCB 原始 test5.jsonl（release_v5）即走这里
            ds = {"test": datasets.load_dataset("json", data_files=jl)["train"]}
        else:
            raise SystemExit(f"[fatal] 本地无 parquet/jsonl：{root}（先把 LCB 数据抓到此目录）")
    else:
        # ⚠️ datasets 5.0 对脚本型数据集会抛 "Dataset scripts are no longer supported"
        ds = datasets.load_dataset(a.hf, version_tag=a.version, trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    print("splits:", list(ds.keys()), "| columns:", ds[split].column_names, flush=True)

    src = ds[split]
    if "question_id" in src.column_names:  # 多版本 jsonl 并集按 question_id 去重（保留首个）
        n0 = len(src)
        src = datasets.Dataset.from_pandas(
            src.to_pandas().drop_duplicates(subset="question_id", keep="first").reset_index(drop=True)
        )
        if len(src) != n0:
            print(f"去重: {n0} -> {len(src)}（按 question_id）", flush=True)

    if a.date_start or a.date_end:  # 防污染时间窗
        if "contest_date" not in src.column_names:  # fail-closed：宁可报错也不输出"未过滤/可能被污染"的全量
            raise SystemExit(f"[fatal] 指定了防污染时间窗，但数据无 contest_date 列（实际列: {src.column_names}）——"
                             f"拒绝输出未过滤的全量。请确认日期列名或去掉 --date_start/--date_end。")
        s, e = a.date_start, a.date_end

        def _in_window(ex):
            d = str(ex.get("contest_date") or "")[:10]
            return bool(d) and (not s or d >= s) and (not e or d <= e)

        n0 = len(src)
        src = src.filter(_in_window)
        print(f"日期窗[{s}~{e}]: {n0} -> {len(src)}", flush=True)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    d = src.map(lambda ex, i: build_row(ex, i, "test", a.data_source), with_indices=True)
    n0 = len(d)
    # 丢 0 测试用例的题：prime_code 对空测试 all([])==True 会判满分 → 假性抬高 pass@1 / GRPO 白送 reward
    d = d.filter(lambda ex: ex["extra_info"]["n_tests"] > 0)
    if len(d) != n0:
        print(f"[warn] 丢弃 {n0 - len(d)} 道 0 测试用例的题（避免假满分）", flush=True)
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    d = d.remove_columns([c for c in d.column_names if c not in keep])
    d.to_parquet(os.path.join(out, "test.parquet"))
    with open(os.path.join(out, "test_example.json"), "w") as f:
        json.dump(d[0], f, ensure_ascii=False, indent=2)
    print(f"livecodebench {a.version}: {len(d)} rows -> {out}/test.parquet", flush=True)
    print("提示：code 评测/GRPO 需 sandbox（verl sandbox_fusion）或 LCB 官方 harness，见 train/grpo.sh。")


if __name__ == "__main__":
    main()
