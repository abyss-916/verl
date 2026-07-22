"""LiveCodeBench → verl RL parquet（供 base eval / GRPO）。
⚠️ 代码判定需执行单测：GRPO/eval 走 verl 的 sandbox_fusion / prime_code（见 train/grpo.sh 注释），
   或用 LiveCodeBench 官方 harness。本脚本只负责把数据整理成 verl parquet（prompt + 测试用例）。

用法（服务器）：
  python prepare_code.py --version release_v5 \
    --out /data/liujiachen/datasets/livecodebench --data_source livecodebench
注意：LiveCodeBench 列名随版本变化，脚本会先打印实际列名，按需在 build_row 里调整。
"""

import argparse
import json
import os

import datasets

PROMPT_TMPL = (
    "You are an expert competitive programmer. Solve the following problem in Python. "
    "Put the final solution in a single ```python code block.\n\n{q}\n{starter}"
)


def build_row(ex, idx, split, data_source):
    q = ex.get("question_content") or ex.get("question") or ex.get("prompt") or ""
    starter = ex.get("starter_code") or ""
    # 公有测试用例（私有多为压缩/base64，实际评测用官方 harness 或 sandbox 解码）
    pub = ex.get("public_test_cases") or ex.get("test_cases") or "[]"
    gt = pub if isinstance(pub, str) else json.dumps(pub, ensure_ascii=False)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": PROMPT_TMPL.format(q=q, starter=starter)}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split, "index": idx,
            "question_id": ex.get("question_id", idx),
            "difficulty": ex.get("difficulty", ""),
            "starter_code": starter,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="livecodebench/code_generation_lite")
    ap.add_argument("--version", default="release_v5", help="LiveCodeBench version_tag")
    ap.add_argument("--out", default="/data/liujiachen/datasets/livecodebench")
    ap.add_argument("--data_source", default="livecodebench")
    a = ap.parse_args()

    ds = datasets.load_dataset(a.hf, version_tag=a.version, trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    print("splits:", list(ds.keys()), "| columns:", ds[split].column_names, flush=True)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    d = ds[split].map(lambda ex, i: build_row(ex, i, "test", a.data_source), with_indices=True)
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    d = d.remove_columns([c for c in d.column_names if c not in keep])
    d.to_parquet(os.path.join(out, "test.parquet"))
    with open(os.path.join(out, "test_example.json"), "w") as f:
        json.dump(d[0], f, ensure_ascii=False, indent=2)
    print(f"livecodebench {a.version}: {len(d)} rows -> {out}/test.parquet", flush=True)
    print("提示：code 评测/GRPO 需 sandbox（verl sandbox_fusion）或 LCB 官方 harness，见 train/grpo.sh。")


if __name__ == "__main__":
    main()
