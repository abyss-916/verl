"""数学数据集 → verl RL parquet（供 GRPO 训练 + 评测）。
改编自 verl/examples/data_preprocess/math_dataset.py。

RL parquet 每行 schema（与 verl 内置一致）：
  data_source / prompt(chat list) / ability / reward_model{style,ground_truth} / extra_info

用法（服务器）：
  python prepare_math.py --hf RUC-AIBOX/OlymMATH --subset EN-HARD \
      --out /data/liujiachen/datasets/olymmath --data_source olymmath
注意：OlymMATH 等自定义集的答案判定走我们的 reward/math_reward.py（math-verify），
      故 data_source 用 'olymmath' 即可，GRPO 时用 custom_reward_function 挂载。
"""

import argparse
import json
import os

import datasets

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# 不同数据集列名不一，按优先级取
Q_KEYS = ["problem", "question", "prompt", "query"]
A_KEYS = ["answer", "final_answer", "solution", "gold"]


def _first(ex, keys):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return None


def make_map_fn(split, data_source):
    def process_fn(example, idx):
        q = _first(example, Q_KEYS)
        a = _first(example, A_KEYS)
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": f"{q} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(a)},
            "extra_info": {"split": split, "index": idx, "question": q},
        }

    return process_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="RUC-AIBOX/OlymMATH", help="HF 数据集名或本地路径")
    ap.add_argument("--subset", default=None, help="HF config/subset（按数据集实际，OlymMATH 如 EN-HARD）")
    ap.add_argument("--train_split", default=None, help="作训练的 split；OlymMATH 常无 train，可复用 test")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--out", default="/data/liujiachen/datasets/olymmath")
    ap.add_argument("--data_source", default="olymmath")
    args = ap.parse_args()

    ds = datasets.load_dataset(args.hf, args.subset) if args.subset else datasets.load_dataset(args.hf)
    print("available splits:", list(ds.keys()), flush=True)

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    test_key = args.test_split if args.test_split in ds else list(ds.keys())[0]
    train_key = args.train_split or ("train" if "train" in ds else test_key)
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    for name, key in [("train", train_key), ("test", test_key)]:
        d = ds[key].map(make_map_fn(name, args.data_source), with_indices=True)
        d = d.remove_columns([c for c in d.column_names if c not in keep])
        path = os.path.join(out, f"{name}.parquet")
        d.to_parquet(path)
        with open(os.path.join(out, f"{name}_example.json"), "w") as f:
            json.dump(d[0], f, ensure_ascii=False, indent=2)
        print(f"{name}: {len(d)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
