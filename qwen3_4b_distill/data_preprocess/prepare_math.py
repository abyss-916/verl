"""数学数据集 → verl RL parquet。改编自 verl/examples/data_preprocess/math_dataset.py。

两种角色（高质量课题的关键：训练集与评测集严格分离）：
  - SEED（训练/蒸馏种子/GRPO prompt）：大数学训练集，如 MATH-lighteval train（~7500）。
  - EVAL（held-out 评测）：OlymMATH 等，绝不进训练。

答案自适应：OlymMATH 用纯 `answer` 列；MATH 用 `solution` 里的 \boxed 抽取。
额外把 level/type/subject 存进 extra_info，供任务三做难度/领域切片归因。

用法（服务器）：
  # 种子（MATH train）
  python prepare_math.py --hf DigitalLearningGmbH/MATH-lighteval --subset default \
      --out /data/liujiachen/datasets/math_seed --data_source math_seed
  # held-out eval（OlymMATH）
  python prepare_math.py --hf RUC-AIBOX/OlymMATH --subset en-hard \
      --out /data/liujiachen/datasets/olymmath --data_source olymmath
"""

import argparse
import json
import os
import re

import datasets

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

Q_KEYS = ["problem", "question", "prompt", "query"]
A_KEYS = ["answer", "final_answer", "gold", "solution"]  # solution 放最后（MATH 无 answer 列时兜底）
META_KEYS = ["level", "type", "subject", "unique_id"]


def _first(ex, keys):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return None


def to_answer(a):
    """把答案统一成"最终答案"：含 \\boxed 的（MATH solution）抽 boxed；否则原样（OlymMATH）。"""
    if a is None:
        return None
    a = str(a)
    if "\\boxed" in a:
        try:
            from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

            b = last_boxed_only_string(a)
            if b:
                return remove_boxed(b).strip()
        except Exception:
            m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", a)
            if m:
                return m[-1].strip()
    return a.strip()


def make_map_fn(split, data_source):
    def process_fn(example, idx):
        q = _first(example, Q_KEYS)
        a = to_answer(_first(example, A_KEYS))
        extra = {"split": split, "index": idx, "question": q}
        for k in META_KEYS:
            if k in example and example[k] is not None:
                extra[k] = example[k]
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": f"{q} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(a)},
            "extra_info": extra,
        }

    return process_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF 数据集名或本地路径")
    ap.add_argument("--subset", default=None, help="HF config：MATH-lighteval=default；OlymMATH=en-hard/en-easy/zh-hard/zh-easy/lean（小写）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_source", required=True, help="reward 路由标识，如 math_seed / olymmath")
    a = ap.parse_args()

    ds = datasets.load_dataset(a.hf, a.subset) if a.subset else datasets.load_dataset(a.hf)
    print("splits:", list(ds.keys()), "| columns:", ds[list(ds.keys())[0]].column_names, flush=True)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    test_key = "test" if "test" in ds else list(ds.keys())[0]
    train_key = "train" if "train" in ds else test_key  # OlymMATH 无 train → 复用 test（仅作占位，不用于训练）
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    for name, key in [("train", train_key), ("test", test_key)]:
        d = ds[key].map(make_map_fn(name, a.data_source), with_indices=True)
        d = d.remove_columns([c for c in d.column_names if c not in keep])
        d.to_parquet(os.path.join(out, f"{name}.parquet"))
        with open(os.path.join(out, f"{name}_example.json"), "w") as f:
            json.dump(d[0], f, ensure_ascii=False, indent=2)
        print(f"{name}: {len(d)} rows -> {out}/{name}.parquet", flush=True)


if __name__ == "__main__":
    main()
