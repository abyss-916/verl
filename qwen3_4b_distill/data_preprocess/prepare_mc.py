"""选择题类科学推理 benchmark → verl RL parquet（held-out eval，扩展项/加分）。
支持 MMLU-Pro（列 question/options/answer(字母)/category）、SuperGPQA（question/options/answer_letter/discipline/...）。

用法（服务器）：
  python prepare_mc.py --hf TIGER-Lab/MMLU-Pro --subset default --out /data/liujiachen/datasets/mmlu_pro --data_source mmlu_pro
  python prepare_mc.py --hf m-a-p/SuperGPQA   --subset default --out /data/liujiachen/datasets/supergpqa --data_source supergpqa
判分：正确选项字母，用 eval/eval_mc.py（抽 \\boxed{字母} 比对）。
"""

import argparse
import ast
import json
import os

import datasets

LETTERS = list("ABCDEFGHIJKLMNOP")
META = ["category", "discipline", "field", "subfield", "difficulty", "src"]


def _options(ex):
    o = ex["options"]
    if isinstance(o, str):
        try:
            o = ast.literal_eval(o)
        except Exception:
            o = [o]
    return list(o)


def make_map_fn(split, data_source):
    def fn(ex, idx):
        q = ex["question"]
        opts = _options(ex)
        letter = ex.get("answer_letter") or ex.get("answer")  # MMLU-Pro:answer=字母；SuperGPQA:answer_letter
        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        prompt = (f"{q}\n\nOptions:\n{body}\n\n"
                  "Answer with the letter of the correct option, and put ONLY the letter within \\boxed{}.")
        extra = {"split": split, "index": idx}
        for k in META:
            if k in ex and ex[k] is not None:
                extra[k] = ex[k]
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "mc",
            "reward_model": {"style": "rule", "ground_truth": str(letter).strip()},
            "extra_info": extra,
        }

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--subset", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_source", required=True)
    a = ap.parse_args()

    ds = datasets.load_dataset(a.hf, a.subset) if a.subset else datasets.load_dataset(a.hf)
    split = "test" if "test" in ds else list(ds.keys())[0]
    print("splits:", list(ds.keys()), "| columns:", ds[split].column_names, flush=True)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    d = ds[split].map(make_map_fn("test", a.data_source), with_indices=True)
    d = d.remove_columns([c for c in d.column_names if c not in keep])
    d.to_parquet(os.path.join(out, "test.parquet"))
    with open(os.path.join(out, "test_example.json"), "w") as f:
        json.dump(d[0], f, ensure_ascii=False, indent=2)
    print(f"{a.data_source}: {len(d)} rows -> {out}/test.parquet", flush=True)


if __name__ == "__main__":
    main()
