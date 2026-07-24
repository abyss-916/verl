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
import time

import datasets

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

Q_KEYS = ["problem", "question", "prompt", "query", "Problem", "Question"]
A_KEYS = ["answer", "final_answer", "gold", "solution", "Answer", "Solution"]  # solution 放最后（MATH 无 answer 列时兜底）；大写兼容 AIME_2024 等
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


def _load_split(source, hf, subset, split):
    """加载单个 split → HF Dataset。
    source=hf：走 datasets（HF/hf-mirror）；source=modelscope：走 MsDataset；
    source=local：从本地目录按文件名匹配 parquet 加载（推荐——先把 parquet 抓到本地再来，绕开脚本型/多文件超时）。
    hf 支持逗号分隔多个候选 id（modelscope 时逐个尝试，任一成功即返回）。
    """
    if source == "local":
        import glob as _glob

        files = sorted(_glob.glob(os.path.join(hf, "**", f"*{split}*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"本地无 {split} 的 parquet: {hf}")
        return datasets.load_dataset("parquet", data_files=files)["train"]
    if source == "modelscope":
        # 用 modelscope CLI 整仓下载 parquet 到本地(国内稳、不执行脚本、不受 datasets 5.0 脚本禁令影响)，再读本地 parquet。
        import glob as _glob
        import subprocess

        data_root = os.environ.get("DATA", "/data/liujiachen/datasets")
        errs = []
        for hid in [x.strip() for x in hf.split(",") if x.strip()]:
            local = os.path.join(data_root, "_ms_" + hid.replace("/", "__"))
            pat = os.path.join(local, "**", f"*{split}*.parquet")
            try:
                files = sorted(_glob.glob(pat, recursive=True))
                if not files:  # 尚未下过 → 整仓拉一次(train/test 一起下)，后续 split 直接命中缓存
                    subprocess.run(["modelscope", "download", "--dataset", hid, "--local_dir", local], check=True)
                    files = sorted(_glob.glob(pat, recursive=True))
                if not files:
                    raise FileNotFoundError(f"{hid}: 下载后仍无 {split} parquet(该仓可能非 parquet 布局)")
                return datasets.load_dataset("parquet", data_files=files)["train"]
            except Exception as e:
                errs.append(f"{hid}: {e}")
        raise RuntimeError("; ".join(errs))
    return datasets.load_dataset(hf, subset, split=split) if subset else datasets.load_dataset(hf, split=split)


def _load_split_retry(source, hf, subset, split, tries=4):
    """带重试（网络抖动）；split 确实不存在则立刻抛出不重试。"""
    last = None
    for i in range(tries):
        try:
            return _load_split(source, hf, subset, split)
        except Exception as e:
            m = str(e).lower()
            if "unknown split" in m or ("split" in m and "not found" in m) or ("splits" in m and "available" in m):
                raise
            last = e
            print(f"[retry {i + 1}/{tries}] source={source} split={split}: {e}", flush=True)
            time.sleep(3)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="数据集名（HF 或 ModelScope，逗号可给多候选）或本地路径")
    ap.add_argument("--subset", default=None, help="config/subset：OlymMATH=en-hard 等（小写）；MATH 一般留空")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_source", required=True, help="reward 路由标识，如 math_seed / olymmath")
    ap.add_argument("--source", default="hf", choices=["hf", "modelscope", "local"], help="源：hf(mirror) / modelscope / local(本地 parquet 目录)")
    a = ap.parse_args()

    subset = a.subset or None
    # 独立加载 train / test：容忍某 split 缺失（OlymMATH 无 train → 复用 test，仅占位不训练）
    train_ds = test_ds = None
    for split in ("train", "test"):
        try:
            d = _load_split_retry(a.source, a.hf, subset, split)
            if split == "train":
                train_ds = d
            else:
                test_ds = d
        except Exception as e:
            print(f"[warn] split={split} 加载失败（可能本就无此 split）：{e}", flush=True)
    if train_ds is None and test_ds is None:
        raise SystemExit("[fatal] train/test 均无法加载——检查数据集名 / 源(--source) / 网络")
    print(f"loaded | train={len(train_ds) if train_ds else 0} test={len(test_ds) if test_ds else 0} | "
          f"columns={(train_ds or test_ds).column_names}", flush=True)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # ⚠️ 只写真正加载到的 split：eval-only 数据集(OlymMATH 只有 test)【绝不】把 test 复制成 train.parquet，
    #    否则被误当种子蒸馏 = 评测集泄漏(蒸出的 CoT 训到 held-out 题上，分数假性冲顶)。
    for name, d0 in [("train", train_ds), ("test", test_ds)]:
        if d0 is None:
            print(f"[skip] 无 {name} split → 不写 {name}.parquet（不拿另一 split 顶替，避免泄漏陷阱）", flush=True)
            continue
        d = d0.map(make_map_fn(name, a.data_source), with_indices=True)
        n0 = len(d)
        # 丢无答案行：to_answer(None) 会让 ground_truth 变字符串 "None"，judge 恒判 0（eval 假性压分 / GRPO 该 prompt 永远 0 reward）
        d = d.filter(lambda ex: ex["reward_model"]["ground_truth"] not in ("None", "", "none"))
        if len(d) != n0:
            print(f"[warn] {name}: 丢弃 {n0 - len(d)} 无答案行", flush=True)
        if len(d) == 0:
            raise SystemExit(f"[fatal] {name}: 过滤后 0 行——检查答案列(A_KEYS)是否匹配该数据集")
        d = d.remove_columns([c for c in d.column_names if c not in keep])
        d.to_parquet(os.path.join(out, f"{name}.parquet"))
        with open(os.path.join(out, f"{name}_example.json"), "w") as f:
            json.dump(d[0], f, ensure_ascii=False, indent=2)
        print(f"{name}: {len(d)} rows -> {out}/{name}.parquet", flush=True)


if __name__ == "__main__":
    main()
