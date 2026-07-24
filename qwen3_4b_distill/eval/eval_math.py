"""数学评测（base / SFT / GRPO 通用）：pass@1、avg@k、pass@k。
- thinking 模式默认开（Qwen3 thinking 下推理类才不被低估；--no_thinking 关）。
- 判定复用 verl 内置 math-verify。
- 输出逐题 jsonl + 汇总 json，便于任务一"与论文对齐"和后续归因。

用法（服务器）：
  python eval_math.py --model /data/liujiachen/models/Qwen3-4B-Base \
    --data /data/liujiachen/datasets/olymmath/test.parquet \
    --n 8 --out /data/liujiachen/logs/eval/olymmath_base
  # avg@64（AIME 类小集降方差）：--n 64
"""

import argparse
import json
import math
import os

import pandas as pd


def pass_at_k(n, c, k):
    """无偏 pass@k 估计（Chen et al. 2021）：n 个样本里 c 个正确。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="RL parquet（prompt + reward_model.ground_truth）")
    ap.add_argument("--out", required=True, help="输出目录前缀")
    ap.add_argument("--n", type=int, default=8, help="每题采样数（pass@1 用均值，pass@k 用无偏估计）")
    ap.add_argument("--k", type=int, default=0, help="pass@k 的 k；默认=n")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比；与他人共卡时调低(如 0.7)")
    ap.add_argument("--temp", type=float, default=0.6)   # thinking 采样默认
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    # 38912 = Qwen3 官方对"数学/编程竞赛类 benchmark"的推荐输出长度；+2048 prompt 后正好顶满
    # max_position_embeddings=40960。再高必须开 YaRN，会改变模型行为、破坏与论文锚点可比性，故到此为止。
    # ⚠️ 别为省时间调小：截断样本必然判错，实测 32768 时 base 仍有 15% 撞顶，8192 时 96% 撞顶（分数假性归零）。
    ap.add_argument("--max_new", type=int, default=38912)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    # 多卡分片：同一份 data 交错切成 num_shards 份（df.iloc[shard::num_shards]，难度自然均衡），
    # 每张卡跑一片、各自 --out，最后用 merge_shards.py 合成一份总 summary。
    ap.add_argument("--shard", type=int, default=0, help="分片编号，0-based")
    ap.add_argument("--num_shards", type=int, default=1, help="总分片数；=1 即不分片")
    a = ap.parse_args()
    k = a.k or a.n

    from vllm import LLM, SamplingParams

    from verl.utils.reward_score.math_verify import compute_score

    df = pd.read_parquet(a.data)
    if a.limit > 0:
        df = df.iloc[: a.limit]
    if a.num_shards > 1:
        df = df.iloc[a.shard :: a.num_shards]
    def _meta(r):
        ex = r["extra_info"] if "extra_info" in df.columns and r["extra_info"] is not None else {}
        try:
            ex = dict(ex)
        except Exception:
            ex = {}
        return {k: ex[k] for k in ("level", "type", "subject", "difficulty") if ex.get(k) is not None}

    items = [(r["prompt"][0]["content"], r["reward_model"]["ground_truth"], _meta(r)) for _, r in df.iterrows()]

    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_new + 2048)
    tok = llm.get_tokenizer()
    ck = {} if a.no_thinking is False else {"enable_thinking": False}
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
        for q, _, _ in items
    ]
    sp = SamplingParams(temperature=a.temp, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_new, n=a.n)
    outs = llm.generate(prompts, sp)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    sum_avg, sum_passk, n_trunc, n_tok, n_gen = 0.0, 0.0, 0, 0, 0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt, meta), o in zip(items, outs):
            corr = [1 if compute_score(s.text, str(gt)) >= 1.0 else 0 for s in o.outputs]
            # finish_reason=="length" ＝撞生成上限被截断。截断样本必然判错，是效度杀手 → 必须逐条统计，
            # 让 summary 自带截断率，任何一次 eval 都能立刻看出分数是不是被预算压出来的。
            trunc = [1 if s.finish_reason == "length" else 0 for s in o.outputs]
            toks = [len(s.token_ids) for s in o.outputs]
            nc, nn = sum(corr), len(corr)
            avg, pk = nc / nn, pass_at_k(nn, nc, k)
            sum_avg += avg
            sum_passk += pk
            n_trunc += sum(trunc)
            n_tok += sum(toks)
            n_gen += nn
            f.write(json.dumps({"question": q, "gt": str(gt), **meta, "n_correct": nc, "n": nn,
                                "avg": avg, f"pass@{k}": pk,
                                "n_truncated": sum(trunc), "new_tokens": toks,
                                "samples": [s.text for s in o.outputs]}, ensure_ascii=False) + "\n")

    N = len(items)
    summary = {
        "model": a.model, "data": a.data, "n_samples": a.n, "thinking": not a.no_thinking,
        "num_questions": N,
        "pass@1 (avg@n)": round(sum_avg / N, 4) if N else 0,
        f"pass@{k}": round(sum_passk / N, 4) if N else 0,
        "max_new": a.max_new,
        "truncated_rate": round(n_trunc / n_gen, 4) if n_gen else 0,
        "mean_new_tokens": round(n_tok / n_gen, 1) if n_gen else 0,
    }
    if a.num_shards > 1:
        summary["shard"] = f"{a.shard}/{a.num_shards}"
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
