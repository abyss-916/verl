"""蒸馏数据度量（student 视角）——对接课题 metrics 阶段与任务三归因。
度量：样本数 / 长度(token) / 多样性(distinct-1,2) / PPL / IFD。
    - PPL、IFD 用 **student 基座**（Qwen3-4B）算，因为课题问的是"数据适不适合 4B 学"。
    - IFD = L(answer|question) / L(answer)：越高=问题对预测答案帮助越小=越难跟随/越有信息量；
      IFD>=1 视为噪声/错配（精确定义参见 Cherry LLM, arXiv:2308.12032）。

用法（服务器）：
  python data_metrics.py --data /data/liujiachen/datasets/distill/standard_cot/train.parquet \
    --model /data/liujiachen/models/Qwen3-4B --limit 500 --out metrics_standard_cot.json
不给 --model 时只算 长度 + 多样性（快，仅需 tokenizer；用 --tokenizer 指定）。
"""

import argparse
import json
from collections import Counter

import pandas as pd


def _qa(row):
    """支持 SFT messages 格式与 RL prompt 格式，返回 (question, answer_text)。"""
    if "messages" in row and row["messages"] is not None:
        msgs = row["messages"]
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return q, a
    # RL 格式：只有问题
    prompt = row["prompt"]
    return prompt[0]["content"], ""


def distinct_n(token_lists, n):
    grams, total = Counter(), 0
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i : i + n])] += 1
            total += 1
    return (len(grams) / total) if total else 0.0


def compute_length_diversity(rows, tok):
    ans_tok = [tok.encode(a) for _, a in rows if a]
    lens = [len(t) for t in ans_tok]
    lens.sort()
    n = len(lens)
    p = lambda q: lens[min(n - 1, int(q * n))] if n else 0
    return {
        "n_samples": len(rows),
        "answer_len_tokens": {"mean": (sum(lens) / n) if n else 0, "median": p(0.5), "p90": p(0.9)},
        "distinct_1": round(distinct_n(ans_tok, 1), 4),
        "distinct_2": round(distinct_n(ans_tok, 2), 4),
    }


def _seq_loss(model, ids, target_start, device):
    import torch

    ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="none"
    )
    loss = loss[max(0, target_start - 1) :]  # 只算 answer 段
    return float(loss.mean()) if loss.numel() else float("nan")


def compute_ppl_ifd(rows, model_path, limit, device="cuda", max_len=8192):
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    ppls, ifds, n_skip = [], [], 0
    for q, a in rows[:limit]:
        if not a:
            continue
        q_ids, a_ids = tok.encode(q), tok.encode(a)
        # 超长跳过：单条前向的 logits 是 [seq × vocab(≈15万)]，16K token 就 ~5G，24G 卡会 OOM。
        # IFD/PPL 是整体分布度量，跳过极少数超长样本不改结论；跳过数记入 json 保持透明。
        if len(q_ids) + len(a_ids) > max_len:
            n_skip += 1
            continue
        l_a_given_q = _seq_loss(model, q_ids + a_ids, len(q_ids), device)  # L(A|Q)
        l_a = _seq_loss(model, a_ids, 0, device)  # L(A)
        if not math.isnan(l_a_given_q):
            ppls.append(math.exp(min(20.0, l_a_given_q)))
        if l_a and not math.isnan(l_a) and not math.isnan(l_a_given_q):
            ifds.append(l_a_given_q / l_a)
    mean = lambda x: (sum(x) / len(x)) if x else None
    return {
        "ppl_student_view_mean": mean(ppls),
        "ifd_mean": mean(ifds),
        "ifd_ge1_ratio": (sum(v >= 1 for v in ifds) / len(ifds)) if ifds else None,  # 噪声占比
        "ppl_ifd_n_used": len(ppls),
        "ppl_ifd_n_skipped_too_long": n_skip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet（SFT messages 或 RL 格式）")
    ap.add_argument("--model", default=None, help="student 基座；给了才算 PPL/IFD")
    ap.add_argument("--tokenizer", default=None, help="仅算长度/多样性时的 tokenizer；默认同 --model 或 Qwen3-4B")
    ap.add_argument("--limit", type=int, default=500, help="PPL/IFD 采样上限（省时）")
    ap.add_argument("--ppl_max_len", type=int, default=8192, help="PPL/IFD 单样本 token 上限，超过跳过防 OOM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    rows = [_qa(r) for _, r in df.iterrows()]

    from transformers import AutoTokenizer

    tok_path = args.tokenizer or args.model or "/data/liujiachen/models/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    result = compute_length_diversity(rows, tok)
    if args.model:
        result.update(compute_ppl_ifd(rows, args.model, args.limit, max_len=args.ppl_max_len))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
