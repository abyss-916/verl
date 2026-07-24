"""选择题评测（MMLU-Pro / SuperGPQA）：vLLM 生成 → 抽答案字母 → accuracy（thinking 模式）。
抽取顺序：\\boxed{X} > "answer is X" > 文中最后一个孤立大写字母。
用法（服务器）：
  python eval_mc.py --model /data/liujiachen/models/Qwen3-4B \
    --data /data/liujiachen/datasets/mmlu_pro/test.parquet --n 1 --out $LOGS/eval/mmlu_pro_base
⚠️ thinking 开着时先长思考再出字母，故 max_new 放大并统计 truncated_rate（截断=丢字母=判错，重演数学 8192 之雷）。
"""

import argparse
import json
import os
import re
from collections import Counter


def extract_letter(text):
    m = re.search(r"\\boxed\{\s*([A-P])\s*\}", text)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*(?:is|:)\s*\(?\s*([A-P])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-P])\b", text)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比;与他人共卡时调低")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max_new", type=int, default=16384, help="thinking+答案;截断率高就调大")
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(a.data)
    if a.limit > 0:
        df = df.iloc[: a.limit]
    items = [(r["prompt"][0]["content"], str(r["reward_model"]["ground_truth"]).strip()) for _, r in df.iterrows()]

    llm = LLM(model=a.model, trust_remote_code=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_new + 2048)
    tok = llm.get_tokenizer()
    ck = {} if not a.no_thinking else {"enable_thinking": False}
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **ck)
               for q, _ in items]
    outs = llm.generate(prompts, SamplingParams(temperature=a.temp, top_p=0.95, max_tokens=a.max_new, n=a.n))

    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out = os.path.expanduser(a.out)
    acc_sum, maj_sum, n_trunc, n_tok, n_gen = 0.0, 0.0, 0, 0, 0
    with open(os.path.join(out, "per_question.jsonl"), "w") as f:
        for (q, gt), o in zip(items, outs):
            preds = [extract_letter(c.text) for c in o.outputs]
            corr = [1 if p == gt else 0 for p in preds]
            acc_sum += sum(corr) / len(corr)
            maj = Counter([p for p in preds if p]).most_common(1)
            maj_sum += 1 if (maj and maj[0][0] == gt) else 0
            n_trunc += sum(1 for c in o.outputs if c.finish_reason == "length")
            n_tok += sum(len(c.token_ids) for c in o.outputs)
            n_gen += len(o.outputs)
            f.write(json.dumps({"gt": gt, "preds": preds,
                                "n_truncated": sum(1 for c in o.outputs if c.finish_reason == "length")},
                               ensure_ascii=False) + "\n")

    N = len(items)
    summary = {"model": a.model, "data": a.data, "n_samples": a.n, "num_questions": N,
               "accuracy": round(acc_sum / N, 4) if N else 0,
               "majority_vote_acc": round(maj_sum / N, 4) if N else 0,
               "max_new": a.max_new,
               "truncated_rate": round(n_trunc / n_gen, 4) if n_gen else 0,
               "mean_new_tokens": round(n_tok / n_gen, 1) if n_gen else 0}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
