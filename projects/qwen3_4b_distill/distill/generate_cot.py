"""teacher 造 CoT 蒸馏数据 → verl SFT messages parquet。

流程（standard_cot）：读 RL parquet(prompt+ground_truth) → teacher(vLLM) 生成 →
  math-verify 过滤"答对"的 → 存 SFT messages 格式 {messages:[user, assistant(teacher CoT)]}。

用法（服务器）：
  python generate_cot.py --method standard_cot \
    --seed /data/liujiachen/datasets/olymmath/train.parquet \
    --teacher /data/liujiachen/models/Qwen3-8B \
    --out /data/liujiachen/datasets/distill/standard_cot --tp 2

三法见 ../../../doc?  →  项目 doc/任务二_方法规格.md：
  standard_cot  已实现
  reverse       RevThink：另造逆向问题 Q_b + 逆向推理 R_b + 一致性过滤（TODO）
  question_aug  Xwin-Math：先造全新题再造答案，无 gold 须补 self-consistency 过滤（TODO）
公平对比铁律：三法共用同一 teacher / 同一 chat 模板 / 同一采样预算。
"""

import argparse
import os

import pandas as pd


def _extract(row):
    """从 RL parquet 一行取 (question, ground_truth)。"""
    prompt = row["prompt"]
    content = prompt[0]["content"]  # [{role:user, content:...}]
    rm = row["reward_model"]
    gt = rm["ground_truth"] if rm is not None else None
    return content, gt


def _save(rows, out, n_total):
    os.makedirs(os.path.expanduser(out), exist_ok=True)
    out = os.path.expanduser(out)
    df = pd.DataFrame(rows)
    n_val = max(1, int(len(df) * 0.05)) if len(df) > 20 else 1
    df.iloc[n_val:].to_parquet(os.path.join(out, "train.parquet"))
    df.iloc[:n_val].to_parquet(os.path.join(out, "val.parquet"))
    print(f"保留 {len(df)}/{n_total} 条（答对过滤后）-> {out}", flush=True)


def gen_standard_cot(args):
    from vllm import LLM, SamplingParams

    from verl.utils.reward_score.math_verify import compute_score

    df = pd.read_parquet(args.seed)
    llm = LLM(
        model=args.teacher,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_len,
    )
    tok = llm.get_tokenizer()

    prompts, metas = [], []
    for _, r in df.iterrows():
        q, gt = _extract(r)
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        metas.append((q, gt))

    sp = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=args.max_new, n=args.n)
    outs = llm.generate(prompts, sp)

    rows = []
    for (q, gt), o in zip(metas, outs):
        for cand in o.outputs:  # n 个候选，留第一个答对的（rejection sampling）
            cot = cand.text
            if gt is not None and compute_score(cot, str(gt)) >= 1.0:
                rows.append({"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": cot},
                ]})
                break
    _save(rows, args.out, len(metas))


def gen_reverse(args):
    raise NotImplementedError(
        "RevThink（doc/任务二_方法规格.md）：用 I_bq 生成逆向问题 Q_b、生成 R_b、用 I_con 一致性过滤，"
        "并产出多目标(Q→R_f, Q→Q_b, Q_b→R_b)训练样本。实现前先看官方 code 确认 Joint 设定。")


def gen_question_aug(args):
    raise NotImplementedError(
        "Xwin-Math（doc/任务二_方法规格.md）：Prompt1 造全新题(FINAL CREATED QUESTION) → Prompt2 造 SOLUTION+FINAL ANSWER；"
        "无 gold，须补 self-consistency/majority-vote 过滤答案；temp=1.0。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["standard_cot", "reverse", "question_aug"], required=True)
    ap.add_argument("--seed", required=True, help="RL parquet（含 prompt / reward_model.ground_truth）")
    ap.add_argument("--teacher", default="/data/liujiachen/models/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)  # 8B 用 TP=2 跨两卡
    ap.add_argument("--temp", type=float, default=0.6)  # thinking 默认；造多样性可调高
    ap.add_argument("--n", type=int, default=1)  # 每题采样数
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_new", type=int, default=4096)
    args = ap.parse_args()
    {"standard_cot": gen_standard_cot, "reverse": gen_reverse, "question_aug": gen_question_aug}[args.method](args)


if __name__ == "__main__":
    main()
