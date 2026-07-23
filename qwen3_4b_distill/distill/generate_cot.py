"""teacher 造蒸馏数据 → verl SFT messages parquet。三法（见项目 doc/任务二_方法规格.md）：
  standard_cot  : teacher 直接对种子题生成 CoT，math-verify 过滤答对（rejection sampling）。
  reverse       : RevThink —— R_f + 逆向问题 Q_b(I_bq) + 逆向推理 R_b + 一致性过滤(I_con)，
                  产多目标样本 (Q→R_f, Q→Q_b, Q_b→R_b)（在 verl SFT 里编码为 3 条 messages 行）。
  question_aug  : Xwin-Math —— Prompt1 造全新题(FINAL CREATED QUESTION) → Prompt2 造解，
                  无 gold 用 self-consistency 多数投票过滤答案。

公平对比铁律：三法共用同一 teacher / 同一 chat 模板 / 同一采样预算。
用法（服务器）：
  python generate_cot.py --method reverse \
    --seed /data/liujiachen/datasets/olymmath/train.parquet \
    --teacher /data/liujiachen/models/Qwen3-8B --out /data/liujiachen/datasets/distill/reverse --tp 2
"""

import argparse
import os
import re
import time
from collections import Counter

import pandas as pd

BOXED_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------- 通用 ----------
def extract_boxed(text):
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return m[-1].strip() if m else None


def verify(pred_text, gold):
    from verl.utils.reward_score.math_verify import compute_score

    try:
        return float(compute_score(pred_text, str(gold))) >= 1.0
    except Exception:
        return False


def msg_row(user, assistant):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


class Teacher:
    def __init__(self, path, tp, max_len, gpu_mem=0.85):
        from vllm import LLM

        self.llm = LLM(
            model=path, trust_remote_code=True, tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem, max_model_len=max_len,
        )
        self.tok = self.llm.get_tokenizer()

    def chat(self, users, temperature, max_tokens, n=1):
        """users: list[str]（user 消息）→ 返回 list[list[str]]（每题 n 个候选）。"""
        from vllm import SamplingParams

        prompts = [
            self.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
            for u in users
        ]
        sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, n=n)
        outs = self.llm.generate(prompts, sp)
        return [[c.text for c in o.outputs] for o in outs]


class APITeacher:
    """OpenAI 兼容 API 后端（DeepSeek / 阿里 DashScope 等），接口与 Teacher.chat 一致——**仅 off-policy**。
    reasoning 模型（如 deepseek-reasoner）的 reasoning_content 会并入 CoT。API key 从环境变量读，别写进代码。"""

    def __init__(self, base_url, model, api_key, workers=16):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.workers = workers

    def _one(self, user, temperature, max_tokens):
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = r.choices[0].message
                text = msg.content or ""
                rc = getattr(msg, "reasoning_content", None)  # DeepSeek-R1 等把思维链单列
                return f"{rc}\n\n{text}" if rc else text
            except Exception:
                if attempt == 3:
                    return ""
                time.sleep(2 ** attempt)  # 退避重试

    def chat(self, users, temperature, max_tokens, n=1):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out = [[] for _ in users]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut2idx = {}
            for i, u in enumerate(users):
                for _ in range(n):
                    fut2idx[ex.submit(self._one, u, temperature, max_tokens)] = i
            for f in as_completed(fut2idx):
                out[fut2idx[f]].append(f.result())
        return out


def read_seed(path):
    df = pd.read_parquet(path)
    items = []
    for _, r in df.iterrows():
        items.append((r["prompt"][0]["content"], r["reward_model"]["ground_truth"] if r["reward_model"] is not None else None))
    return items


def save(rows, out, n_seed, method):
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    n_val = max(1, int(len(df) * 0.05)) if len(df) > 20 else 1
    df.iloc[n_val:].to_parquet(os.path.join(out, "train.parquet"))
    df.iloc[:n_val].to_parquet(os.path.join(out, "val.parquet"))
    print(f"[{method}] 种子 {n_seed} → 产出 {len(df)} 条 messages -> {out}", flush=True)


# ---------- 方法一：Standard CoT ----------
def m_standard(t, items, a):
    outs = t.chat([q for q, _ in items], a.temp, a.max_new, n=a.n)
    rows = []
    for (q, gt), cands in zip(items, outs):
        for text in cands:  # 留第一个答对的
            if gt is not None and verify(text, gt):
                rows.append(msg_row(q, text))
                break
    return rows


# ---------- 方法二：Reverse Thinking (RevThink) ----------
I_BQ = (
    "Your task is to generate an inverse question, based on the input question and its correct answer.\n"
    "Rules:\n"
    "1. Use the correct answer from the input question to create a new, related but inverse question.\n"
    "2. Make sure there exists only one correct answer in your generated question.\n"
    "3. The correct answer in your generated question must be present in the input question.\n"
    "4. The generated question should be semantically different from the input question.\n"
    "Output ONLY the inverse question text, nothing else.\n\n"
    "INPUT: {q}\nThe correct answer is {a}.\nOUTPUT:"
)
I_CON = (
    "You will be given two question-answer pairs (Q1,A1) and (Q2,A2). "
    "Check the consistency between Q1 and A2.\n"
    "If (1) A2 can be found as a given quantity in Q1, and (2) A2 is correct, output 'True'. "
    "Otherwise output 'False'.\n"
    "Output only a single word: True or False.\n\n"
    "Q1: {q1}\nA1: {a1}\nQ2: {q2}\nA2: {a2}\nOutput:"
)


def m_reverse(t, items, a):
    # Step1: R_f，过滤正向正确
    rf = t.chat([q for q, _ in items], a.temp, a.max_new, n=1)
    kept = [(q, gt, c[0]) for (q, gt), c in zip(items, rf) if gt is not None and verify(c[0], gt)]
    if not kept:
        return []
    # Step2: 逆向问题 Q_b
    qb = [c[0].strip() for c in t.chat([I_BQ.format(q=q, a=gt) for q, gt, _ in kept], a.temp, 512, n=1)]
    # Step3: 逆向推理 R_b
    rb = [c[0] for c in t.chat([q + " " + BOXED_INSTR for q in qb], a.temp, a.max_new, n=1)]
    # Step4: 一致性过滤（A2 = R_b 的最终答案，应能在 Q1 中找到且正确）
    con_users = [
        I_CON.format(q1=q, a1=gt, q2=qbi, a2=(extract_boxed(rbi) or rbi.strip()[-64:]))
        for (q, gt, _), qbi, rbi in zip(kept, qb, rb)
    ]
    con = [c[0].strip().lower() for c in t.chat(con_users, 0.0, 8, n=1)]
    rows = []
    for (q, gt, rf_text), qbi, rbi, judge in zip(kept, qb, rb, con):
        if judge.startswith("true"):
            rows.append(msg_row(q, rf_text))                       # (a) Q → R_f
            rows.append(msg_row(q, qbi))                           # (b) Q → Q_b
            rows.append(msg_row(qbi + " " + BOXED_INSTR, rbi))     # (c) Q_b → R_b
    return rows


# ---------- 方法三：Question Augmentation (Xwin-Math) ----------
P1 = (
    "Please act as a professional math teacher. Create ONE new, similar but different math problem "
    "based on the given problem.\n"
    "Principles: (1) self-contained — restate any needed numbers/conditions; (2) reasonable and in line "
    "with common sense; (3) asks for exactly one thing with a single well-defined final answer; "
    "(4) do NOT include the solution in the question.\n"
    "First create it, then verify by solving step by step, then output the final version.\n"
    "Output strictly in this format:\n"
    "CREATED QUESTION: <...>\nVERIFICATION: <solve step by step, fix if needed>\n"
    "FINAL CREATED QUESTION: <the final question>\n\n"
    "Given Question: {q}"
)
P2 = "Please act as a professional math teacher. Solve the problem step by step and put the final answer within \\boxed{{}}.\n\nProblem: {q}"


def m_qaug(t, items, a):
    # Step1: 造全新题（temp=1.0 取多样性）
    p1 = t.chat([P1.format(q=q) for q, _ in items], 1.0, a.max_new, n=a.n)
    newqs = []
    for cands in p1:
        for text in cands:
            m = re.search(r"FINAL CREATED QUESTION:\s*(.+)", text, re.S)
            if m:
                newqs.append(m.group(1).strip())
    if not newqs:
        return []
    # Step2: 造解 + self-consistency 多数投票过滤（无 gold）
    k = max(3, a.n)
    ans = t.chat([P2.format(q=nq) for nq in newqs], 1.0, a.max_new, n=k)
    rows = []
    for nq, cands in zip(newqs, ans):
        boxed = [extract_boxed(c) for c in cands]
        pairs = [(c, b) for c, b in zip(cands, boxed) if b]
        if len(pairs) < 2:
            continue
        maj, cnt = Counter(b for _, b in pairs).most_common(1)[0]
        if cnt < 2:  # 至少 2 次一致才保留
            continue
        sol = next(c for c, b in pairs if b == maj)
        rows.append(msg_row(nq + " " + BOXED_INSTR, sol))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["standard_cot", "reverse", "question_aug"], required=True)
    ap.add_argument("--seed", required=True, help="RL parquet（含 prompt / reward_model.ground_truth）")
    ap.add_argument("--teacher", default="/data/liujiachen/models/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比；与他人共卡时调低(如 0.7)")
    ap.add_argument("--max_new", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 条种子（调试/控预算）")
    # —— API teacher（任务三双轴用；仅 off-policy）——
    ap.add_argument("--teacher_type", choices=["vllm", "api"], default="vllm")
    ap.add_argument("--api_base", default="https://api.deepseek.com",
                    help="OpenAI 兼容 base_url；DashScope=https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api_model", default="deepseek-v4-flash",
                    help="如 deepseek-v4-flash(带thinking,便宜) / deepseek-v4-pro(最强) / qwen3-235b-a22b；旧 deepseek-reasoner 名 2026-07-24 下线")
    ap.add_argument("--api_key_env", default="DEEPSEEK_API_KEY", help="存 API key 的环境变量名")
    ap.add_argument("--workers", type=int, default=16, help="API 并发数")
    a = ap.parse_args()

    items = read_seed(a.seed)
    if a.limit > 0:
        items = items[: a.limit]
    if a.teacher_type == "api":
        key = os.environ.get(a.api_key_env, "")
        assert key, f"未设置环境变量 {a.api_key_env}（API key）"
        t = APITeacher(a.api_base, a.api_model, key, workers=a.workers)
    else:
        t = Teacher(a.teacher, a.tp, a.max_len, a.gpu_mem)
    rows = {"standard_cot": m_standard, "reverse": m_reverse, "question_aug": m_qaug}[a.method](t, items, a)
    save(rows, a.out, len(items), a.method)


if __name__ == "__main__":
    main()
