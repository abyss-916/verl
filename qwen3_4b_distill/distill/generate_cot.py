"""teacher 造蒸馏数据 → verl SFT messages parquet。各方法：
  standard_cot  : teacher 对种子题生成 CoT，math-verify 过滤答对（拒绝采样）。
  shortest_cot  : 每题采样 n 个候选，正确者里留 token 最少的（须 --n>1）。
  reverse       : R_f + 逆向问题 Q_b(I_bq) + 逆向推理 R_b + 一致性过滤(I_con)，产多目标样本 (Q→R_f, Q→Q_b, Q_b→R_b)。
  question_aug  : Prompt1 造全新题(FINAL CREATED QUESTION) → Prompt2 造解，无 gold 时用 self-consistency 多数投票过滤。
  code_cot      : teacher 造代码，prime_code 跑测试用例过滤（种子=prepare_code 的 parquet）。
  mc_cot        : 选择题，抽 \\boxed{字母} 与 gold 比对过滤（种子=prepare_mc 的 parquet）。

用法：
  python generate_cot.py --method reverse \
    --seed $DATA/omni_seed/train.parquet \
    --teacher $MODELS/Qwen3-8B --out $DATA/distill/reverse --tp 2
"""

import argparse
import json
import os
import re
import time
from collections import Counter

import pandas as pd

BOXED_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------- 通用 ----------
def extract_boxed(text):
    """取最后一个 \\boxed{...}，花括号配平支持任意层嵌套。"""
    key = "\\boxed{"
    i = text.rfind(key)
    if i == -1:
        return None
    depth, j = 1, i + len(key)
    start = j
    while j < len(text) and depth:
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1
    return text[start:j - 1].strip() if depth == 0 else None


def strip_think(text):
    """剥掉 Qwen3 的 <think>...</think> 块。"""
    return re.sub(r"(?s)<think>.*?</think>\s*", "", text).strip()


def verify(pred_text, gold):
    from verl.utils.reward_score.math_verify import compute_score

    try:
        return float(compute_score(pred_text, str(gold))) >= 1.0
    except Exception:
        return False


# ---- code / mc 判定：复用 reward/* 的判分器 ----
def extract_code(text):
    """取最后一个 ```python ... ``` 代码块；无围栏则退回整段。"""
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[-1].strip() if m else text.strip()


def _reward_on_path():
    import sys
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # distill/.. = qwen3_4b_distill/
    if d not in sys.path:
        sys.path.insert(0, d)


def verify_code(pred_text, gt):
    """LiveCodeBench：抽代码 → prime_code 本地跑测试用例，全部通过为 True。"""
    _reward_on_path()
    from reward.code_reward import compute_score

    try:
        return float(compute_score("livecodebench", extract_code(pred_text), gt)) >= 1.0
    except Exception:
        return False


def verify_mc(pred_text, gt):
    """选择题：抽答案字母与 gold 比对。"""
    _reward_on_path()
    from reward.mc_reward import compute_score

    try:
        return float(compute_score("mmlu_pro", pred_text, gt)) >= 1.0
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

    def chat_full(self, users, temperature, max_tokens, n=1, enable_thinking=True, system=None):
        """users: list[str] → list[list[dict(text, finish, ntok)]]。
        finish=='length' 表示撞 max_tokens 被截断。
        enable_thinking：推理目标步传 True，辅助解析步传 False。
        system：standard_cot 的 system 指令；None 表示不加 system。"""
        from vllm import SamplingParams

        def _msgs(u):
            head = [{"role": "system", "content": system}] if system else []
            return head + [{"role": "user", "content": u}]
        prompts = [
            self.tok.apply_chat_template(_msgs(u), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)
            for u in users
        ]
        sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, n=n)
        outs = self.llm.generate(prompts, sp)
        return [[{"text": c.text, "finish": c.finish_reason, "ntok": len(c.token_ids)} for c in o.outputs] for o in outs]

    def chat(self, users, temperature, max_tokens, n=1, enable_thinking=True, system=None):
        """只要文本时的薄封装 → list[list[str]]。"""
        return [[c["text"] for c in cs]
                for cs in self.chat_full(users, temperature, max_tokens, n, enable_thinking=enable_thinking, system=system)]


class APITeacher:
    """OpenAI 兼容 API 后端（DeepSeek / DashScope 等），接口与 Teacher.chat 一致。
    reasoning 模型的 reasoning_content 会并入 CoT；API key 从环境变量读取。"""

    def __init__(self, base_url, model, api_key, workers=16):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.workers = workers

    def _one(self, user, temperature, max_tokens, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = r.choices[0].message
                text = msg.content or ""
                rc = getattr(msg, "reasoning_content", None)  # reasoning 模型把思维链单列
                return f"{rc}\n\n{text}" if rc else text
            except Exception:
                if attempt == 3:
                    return ""
                time.sleep(2 ** attempt)  # 退避重试

    def chat(self, users, temperature, max_tokens, n=1, enable_thinking=True, system=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out = [[] for _ in users]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut2idx = {}
            for i, u in enumerate(users):
                for _ in range(n):
                    fut2idx[ex.submit(self._one, u, temperature, max_tokens, system)] = i
            for f in as_completed(fut2idx):
                out[fut2idx[f]].append(f.result())
        return out

    def chat_full(self, users, temperature, max_tokens, n=1, enable_thinking=True, system=None):
        """与 Teacher.chat_full 同签名；API 端不返回 finish_reason/token 数，故留空。
        enable_thinking 仅为签名对齐，此处不透传。"""
        return [[{"text": s, "finish": None, "ntok": None} for s in cs]
                for cs in self.chat(users, temperature, max_tokens, n, system=system)]


def read_seed(path):
    df = pd.read_parquet(path)
    items = []
    for _, r in df.iterrows():
        items.append((r["prompt"][0]["content"], r["reward_model"]["ground_truth"] if r["reward_model"] is not None else None))
    return items


def save(rows, out, n_seed, method):
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    if not rows:                                # 0 产出，拒绝写空数据集
        raise SystemExit(f"[{method}] 0 条产出，拒绝写空数据集：检查 thinking 开关、截断过滤、教师是否正常。")
    df = pd.DataFrame(rows)
    # 打散后再切 val
    if len(df) > 1:
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n_val = max(1, int(len(df) * 0.05)) if len(df) > 20 else 1
    n_val = min(n_val, len(df) - 1)             # 保证 train 至少 1 条
    df.iloc[n_val:].to_parquet(os.path.join(out, "train.parquet"))
    if n_val > 0:
        df.iloc[:n_val].to_parquet(os.path.join(out, "val.parquet"))
    print(f"[{method}] 种子 {n_seed} → 产出 {len(df)} 条 messages（train {len(df) - n_val} / val {n_val}）-> {out}", flush=True)


def gen_stats(out, method, max_new, n_seed, n_kept, n_cand, n_trunc, ntoks):
    """记录生成侧良率/截断率/长度分布 → gen_stats.json。"""
    ntoks = sorted(x for x in ntoks if x)
    q = (lambda r: ntoks[max(0, int(len(ntoks) * r) - 1)]) if ntoks else (lambda r: 0)
    st = {
        "method": method, "max_new": max_new, "n_seed": n_seed, "n_kept": n_kept,
        "yield": round(n_kept / n_seed, 4) if n_seed else 0,
        "n_candidates": n_cand, "n_truncated": n_trunc,
        "truncated_rate": round(n_trunc / n_cand, 4) if n_cand else 0,
        "tok_p50": q(0.5), "tok_p90": q(0.9), "tok_p95": q(0.95), "tok_p99": q(0.99),
        "tok_max": ntoks[-1] if ntoks else 0,
        "tok_over_4096": sum(1 for x in ntoks if x > 4096),
        "tok_over_8192": sum(1 for x in ntoks if x > 8192),
        "tok_over_16384": sum(1 for x in ntoks if x > 16384),
    }
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gen_stats.json"), "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    print("[gen_stats] " + json.dumps(st, ensure_ascii=False), flush=True)


# ---------- 方法一：Standard CoT ----------
def m_standard(t, items, a):
    outs = t.chat_full([q for q, _ in items], a.temp, a.max_new, n=a.n, system=(a.sys_prompt or None))
    rows, n_cand, n_trunc, ntoks = [], 0, 0, []
    for (q, gt), cands in zip(items, outs):
        keep = None
        for c in cands:  # 留第一个"完整(未截断、含 \boxed)且答对"的候选
            n_cand += 1
            ntoks.append(c["ntok"])
            if c["finish"] == "length":  # 截断即不完整，丢弃
                n_trunc += 1
                continue
            if gt is not None and "\\boxed" in c["text"] and verify(c["text"], gt):
                keep = c["text"]
                break
        if keep is not None:
            rows.append(msg_row(q, keep))
    gen_stats(a.out, a.method, a.max_new, len(items), len(rows), n_cand, n_trunc, ntoks)
    return rows


def m_shortest(t, items, a):
    """最短正确 CoT：每题采样 n 个候选，在完整(未截断、含 \\boxed)且答对的候选里保留 token 最少的一个。
    须 --n>1 才有选择空间（n=1 时退化为 standard）。"""
    outs = t.chat_full([q for q, _ in items], a.temp, a.max_new, n=a.n, system=(a.sys_prompt or None))
    rows, n_cand, n_trunc, ntoks = [], 0, 0, []
    for (q, gt), cands in zip(items, outs):
        best = None  # (ntok, text)：当前最短的完整且答对候选
        for c in cands:
            n_cand += 1
            ntoks.append(c["ntok"])
            if c["finish"] == "length":  # 截断即不完整，丢弃
                n_trunc += 1
                continue
            if gt is not None and "\\boxed" in c["text"] and verify(c["text"], gt):
                if best is None or c["ntok"] < best[0]:
                    best = (c["ntok"], c["text"])
        if best is not None:
            rows.append(msg_row(q, best[1]))
    gen_stats(a.out, a.method, a.max_new, len(items), len(rows), n_cand, n_trunc, ntoks)
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
    ntoks, n_cand, n_trunc = [], 0, 0
    # Step1: R_f（正向推理），过滤未截断且含 \boxed 且答对
    rf = t.chat_full([q for q, _ in items], a.temp, a.max_new, n=1)
    kept = []
    for (q, gt), cs in zip(items, rf):
        c = cs[0]; n_cand += 1; ntoks.append(c["ntok"])
        if c["finish"] == "length":                       # 截断的 R_f 丢弃
            n_trunc += 1; continue
        if gt is not None and "\\boxed" in c["text"] and verify(c["text"], gt):
            kept.append((q, gt, c["text"]))
    if not kept:
        gen_stats(a.out, a.method, a.max_new, len(items), 0, n_cand, n_trunc, ntoks)
        return []
    # Step2: 逆向问题 Q_b（短输出 512，不计入截断统计），关 thinking 并剥残留 <think>
    qb = [strip_think(c[0]) for c in t.chat([I_BQ.format(q=q, a=gt) for q, gt, _ in kept],
                                            a.temp, 512, n=1, enable_thinking=False)]
    # Step3: 逆向推理 R_b，同样过滤截断
    rb_out = t.chat_full([q + " " + BOXED_INSTR for q in qb], a.temp, a.max_new, n=1)
    idxs, con_users = [], []
    for i, ((q, gt, _), qbi, cs) in enumerate(zip(kept, qb, rb_out)):
        c = cs[0]; n_cand += 1; ntoks.append(c["ntok"])
        if c["finish"] == "length":
            n_trunc += 1; continue
        idxs.append(i)
        con_users.append(I_CON.format(q1=q, a1=gt, q2=qbi,
                                      a2=(extract_boxed(c["text"]) or c["text"].strip()[-64:])))
    # Step4: 一致性过滤 → 组装多目标样本，关 thinking
    con = [strip_think(c[0]).strip().lower()
           for c in t.chat(con_users, 0.0, 16, n=1, enable_thinking=False)] if con_users else []
    rows = []
    for j, i in enumerate(idxs):
        if con[j].startswith("true"):
            q, gt, rf_text = kept[i]; qbi = qb[i]; rbi = rb_out[i][0]["text"]
            rows.append(msg_row(q, rf_text))                       # (a) Q → R_f
            rows.append(msg_row(q, qbi))                           # (b) Q → Q_b
            rows.append(msg_row(qbi + " " + BOXED_INSTR, rbi))     # (c) Q_b → R_b
    gen_stats(a.out, a.method, a.max_new, len(items), len(rows), n_cand, n_trunc, ntoks)
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
    ntoks, n_cand, n_trunc = [], 0, 0
    # Step1: 造全新题（temp=1.0 取多样性），过滤截断
    p1 = t.chat_full([P1.format(q=q) for q, _ in items], 1.0, a.max_new, n=a.n)
    newqs = []
    for cands in p1:
        for c in cands:
            n_cand += 1; ntoks.append(c["ntok"])
            if c["finish"] == "length":
                n_trunc += 1; continue
            # 先剥 <think>
            txt = strip_think(c["text"])
            idx = txt.rfind("FINAL CREATED QUESTION:")        # 取最后一个标记即最终版
            if idx != -1:
                nq = txt[idx + len("FINAL CREATED QUESTION:"):].strip()
                if nq:
                    newqs.append(nq)
    if not newqs:
        gen_stats(a.out, a.method, a.max_new, len(items), 0, n_cand, n_trunc, ntoks)
        return []
    # Step2: 造解 + self-consistency 多数投票过滤（无 gold）
    k = max(3, a.n)
    ans = t.chat_full([P2.format(q=nq) for nq in newqs], 1.0, a.max_new, n=k)
    rows = []
    for nq, cands in zip(newqs, ans):
        texts = []
        for c in cands:
            n_cand += 1; ntoks.append(c["ntok"])
            if c["finish"] == "length":
                n_trunc += 1; continue
            texts.append(c["text"])
        boxed = [extract_boxed(c) for c in texts]
        pairs = [(c, b) for c, b in zip(texts, boxed) if b]
        if len(pairs) < 2:
            continue
        maj, cnt = Counter(b for _, b in pairs).most_common(1)[0]
        if cnt < 2:  # 至少 2 次一致才保留
            continue
        sol = next(c for c, b in pairs if b == maj)
        rows.append(msg_row(nq + " " + BOXED_INSTR, sol))
    gen_stats(a.out, a.method, a.max_new, len(items), len(rows), n_cand, n_trunc, ntoks)
    return rows


# ---------- 跨能力 Standard-CoT：code / 选择题 ----------
def m_code(t, items, a):
    """LiveCodeBench：teacher 生成思路 + ```python 代码，prime_code 跑测试用例过滤，留第一个未截断且全部通过的。
    种子须为 prepare_code.py 的 parquet（reward_model.ground_truth = 测试用例 json）。"""
    outs = t.chat_full([q for q, _ in items], a.temp, a.max_new, n=a.n, system=(a.sys_prompt or None))
    rows, n_cand, n_trunc, ntoks = [], 0, 0, []
    for (q, gt), cands in zip(items, outs):
        keep = None
        for c in cands:
            n_cand += 1
            ntoks.append(c["ntok"])
            if c["finish"] == "length":  # 截断的代码不完整，丢弃
                n_trunc += 1
                continue
            if gt is not None and "```" in c["text"] and verify_code(c["text"], gt):
                keep = c["text"]
                break
        if keep is not None:
            rows.append(msg_row(q, keep))
    gen_stats(a.out, a.method, a.max_new, len(items), len(rows), n_cand, n_trunc, ntoks)
    return rows


def m_mc(t, items, a):
    """选择题（MMLU-Pro 等）：teacher 生成推理 + \\boxed{字母}，抽字母与 gold 比对过滤，留第一个未截断且答对的。
    种子须为 prepare_mc.py 的 parquet。"""
    outs = t.chat_full([q for q, _ in items], a.temp, a.max_new, n=a.n, system=(a.sys_prompt or None))
    rows, n_cand, n_trunc, ntoks = [], 0, 0, []
    for (q, gt), cands in zip(items, outs):
        keep = None
        for c in cands:
            n_cand += 1
            ntoks.append(c["ntok"])
            if c["finish"] == "length":
                n_trunc += 1
                continue
            if gt is not None and verify_mc(c["text"], gt):
                keep = c["text"]
                break
        if keep is not None:
            rows.append(msg_row(q, keep))
    gen_stats(a.out, a.method, a.max_new, len(items), len(rows), n_cand, n_trunc, ntoks)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["standard_cot", "shortest_cot", "reverse", "question_aug", "code_cot", "mc_cot"], required=True)
    ap.add_argument("--seed", required=True, help="RL parquet（含 prompt / reward_model.ground_truth）")
    ap.add_argument("--teacher", default="$MODELS/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=1)
    # 默认取 Qwen3 的 max_position_embeddings=40960
    ap.add_argument("--max_len", type=int, default=40960)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM 显存占比；与他人共卡时调低(如 0.7)")
    ap.add_argument("--max_new", type=int, default=38912)
    ap.add_argument("--sys_prompt", default="",
                    help="任务三 prompt 轴：standard_cot 的 system 指令(改蒸馏风格)；空=默认无 system")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 条种子（调试/控预算）")
    # —— API teacher ——
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
    rows = {"standard_cot": m_standard, "shortest_cot": m_shortest,
            "reverse": m_reverse, "question_aug": m_qaug,
            "code_cot": m_code, "mc_cot": m_mc}[a.method](t, items, a)
    save(rows, a.out, len(items), a.method)


if __name__ == "__main__":
    main()
