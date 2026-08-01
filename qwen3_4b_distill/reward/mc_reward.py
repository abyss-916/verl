"""选择题（MMLU-Pro / SuperGPQA 等）的可验证奖励——规则判分：抽模型答案字母与正确字母比对。

verl 兼容签名 `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> 0.0/1.0`，
可直接用作 GRPO 的 `reward.custom_reward_function`（`reward_manager=naive` 即可），
也供 `distill/generate_cot.py` 的 `mc_cot` 造数据过滤复用——三处（eval / gen / GRPO）同一套抽取逻辑。

抽取顺序与 `eval/eval_mc.py` 完全一致：\\boxed{X} > "answer is X" > 文中最后一个孤立大写字母（A–P）。
"""

import re


def extract_letter(text):
    """抽答案字母（A–P）：\\boxed{X} 优先，其次 'answer is/：X'，最后退回文中最后一个孤立大写字母。"""
    m = re.search(r"\\boxed\{\s*([A-P])\s*\}", text)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*(?:is|:)\s*\(?\s*([A-P])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-P])\b", text)
    return m[-1] if m else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """规则奖励：抽到的字母 == 正确字母 → 1.0，否则 0.0（抽不到亦 0.0）。"""
    pred = extract_letter(solution_str or "")
    gold = str(ground_truth).strip().upper()
    return 1.0 if pred is not None and pred == gold else 0.0
