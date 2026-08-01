"""reward 判分器单元自检（免 GPU、免数据集）：对合成的"正确解 / 错误解"断言 compute_score = 1.0 / 0.0。
证明 code(prime_code 本地执行单测) / mc(抽字母比对) 的可验证奖励**真的在判分**，而非空壳——
这是"code/MMLU 是真接入、不是只有 base eval"的最快证据（GRPO 用的正是同一个 compute_score）。

用法：
  python tests/reward_unit_check.py --ability code    # 合成 stdin 题：读 n 打印 2n
  python tests/reward_unit_check.py --ability mc      # 合成选择题：\\boxed{字母}
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # → qwen3_4b_distill/


def _extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[-1].strip() if m else text.strip()


def check_mc():
    from reward.mc_reward import compute_score
    gt = "C"
    good = "Work through each option... hence the answer is \\boxed{C}."
    bad = "Work through each option... hence \\boxed{A}."
    g = compute_score("mmlu_pro", good, gt)
    b = compute_score("mmlu_pro", bad, gt)
    print(f"[mc] correct={g} wrong={b}")
    assert g == 1.0, f"mc 正确解应=1.0，实得 {g}"
    assert b == 0.0, f"mc 错误解应=0.0，实得 {b}"


def check_code():
    from reward.code_reward import compute_score
    # 合成 stdin 题（无 fn_name → prime_code 走 standard_input）：读一个整数、输出其 2 倍
    tests = json.dumps({"inputs": ["3\n"], "outputs": ["6\n"]})
    good = "```python\nn = int(input())\nprint(n * 2)\n```"
    bad = "```python\nn = int(input())\nprint(n + 1)\n```"
    g = compute_score("livecodebench", _extract_code(good), tests)
    b = compute_score("livecodebench", _extract_code(bad), tests)
    print(f"[code] correct={g} wrong={b}")
    assert g == 1.0, f"code 正确解应=1.0，实得 {g}（prime_code 执行/格式问题？）"
    assert b == 0.0, f"code 错误解应=0.0，实得 {b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ability", required=True, choices=["code", "mc"])
    ap.add_argument("--data", default=None, help="兼容 smoke 传参；自检用合成样本，不依赖数据集")
    a = ap.parse_args()
    {"mc": check_mc, "code": check_code}[a.ability]()
    print(f"[{a.ability}] reward 单元自检 PASS")


if __name__ == "__main__":
    main()
