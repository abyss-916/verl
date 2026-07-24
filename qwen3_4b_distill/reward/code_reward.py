"""LiveCodeBench 等代码集的可验证奖励——复用 verl 的代码判分器（本地执行单测）。
- 默认用 `prime_code`（APPS 式本地执行，无需 sandbox 服务）。
- 若设了环境变量 `SANDBOX_FUSION_URL`，改用 `sandbox_fusion`（更安全的沙箱）。
GRPO/eval 挂载：custom_reward_function.path=/data/liujiachen/verl/qwen3_4b_distill/reward/code_reward.py

⚠️ 安全（共享服务器！）：`prime_code` 会**本地执行模型生成的任意 Python**——在共享机上有风险。
   正式跑 code 线**强烈建议先起 sandbox** 并设 `SANDBOX_FUSION_URL`，用沙箱执行，别裸跑 prime_code。
⚠️ 测试用例格式：prime_code 期望 APPS 式 {"inputs":[...],"outputs":[...]}（或含 fn_name）。
   LiveCodeBench 的 public_test_cases 格式可能不同，**首次跑必须核对 _normalize_tests 的转换**（见 TODO）。
"""

import json
import os


def _normalize_tests(ground_truth):
    """把 ground_truth（JSON 字符串/对象）转成打分器期望的测试用例结构。
    TODO(首跑后据 LCB 实际格式调整)：LCB 常见为 [{"input":..,"output":..,"testtype":..}, ...]，
    prime_code 期望 {"inputs":[...],"outputs":[...]}。这里做一个尽力转换。"""
    t = ground_truth
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except Exception:
            return ground_truth
    if isinstance(t, list) and t and isinstance(t[0], dict) and "input" in t[0]:
        return {"inputs": [c.get("input") for c in t], "outputs": [c.get("output") for c in t]}
    return t


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    tests = _normalize_tests(ground_truth)
    url = os.environ.get("SANDBOX_FUSION_URL")
    try:
        if url:
            from verl.utils.reward_score import sandbox_fusion

            r = sandbox_fusion.compute_score(url, None, 1024, solution_str, tests, continuous=True)
        else:
            from verl.utils.reward_score import prime_code

            r = prime_code.compute_score(solution_str, tests, continuous=True)
        return float(r[0] if isinstance(r, tuple) else r)
    except Exception:
        return 0.0
