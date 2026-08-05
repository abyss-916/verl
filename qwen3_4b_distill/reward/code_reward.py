"""LiveCodeBench 等代码集的可验证奖励，复用 verl 的 prime_code 判分器。
- 默认 `prime_code`（本地执行，带超时，无需 sandbox 服务）。
- 设 `SANDBOX_FUSION_URL` 则改用 `sandbox_fusion`（沙箱执行）。
GRPO/eval 挂载：custom_reward_function.path=<repo>/qwen3_4b_distill/reward/code_reward.py

ground_truth 由 `prepare_code.py` 预转为 prime_code 格式（json 串）：
    stdin      : {"inputs":[...], "outputs":[...]}
    functional : {"inputs":[[...],...], "outputs":[...], "fn_name":"..."}
安全：prime_code 本地执行模型代码，大规模运行前建议起 sandbox 并设 `SANDBOX_FUSION_URL`。
"""

import json
import os
import sys
import types

# pyext 兼容垫片（py3.12）
if "pyext" not in sys.modules:
    _pyext = types.ModuleType("pyext")

    class _RuntimeModule:
        @staticmethod
        def from_string(name, docstring, source):
            mod = types.ModuleType(str(name), docstring)
            exec(source, mod.__dict__)
            return mod

    _pyext.RuntimeModule = _RuntimeModule
    sys.modules["pyext"] = _pyext


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    tests = ground_truth
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except Exception:
            return 0.0
    if not isinstance(tests, dict) or "inputs" not in tests:
        return 0.0
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
