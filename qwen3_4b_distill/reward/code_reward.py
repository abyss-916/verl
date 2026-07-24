"""LiveCodeBench 等代码集的可验证奖励——复用 verl 的代码判分器（基于 LiveCodeBench 的 prime_code）。
- 默认用 `prime_code`（APPS/LCB 式本地执行，带超时；无需 sandbox 服务）。
- 若设了环境变量 `SANDBOX_FUSION_URL`，改用 `sandbox_fusion`（沙箱执行，更安全）。
GRPO/eval 挂载：custom_reward_function.path=/data/liujiachen/verl/qwen3_4b_distill/reward/code_reward.py

ground_truth 由 `prepare_code.py` 预先转成 prime_code 期望的格式（json 字符串）：
    stdin      : {"inputs":[输入串,...], "outputs":[期望输出串,...]}
    functional : {"inputs":[[参数,...],...], "outputs":[期望返回,...], "fn_name":"方法名"}
故本文件不再猜测格式，直接解析后交给执行器。

⚠️ 安全（共享服务器！）：`prime_code` 本地执行模型生成的代码——正式大规模跑前建议起 sandbox 并设
   `SANDBOX_FUSION_URL`；小规模验证/评测可用 prime_code（自带 signal 超时）。
"""

import json
import os


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
