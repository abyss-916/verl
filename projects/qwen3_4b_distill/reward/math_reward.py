"""OlymMATH 等自定义 math 集的可验证奖励——复用 verl 内置 math-verify。

verl 的 RewardManager 会以固定签名调用：
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
GRPO/eval 挂载方式（命令行）：
    custom_reward_function.path=/data/liujiachen/verl/projects/qwen3_4b_distill/reward/math_reward.py
    custom_reward_function.name=compute_score
"""

from verl.utils.reward_score.math_verify import compute_score as _math_verify_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # _math_verify_score 内部把 ground_truth 包成 \boxed{} 后用 math-verify 比对，答对 1.0 否则 0.0，
    # 并在子进程里做 30s 超时保护。
    try:
        return float(_math_verify_score(solution_str, str(ground_truth)))
    except Exception:
        return 0.0
