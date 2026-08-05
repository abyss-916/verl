#!/usr/bin/env bash
# 环境配置：按 verl 官方流程（docs/start/install.rst）建 conda 环境并安装 verl 与后端（vLLM）。
# 用法：
#   bash run/setup.sh                 # 建默认环境 verl 并安装
#   ENV_NAME=verl_grpo bash run/setup.sh   # 建另一环境名
#
# 说明：本脚本给的是 verl 原生安装命令。本项目实跑用两个隔离环境（verl / verl_grpo）；
# 因 glibc 2.31 需把 vLLM 钉在 ≤0.12、flash-attn 用预编译轮子，且有 2×3090 无 NVLink 的崩溃修复 env——
# 这些项目专属约束见 run/env.sh 顶部与 doc，不在本脚本内。
set -euo pipefail

ENV_NAME=${ENV_NAME:-verl}
VERL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)   # run/../.. = verl 仓根

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python==3.12
conda activate "$ENV_NAME"

USE_MEGATRON=0 bash "$VERL_ROOT/scripts/install_vllm_sglang_mcore.sh"   # 装 vLLM 等后端（不含 Megatron）
pip install --no-deps -e "$VERL_ROOT"                                   # 以可编辑方式安装 verl 本体

echo "[setup] 环境 $ENV_NAME 就绪。项目专属约束（vLLM≤0.12、flash-attn 预编译轮子、崩溃修复 env）见 run/env.sh 与 doc。"
