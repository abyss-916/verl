#!/usr/bin/env bash
# 公共路径与默认值，由 run/*.sh 自动 source，仅 export 到当前会话。
# 勿写入 API key 等机密，改用当场 export：export DEEPSEEK_API_KEY=... / export DASHSCOPE_API_KEY=...
export PROJ=${PROJ:-/data/liujiachen/verl/qwen3_4b_distill}
export MODELS=${MODELS:-/data/liujiachen/models}
export DATA=${DATA:-/data/liujiachen/datasets}
export CKPT=${CKPT:-/data/liujiachen/checkpoints}
export LOGS=${LOGS:-/data/liujiachen/logs}

# 缓存/临时/日志统一重定向到 /data
export HF_HOME=${HF_HOME:-/data/liujiachen/hf}                       # HF 模型/数据/token 缓存
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}             # 经镜像下载 datasets（MATH/OlymMATH/MMLU-Pro 等）
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/data/liujiachen/modelscope}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data/liujiachen/.cache}     # torch/triton/vllm 默认缓存位置
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}           # pip 缓存
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}  # 国内 pip 镜像
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}  # JIT 编译的算子（flashinfer 等）
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}  # torch.compile 缓存
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv}        # CUDA JIT 缓存
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}        # 放宽 HF 下载超时
export TMPDIR=${TMPDIR:-/data/liujiachen/tmp}                        # ray/临时文件
export RAY_TMPDIR=${RAY_TMPDIR:-/data/liujiachen/tmp/ray}
export WANDB_DIR=${WANDB_DIR:-$LOGS/wandb}
export WANDB_MODE=${WANDB_MODE:-offline}   # 默认离线（免登录）；上传设 WANDB_MODE=online 并先 wandb login
# ── verl GRPO 必需的运行时修复 ──
export JE_ARROW_MALLOC_CONF=${JE_ARROW_MALLOC_CONF:-background_thread:false}  # 关闭 pyarrow 内嵌 jemalloc 的后台线程
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}     # 禁用 P2P，退回 SHM 通信
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}   # 关闭 NCCL cuMem
# 勿设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（与 vLLM 内存池不兼容）
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}  # HF 快速分词并行
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$RAY_TMPDIR" "$WANDB_DIR" 2>/dev/null || true

export STUDENT_BASE=${STUDENT_BASE:-$MODELS/Qwen3-4B}   # student 固定为 Qwen3-4B
export TEACHER=${TEACHER:-$MODELS/Qwen3-8B}

# code 线（LiveCodeBench）
export CODE_HF=${CODE_HF:-livecodebench/code_generation_lite}
export CODE_VERSION=${CODE_VERSION:-release_v5}
# 扩展 benchmark（run/eval_extended.sh 使用）
export MMLU_PRO_HF=${MMLU_PRO_HF:-TIGER-Lab/MMLU-Pro}
export SUPERGPQA_HF=${SUPERGPQA_HF:-m-a-p/SuperGPQA}
export AIME_HF=${AIME_HF:-yentinglin/aime_2025}   # AIME2025，列 problem/answer；置空则跳过
export AIME_SUBSET=${AIME_SUBSET:-}

# 任务三 API teacher；key 经环境变量注入，勿写进仓库
export DEEPSEEK_API_BASE=${DEEPSEEK_API_BASE:-https://api.deepseek.com}
export QWEN_API_BASE=${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}

# ── 数据角色（训练/评测严格分离）──
export SEED_SOURCE=${SEED_SOURCE:-modelscope}   # SEED：训练/蒸馏种子 + GRPO prompt（初期 MATH-lighteval）
export SEED_HF=${SEED_HF:-AI-ModelScope/MATH-lighteval}
export SEED_SUBSET=${SEED_SUBSET:-}
export SEED_DIR=${SEED_DIR:-$DATA/math_seed}
# 主线种子 = Omni-MATH d4–5（omni_seed，500 条）
export OMNI_SEED_DIR=${OMNI_SEED_DIR:-$DATA/omni_seed}
# EVAL：held-out 评测（OlymMATH），不进入训练
export EVAL_SOURCE=${EVAL_SOURCE:-hf}
export EVAL_HF=${EVAL_HF:-RUC-AIBOX/OlymMATH}
export EVAL_SUBSET=${EVAL_SUBSET:-en-hard}
export EVAL_DIR=${EVAL_DIR:-$DATA/olymmath}

# ── 能力域切换：按 ABILITY 切 reward/种子/评测集/评测脚本/造数方法（默认 math）──
export ABILITY=${ABILITY:-math}
case "$ABILITY" in
  code) export REWARD_FN=$PROJ/reward/code_reward.py; export REWARD_MGR=prime; export ABILITY_SEED_DIR=$DATA/livecodebench; export ABILITY_EVAL_DIR=$DATA/livecodebench; export EVAL_PY=eval_code.py; export DISTILL_METHOD=code_cot ;;
  mc)   export REWARD_FN=$PROJ/reward/mc_reward.py;   export REWARD_MGR=naive; export ABILITY_SEED_DIR=$DATA/mmlu_pro;      export ABILITY_EVAL_DIR=$DATA/mmlu_pro;      export EVAL_PY=eval_mc.py;   export DISTILL_METHOD=mc_cot ;;
  *)    [ "$ABILITY" = math ] || echo "[env] 未知 ABILITY=$ABILITY（math|code|mc），按 math 处理" >&2
        export ABILITY=math; export REWARD_FN=$PROJ/reward/math_reward.py; export REWARD_MGR=naive; export ABILITY_SEED_DIR=$OMNI_SEED_DIR; export ABILITY_EVAL_DIR=$EVAL_DIR; export EVAL_PY=eval_math.py; export DISTILL_METHOD=standard_cot ;;
esac

# 取训练输出目录下最新 global_step 的 HF 权重目录
latest_hf() { ls -d "$1"/global_step_*/huggingface 2>/dev/null | sort -V | tail -1; }
export -f latest_hf

mkdir -p "$DATA" "$CKPT" "$LOGS/eval" "$LOGS/run" 2>/dev/null || true
echo "[env] PROJ=$PROJ STUDENT=$STUDENT_BASE TEACHER=$TEACHER"
