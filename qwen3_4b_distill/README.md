# qwen3_4b_distill

**Qwen3-4B 推理能力的蒸馏与强化学习后训练研究**，构建于 [verl](https://github.com/volcengine/verl) fork 之上。

本目录是课题的**自研代码层**：数据接入、蒸馏数据构造、数据度量与归因、评测、以及全流程编排；训练 / RL / rollout 复用 verl 框架本体。研究主线不止于报告 accuracy，而在于**解释"表现为什么变"**——把蒸馏数据的可度量属性（长度、多样性、困惑度、指令跟随难度）与学生模型的下游表现关联起来。

> 代码在本仓（`abyss-916/verl`）的 `qwen3_4b_distill/` 下；过程记录、方法学与交付报告在配套文档仓（`abyss-916/project` 的 `doc/`、`material/`）。二者隔离，避免代码与文档相互污染。

---

## 目录结构

```
qwen3_4b_distill/
├── data_preprocess/     数据集 → verl parquet（RL / eval 格式）
│   ├── prepare_math.py      数学：MATH 种子（训练）/ OlymMATH（held-out 评测），答案自适应抽取
│   ├── prepare_code.py      代码：LiveCodeBench（防污染窗）
│   └── prepare_mc.py        选择题：MMLU-Pro / SuperGPQA / AIME（扩展 benchmark）
├── distill/
│   └── generate_cot.py      教师造 CoT + 可验证过滤 → SFT messages parquet（六种方法，见下）
├── train/
│   ├── sft.sh               off-policy 序列蒸馏（LoRA，2×3090 适配）
│   ├── whole_conv_sft_dataset.py  整段渲染 SFT 数据集（保住 Qwen3 <think>，见文件内说明）
│   ├── grpo.sh              GRPO 后训练（LoRA 叠 SFT-merged，2×3090 适配）
│   └── opd.sh               On-Policy Distillation（弃跑，见"已知取舍"）
├── reward/                  可验证奖励（GRPO / eval 共用，与判定同源）
│   ├── math_reward.py           math-verify
│   ├── code_reward.py           prime_code 跑单测
│   └── mc_reward.py             \boxed{字母} 匹配
├── eval/                    held-out 评测（base / SFT / GRPO 通用）
│   ├── eval_math.py             pass@1 / avg@k / pass@k（thinking，含截断率）
│   ├── eval_code.py / eval_mc.py
│   ├── base_at_k.py             由既有样本重算不同 k 的指标，不重跑生成
│   └── merge_shards.py          多卡分片结果合并（与单卡等价）
├── metrics/                 数据度量与归因（课题核心）
│   ├── data_metrics.py          length / distinct-n / PPL / IFD（均以 student 基座视角计算）
│   ├── compare_methods.py       各方法数据侧指标对比表
│   ├── attribution.py           数据属性 ↔ 下游表现 的相关性归因
│   └── slice_eval.py            逐学科 Δ + 配对 McNemar + 错例（深度归因）
├── run/                     编排脚本（详见"复现"）
├── tests/                   reward 判定器单元自检
└── README.md
```

---

## 运行环境

- **硬件**：2 × RTX 3090（24 GB，**无 NVLink**，共享机）。所有训练 / 生成 / 评测在服务器上进行；本地仅编辑代码，改后推送、服务器 `git pull`。
- **隔离 conda 环境**（glibc 2.31 限定 vLLM ≤ 0.12）：
  - `verl_grpo`：GRPO / rollout（torch 2.9 + cu128 / vLLM 0.12 / flash-attn 2.8.3）。
  - `verl`：SFT / 评测。
- **2×3090 无 NVLink 的必需运行时修复**（`run/env.sh` 已固化，`source` 即带上）：
  - `JE_ARROW_MALLOC_CONF=background_thread:false` —— pyarrow 内嵌 jemalloc 后台线程在 Ray fork 后段错。
  - `NCCL_P2P_DISABLE=1` + `NCCL_CUMEM_ENABLE=0` —— 无 NVLink 两卡 peer-access 不支持。
  - **切勿设** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —— 与 vLLM CuMem 内存池不兼容，会在引擎初始化时断言失败。

路径经环境变量集中约定（`run/env.sh`）：`PROJ` / `MODELS` / `DATA` / `CKPT` / `LOGS`，默认根为 `/data/liujiachen/`。

---

## 数据角色（训练与评测严格分离，防泄漏）

| 角色 | 数据 | 用途 |
|---|---|---|
| **SEED**（蒸馏种子 + GRPO prompt） | **Omni-MATH 难度 4–5**（`omni_seed`，每法 500 条） | 造蒸馏数据、GRPO 采样。初期用 MATH-lighteval，因 instruct-4B 已饱和而切换至难度匹配的 Omni（详见报告 §5.3）。 |
| **EVAL**（held-out） | **OlymMATH en-hard**（100 题，仅 test） | base / SFT / GRPO 全部在此评测，**绝不进训练**。 |

评测统一 `max_new=38912`（thinking 需长预算）；base 用 avg@8，SFT / GRPO 用 avg@4。

---

## 蒸馏方法（`distill/generate_cot.py`）

各方法**共用同一教师、同一模板、同一采样预算**（公平对比），判定器随能力切换（math-verify / prime_code / 字母匹配），与评测同源。

| 方法 | 来源思路 | 说明 |
|---|---|---|
| `standard_cot` | 标准拒绝采样 | 教师直接生成 CoT，答对者留用（基线）。 |
| `reverse` | RevThink | 前向解 + 逆向问题 + 逆向推理 + 一致性过滤 → 多目标样本。 |
| `question_aug` | Xwin-Math | 造新题 + 造解，无 gold 用 self-consistency 多数投票过滤。 |
| `shortest_cot` | Concise-CoT | 每题采样多候选，正确者里留 token 最少的（测"短而对"）。 |
| `code_cot` / `mc_cot` | — | LiveCodeBench / 选择题的对应造数据。 |

**教师轴**（任务三开放研究）：强度 8B → 14B → 32B-AWQ；专精 Qwen2.5-Math-7B-Instruct；跨族 DeepSeek-V4-pro（API）；顶点 235B（API 小样本）。

---

## 流水线与复现

完整链路：`prepare → distill → SFT → merge → eval`，可选 `→ GRPO → merge → eval`；配合 `metrics/` 做数据度量与归因。

**分步（逐步跑，每步先查 `nvidia-smi` 定 `gpu_mem`；长任务用 `setsid` 后台）**：

```bash
source run/env.sh
bash run/01_task1_data_and_base_eval.sh                       # 备 SEED + EVAL 数据 + base eval
METHOD=omni_standard_cot LIMIT=500 bash run/gen_distill.sh    # 造蒸馏数据（冒烟→正式）
EXP=sft_omni_standard_cot DATA_DIR=$DATA/distill/omni_standard_cot bash train/sft.sh   # SFT（首次先 TEST=1）
METHODS="omni_standard_cot" bash run/07_merge.sh              # LoRA 折叠 → 完整模型
python eval/eval_math.py --model $CKPT/sft_omni_standard_cot_merged \
  --data $EVAL_DIR/test.parquet --n 4 --out $LOGS/eval/olymmath_sft_omni_standard_cot
python metrics/data_metrics.py --data $DATA/distill/omni_standard_cot/train.parquet \
  --model $STUDENT_BASE --out $LOGS/metrics_omni_standard_cot.json
```

**GRPO 短 PoC（一键自动接续，`run/10_grpo_chain.sh`）**：

```bash
# GRPO(LoRA 叠 SFT-merged) → merge(折叠) → eval，一次启动、串行跑完；每步 fail-loud 门控
STEPS=5 setsid bash run/10_grpo_chain.sh > $LOGS/run/10_grpo_chain.log 2>&1 < /dev/null &
# 失败后复用已训 ckpt、只补 merge+eval： RESUME=1 setsid bash run/10_grpo_chain.sh ...
```

### 关键训练配置

- **SFT**：LoRA rank 32 / alpha 32 / all-linear，lr 2e-4，5 epoch，`max_length` 40960，整段渲染保 `<think>`（`whole_conv_sft_dataset.py`）。
- **GRPO**（短 PoC）：LoRA r32/α32 叠在 SFT-merged 之上；reward = math-verify；prompt 池 = Omni d4–5；**KL 至冻结 base**（关 LoRA 即参考模型，DeepSeekMath 式，`kl_loss_coef=0.001` + `low_var_kl`）；rollout TP=2、`RESP=16384`、N=5、GM=0.70；param/optimizer offload + `enforce_eager` + `use_fused_kernels`（融合 LM-head+CE，绕开长响应下的 logits 显存峰值）。2×3090 上属短程可行性演示，非收敛结果。

---

## 编排脚本（`run/`）

| 脚本 | 作用 |
|---|---|
| `env.sh` | 公共路径 + 缓存重定向 + 崩溃修复 env（被所有脚本 source） |
| `00_smoke.sh` | 环境自检 + verl 入口核对 |
| `01_task1_data_and_base_eval.sh` | 任务一：数据接入 + base eval |
| `gen_distill.sh` | 造蒸馏数据安全启动器（冒烟→正式；换 METHOD / TEACHER / LIMIT 通用） |
| `06_sft_eval_all.sh` | 多方法下游 SFT-eval 满配对比（两卡分片、串行） |
| `07_merge.sh` | LoRA 折叠为完整 HF 模型（含权重移动自检；PPO ckpt 自动取 `actor/` 子目录） |
| `10_grpo_chain.sh` | GRPO → merge → eval 一键自动接续 |
| `make_manifest.py` | 单实验完整记录（dataset / teacher / 方法 / 采样 / filter / 结果 / 论文对齐） |
| `05_extended.sh` / `08_expansion.sh` / `09_smoke_pipeline.sh` | 扩展 benchmark / 归因扩展批 / 端到端冒烟 |

---

## 结果与分析

下游对比（OlymMATH-hard，n=4，pass@1）：`question_aug` 17.75 是唯一稳超 base(15.75) 的方法，`reverse` 11.0 为多任务稀释的负结果，跨族 DeepSeek 因文体长度爆炸大量截断而崩。归因上 **PPL 是最强预测子**（Pearson r ≈ −0.82），预设的 IFD 反而弱。完整数据、表格与论证见文档仓的 `material/实验报告`。

---

## 已知取舍

- **OPD（On-Policy Distillation）弃跑**：2×3090 上 teacher+student+vLLM 共显存 OOM，且跨族异 tokenizer 无法 on-policy；停在设计与脚本阶段，报告如实标注。
- **GRPO 为短 PoC**：无 NVLink 下 TP=2 逐 token 跨卡 all-reduce 使 rollout 偏慢，按算力约束截为短程演示。
- 扩展 benchmark（SuperGPQA / AIME）代码就绪但未跑，留作上限。

## 致谢

训练 / RL / rollout 基于 [verl](https://github.com/volcengine/verl)。判定复用 verl 内置 math-verify / prime_code。
