# projects/qwen3_4b_distill — 夏令营课题代码（在 verl fork 之上）

课题：研究 **Qwen3-4B** 的蒸馏 pipeline `distill → metrics → sft → sft_eval → grpo → grpo_eval`，
核心理念**不只报 accuracy，要解释为什么变**。verl 负责训练/RL/评测，本目录负责**数据构造 + 数据度量 + 编排 + 归因**。

> 本机只写/改代码；配置、下模型、训练/评测都在**服务器 2×RTX3090** 上跑。改完 push 到 `abyss-916/verl`。

## 服务器路径约定
| 变量 | 默认 |
|---|---|
| verl 仓 | `/data/liujiachen/verl`（本目录在其下 `projects/qwen3_4b_distill/`） |
| 模型 | `/data/liujiachen/models/`（`Qwen3-4B-Base`、`Qwen3-8B`…） |
| 数据 | `/data/liujiachen/datasets/`（verl parquet） |
| ckpt | `/data/liujiachen/checkpoints/` |

## 流水线与文件（对应 doc/verl框架解读.md §14.2）
| 步 | 文件 | 作用 |
|---|---|---|
| 1 数据接入 | `data_preprocess/prepare_math.py` | 数据集 → verl **RL parquet**（含 ground_truth，供 GRPO/eval） |
| 2 造蒸馏数据 | `distill/generate_cot.py` | teacher(8B) 造 CoT + math-verify 过滤 → **SFT messages parquet**（standard_cot/reverse/question_aug） |
| 3 SFT | `train/sft.sh` | off-policy 序列蒸馏训 Qwen3-4B-Base |
| — reward | `reward/math_reward.py` | OlymMATH 等自定义集的可验证奖励（复用 verl math-verify） |
| 4 GRPO | `train/grpo.sh` | GRPO 后训练（**2×3090 适配**） |
| 5 OPD(加分) | `train/opd.sh` | On-Policy Distillation（logit KD，**stretch**，可能 OOM） |
| 6 度量 | `metrics/data_metrics.py` `metrics/compare_methods.py` | 各方法数据 length/diversity/PPL/IFD（**student 视角**）+ 三法对比表 |
| 7 评测 | `eval/eval_math.py` | pass@1 / avg@k / pass@k（thinking），逐题 jsonl + 论文对齐 |
| code | `data_preprocess/prepare_code.py` | LiveCodeBench → RL parquet（评测/GRPO 需 sandbox） |

## 一键编排（run/）——服务器上按 `RUNBOOK.md` 顺序跑
| 脚本 | 阶段 |
|---|---|
| `run/00_smoke.sh` | 环境自检 + verl 入口核对（M0→M1） |
| `run/01_task1_data_and_base_eval.sh` | 任务一：数据 + base eval |
| `run/02_task2_methods.sh` | 任务二：三法造数据 + 度量 + SFT + sft_eval + 对比 |
| `run/03_grpo.sh` | GRPO + grpo_eval（后台跑） |
| `run/04_task3_teacher_scan.sh` | 任务三：teacher 强度扫描（off-policy） |

`distill/generate_cot.py` 三法（standard_cot / reverse / question_aug）**均已实现**。

## 2×3090 铁律
- **任何训练/生成首次先 `TEST=1`**（几十条/小 batch/短 response）验证不 OOM，再放大。
- GRPO：去 ref(`use_kl_loss=False`) + 全 offload + rollout `TP=1` + 小 batch；单次量级数天。
- 全流程服务器 `tmux/nohup` 后台跑，睡前启动、醒来看结果。

## 数据角色（训练/评测严格分离）
- **SEED（训练/蒸馏种子 + GRPO prompt）= MATH-lighteval train ~7500** → `$SEED_DIR`（服务 task2 scaling）。
- **EVAL（held-out）= OlymMATH en-hard** → `$EVAL_DIR`（只评测，绝不进训练；所有分报在它上）。

## 一条最短跑通路径（math 主线，走 run 编排）
```bash
source run/env.sh
bash run/01_task1_data_and_base_eval.sh          # 备 SEED+EVAL 数据 + base eval(held-out)
LIMIT=200 TEST=1 bash run/02_task2_methods.sh    # 三法造数据+度量+SFT+sft_eval(held-out)，先小规模跑通
TEST=1 bash run/03_grpo.sh                        # GRPO(prompt=MATH) + grpo_eval(held-out)，先验证不 OOM
```
详见 `RUNBOOK.md`。
