# qwen3_4b_distill — 夏令营课题代码（在 verl fork 之上）

课题：研究 **Qwen3-4B** 的蒸馏 pipeline `distill → metrics → sft → sft_eval → grpo → grpo_eval`，
核心理念**不只报 accuracy，要解释为什么变**。verl 负责训练/RL/评测，本目录负责**数据构造 + 数据度量 + 编排 + 归因**。

> 本机只写/改代码；配置、下模型、训练/评测都在**服务器 2×RTX3090** 上跑。改完 push 到 `abyss-916/verl`。

## 服务器路径约定
| 变量 | 默认 |
|---|---|
| verl 仓 | `/data/liujiachen/verl`（本目录在其下 `qwen3_4b_distill/`） |
| 模型 | `/data/liujiachen/models/`（`Qwen3-4B`、`Qwen3-8B`…） |
| 数据 | `/data/liujiachen/datasets/`（verl parquet） |
| ckpt | `/data/liujiachen/checkpoints/` |

## 流水线与文件（对应 doc/verl框架解读.md §14.2）
| 步 | 文件 | 作用 |
|---|---|---|
| 1 数据接入 | `data_preprocess/prepare_math.py` | 数据集 → verl **RL parquet**（含 ground_truth，供 GRPO/eval） |
| 2 造蒸馏数据 | `distill/generate_cot.py` | teacher(8B) 造 CoT + math-verify 过滤 → **SFT messages parquet**（standard_cot/reverse/question_aug） |
| 3 SFT | `train/sft.sh` | off-policy 序列蒸馏训 Qwen3-4B |
| — reward | `reward/math_reward.py` | OlymMATH 等自定义集的可验证奖励（复用 verl math-verify） |
| 4 GRPO | `train/grpo.sh` | GRPO 后训练（**2×3090 适配**，从 SFT ckpt 起，prompt=MATH 种子防泄漏） |
| 5 度量 | `metrics/data_metrics.py` `metrics/compare_methods.py` | 各方法/教师数据 length/diversity/PPL/IFD（**student 视角**）+ 对比表 |
| 6 评测 | `eval/eval_math.py`（+ `merge_shards.py` 分片合并） | pass@1 / avg@k / pass@k（thinking，含截断率），逐题 jsonl + 论文对齐 |
| 7 记录 | `run/make_manifest.py` | 每实验统一记录（dataset/teacher/方法/采样/filter/数据统计/结果/论文对齐） |
| code | `data_preprocess/prepare_code.py` `eval/eval_code.py` `reward/code_reward.py` | LiveCodeBench（需 parquet 源 + sandbox） |

## 编排脚本（run/）
| 脚本 | 阶段 |
|---|---|
| `run/00_smoke.sh` | 环境自检 + verl 入口核对（M0→M1） |
| `run/01_task1_data_and_base_eval.sh` | 任务一：数据接入 + base eval |
| `run/gen_distill.sh` | 造蒸馏数据的**安全启动器**（tp=2 满预算，冒烟→正式；standard_cot/reverse/question_aug + 换教师通用） |
| `run/03_grpo.sh` | GRPO + grpo_eval（从 SFT ckpt 起，后台跑） |
| `run/make_manifest.py` | 每实验统一记录（满足课题§1） |

> **执行纪律（共享服务器 + 质量第一）**：不用"一键跑全部"的自动链；**每步先查 `nvidia-smi` 定 gpu_mem、造数据先冒烟**，逐步跑（造数据用 `gen_distill.sh`，训练 `train/sft.sh`，评测 `eval/eval_math.py`，度量 `metrics/*.py`）。任务二/三/scaling 都用 `gen_distill.sh`（换 METHOD / TEACHER / LIMIT）+ 分步命令组合。

`distill/generate_cot.py` 三法（standard_cot / reverse / question_aug）+ 多教师/API teacher（`--teacher_type api`）**均已实现**。
归因：`metrics/attribution.py`（数据指标↔表现相关）+ `metrics/slice_eval.py`（**深度归因**：`--by subject` 逐学科 Δ + `--vs` 配对 McNemar + 错例）。
code 线：`prepare_code`/`eval_code`/`code_reward`（就绪门：LCB parquet 源 + sandbox + 测试用例格式首跑核对）。

**T4 加分/stretch（不在 committed，有余量才跑；已写好、留作上限）**：
`train/opd.sh`（On-Policy Distillation，logit KD；2×3090 可能 OOM，炸了也作"尝试+分析"写进报告）｜
`data_preprocess/prepare_mc.py`+`eval/eval_mc.py`+`run/05_extended.sh`（MMLU-Pro/SuperGPQA/AIME 扩展 benchmark base eval）。

## 2×3090 铁律
- **任何训练/生成首次先 `TEST=1`**（几十条/小 batch/短 response）验证不 OOM，再放大。
- GRPO：去 ref(`use_kl_loss=False`) + 全 offload + rollout `TP=1` + 小 batch；完整版数天，默认只跑短 PoC(几百步~1晚，非必交加分)。
- 全流程服务器 `tmux/nohup` 后台跑，睡前启动、醒来看结果。

## 数据角色（训练/评测严格分离）
- **SEED（训练/蒸馏种子 + GRPO prompt）= MATH-lighteval train ~7500** → `$SEED_DIR`（服务 task2 scaling）。
- **EVAL（held-out）= OlymMATH en-hard** → `$EVAL_DIR`（只评测，绝不进训练；所有分报在它上）。

## 一条最短跑通路径（math 主线，逐步跑；每步先查 nvidia-smi）
```bash
source run/env.sh
bash run/01_task1_data_and_base_eval.sh                                  # 备 SEED+EVAL 数据 + base eval(held-out)
GPU_MEM=0.9 METHOD=standard_cot LIMIT=2000 bash run/gen_distill.sh       # 造数据(冒烟→正式，满预算)
EXP=sft_standard_cot DATA_DIR=$DATA/distill/standard_cot bash train/sft.sh   # SFT(首次先 TEST=1)
python eval/eval_math.py --model "$(latest_hf $CKPT/sft_standard_cot)" --data $EVAL_DIR/test.parquet --n 8 --out $LOGS/eval/olymmath_sft_standard
python metrics/data_metrics.py --data $DATA/distill/standard_cot/train.parquet --model $STUDENT_BASE --out $LOGS/metrics_standard_cot.json
# GRPO(upward)： TEST=1 bash run/03_grpo.sh   先验不 OOM
```
详见 `项目 doc/RUNBOOK.md` 与 `项目 doc/课题要求对照与交付.md` 的排期。
