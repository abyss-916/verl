# RUNBOOK — 服务器上按此顺序跑（2×3090）

> 全流程用 `tmux`/`nohup` 后台跑，长任务睡前启动、醒来看结果。**任何训练/生成首次先 `TEST=1`** 验证不 OOM 再放大。
> 路径默认在 `run/env.sh`，按服务器实际改（默认 `/data/liujiachen/...`）。

## 数据角色（重要——高质量课题的前提：训练/评测严格分离）
- **SEED（训练/蒸馏种子 + GRPO prompt）= MATH-lighteval train（~7500）** → `$SEED_DIR`（`math_seed/`）。teacher 在其上造 CoT（任务二三法 + 任务三 teacher 扫描），GRPO 也用它当 prompt。用 `LIMIT` 控预算/做 scaling。
- **EVAL（held-out 评测）= OlymMATH en-hard（200，仅 test）** → `$EVAL_DIR`（`olymmath/`）。**只评测、绝不进训练**，所有 base/SFT/GRPO 分都报在它上，保证与论文对齐可信。
- MATH 与 OlymMATH 问题不重叠（MATH=2021、OlymMATH=2025 且做过去污）→ 干净的"训 MATH、测 olympiad"迁移评测。
- 附带：`math_seed/test.parquet`（MATH test，held-out from train）可作**同分布** in-domain eval（`--limit 500` 近似 MATH-500）。

## 0. 前置（M0→M1）
```bash
# env: conda activate llm；已按 doc/环境依赖安装清单.md 装好；verl 在 /data/liujiachen/verl
# 下模型（见项目 doc/下载部署清单.md）：Qwen3-4B-Base（student）、Qwen3-8B（teacher）
cd /data/liujiachen/verl/qwen3_4b_distill
bash run/00_smoke.sh          # 自检 + verl 入口核对
```

## 1. 任务一：数据 + base eval（7/23–24）
```bash
bash run/01_task1_data_and_base_eval.sh
# 看 $LOGS/eval/olymmath_base/summary.json，对齐锚点（OlymMATH HARD-EN≈13.9，见 Qwen3报告笔记）
```

## 2. 任务二：三法造数据 + 度量 + SFT（7/25–26）
```bash
# 首轮跑通（小规模）：
LIMIT=200 TEST=1 bash run/02_task2_methods.sh
# 校准无误后正式跑（去掉 TEST，按预算设 LIMIT）：
LIMIT=2000 bash run/02_task2_methods.sh
# 产出：$LOGS/compare_methods.md（三法数据对比）+ 各法 sft_eval
```

## 3. GRPO（7/26 起，长杆早启，后台跑数天）
```bash
TEST=1 bash run/03_grpo.sh                       # 先验证不 OOM
nohup bash run/03_grpo.sh > $LOGS/grpo.log 2>&1 &  # 正式，后台
```

## 4. 任务三：teacher 扫描（7/27–29）
```bash
LIMIT=2000 TEACHERS="Qwen3-8B" bash run/04_task3_teacher_scan.sh
# 更强 teacher：先下 Qwen3-32B(4bit) 或配 API，加进 TEACHERS
# on-policy 对照（stretch，可能 OOM）：
TEST=1 EXP=opd_4b_from_8b DATA_DIR=$DATA/olymmath bash train/opd.sh
```

## 5. 回填报告
把 `$LOGS/eval/*/summary.json`、`compare_methods.md`、metrics json 的结果填进 `项目/material/实验报告.md`。

---

## 常见坑
- **SFT/GRPO ckpt（已解决）**：训练脚本已加 `checkpoint.save_contents=[...,hf_model]`，HF 全权重会存到 `<ckpt>/global_step_<N>/huggingface/`。run 脚本用 `latest_hf` 自动指到最新那个供 eval / 从 SFT 起 GRPO。代价：每个 save 存一份全权重占盘（默认 keep 全部），盘紧可加 `trainer.max_actor_ckpt_to_keep` / `checkpoint.max_ckpt_to_keep`。
- **OlymMATH（已解决）**：列 = `problem/answer/subject/unique_id`（脚本默认已对）；**config 小写** `en-hard/en-easy/zh-hard/zh-easy/lean`；**仅 test split**（train 复用 test）。默认 `MATH_SUBSET=en-hard`。
- **code（LiveCodeBench）**：评测/GRPO 需 sandbox。起 verl `sandbox_fusion` 服务后按 `train/grpo.sh` 注释接；或用 LCB 官方 harness。math 是主线、可独立跑通。
- **OOM**：降 `MB`/`RESP`/`train_batch_size`；GRPO 确认 `param_offload/optimizer_offload=True`、rollout `TP=1`、`gpu_memory_utilization` 调低；OPD 不行就退回 SFT。
- **wandb**：不想上传设 `WANDB_MODE=offline` 或把 `trainer.logger='["console"]'`。

## 对应计划
见项目 `项目计划.md` 日历排期；本 RUNBOOK 的 1–5 对应其 7/23–31 主线。
