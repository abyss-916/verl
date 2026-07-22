# RUNBOOK — 服务器上按此顺序跑（2×3090）

> 全流程用 `tmux`/`nohup` 后台跑，长任务睡前启动、醒来看结果。**任何训练/生成首次先 `TEST=1`** 验证不 OOM 再放大。
> 路径默认在 `run/env.sh`，按服务器实际改（默认 `/data/liujiachen/...`）。

## 0. 前置（M0→M1）
```bash
# env: conda activate llm；已按 doc/环境依赖安装清单.md 装好；verl 在 /data/liujiachen/verl
# 下模型（见项目 doc/下载部署清单.md）：Qwen3-4B-Base（student）、Qwen3-8B（teacher）
cd /data/liujiachen/verl/projects/qwen3_4b_distill
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
- **SFT ckpt 加载**：verl `sft_trainer` 存的 checkpoint 若非 HF 目录，vLLM eval 前需指到具体 `global_step_*` 子目录，或用 `verl.model_merger` 合并成 HF 权重。首次 SFT 后 `ls $CKPT/sft_standard_cot/` 确认结构。
- **OlymMATH 列名**：`prepare_math.py` 按常见名（problem/answer）猜；`load_dataset` 后脚本会打印 splits，若列名不符改 `Q_KEYS/A_KEYS`。
- **code（LiveCodeBench）**：评测/GRPO 需 sandbox。起 verl `sandbox_fusion` 服务后按 `train/grpo.sh` 注释接；或用 LCB 官方 harness。math 是主线、可独立跑通。
- **OOM**：降 `MB`/`RESP`/`train_batch_size`；GRPO 确认 `param_offload/optimizer_offload=True`、rollout `TP=1`、`gpu_memory_utilization` 调低；OPD 不行就退回 SFT。
- **wandb**：不想上传设 `WANDB_MODE=offline` 或把 `trainer.logger='["console"]'`。

## 对应计划
见项目 `项目计划.md` 日历排期；本 RUNBOOK 的 1–5 对应其 7/23–31 主线。
