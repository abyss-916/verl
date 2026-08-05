#!/usr/bin/env bash
# SFT 一条龙：训练 → merge 折叠 → held-out 评测，启动一次跑完（全程 conda verl 环境）。
# 用法（先 conda activate verl；首次可先 TEST=1 bash train/sft.sh 验证不 OOM）：
#   METHOD=omni_standard_cot setsid bash run/sft.sh > $LOGS/run/sft_omni_standard_cot.log 2>&1 < /dev/null &
# 可覆盖：METHOD（方法名）/ DATA_DIR（蒸馏数据目录，缺省 $DATA/distill/<METHOD>）/ N（eval 采样，缺省 4）；
#   train/sft.sh 的超参 env 也会透传。
set -o pipefail
source "$(dirname "$0")/env.sh"

METHOD=${METHOD:-omni_standard_cot}
DATA_DIR=${DATA_DIR:-$DATA/distill/$METHOD}
EXP=sft_${METHOD}
EVAL_N=${N:-4}

echo ">>>>> [1/3] SFT 训练开始 $(date '+%F %T')  exp=$EXP data=$DATA_DIR"
EXP="$EXP" DATA_DIR="$DATA_DIR" bash "$PROJ/train/sft.sh"
GSTEP=$(ls -d "$CKPT/$EXP"/global_step_* 2>/dev/null | sort -V | tail -1)
[ -n "$GSTEP" ] || { echo "!! [1/3] SFT 未产出 ckpt（见上方日志），终止"; exit 1; }
echo ">>>>> [1/3] SFT 完成 $(date '+%F %T')  ckpt=$GSTEP"

echo ">>>>> [2/3] merge 折叠 $(date '+%F %T')"
PREFIX=sft_ METHODS="$METHOD" bash "$PROJ/run/merge.sh"
[ -d "$CKPT/${EXP}_merged" ] || { echo "!! [2/3] 未产出 $CKPT/${EXP}_merged，终止"; exit 1; }

echo ">>>>> [3/3] eval $(date '+%F %T')"
PREFIX=sft_ METHODS="$METHOD" N="$EVAL_N" bash "$PROJ/run/eval.sh"
echo ">>>>> SFT 全部完成 $(date '+%F %T') → $LOGS/eval/olymmath_${EXP}/summary.json"
