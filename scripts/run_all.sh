#!/bin/bash
# Full pipeline, resumable: every step is skipped when its output already exists. Run from the repo root:
#   setsid nohup scripts/run_all.sh > logs/run_all.log 2>&1 &
cd "$(dirname "$0")/.." && source scripts/env.sh
set -eo pipefail   # a failed step stops the pipeline; ALL_DONE is only printed if every step succeeded
TASKS_EXTRA=${TASKS_EXTRA:-"1 3 7"}
mkdir -p logs results ckpt
TASK=5; EPS=${EPS:-30}; STEPS=${STEPS:-3000}
EVAL_SEEDS="2001 3001 4001"
ev() { [ -f logs/eval_$1_s$2/eval_info.json ] || scripts/run_eval.sh "$@"; }
load() { uptime | sed 's/.*load average: \([0-9]*\).*/\1/'; }
for s in $EVAL_SEEDS; do ev pretrained $s $EPS; done
for ts in 0 1; do
  [ -f ckpt/lora_s$ts/adapter.safetensors ] || { while [ "$(load)" -gt 30 ]; do sleep 60; done
    heavy $PY scripts/train.py lora_s$ts $TASK $ts $STEPS 16 2>&1 | grep -v Warn > logs/train_lora_s$ts.log; }
done
for ts in 0 1; do for s in $EVAL_SEEDS; do ev lora_s$ts $s $EPS ckpt/lora_s$ts/adapter.safetensors; done; done
# forgetting check: other tasks, 10 episodes, one eval seed
for t in $TASKS_EXTRA; do
  ev pretrained_t$t 2001 10 "" $t
  for ts in 0 1; do ev lora_s${ts}_t$t 2001 10 ckpt/lora_s$ts/adapter.safetensors $t; done
done
echo ALL_DONE
