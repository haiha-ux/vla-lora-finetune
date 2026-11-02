#!/bin/bash
# usage: scripts/run_eval.sh <tag> <eval_seed> <episodes> [adapter.safetensors] [task_id]
# Fresh layouts: --env.init_states=false (random placement from the eval seed) instead of the 50 fixed init states.
cd "$(dirname "$0")/.." && source scripts/env.sh
TAG=$1; SEED=$2; N=$3; TASK=${5:-5}
[ -n "$4" ] && export LORA_ADAPTER=$4
mkdir -p logs results/eval
export TIMING_OUT=logs/timing_${TAG}_s${SEED}.json
timeout 7200 nice -n 19 ionice -c3 $PY scripts/eval_lora.py \
  --policy.path=HuggingFaceVLA/smolvla_libero --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[$TASK]" --env.init_states=false \
  --eval.batch_size=1 --eval.n_episodes=$N --env.max_parallel_tasks=1 \
  --seed=$SEED --output_dir=logs/eval_${TAG}_s${SEED}
