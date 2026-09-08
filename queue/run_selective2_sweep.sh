#!/usr/bin/env bash
# Phase-local memory sweep, --hidden-states selective, into record_selective/.
#
# Same dataset, sample count, seed, block size, max_new_tokens and reasoning
# setting as the selective sweep this supersedes, so acceptance and peak
# allocated are directly comparable. What is new is the instrumentation:
# per-phase simultaneous memory, the first draft call separated from the steady
# state, and CUDA-event timers inside the drafter. All of it is measurement
# only -- verified bit-identical tokens, acceptance and peak on both presets.
#
# Four idle A6000s, one job per card:
#   GPU $1  qwen3-8b    4k 8k 16k 32k
#   GPU $2  qwen3.5-9b  4k 8k 16k 32k
#   GPU $3  qwen3-8b    64k, single card, --rope-scaling yarn (40960 trained
#           positions on target and draft alike, so 64k needs interpolation)
#   GPU $3,$4  qwen3.5-9b 64k, once the 8B 64k run frees the card. Its fp32
#           gated-delta-rule prefill fallback puts 64k past one card whatever
#           the drafter costs, so this one is sharded and its timings are
#           pipeline-parallel -- compare its acceptance and memory, not its
#           latency, against the single-GPU rows.
set -u
cd /home/seoyounglee/dflash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REC=record_selective
LOGDIR=logs/selective2
Q=queue/selective2.log
mkdir -p "$REC" "$LOGDIR" queue

say() { echo "$(date '+%m-%d %H:%M:%S') [$1] ${*:2}" >> "$Q"; }

COMMON=(--max-samples 32 --max-new-tokens 512 --reasoning off
        --hidden-states selective --profile-draft-memory --record-dir "$REC")

run() {             # run <model> <ctx> <gpus> [extra flags...]
  local model=$1 ctx=$2 gpus=$3; shift 3
  local log="$LOGDIR/${model}_${ctx}.log" dm=""
  [ "$(echo "$gpus" | tr ',' '\n' | grep -c .)" -gt 1 ] && dm="--device-map balanced"
  say "$model" "START ctx=$ctx on GPU $gpus $dm $*"
  CUDA_VISIBLE_DEVICES="$gpus" python -m dflash.cli benchmark transformers \
    --model-preset "$model" --context-length "$ctx" \
    "${COMMON[@]}" $dm "$@" > "$log" 2>&1
  local rc=$?
  say "$model" "DONE ctx=$ctx (exit $rc)"
  return $rc
}

short_sweep() {     # short_sweep <model> <gpu> <spare>
  local model=$1 gpu=$2 spare=$3
  for ctx in 4096 8192 16384 32768; do
    run "$model" "$ctx" "$gpu"
    if [ $? -ne 0 ] && grep -qi "out of memory" "$LOGDIR/${model}_${ctx}.log"; then
      say "$model" "OOM ctx=$ctx -- retrying sharded over $gpu,$spare"
      run "$model" "$ctx" "$gpu,$spare"
    fi
  done
  say "$model" "4k-32k finished"
}

A=$1; B=$2; C=$3; D=$4
say queue "start: 8b->GPU $A, 9b->GPU $B, 64k->GPU $C then $C,$D"
short_sweep qwen3-8b   "$A" "$D" &
P1=$!
short_sweep qwen3.5-9b "$B" "$D" &
P2=$!

# 64k, in series on the last two cards: 8B alone first, then 9B across both.
(
  run qwen3-8b 65536 "$C" --rope-scaling yarn
  if [ $? -ne 0 ] && grep -qi "out of memory" "$LOGDIR/qwen3-8b_65536.log"; then
    say qwen3-8b "OOM at 64k on one card -- retrying sharded over $C,$D"
    run qwen3-8b 65536 "$C,$D" --rope-scaling yarn
  fi
  run qwen3.5-9b 65536 "$C,$D"
) &
P3=$!

wait $P1 $P2 $P3
say queue "sweep drained"
