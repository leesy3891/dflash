#!/usr/bin/env bash
# Context sweep with --hidden-states selective, written to record_selective/.
#
# Phase 1 (this script): 4k..32k, one dedicated GPU per model, the two models
# in parallel. Every run is single-GPU, so peak_memory_gb is exactly
# torch.cuda.max_memory_allocated and peak_site names the operation that set it.
#
# A 32k Qwen3.5-9B run is the one job that may not fit a 48 GB card: selective
# takes its projected peak to ~44 GB, and the fp32 gated-delta-rule prefill
# fallback is what decides it. If that run OOMs it is retried once, sharded
# over the model's own GPU plus the spare -- sharding changes no arithmetic
# (acceptance is identical) but makes the timings pipeline-parallel.
set -u
cd /home/seoyounglee/dflash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REC=record_selective
LOGDIR=logs/selective
Q=queue/selective.log
mkdir -p "$REC" "$LOGDIR" queue

say() { echo "$(date '+%m-%d %H:%M:%S') [$1] ${*:2}" >> "$Q"; }

run_model() {
  local model=$1 gpu=$2 spare=$3
  for ctx in 4096 8192 16384 32768; do
    local log="$LOGDIR/${model}_${ctx}.log"
    say "$model" "START ctx=$ctx on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -m dflash.cli benchmark transformers \
      --model-preset "$model" --context-length "$ctx" \
      --max-samples 32 --max-new-tokens 512 --reasoning off \
      --hidden-states selective --record-dir "$REC" \
      > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] && grep -qi "out of memory" "$log"; then
      say "$model" "OOM ctx=$ctx on one GPU -- retrying sharded over $gpu,$spare"
      CUDA_VISIBLE_DEVICES="$gpu,$spare" python -m dflash.cli benchmark transformers \
        --model-preset "$model" --context-length "$ctx" \
        --max-samples 32 --max-new-tokens 512 --reasoning off \
        --hidden-states selective --record-dir "$REC" \
        --device-map balanced \
        > "$log" 2>&1
      rc=$?
    fi
    say "$model" "DONE ctx=$ctx (exit $rc)"
  done
  say "$model" "sweep finished"
}

say queue "phase 1 start: 4k-32k, selective, GPU $1 -> qwen3-8b, GPU $2 -> qwen3.5-9b"
run_model qwen3-8b   "$1" "$3" &
P1=$!
run_model qwen3.5-9b "$2" "$3" &
P2=$!
wait $P1 $P2
say queue "phase 1 drained"
