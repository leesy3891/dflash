#!/usr/bin/env bash
# Phase 2 of the selective sweep: 64k for both presets, into record_selective/.
#
# Qwen3-8B goes first and alone, on the spare card, in parallel with phase 1:
# selective removes 16 GB of hidden-state tuple at this length, which is what
# makes a single-card 64k run plausible where the full-mode record needed two.
# If it does not fit it is retried sharded once phase 1 has freed its GPUs.
#
# Qwen3-8B also needs --rope-scaling yarn: max_position_embeddings is 40960 on
# both target and draft, so 64k positions are outside what either was trained
# on and RoPE would silently extrapolate. Same flag the full-mode 64k record
# used, so the two stay comparable. Qwen3.5-9B trains to 262144 and needs none.
#
# Qwen3.5-9B waits for phase 1 to drain and then takes two cards: its fp32
# gated-delta-rule prefill fallback is ~2.3 GB at 4k and scales with sequence
# length, which puts a 64k run well past one card no matter what the drafter
# costs.
set -u
cd /home/seoyounglee/dflash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REC=record_selective
LOGDIR=logs/selective
Q=queue/selective.log
SPARE=$1          # card free right now
mkdir -p "$REC" "$LOGDIR"

say() { echo "$(date '+%m-%d %H:%M:%S') [$1] ${*:2}" >> "$Q"; }

# A GPU is idle when nvidia-smi lists no compute app on it and it holds under
# 100 MiB -- the same test the full-mode queue used, so a run never lands on a
# card someone else is already using.
idle_gpus() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
  while IFS=, read -r idx uuid used; do
    idx=${idx// /}; uuid=${uuid// /}; used=${used// /}
    if [ "$used" -lt 100 ] && ! echo "$busy" | grep -q "$uuid"; then echo "$idx"; fi
  done | paste -sd, -
}

wait_for_idle() {   # wait_for_idle <n> -> echoes a comma list of n idle GPUs
  local need=$1 free n
  while :; do
    free=$(idle_gpus); n=0
    [ -n "$free" ] && n=$(echo "$free" | tr ',' '\n' | grep -c .)
    if [ "$n" -ge "$need" ]; then echo "$free" | cut -d, -f1-"$need"; return; fi
    sleep 120
  done
}

run64k() {          # run64k <model> <gpus> <extra flags...>
  local model=$1 gpus=$2; shift 2
  local log="$LOGDIR/${model}_65536.log"
  local dm=""
  [ "$(echo "$gpus" | tr ',' '\n' | grep -c .)" -gt 1 ] && dm="--device-map balanced"
  say "$model" "START ctx=65536 on GPU $gpus $dm"
  CUDA_VISIBLE_DEVICES="$gpus" python -m dflash.cli benchmark transformers \
    --model-preset "$model" --context-length 65536 \
    --max-samples 32 --max-new-tokens 512 --reasoning off \
    --hidden-states selective --record-dir "$REC" $dm "$@" \
    > "$log" 2>&1
  local rc=$?
  say "$model" "DONE ctx=65536 (exit $rc)"
  return $rc
}

# --- Qwen3-8B: single card now, on the spare -----------------------------------
run64k qwen3-8b "$SPARE" --rope-scaling yarn
RC8=$?
if [ $RC8 -ne 0 ] && grep -qi "out of memory" "$LOGDIR/qwen3-8b_65536.log"; then
  say qwen3-8b "OOM on one card -- will retry sharded after 9B"
  RETRY8=1
else
  RETRY8=0
fi

# --- Qwen3.5-9B: two cards, after phase 1 has drained --------------------------
say qwen3.5-9b "waiting for 2 idle GPUs"
GPUS=$(wait_for_idle 2)
run64k qwen3.5-9b "$GPUS"

if [ "$RETRY8" = "1" ]; then
  say qwen3-8b "retrying 64k sharded"
  GPUS=$(wait_for_idle 2)
  run64k qwen3-8b "$GPUS" --rope-scaling yarn
fi
say queue "phase 2 drained"
