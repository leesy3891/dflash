#!/usr/bin/env bash
# Re-measure 32k and 64k with hidden_states=full, the reference behaviour the
# 4k/8k/16k records already use. Waits for COMPLETELY IDLE GPUs -- no compute
# processes and <100 MiB resident -- so a run never lands on a shared card and
# is never crowded out mid-flight the way the earlier NarrativeQA run was.
#
# Jobs are run one at a time, in order. A job that fails is logged and skipped.
set -u
cd /home/seoyounglee/dflash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=queue/queue.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$Q"; }

idle_gpus() {
  # A GPU is idle when nvidia-smi lists no compute apps on it and it holds
  # under 100 MiB (the driver's own few MiB, nothing else).
  local busy used idx
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
  while IFS=, read -r idx uuid used; do
    idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' ')
    used=$(echo "$used" | tr -d ' ')
    if [ "$used" -lt 100 ] && ! echo "$busy" | grep -q "$uuid"; then echo "$idx"; fi
  done | paste -sd, -
}

# name | ctx | gpus needed | extra flags | record dir
JOBS=(
  "qwen3-8b|32768|1||record"
  "qwen3-8b|65536|2|--rope-scaling yarn|record"
  "qwen3.5-9b|32768|2||record"
  "qwen3.5-9b|65536|2||record"
  "qwen3-8b|32768|1|--context-task narrativeqa|record/narrativeqa"
  "qwen3.5-9b|32768|2|--context-task narrativeqa|record/narrativeqa"
)

say "queue started, ${#JOBS[@]} jobs, hidden_states=full"
for job in "${JOBS[@]}"; do
  IFS='|' read -r model ctx need extra rec <<< "$job"
  tag="${model}_${ctx}$(echo "$extra" | grep -q narrativeqa && echo _narrativeqa)_full"
  log="logs/${tag}.log"
  say "QUEUED $tag (needs $need idle GPU(s))"
  while :; do
    free=$(idle_gpus); n=0
    [ -n "$free" ] && n=$(echo "$free" | tr ',' '\n' | grep -c .)
    if [ "$n" -ge "$need" ]; then
      pick=$(echo "$free" | cut -d, -f1-"$need")
      dm=""; [ "$need" -gt 1 ] && dm="--device-map balanced"
      say "START $tag on GPU $pick"
      mkdir -p "$rec"
      CUDA_VISIBLE_DEVICES="$pick" python -m dflash.cli benchmark transformers \
        --model-preset "$model" --context-length "$ctx" \
        --max-samples 32 --max-new-tokens 512 --reasoning off \
        --hidden-states full --record-dir "$rec" $dm $extra \
        > "$log" 2>&1
      say "DONE $tag (exit $?)"
      break
    fi
    sleep 120
  done
done
say "queue drained"
