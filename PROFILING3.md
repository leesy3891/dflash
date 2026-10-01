# Batch architecture profiling: three fixed GPUs

## Scope and provenance

Implementation is on `batch`, starting from local and remote
`648f97c283cb636aa35e2a7a7f6c3e489f17050c`. No other branch was merged,
cherry-picked, or checked out. Existing deleted logs and untracked experiments
were already present and were not modified. New results go to `record_arch_batch/`.
This is a new engine in `dflash/batch.py`, not an implementation recovered from
previous batch planning documents. B=1 is used only for this engine's correctness
smoke/bridge checks; the scientific sweep is B=2,4,8,16,32.

The fixed physical devices are RTX A6000 GPUs 1,2,3:

| Logical | Physical | GPU UUID |
|---|---|---|
| 0 | 1 | GPU-ef8b9800-1a20-174f-32b5-b3e9935e1db2 |
| 1 | 2 | GPU-8911bb88-5fdb-401b-808e-4be44cc6acd5 |
| 2 | 3 | GPU-8c8a823b-24e8-e395-532d-b619ae2522d5 |

CUDA reports 50,925,535,232 bytes per selected GPU; nvidia-smi reports 49,140 MiB.
Use UUIDs, not ordinal `CUDA_VISIBLE_DEVICES=1,2,3`: this machine's CUDA ordinal
and NVML index enumeration did not agree during validation. The ordinal attempt
failed with invalid device ordinal; the UUID initialization check passed on all
three devices. Actual topology/P2P results are recorded in each manifest.

Contiguous decoder layers are placed by `min(layer * 3 // num_layers, 2)`.
All three GPUs own target layers. Embeddings and remaining wrapper modules are
on GPU 0, final norm/head on GPU 2, one drafter on GPU 0. AR and DFlash use the
same target map. No CPU weight offload, quantization, DP replicas, TP, EP,
prefix cache reuse, or continuous batching is used. MoE layer GEMM shapes retain
global B and verify width; requests are never distributed as independent replicas.

## Research design

| Model | Role | Primary measurements |
|---|---|---|
| Qwen3-8B | Full attention reference | AR-relative latency, target/draft KV slopes, phase costs |
| Qwen3.5-9B | GDN hybrid | Relative conditioning/hidden overhead as target KV shrinks; phase/layer scaling |
| Qwen3.5-35B-A3B | GDN + MoE | Same memory analysis plus routing union, load balance, expert/dispatch kernels and counter evidence |

Targets/drafters are exactly the pairs in the request. 35B target uses
`~/models/Qwen3.5-35B-A3B`. All six checkpoint shard/index/header/offset checks
passed before GPU execution. Payload SHA hashing is not performed: the inventory
is a structural integrity check. Loading additionally rejects missing/mismatched
weights and drafter extras. Revisions, shard sizes and header hashes are recorded.
The 35B drafter condition above 40,000 tokens is labelled outside its reported
training length. Native checkpoint RoPE is preserved; Qwen3 lengths beyond its
native window are separately labelled as extrapolation.

Candidate S={4096,8192,16384,32768,65536}, B={2,4,8,16,32}, output=256,
block=16. Initial representative points are (S,B)=(4k,2),(4k,8),(16k,2),
(16k,8),(4k,32). Memory/trace passes cover (4k,2),(4k,8),(16k,8).
MoE routing is a separate pass; Nsight Systems and NCU cover (4k,8).
Natural EOS uses (4k,8). Independent width={4,8} sweeps use (4k,8), and
output={1024,4096} uses (4k,2), not a Cartesian product. `--extended` adds
S=8k/32k/64k at B=2/8, and missing B=4/16/32 points at S=4k/16k.
The manifest enumerates scheduled points; status records distinguish executed,
OOM, preflight-rejected, unsupported, failed and skipped points.

A lower-bound per-GPU cache/context budget is checked after weights load and
before warmup. It includes preallocated attention KV, window-bounded drafter KV,
and selected context features. It deliberately excludes transient peaks and can
only reject an impossible point, not guarantee fit. Actual OOM is recorded.
A compute/bandwidth transition is assessed per phase × mixer × FFN × layer;
no single model-wide transition batch is imposed. Use intermediate B=4/16 and
width sweeps to refine observed boundaries. Capacity failure is not evidence of
a bandwidth limit. Cross-model absolute latency is not a causal GDN/MoE effect.

## Engine and correctness

`BatchEngine` uses row-specific prompt length, target position, draft history
length, accepted prefix, correction/bonus, EOS, cap and active state. Each
speculative target forward has shape [global B, block, hidden]. Block includes
one anchor, so widths 4/8/16 propose 3/7/15 draft tokens. The anchor was already
returned by prefill/previous commit and is never counted twice.

Prefill policy is **sequential, unchunked**, identical for AR/DFlash. Each row
has an independent temporary HF cache; its state is copied to the preallocated
batch cache. This policy avoids unvalidated GDN chunk boundaries and records
its staging/duplication costs. Batch target KV scatters in place at each row's
logical position, without full-history cat/copy. Drafter full-attention history
is preallocated; sliding layers retain a physical ring bounded by their window.
An attention scratch concatenation joins history and noise and is overhead.
Masks are built for the actual row positions and each layer's causal/window
semantics; full/sliding masks are never keyed only by shape.

HF's installed GDN cache crop restores conv history but not recurrent state.
This engine never relies on that for rejection. It snapshots fixed conv and
recurrent state before verification and captures each GDN mixer's input.
For partial acceptance it restores and replays only the GDN mixer at one token
per call, with global B retained and row states frozen after each accepted prefix.
Attention and FFN are not replayed. kept=0/all can restore/select the saved/final
state directly. The GDN replay/recording memory and its compute are explicit
DFlash overhead, not included in target verify time. AR with all rows active
has no transaction/replay. Finished static rows still enter actual kernels;
their output and logical state are frozen, and padding/unused input work is
recorded. No replacement request is inserted.

CPU tests use explicit Torch GDN fallback in FP32, not CUDA FLA. They cover
Qwen3/Qwen3.5 B1/B3, ragged lengths/acceptance, cacheless greedy equality,
full/sliding/causal masks, cache lifetime, EOS and observer on/off. Direct GDN
state+next-logit tests cover kept=[0,0,0], [1,2,3], [4,4,4], [0,2,4] with
atol/rtol 2e-5 for states and 3e-5 for next logits. Greedy token equality is exact.
Initial 18 CPU tests passed; see `record_arch_batch/cpu-tests.log`.

GPU queue gates each model on real BF16 B1/B2 AR/DFlash output equality,
row0 B1/B2 output and acceptance, natural EOS, and profiler on/off equality.
AR is a separate target-only subprocess, never drafter-loaded block=1 memory.
A failing gate skips that model's scientific sweep with evidence. CPU success
is not claimed as proof of checkpoint CUDA/FLA correctness.

## Workload and metrics

Frozen JSON stores 32 distinct LongBench rows, task mix (narrativeqa, qasper,
multifieldqa_en, hotpotqa), original IDs/indices/content hashes, composition
contributors, archive hash, tokenizer revision and exact token hash. Contexts
are composed from distinct source documents if necessary, and token-level
head/tail trimming makes every row exactly S without re-encoding drift. This
is a shape workload with a recorded Question/Answer template, not LongBench
accuracy evaluation. AR and DFlash read identical frozen IDs and seed=42.
Natural-EOS pass preserves completion times. Prompts are not B copies of one row.

TTFT is from prepared IDs to each row's first usable token. It includes that
row's wait behind sequential prefill. First drafter prefill occurs after those
first tokens, so is not automatically added to TTFT. First speculative commit
is recorded separately. TPOT=(last-first output timestamp)/(output count-1);
all tokens in one speculative commit share the commit availability time.
E2E tok/s includes every output token. `decode_tok_s` counts only outputs committed
after prefill; `batch_decode_throughput_all_output_tok_s` supplies the requested
all-output-token/decode-wall convention as a separate explicit denominator.
Do not silently exchange them. Per-request TPOT p50/p95 is separate from global
throughput. Acceptance is actual accepted drafts / actual proposed drafts;
commits are clipped at EOS/cap. Tail padding counts are token-work counts, not
measured wasted milliseconds or an isolated tail-free counterfactual.

## Profiling and aggregation

Perf uses one full-condition warmup and at least three measured repeats,
median/stdev/min/max; no observer/module hooks or per-layer synchronization.
Memory, trace, routing, Nsys and NCU are separate subprocess passes. Diagnostics
also compare outputs and acceptance to an uninstrumented full-condition warmup.
`profiler_perturbation.csv` compares matched diagnostic E2E against the independent
perf median; diagnostic latency never enters performance medians.

NVTX/record_function ranges nest phase → layer → module. Diagnostics carry
run/request/step/phase/parent/role/layer/module/device/stream/shape/backend.
Mixer kind comes from config/module, FFN classification independently inspects
actual experts. Same-device CUDA event times are inclusive and may include waits;
CPU exclusive/inclusive time is separately labelled. No elapsed_time crosses GPUs.
Chrome trace analysis connects kernel external/launch IDs to the deepest module
range and attributes each kernel once. Phase/layer heatmaps use that exclusive
attribution, not parent+child sums. Busy and memcpy interval unions, overlap,
global no-work gaps and unattributed kernels remain separate. Unknown gaps are
not automatically labelled DRAM stalls or structural waits; inspect Nsys launch,
transfer and dependency evidence. Three GPU busy sums are never wall time.

Memory snapshots count storage once across component groups per GPU: target
attention KV, GDN conv/recurrent, rollback buffers/verify inputs, draft KV,
selected hidden, concatenated feature and weights. Boundary resident snapshots
are not component peaks. Per-GPU allocator peaks and their (nonconcurrent) sum
are recorded separately. The memory pass records timestamped allocator history and reconstructs aggregate
allocated bytes. Equal cross-device timestamps produce explicit lower/upper bounds;
missing timestamps, ring wrap or inconsistent final counters produce NA. Perf does
not enable allocator history. Boundary snapshots remain distinct from exact peaks.
Transient bytes that are not captured at boundaries remain unattributed. Do not
sum independently measured component maxima to manufacture a total peak.

The installed NCU is `/opt/nvidia/nsight-compute/2024.1.1/ncu`. Separate kernel
replay collects SpeedOfLight, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis,
Occupancy, WarpStateStats and tensor roofline sections for the first 24 matching
GEMM/attention/GDN kernels per GPU in measured verify/AR decode NVTX ranges. Selection is
bounded and must not be presented as all-layer counter coverage. Replay serializes
kernels; cache-control=none and clock-control=none avoid explicit flush/clock
changes but do not eliminate replay perturbation. NCU reports/csv contain the
actual counters; absent metrics retain NA/reasons. GPU utilization is scheduling
telemetry only. MoE expert IDs, union and tokens/expert are measured separately;
union is not converted to measured weight DRAM bytes. Rejected input-token counts
are not rejection-rate × verify latency “wasted time”; router/expert GEMM FLOPs are formula estimates. A separate post-generation
first-step gathered-suffix expert microbenchmark records three CUDA-event samples
per MoE layer. Its smaller shape and warm cache do not measure in-situ waste.

## Commands and resume

```bash
cd /home/seoyounglee/dflash
PY=/home/seoyounglee/venvs/dflash-fla/bin/python
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=record_arch_batch/test_deps:. \
  "$PY" -m pytest tests/test_batch_engine.py tests/test_batch_profiling_v3.py -q

# Creates a persisted sparse plan, checks idle GPU UUIDs before every job,
# runs immediately when available, otherwise polls every 30 seconds.
"$PY" -m dflash.batch_queue start --root record_arch_batch/campaign3 --extended
"$PY" -m dflash.batch_queue status --root record_arch_batch/campaign3
"$PY" -m dflash.batch_report record_arch_batch/campaign3
```

Queue writes PID, worker log, per-job commands, statuses and model gates. It uses
cooperative file locks, not a cluster scheduler's atomic GPU allocation; other
programs may race unless they respect the same locks. GPU query failure waits
rather than assuming idle. Completed/OOM/failed jobs are preserved on resume.
A source hash or branch change blocks new jobs; create a new campaign root after
reviewed changes. Don't edit running engine code and silently mix results.
To retry a failed condition, use a new output directory with its recorded command.

Outputs: `runs/*/{manifest,preflight,status,summary}.json`, `raw.jsonl`,
`diagnostic-*/{events,memory,routing}.jsonl`, Chrome trace, Nsys `.nsys-rep`/SQLite,
NCU `.ncu-rep`/CSV; report status/AR comparison/phase/layer/operator/routing/memory/
perturbation CSV; per-trace operator/idle/transfer/gap CSV, three-GPU timeline and
phase-layer heatmap. Roofline CSV remains NA until matched NCU counters are
available; NCU's native report contains its measured roofline sections.

## Current limits and claims that require further work

The other branch's B=1 experiment is not rerun or imported. No main bridge
measurement exists in this campaign: schema/kernel/placement/revision matching
is required before comparing, and engines must retain separate result rows.
No main result with suspected GDN rollback bugs may serve as baseline.

This implementation is a static batch characterization of Transformers layer
sharding. Sequential prefill, GDN replay, explicit mask/scratch construction,
Python row bookkeeping and structural shard waits affect measured speedups.
It is not a serving performance claim. vLLM/SGLang, TP/EP and continuous batching
need separate correctness and performance validation. Automatic causal gap attribution, full operator-specific roofline normalization,
adaptive boundary selection and a main B=1 bridge are not yet completed. These limitations
must remain visible in analysis rather than becoming inferred measured results.

Development validation: Qwen3-8B B=2/S=128/output=16 completed three measured
DFlash repeats on the fixed three GPUs with identical outputs/acceptance. This
is a smoke result, not a research-grid speedup or an AR correctness comparison.
See `record_arch_batch/validation/qwen3-8b-dflash-b2-v4/`. Earlier failed
development attempts are preserved and excluded from the campaign.

Queue registration: `record_arch_batch/campaign3/plan.json` contains 198 jobs;
worker was restarted as PID 3735128 after the NVML query fix. Inspect `queue_status.json` for live status
and `report/status.csv` for executed versus queued coverage. Qwen3.5-9B B=2,
S=128, output=16 memory validation completed with profiler output/acceptance
equality and timestamp-reconstructed concurrent allocator peak 21,712,023,744
bytes. Its diagnostic latency is excluded from performance results. The 35B
real-checkpoint routing validation was started separately before this queue.

GPU queue probe verification: while the 35B validation occupied the selected
GPU set, the worker recorded `waiting_for_gpus` with the actual process PID
and per-GPU memory readings. The NVML probe uses physical indices 1/2/3 and
checks their UUIDs; CUDA subprocesses continue using those UUIDs. The original
pre-execution plan is preserved as `plan.before-nvml-query-fix.json`.
