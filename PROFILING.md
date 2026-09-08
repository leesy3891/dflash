# Context-Length Profiling

How to measure DFlash acceptance, latency and memory as a function of input
context length, and what the numbers mean.

## Code map

| File | Role |
| --- | --- |
| `dflash/cli.py` | Argument parsing; resolves `--model-preset` into a target/draft pair and routes `benchmark` to the right runner. |
| `dflash/benchmark.py` | Dataset registry, model loading, and the runners — `_run_context_length` drives the context-length sweep, the other three are the pre-existing dataset benchmarks. |
| `dflash/model.py` | The DFlash draft model and `dflash_generate`, the draft/verify loop; with `return_stats=True` it also emits acceptance, timing, and memory counters. |
| `dflash/context.py` | Builds LongBench-E prompts trimmed to hit an exact context length, using LongBench's official per-task templates. |
| `dflash/record.py` | Aggregates per-sample metrics into summaries and writes the `record/*.json` file. |
| `dflash/model_mlx.py` | The MLX (Apple Silicon) backend; not used by context-length profiling. |

## Input

```
dflash benchmark transformers --model-preset <name> --context-length <N> [options]
```

| Argument | Meaning |
| --- | --- |
| `--model-preset` | `qwen3-8b` or `qwen3.5-9b`; fills in `--model` and `--draft`. Use `--model`/`--draft` directly for anything else. |
| `--context-length` | Input tokens per prompt; every prompt is fitted to it. Required to enter this mode. |
| `--max-samples` | Prompts to run (default 32). |
| `--max-new-tokens` | Decode cap per prompt. Keep it at 256+ so per-token latency is not dominated by prefill. |
| `--reasoning` | `off`/`on`, or a model-specific level. **Set this deliberately** — see Caveats. |
| `--context-task` | Task to run. Default (and `all`): the LongBench-E English suite at or below 16k, `long` above it. Accepts a task name, a comma-separated list, or a group (`all-e`, `all-en`, `paper`, `long`). The paper's long-context tasks are `hotpotqa`, `qasper`, `gov_report`. |
| `--context-split` | `auto` (default), `e`, `full`. `auto` reads LongBench-E at or below 16k and the full split above it. |
| `--context-extend` | `auto` (default), `on`, `off`. Whether a task whose context is a sequence of units may borrow units from other documents of the same task to reach the target. `auto` means on above 16k. |
| `--context-dry-run` | Fit the prompts, print the feasibility table, exit. Loads the tokenizer only, no GPU. |
| `--hidden-states` | `full` (default) or `selective`. How the target's residual streams reach the drafter — see *Hidden states: how many layers are actually resident* below. |
| `--prefill-chunk` | Prefill the target in slices of this many tokens. Full-attention targets only; refused on a hybrid one — see *Chunked prefill*. |
| `--device-map` | Shard the target across the visible GPUs (`auto`, `balanced`); requires `accelerate`. Buys prefill headroom; makes timings pipeline-parallel. |
| `--max-memory` | Per-GPU weight budget for `--device-map`, e.g. `0=20GiB,1=32GiB`. Defaults to each card's *free* memory less 2 GiB, so a shared machine is not handed a card someone else is using. |
| `--rope-scaling` | `none` (default) or `yarn`. Widens RoPE on target *and* draft so a context past the trained window is interpolated. Qwen3-8B needs it at 64k. |
| `--no-baseline` | Skip the `block_size=1` run. Halves runtime, drops the speedup number. |
| `--profile-draft-memory` | Also measure the drafter's activation peak. Perturbs latency slightly; leave off when timing is the point. |
| `--record-dir` | Output directory (default `record`). |
| `--block-size` | Override the draft's block size (default: the checkpoint's, 16 for both presets). |

Prompts are drawn from LongBench-E, the length-balanced split of LongBench, read
straight out of the Hub archive (`datasets` will not run its loading script).
Each document is middle-truncated — head plus tail, as LongBench itself does —
and the trim is re-fitted until the *templated, tokenized* prompt hits
`--context-length`. Because decoding a token slice and re-encoding it is not
token-identical at the boundary, the fit is exact for almost every prompt and
accepted within ±16 tokens otherwise. Documents too short to reach the target
are skipped and reported. Sample selection is seeded, so a repeated run at the same context
length uses the same prompts.

## Reaching 32k and 64k

LongBench caps its own documents, and the cap is well under 64k. Measured with
the Qwen3-8B tokenizer over every English task, counting documents whose
`context` field alone clears each target (the templated prompt needs a few
hundred tokens more):

| Task | split | n | p50 | max | >=16k | >=32k | >=64k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narrativeqa | full | 200 | 31296 | 65301 | 139 | **85** | 0 |
| gov_report | full | 200 | 8902 | 52521 | 27 | 3 | 0 |
| gov_report | e | 300 | 7185 | 28575 | 32 | 0 | 0 |
| qmsum | full | 200 | 12970 | 30377 | 52 | 0 | 0 |
| musique | full | 200 | 16733 | 17824 | 151 | 0 | 0 |
| hotpotqa | full | 200 | 14982 | 17578 | 63 | 0 | 0 |
| 2wikimqa | e | 300 | 8160 | 17169 | 27 | 0 | 0 |
| multifieldqa_en | full | 150 | 7368 | 16446 | 1 | 0 | 0 |
| qasper | e | 224 | 5604 | 21879 | 14 | 0 | 0 |
| triviaqa | e | 300 | 8289 | 36186 | 44 | 1 | 0 |
| trec | e | 300 | 7974 | 17328 | 17 | 0 | 0 |
| samsum | e | 300 | 9062 | 18191 | 7 | 0 | 0 |
| passage_count | full | 200 | 15822 | 29538 | 91 | 0 | 0 |
| passage_retrieval_en | full | 200 | 12581 | 15344 | 0 | 0 | 0 |
| multi_news | e | 294 | 6728 | 41048 | 23 | 4 | 0 |
| lcc | e | 300 | 13178 | 57962 | 122 | 10 | 0 |
| repobench-p | e | 300 | 12715 | 40567 | 115 | 11 | 0 |

So, naturally: **32k is NarrativeQA and little else** (85 documents, then a
long tail of 3-11 from gov_report, multi_news, LCC and RepoBench-P), and **64k
is nothing at all** — the longest English document in LongBench is a 65301-token
NarrativeQA story, 235 tokens short of 64Ki.

Above 16k the harness therefore does two things, both on by default and both
recorded in the output file:

1. **Reads the full split rather than LongBench-E** (`--context-split`), because
   that is where the long documents are. NarrativeQA, MuSiQue and QMSum only
   exist there at all.
2. **Composes** (`--context-extend`), for the tasks whose context is a sequence
   of independent units: multi-document QA (`Passage N:` blocks), the synthetic
   retrieval tasks (`Paragraph N:` lines), and the few-shot tasks (one example
   per unit). Units from *other* documents of the same task are drawn in, the
   document's own units are scattered among them rather than left in a block at
   the head, numbered markers are renumbered to run 1..N, and the tail is
   trimmed so the prompt lands on the target. `passage_retrieval_en` also
   switches to a prompt that states the real paragraph count instead of
   LongBench's hardcoded "30".

`samples[].composed` records which prompts were built this way, and
`context_task_report` holds the per-task natural/composed/skipped breakdown.

Composition is a length stressor, not a scoring harness: this benchmark never
scores an answer, and a composed `passage_count` prompt has a different unique-
paragraph count from the one LongBench labelled. What it does preserve is the
shape of the task at the target length — the model still has to find evidence
scattered through 64k of same-genre distractors.

Check what a length will actually yield before spending a GPU on it:

```bash
dflash benchmark transformers --model-preset qwen3-8b \
    --context-length 65536 --max-samples 32 --context-dry-run
```

## Hidden states: how many layers are actually resident

DFlash conditions the drafter on a handful of the target's residual streams, but
the reference implementation keeps all of them alive to get there. In
`dflash_generate` the target is called with `output_hidden_states=block_size > 1`
and `extract_context_feature` then indexes the layers it wants out of the
returned tuple — so every layer is materialised and most are discarded:

| Preset | decoder layers | tuple returned | injected (`target_layer_ids`) | used |
| --- | ---: | ---: | --- | ---: |
| Qwen3-8B | 36 | 37 | `[1, 9, 17, 25, 33]` → 5 | 13.5% |
| Qwen3.5-9B | 32 | 33 | `[1, 5, 9, 13, 17, 21, 25, 29]` → 8 | 24.2% |

The tuple is `L + 1` tensors of `(1, S, d)` — the embedding output plus one per
layer. At `S = 65536` that is 18.5 GiB on Qwen3-8B where the drafter needs 2.5,
and on a 48 GB card it is the single reason a 64k run does not fit.

**`full` is the default** because it is what the reference implementation does
and what every record in the context sweep is measured with. Do not mix modes
within a sweep: `max_target_hidden_states_gb` counts `L + 1` layers under `full`
and `len(target_layer_ids)` under `selective`, so the same run reports an
eight-fold difference in that term for reasons that have nothing to do with
context length.

`--hidden-states selective` registers forward hooks on just the
wanted layers instead, so the rest are freed as the forward walks the stack.
The captured tensors are the same ones — `hidden_states[i + 1]` is the output of
layer `i` — and `build_target_layer_ids` never selects the last layer, the only
entry HuggingFace normalises before reporting. Verified on both presets: the
context feature is bit-identical, and greedy generation returns the same token
ids and the same acceptance lengths.

What changes is `max_target_hidden_states_gb`, which now reports what the run
actually paid rather than the full tuple:

| Preset | mode | `max_target_hidden_states_gb` at 1.2k |
| --- | --- | --- |
| qwen3-8b | full | 0.344 |
| qwen3-8b | selective | 0.046 |
| qwen3.5-9b | full | 0.306 |
| qwen3.5-9b | selective | 0.074 |

Records carry a `hidden_states` field so the two are never compared by accident.

One consequence worth knowing when reading `peak_site`: under `full` the peak
moves out of prefill. The `output` object holding the whole tuple stays
referenced through the first decode iteration, so the peak lands at
`decode: draft forward, decode token 0` rather than at the last prefill layer,
where `selective` puts it.

## Where the peak lands

`torch.cuda.max_memory_allocated` is a per-device maximum over the whole run.
Summing it across a sharded target adds maxima that never coexisted. Measured on
a 32k Qwen3.5-9B prefill split over two cards, sampling both devices at every
decoder layer:

```
per-device peak                     dev0 20.69 | dev1 24.51
sum of per-device peaks             45.20 GB     <- what used to be reported
max over time of (dev0 + dev1) live 32.91 GB     <- the real footprint
over-count                          12.29 GB  (27%)
```

Device 0 peaks while running its own layers and has fallen back to 14.29 GB long
before device 1 peaks. `PeakTracker` fixes this by cutting the run into short
intervals: within one interval only one device is doing work, so the sum of
per-interval device maxima is tight, and the maximum over intervals is reported.
On a single device the two agree exactly — a maximum over time is the maximum
over any partition of it — so **single-GPU records are unchanged**, and the
32768/65536 sharded figures published before this fix are over-counts.

Intervals are bounded at each drafter and target operation, and additionally at
every decoder layer during prefill. Layer-level bounds stay on through decode
only when the target is sharded, where the timings are pipeline-parallel and not
comparable to a single-GPU run anyway.

Each interval carries the operation name plus the decode token and drafter step
in flight, so `peak_site` says where the peak came from:

```
Peak memory (allocated / reserved)      25.36 / 27.09 GB
Peak hit at                             decode: draft forward, decode token 0, drafter step 0 (sample 1)
```

The tracker reads the allocator through `torch._C._cuda_memoryStats` rather than
`torch.cuda.memory_allocated`: the public wrapper rebuilds and flattens the whole
stats dict on every call, 81 us against 10 us measured, which at one read per
decoder layer would land in the per-token latency it is meant to measure. A/B
over 512 generated tokens: 32.700 ms/token with the tracker off, 32.610 with it
on.

## Chunked prefill, and where it is valid

`--prefill-chunk N` feeds the target its prompt in slices, so an attention
kernel's activation is bounded by the chunk instead of the context. It exists
because at long context the prefill activation, not the KV cache, is the largest
term: at 32k the Qwen3.5-9B **baseline** — which never touches the drafter —
already peaks at 38.8 GB while every tensor the record accounts for sums to
~27 GB. The missing ~14 GB is `torch_chunk_gated_delta_rule`, the pure-PyTorch
fallback for Qwen3.5's linear attention, which casts q/k/v/beta/g to float32 and
materialises them at full sequence length.

It is only valid for a **full-attention** target, and the harness refuses it
otherwise. Measured over a 13217-token prompt, comparing chunked against
one-shot prefill on the target's own next-token argmax at every position:

| Target | chunk 4096 | chunk 8192 |
| --- | --- | --- |
| Qwen3-8B (36 full-attention layers) | **100.000%** | **100.000%** |
| Qwen3.5-9B (24 of 32 layers recurrent) | 93.191% | 93.690% |

So for Qwen3-8B chunking is exact where it counts, and for Qwen3.5-9B it changes
what the model predicts at roughly one position in fourteen — the drafter is
conditioned on those residual streams, so a chunked run there would report an
acceptance length for a model that was never actually evaluated. Comparing raw
hidden states instead of predictions is misleading in both directions: Qwen3's
massive-activation dimensions carry values in the thousands, so a single
position can show a cosine similarity of 0.32 while every prediction still
agrees.

Qwen3.5-9B at 64k therefore needs the memory from somewhere else. Two ways out,
neither free:

* `--device-map balanced` shards the target over several GPUs (needs
  `pip install accelerate`; it only places modules and adds device hooks, so no
  kernel and no arithmetic changes). Verified on Qwen3.5-9B: sharded and
  single-GPU runs return **identical token ids and identical mean acceptance**
  (6.300 vs 6.300) at both `block_size` 1 and 16, so **acceptance length stays
  comparable** with the single-GPU rows — but the forward now crosses a device boundary mid-stack, so
  `mean_ttft_s`, `aggregate_time_per_output_token_s` and
  `decode_throughput_tok_s` are pipeline-parallel figures and must not be read
  against single-GPU ones. Records carry `device_map`, `target_device_map` and
  `num_devices` so a sharded run is never mistaken for a single-card one, and
  the memory counters sum over every device rather than reporting card 0.
* Installing the `kernels` package lets transformers fetch the `fla` Triton
  kernel instead of the fp32 fallback, which removes the problem outright — at
  the cost of running 64k on a different linear-attention kernel from the rest
  of the sweep.

## Output

One JSON file per run at `record/<model>_<context-length>_<date>.json`, e.g.
`record/qwen3.5-9b_4096_20260906-020610.json`.

```jsonc
{
  "git_commit": "07ebd93",
  "model": "Qwen/Qwen3.5-9B", "draft": "z-lab/Qwen3.5-9B-DFlash",
  "context_length": 4096, "context_tasks": ["gov_report", ...],
  "context_split": "auto", "context_extend": false,
  "context_task_report": { "gov_report": {"natural": 3, "composed": 0, "skipped": 1, ...}, ... },
  "num_composed_samples": 0, "hidden_states": "selective",
  "block_size": 16, "gamma": 15,
  "max_new_tokens": 256, "temperature": 0.0, "reasoning": "off",
  "device": "NVIDIA RTX A6000", "torch_version": "2.13.0+cu129",

  "summary": {
    "dflash":   { /* all metrics below */ },
    "baseline": { /* latency, tokens and memory only */ }
  },
  "decoding_speedup": 2.43,
  "samples": [ { "index": 0, "task": "gov_report",
                 "split": "e", "composed": false,
                 "dflash": {...}, "baseline": {...} }, ... ]
}
```

`summary.dflash` and `summary.baseline` share these keys:

| Key | Meaning |
| --- | --- |
| `total_input_tokens`, `mean_input_tokens` | Input tokens. The fit lands on `context_length` exactly for almost every prompt and is guaranteed within ±16; `samples[].fitted_input_tokens` records what each one actually got. |
| `total_output_tokens`, `mean_output_tokens` | Generated tokens. |
| `total_latency_s`, `mean_latency_s` | End-to-end per prompt, prefill through last token. |
| `mean_ttft_s`, `p50_ttft_s`, `p95_ttft_s` | Time to first token — prefill plus the first sample, excluding tokenization. |
| `aggregate_time_per_output_token_s` | Decode time over decode tokens, pooled across samples. The headline per-token latency. |
| `mean_time_per_output_token_s` | Same quantity averaged per sample rather than pooled. |
| `decode_throughput_tok_s` | Reciprocal of the aggregate figure. |
| `peak_memory_gb` | Largest **simultaneous** allocation across devices — see *Where the peak lands* below. On one GPU this is exactly `torch.cuda.max_memory_allocated`. |
| `peak_site` | Which operation set that peak: `operation`, `decode_token`, `draft_step`, `layer`, the `sample` it came from, and `per_device_gb`. |
| `peak_memory_sum_device_maxima_gb` | The naive figure — per-device maxima summed regardless of whether they coexisted. Equal to `peak_memory_gb` on one device; larger when sharded, and the gap is pure over-count. |
| `peak_memory_reserved_gb` | `torch.cuda.max_memory_reserved` — what the caching allocator holds. This is the number to compare against `nvidia-smi`, which additionally includes the CUDA context (a few hundred MB). |
| `max_target_cache_gb` | Largest target KV cache observed. |

`summary.dflash` adds the drafter-only metrics. The baseline never calls the
drafter, so it carries `mean_decode_steps` instead and omits these entirely
rather than reporting zeros:

| Key | Meaning |
| --- | --- |
| `acceptance_rate` | Accepted tokens over proposed tokens. Denominator is `gamma` per step. |
| `mean_acceptance_length` | Tokens committed per verify step, bonus token included. This is the figure the DFlash paper reports. |
| `acceptance_length_histogram` | Distribution over 0..`block_size` of tokens committed per step. |
| `gamma` | Tokens proposed per full block, `block_size - 1`. |
| `total_full_gamma_proposals` | Steps where the drafter proposed a full `gamma` tokens. |
| `total_draft_calls`, `total_verify_steps` | Draft forwards, and verify steps overall. |
| `draft_weight_gb` | Draft parameters and buffers. |
| `max_draft_cache_gb` | Largest draft KV cache observed — this is what grows with context. |
| `max_draft_activation_gb` | Draft activation peak, only under `--profile-draft-memory`; otherwise `null`. |
| `max_target_hidden_states_gb` | Target hidden states DFlash forces the target to emit (`output_hidden_states`) so its context feature can be injected into every draft layer. Usually the **largest** overhead term. |
| `max_context_feature_gb` | The concatenated per-layer target feature actually injected into the drafter. |
| `draft_overhead_gb` | Sum of the five terms above — the real memory cost of running the drafter. |
| `draft_forward_s`, `context_feature_s` | GPU time in the draft forward, and in building the injected context feature. Measured with CUDA events, so async work is attributed correctly. |
| `drafter_latency_s`, `drafter_share_of_decode` | The two above combined, absolute and as a fraction of decode time. |
| `target_forward_s`, `target_share_of_decode` | Same for the target's verify forwards. |

Per-sample entries under `samples` carry the same fields for one prompt, plus
`task` and the per-step `accepted_lengths` / `acceptance_lengths` arrays, so
results can be sliced by task or by step position after the fact.

## How the memory numbers are computed

Four primitives, all counting **CUDA storages deduplicated by `data_ptr`**, so a
tensor and its views are never counted twice:

| Helper | `model.py` | Used for |
| --- | --- | --- |
| `module_bytes(m)` | :247 | Draft weights — sums `m.parameters()` + `m.buffers()`. |
| `_cache_bytes(cache)` | :157 | Draft / target KV cache — walks `cache.layers`, so it also picks up the conv and recurrent state of hybrid targets. |
| `_tensor_bytes(ts)` | :184 | Target hidden states and the injected context feature. |
| `torch.cuda.max_memory_allocated` / `max_memory_reserved` | :500-501 | Process peak, live tensors vs allocator pool. |

Write `S` for the fitted context length, `p` for bytes per element (2 for
bfloat16), and for the target `L_t` layers / `d` hidden / `H_kv` KV heads /
`d_head`; for the draft `L_d` layers and `n_inj = len(target_layer_ids)` injected
layers. Verified against Qwen3-8B at `S = 8192` (`L_t=36, L_d=5, d=4096, H_kv=8,
d_head=128, n_inj=5`, draft `P = 1.049e9` params):

| Metric | Where | Formula | Predicted | Measured |
| --- | --- | --- | --- | --- |
| `draft_weight_gb` | `benchmark.py` calls `module_bytes(draft)` | `P · p` | 1.953 GiB | 1.95 |
| `max_target_cache_gb` | :504 | `2 · L_t · S · H_kv · d_head · p` | 1.160 GiB | 1.16 |
| `max_draft_cache_gb` | :503 | `2 · L_d · S · H_kv · d_head · p` | 0.161 GiB | 0.16 |
| `max_target_hidden_states_gb` | :336, :461 | `(L_t + 1) · S · d · p` | 2.313 GiB | 2.31 |
| `max_context_feature_gb` | :343, :468 | `S · (n_inj · d) · p` | 0.313 GiB | 0.31 |

The two KV-cache rows use the running sequence length (`S` plus tokens generated
so far), and a sliding-window draft layer caps its own contribution at the window
rather than `S` — which is why the Qwen3.5-9B draft cache stays nearly flat as
context grows while the Qwen3-8B one scales linearly.

`max_target_hidden_states_gb` is the `L_t + 1` hidden states the target must emit
(`output_hidden_states=True`) purely so DFlash can build its injected feature,
and `max_context_feature_gb` is that feature. Both scale with `S` and neither is
paid by the baseline, so both are drafter overhead.

**Draft activation** (`--profile-draft-memory` only, :374-398) is measured around
the draft forward rather than derived:

```
activation = max over draft calls of [ peak_during_call − max(allocated_before, allocated_after) ]
```

Subtracting `allocated_after` matters: the *first* draft call populates the
entire draft KV cache, which is persistent, not activation. Without the
subtraction that one step reports 325.5 MB at 8k — 162.5 MB of cache plus
163 MB of real activation — while every later step reports 16.7 MB. Netting out
what the call leaves behind yields 163 MB and stops `draft_overhead_gb` from
counting the KV cache twice.

**Total** (`record.py:170`), each term a max over samples:

```
draft_overhead = draft_weight + draft_cache + target_hidden_states
               + context_feature + draft_activation
```

At 8k on Qwen3-8B: `1.95 + 0.16 + 2.31 + 0.31 + 0.16 = 4.90 GB`. It is an upper
bound, not a simultaneous peak — the terms do not all reach their maximum at the
same instant.

**Peak allocated vs reserved** (:500-501). `max_memory_allocated` counts live
tensors; `max_memory_reserved` counts the caching allocator's pool, which is
what `nvidia-smi` sees, plus a few hundred MB of CUDA context on top:

```
nvidia-smi  ≈  peak_memory_reserved_gb  +  CUDA context
22.3 GiB    ≈  22.01 GB                 +  ~0.3 GB
```

Both are reset per sample, so they are per-request peaks, not run-wide ones.

## Running

Both recipes below were verified on an RTX A6000. Weights land in `HF_HOME`.

**Qwen3-8B** (target `Qwen/Qwen3-8B`, draft `z-lab/Qwen3-8B-DFlash-b16`):

```bash
CUDA_VISIBLE_DEVICES=0 dflash benchmark transformers \
    --model-preset qwen3-8b --context-length 4096 \
    --max-samples 32 --max-new-tokens 512 --reasoning off \
    --profile-draft-memory
```

**Qwen3.5-9B** (target `Qwen/Qwen3.5-9B`, draft `z-lab/Qwen3.5-9B-DFlash`):

```bash
CUDA_VISIBLE_DEVICES=1 dflash benchmark transformers \
    --model-preset qwen3.5-9b --context-length 4096 \
    --max-samples 32 --max-new-tokens 512 --reasoning off \
    --profile-draft-memory
```

Repeat with `--context-length 8192`, `16384`, `32768` and `65536` for the
sweep. Runs are independent, so the two models can occupy different GPUs at the
same time. Budget roughly 20-25 minutes per (model, context length) at 4k-16k
with 32 samples, 512 new tokens and the baseline enabled; 32k is about twice
that and 64k about four times.

To run a full sweep unattended and keep the logs:

```bash
mkdir -p logs
for L in 4096 8192 16384 32768 65536; do
    CUDA_VISIBLE_DEVICES=0 dflash benchmark transformers \
        --model-preset qwen3-8b --context-length $L \
        --max-samples 32 --max-new-tokens 512 --reasoning off \
        > logs/qwen3-8b_$L.log 2>&1
done
```

### What each length needs

Peak allocated memory, 32 samples, `--profile-draft-memory` off, at the default
`--hidden-states full`. 4k-16k are measured. 32k and 64k are projected from the
16k records plus the `L + 1` hidden-state term for that length, and are being
re-measured by `queue/run_full_sweep.sh`.

| Context | qwen3-8b | qwen3.5-9b | Fits a 48 GB A6000 |
| --- | --- | --- | --- |
| 4k | 19 GB | 23 GB | yes |
| 8k | 21 GB | 26 GB | yes |
| 16k | 25 GB | 33 GB | yes |
| 32k | ~34 GB | ~47 GB | 8b yes; 9b marginal, shard it |
| 64k | ~51 GB | ~67 GB | no — two cards for both |

The 32k/64k rows are dominated by the hidden-state tuple: 9.25 / 18.5 GiB on
qwen3-8b and 8.25 / 16.5 GiB on qwen3.5-9b. `--hidden-states selective` removes
roughly seven eighths of that and puts every row back on one card, at the cost of
producing a record that is not comparable with the rest of the sweep.

**Qwen3-8B cannot honestly be run at 64k.** Its
`max_position_embeddings` is 40960, and the DFlash draft checkpoint inherits the
same value. Nothing raises an error — RoPE happily extrapolates — but positions
past 40960 are outside what either model was trained on, so both the target's
output and the drafter's agreement with it are extrapolation artifacts rather
than a measurement of DFlash at 64k. Running it needs YaRN on the target *and*
the draft, which is a different experiment. Qwen3.5-9B has
`max_position_embeddings = 262144` and is unaffected.

## Checking progress

A run prints a `tqdm` bar to stderr, so with the command in the foreground
progress is already visible. For a backgrounded or redirected run:

```bash
# Latest progress line (tqdm uses \r, so translate it to newlines)
tr '\r' '\n' < logs/qwen3-8b_4096.log | grep 'ctx=' | tail -1

# Follow it live
tail -f logs/qwen3-8b_4096.log

# GPU utilisation and memory, refreshing every 2s
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 2

# Records written so far
ls -lt record/ | head

# Headline numbers from a finished record
python -c "
import json,sys; d=json.load(open(sys.argv[1])); s=d['summary']['dflash']
print(d['model_name'], d['context_length'],
      'accept_len=%.2f' % s['mean_acceptance_length'],
      'rate=%.3f' % s['acceptance_rate'],
      'tpot=%.2fms' % (s['aggregate_time_per_output_token_s']*1000),
      'speedup=%.2fx' % d.get('decoding_speedup', float('nan')))
" record/qwen3-8b_4096_*.json
```

## Caveats

**Decode-side numbers do not cross the 16k/32k boundary.** Mean output length
collapses there — 218 tokens at 16k to 12 at 32k for Qwen3-8B, 246 to 24 for
Qwen3.5-9B — and it is the task mix, not the models. Above 16k the only tasks
that reach the target are short-answer ones (HotpotQA, MuSiQue, PassageCount,
PassageRetrieval, TREC, TriviaQA, SAMSum; `max_gen` 32-128), while 4k-16k still
had GovReport, MultiNews and Qasper at `max_gen` 512. With a dozen decode steps
per sample, `aggregate_time_per_output_token_s`, `decode_throughput_tok_s` and
`decoding_speedup` are dominated by the first few steps and are not comparable
with the shorter lengths. `mean_acceptance_length`, `acceptance_rate` and every
memory figure are unaffected and do compare.

The `record/narrativeqa/` runs are the one place this does not bite at 32k:
NarrativeQA is the only task with enough documents past 32k to fill a run
naturally, so those records are 32 real documents with no composition.

**`--reasoning` changes the result.** Both drafts here are non-thinking
checkpoints, and Qwen3-8B defaults to thinking on. Leaving it unset also changes
how much the model generates. Set it explicitly; the value is recorded in the
output file.

**`acceptance_rate` has `gamma` in its denominator.** DFlash proposes all 15
tokens every step regardless of how many survive, so even a healthy
`mean_acceptance_length` of 5 is only a 0.27 acceptance rate. Compare
`mean_acceptance_length` against the paper, and use `acceptance_rate` for
relative comparisons.

**Drafter memory is reported in parts, not as one number.** The CUDA allocator
cannot attribute a peak to one of two models sharing a device, and the draft
weights stay resident during the baseline run as well — so the DFlash-minus-
baseline peak delta understates the cost. Use `draft_overhead_gb`, which adds up
weights, KV cache, activations, and the two target-KV-injection terms.

**The injection terms dominate, and they are why the process is bigger than
weights + KV + activations.** To inject the target's context into each draft
layer, DFlash runs the target with `output_hidden_states=True` and concatenates
the selected layers. At 8k on Qwen3-8B that is 2.31 GB of hidden states plus
0.31 GB of context feature — more than the 1.95 GB of draft weights. The
baseline pays neither, which is why its allocated peak is ~2 GB lower.

**`nvidia-smi` will always read higher than `peak_memory_gb`.** Allocated counts
live tensors only. Compare against `peak_memory_reserved_gb` instead, and expect
`nvidia-smi` to sit a few hundred MB above even that for the CUDA context. At 8k
on Qwen3-8B: 21.29 GB allocated, 22.01 GB reserved, ~22.3 GiB in `nvidia-smi`.

**The two presets behave very differently, and that is real.** Measured on
`gov_report`, mean acceptance length against the paper's Table 4 Base column
(which is for a Qwen3.5-27B drafter, not either preset here):

| Context | Qwen3-8B | Qwen3.5-9B | Paper (27B, Base) | Paper (27B, Long) |
| --- | --- | --- | --- | --- |
| 1K | 2.87 | 7.39 | 4.53 | 4.53 |
| 4K | 2.46 | 8.46 | 3.93 | 4.25 |
| 8K | 2.13 | 8.43 | 3.32 | 4.04 |
| 16K | — | 5.86 | 2.67 | 3.81 |

Qwen3.5-9B lands *above* the paper's base drafter and holds flat through 8k;
Qwen3-8B lands well below it. So a low number from `qwen3-8b` is a property of
the `Qwen3-8B-DFlash-b16` checkpoint, not of this harness. Two things were ruled
out directly: switching the source dataset changed nothing, and head-only versus
middle truncation changed nothing (2.11 vs 2.13 at 8k). The likely mechanism is
attention shape — all five Qwen3-8B draft layers are full-attention, so draft
cost and drift both grow with context, while the Qwen3.5-9B draft keeps five of
six layers on a 4096 sliding window.
