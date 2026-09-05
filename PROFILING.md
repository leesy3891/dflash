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
| `--context-task` | Comma-separated LongBench-E tasks, or `all`. Default: `gov_report,multi_news,qasper,multifieldqa_en,lcc,repobench-p`. |
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

## Output

One JSON file per run at `record/<model>_<context-length>_<date>.json`, e.g.
`record/qwen3.5-9b_4096_20260906-020610.json`.

```jsonc
{
  "git_commit": "07ebd93",
  "model": "Qwen/Qwen3.5-9B", "draft": "z-lab/Qwen3.5-9B-DFlash",
  "context_length": 4096, "context_tasks": ["gov_report", ...],
  "block_size": 16, "gamma": 15,
  "max_new_tokens": 256, "temperature": 0.0, "reasoning": "off",
  "device": "NVIDIA RTX A6000", "torch_version": "2.13.0+cu129",

  "summary": {
    "dflash":   { /* all metrics below */ },
    "baseline": { /* latency, tokens and memory only */ }
  },
  "decoding_speedup": 2.43,
  "samples": [ { "index": 0, "task": "gov_report",
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
| `peak_memory_gb` | `torch.cuda.max_memory_allocated`, reset before each sample. |
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

Per-sample entries under `samples` carry the same fields for one prompt, plus
`task` and the per-step `accepted_lengths` / `acceptance_lengths` arrays, so
results can be sliced by task or by step position after the fact.

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

Repeat with `--context-length 8192` and `16384` for the sweep. Runs are
independent, so the two models can occupy different GPUs at the same time.
Budget roughly 20-25 minutes per (model, context length) at 32 samples and 512
new tokens with the baseline enabled.

To run a full sweep unattended and keep the logs:

```bash
mkdir -p logs
for L in 4096 8192 16384; do
    CUDA_VISIBLE_DEVICES=0 dflash benchmark transformers \
        --model-preset qwen3-8b --context-length $L \
        --max-samples 32 --max-new-tokens 512 --reasoning off \
        > logs/qwen3-8b_$L.log 2>&1
done
```

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
baseline peak delta covers cache and activations only, not weights. Read
`draft_weight_gb`, `max_draft_cache_gb` and `max_draft_activation_gb` together.

**The two presets behave very differently, and that is real.** On identical
LongBench prompts at 4k, Qwen3.5-9B reaches ~9.2 mean acceptance length and
~2.4x decoding speedup, while Qwen3-8B reaches ~2.5 and ~1.4x. The Qwen3-8B
draft also degrades with context (2.87 → 2.46 → 2.13 mean acceptance length at
1k → 4k → 8k on `gov_report`) and turns into a slowdown at 16k, because all five
of its draft layers are full-attention; the Qwen3.5-9B draft keeps five of six
layers on a 4096 sliding window, so its draft cost stays flat as context grows.
