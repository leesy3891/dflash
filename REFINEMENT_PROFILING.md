# Local causal refinement for DFlash: implementation and profiling

Training-free, block-local refinement of DFlash draft proposals, and a separate
latency/memory profile of what it costs. Qwen3-8B + `z-lab/Qwen3-8B-DFlash-b16`,
greedy, `--hidden-states selective`, same LongBench prompts and seed as
`record_selective` (PROFILING2.md §7). Records are in `record_refine/`, tables
come from `python -m dflash.refine_report record_refine`. §7 repeats the study
on Qwen3.5-9B, where the target's gated-delta-rule rollback bug makes the
acceptance comparison harder to read.

## 1. Implementation changes

| file | change |
|---|---|
| `dflash/refine.py` (new) | `RefineConfig`, `local_causal_mask`, `local_refine()`: the four refinement stages |
| `dflash/model.py` | `_REFINE_CAPTURE` handle; the drafter's last layer stores Q, block K/V, RoPE cos/sin, its input hidden and the pre-norm output during the one forward DFlash already runs; `dflash_generate(..., local_refine=)` runs the refinement after the draft logits; per-stage CUDA-event timers; a `decode: local refine` phase for PeakTracker/PhaseMemory; forward hooks that count drafter and full-vocabulary LM-head forwards per phase; rerank diagnostics |
| `dflash/record.py` | per-sample refine fields, per-call aggregates (`mean_refine_*_ms`, `refine_peak_transient_bytes`, `refine_share_of_decode`), module-call totals, comparison printout against plain DFlash |
| `dflash/cli.py` | `--local-refine`, `--refine-top-k 16`, `--refine-window {1,2,4,...,full}` and `--refine-alpha` (both take comma lists in `benchmark`, crossed) |
| `dflash/benchmark.py` | baseline, DFlash and every refine window run in one process over the same prompts; per-sample `output_token_ids` kept for exactness checks; `decoding_speedups` per configuration |
| `dflash/refine_report.py` (new) | the tables in this document |
| `tests/test_local_refine.py` (new) | tiny fp32 CPU models: exact greedy output for w=1,2,4,full; alpha=0 keeps argmax and re-projected K/V reproduce the captured ones; position 1 never changes; later logits never change earlier choices; mask shape |
| `queue/run_refine_sweep.sh`, `queue/run_refine_alpha_sweep.sh`, `queue/run_refine_best_alpha.sh`, `queue/run_refine_9b.sh` (new) | the sweeps below; the first two take `PRESET` and `RECORD_DIR` |

The plain DFlash path is unchanged when `--local-refine` is off: the capture
handle is `None`, so the only added work is one global lookup per draft layer.
The batched engine (`dflash/batch.py`) never sets the handle, and its CPU tests
still pass.

### Cache / rollback invariant

`past_key_values_draft`, `_crop_to` and every crop site are untouched. The
refinement reads the captured tensors and allocates only block-local
temporaries (at most `(L-1) x K x H` rows for L = 16, K = 16, H = 4096), which
are freed when `local_refine` returns; the capture dict is cleared at the same
point. Nothing is written to either KV cache, so the draft cache length after
each step is the same as in plain DFlash, and the target verify decides which
tokens are kept exactly as before.

## 2. Refinement computation path

```
draft forward (1x, captures last-layer Q, block K/V, h_in, h_prenorm)
  -> full LM head (1x, T=L-1 positions x V)
  -> [refine_topk_soft_embedding]  topk(logits, K); p = softmax(topk values)
                                   e_soft[t] = sum_k p_t,k * E[idx_t,k]      (target input embedding)
                                   delta_e[t] = (e_soft[t] - E[MASK]) * input_embedding_scale
  -> [refine_kv_projection]        h'[s] = h_in[s] + alpha * delta_e[s], s = 1..L-2
                                   K' = RoPE(k_norm(k_proj(input_norm(h')))), V' = v_proj(input_norm(h'))
                                   position 0 (verified anchor) reuses the captured K/V
  -> [refine_attention]            queries: captured Q at positions 1..L-1
                                   keys:    block positions 0..L-2, strictly earlier, within `window`
                                   one SDPA over a batch of 2: (Q, K', V') and (Q, K, V)
                                   delta_attn = o_proj(A' - A)
  -> [refine_candidate_rerank]     d = norm(h_prenorm + delta_attn) - norm(h_prenorm)
                                   delta_logit = W_lm[idx] . d * output_multiplier   (T x K rows only)
                                   token = idx[argmax(topk values + delta_logit)]
  -> target verify (unchanged)
```

Design choices and approximations:

- **Q is reused**, not recomputed: the layer's Q depends on each position's own
  input, and the patch is meant to change what a position *sees* from its
  predecessors, not what it asks for.
- **Delta of attention, not a replacement.** The captured layer output already
  contains the full attention over context and block. Only the local part is
  recomputed, with and without the patch, and the difference is added. Context
  keys and the draft KV cache are never attended again.
- **Strictly earlier positions.** `window=w` lets block position t see
  positions t-w..t-1. Position 1's only predecessor is the unpatched anchor, so
  position 1 is never changed. With `w=1` each softmax has a single key, so the
  delta is `o_proj(V'[t-1] - V[t-1])` and Q does not affect the result.
- **First-order through the last layer.** The perturbation reaches the output
  via the residual path; the last layer's MLP is not re-run. The final RMSNorm
  is applied exactly.
- **No full-vocabulary product.** The rerank multiplies against `T x K` LM-head
  rows gathered by index (`F.embedding` on `lm_head.weight`); the LM-head module
  itself is never called.
- Greedy only (`temperature > 0` raises), `DFlashDraftModel` only (DFlash2 has
  its own candidate selector).

## 3. Experiment

Four idle RTX A6000 (49140 MiB, 2 MiB used before start), torch 2.13.0+cu129,
transformers 5.16.1. `queue/run_refine_sweep.sh 0 1 2` ran one context length
per card in parallel:

```
python -m dflash.cli benchmark transformers \
  --model-preset qwen3-8b --context-length {4096|16384|32768} \
  --max-samples 32 --max-new-tokens 512 --reasoning off \
  --hidden-states selective --profile-draft-memory \
  --local-refine --refine-top-k 16 --refine-window 1,2,4 --refine-alpha 1.0 \
  --record-dir record_refine
```

Same prompts, sample count, seed (`torch.manual_seed(0)`), block size 16,
`max_new_tokens` and reasoning setting as `record_selective`. Within one record,
every sample is decoded by baseline, DFlash and the three windows back to back.
Plain DFlash reproduces the `record_selective` acceptance exactly at every
length: 3.1101 / 3861 accepted / 27009 proposed at 4k, 2.505 at 16k and 1.480
at 32k.

**Timing control.** The 4k point was run a second time alone on GPU 0
(`record_refine/timing_control/`, log `logs/refine/qwen3-8b_4096_isolated.log`),
because the parallel 4k run showed a TPOT spread that refinement cannot cause
(§4.4). Acceptance, proposals and outputs in the rerun match the parallel run
exactly. The 4k latency rows below come from this isolated run; the 16k and
32k rows come from the parallel sweep, where target verify per step matched
within 0.2% across configurations.

## 4. Results

### 4.1 No second drafter forward, no second full-vocabulary LM-head forward

**In code.** `local_refine` (`dflash/refine.py`) calls, in order: `torch.topk`,
`F.embedding` on the input embedding table, `input_layernorm`, `k_proj`,
`k_norm`, `v_proj`, one `F.scaled_dot_product_attention`, `o_proj`, the final
`model.norm` twice on `(1, L-1, H)`, and `F.embedding` on `lm_head.weight` to
gather `T x K` rows. It never calls `model(...)`, a decoder layer, `q_proj`,
the MLP, or the LM-head module, and it takes no KV cache argument.

**In the profile.** Forward hooks count every call of the drafter module and
every call of the target's output head that returns a full-vocabulary tensor
(`shape[-1] == 151936`), tagged with the phase it ran in. The refinement runs
inside its own phase, `decode: local refine`.

| ctx | config | draft calls | drafter forwards (all phases) | full-vocab head: draft logits | full-vocab head: target verify | drafter fwd in refine | full-vocab head in refine | phases seen |
|---|---|---|---|---|---|---|---|---|
| 4k | DFlash | 1816 | 1816 | 1816 | 1816 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 4k | +refine w=1 | 1812 | 1812 | 1812 | 1812 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 4k | +refine w=2 | 1817 | 1817 | 1817 | 1817 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 4k | +refine w=4 | 1815 | 1815 | 1815 | 1815 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 16k | DFlash | 2769 | 2769 | 2769 | 2769 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 16k | +refine w=1 | 2766 | 2766 | 2766 | 2766 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 16k | +refine w=2 | 2756 | 2756 | 2756 | 2756 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 16k | +refine w=4 | 2763 | 2763 | 2763 | 2763 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 32k | DFlash | 244 | 244 | 244 | 244 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 32k | +refine w=1 | 245 | 245 | 245 | 245 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 32k | +refine w=2 | 244 | 244 | 244 | 244 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |
| 32k | +refine w=4 | 244 | 244 | 244 | 244 | 0 | 0 | decode: draft forward, decode: draft logits, decode: first draft forward, decode: target verify, prefill: target forward |

In every configuration the drafter ran exactly once per draft call and the
full-vocabulary head exactly once per draft call (draft logits) plus once per
verify. The counts inside `decode: local refine` are 0 in all 9 refine
configurations, and that phase never appears among the phases in which a
drafter or full-vocabulary head forward was seen.

### 4.2 Exactness

| ctx | config | outputs == DFlash | outputs == baseline | output tokens |
|---|---|---|---|---|
| 4k | DFlash | 32/32 | 19/32 | 5680 |
| 4k | +refine w=1 | 32/32 | 19/32 | 5680 |
| 4k | +refine w=2 | 32/32 | 19/32 | 5680 |
| 4k | +refine w=4 | 32/32 | 19/32 | 5680 |
| 16k | DFlash | 32/32 | 21/32 | 6968 |
| 16k | +refine w=1 | 32/32 | 21/32 | 6968 |
| 16k | +refine w=2 | 32/32 | 21/32 | 6968 |
| 16k | +refine w=4 | 32/32 | 21/32 | 6968 |
| 32k | DFlash | 32/32 | 31/32 | 393 |
| 32k | +refine w=1 | 32/32 | 31/32 | 393 |
| 32k | +refine w=2 | 32/32 | 31/32 | 393 |
| 32k | +refine w=4 | 32/32 | 31/32 | 393 |

Refined output equals plain DFlash output on all 96 sample x window pairs per
context length. The `== baseline` column is the pre-existing numerical
divergence between 16-token verify and 1-token decode in bf16 (PROFILING2.md
§5 already shows 5680 vs 5640 output tokens at 4k). Refinement does not change
it: the same samples diverge with and without it.

### 4.3 Stage latency

4k (isolated run):

| ctx | config | calls | soft embedding ms | K/V projection ms | attention ms | candidate rerank ms | total ms / draft call | % of DFlash steady draft fwd | % of decode |
|---|---|---|---|---|---|---|---|---|---|
| 4k | +refine w=1 | 1812 | 0.094 | 0.113 | 0.140 | 0.083 | 0.438 | 7.2 | 0.80 |
| 4k | +refine w=2 | 1817 | 0.094 | 0.113 | 0.140 | 0.083 | 0.438 | 7.2 | 0.80 |
| 4k | +refine w=4 | 1815 | 0.095 | 0.113 | 0.141 | 0.085 | 0.442 | 7.3 | 0.81 |

16k and 32k (parallel sweep):

| ctx | config | calls | soft embedding ms | K/V projection ms | attention ms | candidate rerank ms | total ms / draft call | % of DFlash steady draft fwd | % of decode |
|---|---|---|---|---|---|---|---|---|---|
| 16k | +refine w=1 | 2766 | 0.094 | 0.113 | 0.176 | 0.083 | 0.475 | 6.7 | 0.44 |
| 16k | +refine w=2 | 2756 | 0.094 | 0.113 | 0.141 | 0.083 | 0.439 | 6.2 | 0.41 |
| 16k | +refine w=4 | 2763 | 0.094 | 0.114 | 0.141 | 0.083 | 0.439 | 6.2 | 0.41 |
| 32k | +refine w=1 | 245 | 0.097 | 0.117 | 1.786 | 0.087 | 2.094 | 22.5 | 1.09 |
| 32k | +refine w=2 | 244 | 0.097 | 0.117 | 0.145 | 0.086 | 0.450 | 4.8 | 0.24 |
| 32k | +refine w=4 | 244 | 0.097 | 0.117 | 0.145 | 0.086 | 0.451 | 4.8 | 0.24 |

The refinement costs 0.44 ms per draft call, independent of context length and
window. Nothing it touches grows with S. The four stages are roughly 0.09 /
0.11 / 0.14 / 0.08 ms. At this size the cost is kernel-launch overhead, about
30 small kernels, rather than arithmetic. The window changes only the boolean
mask of a 15x15 attention, so w=1, 2 and 4 cost the same.

One outlier: the 32k w=1 attention mean of 1.79 ms comes from a single call of
405 ms in sample 2. Without it the mean is 0.15 ms, matching w=2 and w=4. The
stall is a one-off host or driver stall, not a property of the stage. Likewise
the 32k w=4 steady draft forward mean of 14.25 ms comes from one 152 ms draft
forward in sample 1, which is outside the refinement.

### 4.4 TPOT, decode latency and speedup

4k, isolated:

| ctx | config | accept len | accepted / proposed | accept rate | TPOT ms | decode s | speedup | steady draft fwd ms | refine % of decode | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 4k | baseline | 1.00 | - | - | 31.74 | 179.0 | 1.00x | - | - | 18.190 |
| 4k | DFlash | 3.110 | 3861 / 27009 | 0.1430 | 17.47 | 99.2 | 1.816x | 6.05 | - | 18.347 |
| 4k | +refine w=1 | 3.117 | 3865 / 26949 | 0.1434 | 17.50 | 99.4 | 1.813x | 6.15 | 0.80 | 18.347 |
| 4k | +refine w=2 | 3.108 | 3860 / 27026 | 0.1428 | 17.52 | 99.5 | 1.812x | 6.09 | 0.80 | 18.347 |
| 4k | +refine w=4 | 3.112 | 3862 / 26994 | 0.1431 | 17.51 | 99.5 | 1.812x | 6.13 | 0.81 | 18.347 |

16k and 32k, parallel sweep:

| ctx | config | accept len | accepted / proposed | accept rate | TPOT ms | decode s | speedup | steady draft fwd ms | refine % of decode | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 16k | baseline | 1.00 | - | - | 38.46 | 268.9 | 1.00x | - | - | 21.103 |
| 16k | DFlash | 2.505 | 4192 / 41106 | 0.1020 | 42.68 | 297.4 | 0.901x | 7.10 | - | 21.728 |
| 16k | +refine w=1 | 2.508 | 4195 / 41044 | 0.1022 | 42.97 | 299.4 | 0.895x | 7.10 | 0.44 | 21.728 |
| 16k | +refine w=2 | 2.517 | 4205 / 40911 | 0.1028 | 42.77 | 298.0 | 0.899x | 7.11 | 0.41 | 21.728 |
| 16k | +refine w=4 | 2.510 | 4199 / 41012 | 0.1024 | 42.85 | 298.6 | 0.898x | 7.11 | 0.41 | 21.728 |
| 32k | baseline | 1.00 | - | - | 44.79 | 17.6 | 1.00x | - | - | 24.987 |
| 32k | DFlash | 1.480 | 146 / 3660 | 0.0399 | 117.65 | 46.2 | 0.381x | 9.32 | - | 26.237 |
| 32k | +refine w=1 | 1.473 | 145 / 3675 | 0.0395 | 119.50 | 47.0 | 0.375x | 9.32 | 1.09 | 26.237 |
| 32k | +refine w=2 | 1.480 | 146 / 3660 | 0.0399 | 118.04 | 46.4 | 0.379x | 9.33 | 0.24 | 26.237 |
| 32k | +refine w=4 | 1.480 | 146 / 3660 | 0.0399 | 118.39 | 46.5 | 0.378x | 14.25 | 0.24 | 26.237 |

| ctx | config | target verify ms/step | steady draft fwd ms | refine s (total) | decode s | refine / decode |
|---|---|---|---|---|---|---|
| 4k iso | DFlash | 45.41 | 6.05 | - | 99.25 | - |
| 4k iso | +refine w=1 | 45.11 | 6.15 | 0.794 | 99.42 | 0.80% |
| 4k iso | +refine w=2 | 44.97 | 6.09 | 0.796 | 99.51 | 0.80% |
| 4k iso | +refine w=4 | 45.02 | 6.13 | 0.802 | 99.47 | 0.81% |
| 4k par | DFlash | 54.00 | 6.91 | - | 118.73 | - |
| 4k par | +refine w=1 | 47.63 | 6.74 | 0.827 | 104.75 | 0.79% |
| 4k par | +refine w=2 | 49.51 | 6.63 | 0.868 | 109.45 | 0.79% |
| 4k par | +refine w=4 | 51.41 | 6.64 | 0.895 | 113.90 | 0.79% |
| 16k | DFlash | 97.23 | 7.10 | - | 297.41 | - |
| 16k | +refine w=1 | 97.41 | 7.10 | 1.312 | 299.43 | 0.44% |
| 16k | +refine w=2 | 97.31 | 7.11 | 1.209 | 298.02 | 0.41% |
| 16k | +refine w=4 | 97.27 | 7.11 | 1.212 | 298.58 | 0.41% |
| 32k | DFlash | 165.71 | 9.32 | - | 46.24 | - |
| 32k | +refine w=1 | 165.66 | 9.32 | 0.513 | 46.96 | 1.09% |
| 32k | +refine w=2 | 165.67 | 9.33 | 0.110 | 46.39 | 0.24% |
| 32k | +refine w=4 | 165.68 | 9.33 | 0.110 | 46.53 | 0.24% |

- **The parallel 4k TPOT gain was noise.** There, refine configurations showed
  TPOT 18.4-20.1 ms against DFlash's 20.9 ms. The gap is entirely in target
  verify per step (54.0 ms vs 47.6-51.4 ms), which refinement does not touch. Run alone, the same prompts give
  17.47 ms for DFlash and 17.50-17.52 ms for refinement.
- **Net effect is a slowdown of about 2.5% at 4k.** The single isolated run
  above shows only +0.2%, but the 4k alpha sweep (§6) repeats the measurement
  in four more independent processes. There DFlash takes 52.7-53.1 ms per verify
  step and refinement 54.0-54.5 ms, so refinement costs 1.3-1.5 ms per step and
  TPOT rises by 2.3-2.8%. The isolated run's DFlash step was 54.65 ms, the
  slowest DFlash reading of all five processes, which is why its gap looked
  small. The refinement stages account for 0.44 ms of the 1.3-1.5 ms, and the
  capture in the draft forward for about 0.1 ms. The remaining 0.5-0.7 ms shows
  up as slower target verify, which refinement does not execute. Its cause is
  not established; see §6.3.
- **At 16k and 32k** TPOT rises by 0.2-0.7% and 0.3-0.6% in the parallel sweep,
  or 1.6% for 32k w=1 including the 405 ms stall. A step there is 97-166 ms, so
  a fixed ~1.4 ms per-step cost is 0.8-1.5%, the same order as these readings.
- **Steady draft forward** moves by less than 0.1 ms, within noise. The capture
  adds five small clones of block-sized tensors to the last draft layer.

### 4.5 Memory

| ctx | config | peak GiB | vs baseline GiB | vs DFlash GiB | peak phase | steady draft transient MiB | refine_peak_transient MiB | refine phase peak GiB |
|---|---|---|---|---|---|---|---|---|
| 4k | DFlash | 18.3467 | +0.1562 | +0.0000 | prefill: target forward | 9.65 | - | - |
| 4k | +refine w=1 | 18.3467 | +0.1562 | +0.0000 | prefill: target forward | 9.37 | 7.35 | 18.028 |
| 4k | +refine w=2 | 18.3467 | +0.1562 | +0.0000 | prefill: target forward | 9.37 | 7.35 | 18.028 |
| 4k | +refine w=4 | 18.3467 | +0.1562 | +0.0000 | prefill: target forward | 9.37 | 7.35 | 18.028 |
| 16k | DFlash | 21.7283 | +0.6250 | +0.0000 | prefill: target forward | 33.54 | - | - |
| 16k | +refine w=1 | 21.7283 | +0.6250 | +0.0000 | prefill: target forward | 33.27 | 7.07 | 20.420 |
| 16k | +refine w=2 | 21.7283 | +0.6250 | +0.0000 | prefill: target forward | 33.27 | 7.07 | 20.420 |
| 16k | +refine w=4 | 21.7283 | +0.6250 | +0.0000 | prefill: target forward | 33.27 | 7.07 | 20.420 |
| 32k | DFlash | 26.2367 | +1.2500 | +0.0000 | prefill: target forward | 64.64 | - | - |
| 32k | +refine w=1 | 26.2367 | +1.2500 | +0.0000 | prefill: target forward | 64.27 | 6.43 | 23.608 |
| 32k | +refine w=2 | 26.2367 | +1.2500 | +0.0000 | prefill: target forward | 64.27 | 6.43 | 23.608 |
| 32k | +refine w=4 | 26.2367 | +1.2500 | +0.0000 | prefill: target forward | 64.27 | 6.43 | 23.608 |

- **The run's peak is unchanged to the byte** at every context length and
  window, because it sits in the target's prefill (`prefill: target forward`),
  which refinement does not touch.
- **Refinement transient is 6.4-7.4 MiB**, measured as the `decode: local
  refine` interval peak minus the larger of the allocations on entry and exit.
  The arithmetic closes: candidate embeddings in bf16 take 15x16x4096x2 bytes, or
  1.88 MiB. Candidate LM-head rows take 1.88 MiB in bf16 and 3.75 MiB after the
  float32 upcast, plus a few MiB of top-K, norm and SDPA workspace. It does not grow with context.
- **Refine phase peak** is 18.03 / 20.42 / 23.61 GiB. It is the largest
  allocation seen inside the refinement across all calls, and it stays 0.3-2.6
  GiB below the run peak.
- **No persistent memory is added.** The draft KV cache (`max_draft_cache_gb`)
  and the steady context feature are identical to plain DFlash, and the
  steady draft transient differs by less than 0.4 MiB.

### 4.6 Acceptance

| ctx | config | mean accept len (DFlash -> refine) | accepted tokens | verify steps | proposals changed by rerank | changed at verified positions | helped | hurt |
|---|---|---|---|---|---|---|---|---|
| 4k | +refine w=1 | 3.110 -> 3.117 (+0.007) | 3861 -> 3865 | 1816 -> 1812 | 1043 (3.87%) | 46 | 5 | 5 |
| 4k | +refine w=2 | 3.110 -> 3.108 (-0.002) | 3861 -> 3860 | 1816 -> 1817 | 1128 (4.17%) | 50 | 8 | 7 |
| 4k | +refine w=4 | 3.110 -> 3.112 (+0.002) | 3861 -> 3862 | 1816 -> 1815 | 1069 (3.96%) | 36 | 4 | 1 |
| 16k | +refine w=1 | 2.505 -> 2.508 (+0.003) | 4192 -> 4195 | 2769 -> 2766 | 1422 (3.46%) | 79 | 13 | 9 |
| 16k | +refine w=2 | 2.505 -> 2.517 (+0.012) | 4192 -> 4205 | 2769 -> 2756 | 1409 (3.44%) | 89 | 17 | 7 |
| 16k | +refine w=4 | 2.505 -> 2.510 (+0.005) | 4192 -> 4199 | 2769 -> 2763 | 1429 (3.48%) | 88 | 16 | 8 |
| 32k | +refine w=1 | 1.480 -> 1.473 (-0.006) | 146 -> 145 | 244 -> 245 | 109 (2.97%) | 5 | 0 | 2 |
| 32k | +refine w=2 | 1.480 -> 1.480 (+0.000) | 146 -> 146 | 244 -> 244 | 104 (2.84%) | 4 | 0 | 1 |
| 32k | +refine w=4 | 1.480 -> 1.480 (+0.000) | 146 -> 146 | 244 -> 244 | 101 (2.76%) | 4 | 0 | 1 |

`proposals changed by rerank` counts draft positions where the refined argmax
differs from the original argmax. `changed at verified positions` restricts that
count to positions up to the first rejection, where the target's greedy token
for the true prefix is known. `helped` counts positions where the refined token
matches the target and the original did not; `hurt` counts the reverse.

- Refinement changes 2.8-4.2% of proposals. Most changes land after the first
  rejection, where they cannot affect acceptance.
- Mean acceptance length moves by -0.006 to +0.012, and accepted tokens move by
  -1 to +13 out of 146-4205. Helped and hurt counts are of the same order, for
  example 17 vs 7 at 16k w=2 and 5 vs 5 at 4k w=1. Across all 9 points the
  total is 63 helped against 41 hurt.
- **No window gives a meaningful acceptance change** at top_k=16, alpha=1.0.
  The best point, 16k w=2, adds +0.012 accepted tokens per step. That is 0.5%
  of acceptance length, too small to cover the refinement's 0.4% share of
  decode.

## 5. Conclusion

- The refinement meets its structural constraints. Code and hook counts both
  show one drafter forward and one full-vocabulary head forward per draft call.
  There are 0 of either inside the refinement. Caches and rollback are
  untouched. Output is identical to DFlash on every sample.
- Its cost is small and flat. It takes 0.44 ms per draft call, 7% of a steady
  draft forward at 4k and 5% at 32k, plus a 6-7 MiB transient, with no change
  to the peak.
- At top_k=16 it does not buy acceptance at any alpha in 0.25-4 or at 8k and
  16k with the best alpha (§6). Mean
  acceptance length moves within ±0.012, so TPOT gets worse, by about 2.5% at
  4k where a verify step is shortest.
- Levers worth trying next: `--refine-window full`; re-running the last layer's MLP on the 15 block
  positions instead of the first-order residual approximation; and a cheaper
  kernel path, since 0.44 ms is launch overhead that a fused or compiled
  version would largely remove.

## 6. Alpha sweep (4k) and best alpha at 8k / 16k

`queue/run_refine_alpha_sweep.sh` ran alpha = 0.25, 0.5, 2 and 4 at 4k, one
alpha per GPU, windows 1, 2 and 4, top_k=16. Each process decodes baseline and
DFlash on the same prompts, so every refine row has a DFlash reference measured
in the same process. `--refine-alpha` now takes a comma list, crossed with
`--refine-window`. The alpha=1 rows are the isolated 4k run from §3.
`queue/run_refine_best_alpha.sh` then ran the best alpha at 8k and 16k on two
GPUs in parallel. Records: `record_refine/alpha_sweep/a{alpha}/`,
`record_refine/best_alpha/`.

### 6.1 Acceptance by alpha, 4k

| alpha | window | accept len (DFlash 3.110) | accepted tokens (DFlash 3861) | changed proposals | helped / hurt | TPOT ms (DFlash, same process) | step ms (DFlash) |
|---|---|---|---|---|---|---|---|
| 0.25 | 1 | 3.114 | 3863 | 3.34% | 6 / 5 | 17.41 (16.98) | 54.52 (53.09) |
| 0.25 | 2 | 3.107 | 3859 | 3.89% | 5 / 7 | 17.42 (16.98) | 54.42 (53.09) |
| 0.25 | 4 | 3.102 | 3855 | 3.97% | 5 / 6 | 17.45 (16.98) | 54.42 (53.09) |
| 0.5 | 1 | 3.103 | 3856 | 3.88% | 3 / 8 | 17.36 (16.89) | 54.17 (52.82) |
| 0.5 | 2 | 3.105 | 3858 | 3.98% | 6 / 8 | 17.32 (16.89) | 54.08 (52.82) |
| 0.5 | 4 | 3.100 | 3854 | 3.99% | 5 / 5 | 17.37 (16.89) | 54.16 (52.82) |
| 1 | 1 | 3.117 | 3865 | 3.87% | 5 / 5 | 17.50 (17.47) | 54.87 (54.65) |
| 1 | 2 | 3.108 | 3860 | 4.17% | 8 / 7 | 17.52 (17.47) | 54.76 (54.65) |
| 1 | 4 | 3.112 | 3862 | 3.96% | 4 / 1 | 17.51 (17.47) | 54.81 (54.65) |
| 2 | 1 | 3.117 | 3865 | 4.01% | 8 / 7 | 17.34 (16.91) | 54.36 (52.88) |
| 2 | 2 | 3.117 | 3865 | 4.11% | 5 / 8 | 17.29 (16.91) | 54.18 (52.88) |
| 2 | 4 | 3.112 | 3862 | 4.04% | 6 / 5 | 17.32 (16.91) | 54.20 (52.88) |
| 4 | 1 | 3.114 | 3863 | 4.08% | 6 / 9 | 17.29 (16.85) | 54.14 (52.70) |
| 4 | 2 | 3.110 | 3861 | 4.10% | 5 / 6 | 17.28 (16.85) | 54.05 (52.70) |
| 4 | 4 | 3.117 | 3864 | 3.84% | 7 / 7 | 17.21 (16.85) | 53.96 (52.70) |

Per alpha, averaged or summed over the three windows:

| alpha | mean accept len over w=1,2,4 | accepted tokens summed over w | helped / hurt summed over w |
|---|---|---|---|
| 0.25 | 3.1073 | 11577 (DFlash x3 = 11583) | 16 / 18 |
| 0.5 | 3.1027 | 11568 (DFlash x3 = 11583) | 14 / 21 |
| 1 | 3.1124 | 11587 (DFlash x3 = 11583) | 17 / 13 |
| 2 | 3.1153 | 11592 (DFlash x3 = 11583) | 19 / 20 |
| 4 | 3.1136 | 11588 (DFlash x3 = 11583) | 18 / 22 |

- **No alpha moves acceptance outside noise.** The spread in accepted tokens is
  3854-3865 against DFlash's 3861, at most 11 tokens or 0.3%. Helped and hurt
  counts are balanced at every alpha.
- **alpha=2 is best by a hair.** It has the highest mean acceptance length
  over windows, 3.1153 against 3.1124 at alpha=1, and the most accepted tokens,
  11592 against 11583 for DFlash. That is 9 tokens over three windows and 32
  prompts, too small to call a real effect. It was used for 8k and 16k as the
  sweep's best point, not because it is clearly better.
- Small alphas (0.25, 0.5) are slightly worse than DFlash on every window but one.
  Larger alphas change slightly more proposals, 3.8-4.1% against 3.3-4.0%.
  That fits the patch mattering more as it grows, but with no acceptance to
  show for it.

### 6.2 Best alpha (alpha=2) at 8k and 16k

| ctx | config | accept len | accepted / proposed | TPOT ms | speedup | refine ms / call | refine % of decode | verify ms/step | step ms | refine transient MiB | outputs == DFlash | drafter / full-vocab head fwd in refine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8k | baseline | 1.000 | - | 33.39 | 1.000x | - | - | - | - | - | - | - |
| 8k | DFlash | 3.085 | 3967 / 28079 | 21.70 | 1.539x | - | - | 58.36 | 67.31 | - | 32/32 | - |
| 8k | refine w=1 a=2 | 3.084 | 3965 / 28077 | 22.14 | 1.508x | 0.591 | 0.86% | 58.92 | 68.66 | 7.35 | 32/32 | 0 / 0 |
| 8k | refine w=2 a=2 | 3.077 | 3962 / 28154 | 22.14 | 1.508x | 0.438 | 0.64% | 58.91 | 68.49 | 7.35 | 32/32 | 0 / 0 |
| 8k | refine w=4 a=2 | 3.077 | 3962 / 28154 | 22.11 | 1.510x | 0.438 | 0.64% | 58.84 | 68.42 | 7.35 | 32/32 | 0 / 0 |
| 16k | baseline | 1.000 | - | 38.42 | 1.000x | - | - | - | - | - | - | - |
| 16k | DFlash | 2.505 | 4192 / 41106 | 42.74 | 0.899x | - | - | 97.27 | 107.54 | - | 32/32 | - |
| 16k | refine w=1 a=2 | 2.508 | 4195 / 41061 | 42.96 | 0.894x | 0.545 | 0.50% | 97.36 | 108.23 | 7.07 | 32/32 | 0 / 0 |
| 16k | refine w=2 a=2 | 2.516 | 4204 / 40926 | 42.80 | 0.898x | 0.438 | 0.41% | 97.41 | 108.17 | 7.07 | 32/32 | 0 / 0 |
| 16k | refine w=4 a=2 | 2.514 | 4202 / 40938 | 42.82 | 0.897x | 0.438 | 0.41% | 97.36 | 108.13 | 7.07 | 32/32 | 0 / 0 |

16k, alpha=1 (§4) against alpha=2, same prompts:

| window | alpha=1 accept len | alpha=2 accept len | alpha=1 accepted | alpha=2 accepted | alpha=1 helped / hurt | alpha=2 helped / hurt |
|---|---|---|---|---|---|---|
| 1 | 2.508 | 2.508 | 4195 | 4195 | 13 / 9 | 14 / 9 |
| 2 | 2.517 | 2.516 | 4205 | 4204 | 17 / 7 | 20 / 6 |
| 4 | 2.510 | 2.514 | 4199 | 4202 | 16 / 8 | 15 / 11 |

- **8k: acceptance is flat to slightly lower.** Mean acceptance length is
  3.077-3.084 against DFlash's 3.085, and TPOT rises 1.9-2.1%. Speedup over
  baseline drops from 1.539x to 1.508-1.510x.
- **16k: the best point of the whole study, and still a net loss.** Mean
  acceptance length rises by up to +0.011 at w=2, which is 12 more accepted
  tokens and 12 fewer verify steps. Even so, TPOT rises 0.2-0.5% and speedup
  moves from 0.899x to 0.894-0.898x. alpha=2 reproduces alpha=1 at 16k to within
  3 tokens on every window.
- The structural checks hold at both lengths. Drafter and full-vocabulary head
  forwards inside the refinement are 0 / 0, outputs equal DFlash on 32/32
  samples, and the transient is 7.1-7.4 MiB with no change to the peak.
- The w=1 rows at 8k and 16k have a mean of 0.55-0.59 ms per call against 0.44
  ms for w=2 and w=4. Each comes from a single stalled call, in the
  soft-embedding stage at 8k and the attention stage at 16k, the same kind of
  one-off as the 32k stall in §4.3.

### 6.3 Where the per-step overhead goes

| run | DFlash verify ms/step | refine verify ms/step | DFlash step ms | refine step ms | refine stages ms/step | step delta ms | TPOT delta |
|---|---|---|---|---|---|---|---|
| 4k alpha=0.25 | 44.24 | 44.82-44.92 | 53.09 | 54.42-54.52 | 0.44-0.44 | +1.33 to +1.42 | +2.6% to +2.8% |
| 4k alpha=0.5 | 44.21 | 44.75-44.83 | 52.82 | 54.08-54.17 | 0.44-0.44 | +1.26 to +1.35 | +2.5% to +2.9% |
| 4k isolated (alpha=1) | 45.41 | 44.97-45.11 | 54.65 | 54.76-54.87 | 0.44-0.44 | +0.11 to +0.22 | +0.2% to +0.3% |
| 4k alpha=2 | 44.28 | 44.85-45.01 | 52.88 | 54.18-54.36 | 0.44-0.44 | +1.30 to +1.48 | +2.2% to +2.6% |
| 4k alpha=4 | 44.10 | 44.65-44.81 | 52.70 | 53.96-54.14 | 0.43-0.44 | +1.27 to +1.45 | +2.2% to +2.6% |
| 8k alpha=2 w=1 | 58.36 | 58.92 | 67.31 | 68.66 | 0.59 | +1.35 | +2.1% |
| 8k alpha=2 w=2 | 58.36 | 58.91 | 67.31 | 68.49 | 0.44 | +1.18 | +2.0% |
| 8k alpha=2 w=4 | 58.36 | 58.84 | 67.31 | 68.42 | 0.44 | +1.11 | +1.9% |
| 16k alpha=2 w=1 | 97.27 | 97.36 | 107.54 | 108.23 | 0.55 | +0.69 | +0.5% |
| 16k alpha=2 w=2 | 97.27 | 97.41 | 107.54 | 108.17 | 0.44 | +0.63 | +0.2% |
| 16k alpha=2 w=4 | 97.27 | 97.36 | 107.54 | 108.13 | 0.44 | +0.59 | +0.2% |

- **The refinement costs about 1.1-1.5 ms per verify step at 4k and 8k.** In
  every multi-process measurement except the isolated alpha=1 run, TPOT rises
  1.9-2.9%. The isolated run is the outlier because its DFlash reference was
  the slowest of the five 4k processes.
- **Only a third of that is the refinement stages.** The stage timers give 0.44
  ms, and the capture in the draft forward adds about 0.1 ms. The remaining
  0.5-0.7 ms appears as slower target verify, although verify runs no
  refinement code. At 16k the verify gap shrinks to 0.1 ms, so the step delta
  is 0.6-0.7 ms, close to stages plus capture.
- **Likely cause, not verified:** at 4-8k a verify step is CPU launch-bound.
  CUDA-event time then includes host gaps between kernel launches, so
  host-side work that refinement adds, such as extra Python objects, garbage
  collection and allocator state, can land inside the verify window. At 16k the
  verify kernels are long enough to hide this. Settling it would take a run
  with `gc.disable()` or a CUDA-graph verify path.

### 6.4 Conclusion of the sweep

At top_k=16, the training-free local refinement does not improve acceptance for
any alpha in 0.25-4 or window in 1, 2, 4 at 4k, 8k or 16k. Its largest effect
is +0.011 mean acceptance length, 0.4%, at 16k. Against that it adds 0.6-1.5
ms per verify step, so TPOT is 0.2-2.9% worse everywhere measured. The
soft-token patch changes 3-4% of proposals, and those changes help about as
often as they hurt. The signal the block's own earlier draft positions carry
through one attention layer, without re-running the MLP, is too weak to fix
the drafter's errors.

## 7. Qwen3.5-9B

Same code and conditions on `Qwen/Qwen3.5-9B` + `z-lab/Qwen3.5-9B-DFlash`:
4k / 16k / 32k at alpha=1 (`record_refine_9b/`), then the 4k alpha sweep
(`record_refine_9b/alpha_sweep/`). Both run by `queue/run_refine_9b.sh`.
The 9B drafter has 6 layers, 5 of them `sliding_attention` with the last one
`full_attention`, so the capture sits on a full-attention layer as on 8B.

### 7.1 Read these numbers with the GDN rollback bug in mind

**On 9B the DFlash path in `dflash/model.py` is not lossless.** `_crop_to`
crops a linear-attention layer's `conv_states` but not its `recurrent_states`
(transformers 5.16.1), so every rejected token stays folded into the target's
24 gated-delta-rule layers. Verification is therefore not exact, and changing
what the drafter proposes changes the final output rather than only the
acceptance pattern. The exactness column shows it: on 8B refined output equals
DFlash output on 32/32 samples at every setting, while on 9B it equals it on
only 16-31 of 32.

That makes the headline acceptance deltas meaningless as a measure of the
refinement: they mix a different proposal with a different decoding
trajectory. The comparison below is therefore restricted to the samples where
refined and plain DFlash produced the identical output, where acceptance is
the only thing that can differ.

### 7.2 Headline, all 32 samples (trajectory-confounded)

| ctx | config | accept len | TPOT ms | speedup | refine ms / call | refine % of decode | peak GiB | refine transient MiB | outputs == DFlash | drafter / head fwd in refine |
|---|---|---|---|---|---|---|---|---|---|---|
| 4k | baseline | 1.000 | 39.29 | 1.000x | - | - | 21.599 | - | - | - |
| 4k | DFlash | 8.976 | 16.58 | 2.370x | - | - | 21.818 | - | 32/32 | - |
| 4k | refine w=1 | 9.394 | 16.08 | 2.444x | 0.473 | 0.31% | 21.818 | 7.35 | 19/32 | 0 / 0 |
| 4k | refine w=2 | 8.905 | 16.95 | 2.317x | 0.471 | 0.31% | 21.818 | 7.35 | 19/32 | 0 / 0 |
| 4k | refine w=4 | 9.825 | 15.41 | 2.550x | 0.472 | 0.31% | 21.818 | 7.34 | 21/32 | 0 / 0 |
| 16k | baseline | 1.000 | 39.68 | 1.000x | - | - | 28.966 | - | - | - |
| 16k | DFlash | 8.832 | 17.34 | 2.288x | - | - | 29.841 | - | 32/32 | - |
| 16k | refine w=1 | 8.506 | 18.29 | 2.169x | 0.475 | 0.30% | 29.841 | 7.31 | 17/32 | 0 / 0 |
| 16k | refine w=2 | 8.970 | 17.37 | 2.285x | 0.475 | 0.30% | 29.841 | 7.34 | 16/32 | 0 / 0 |
| 16k | refine w=4 | 8.943 | 17.39 | 2.281x | 0.475 | 0.30% | 29.841 | 7.33 | 17/32 | 0 / 0 |
| 32k | baseline | 1.000 | 36.32 | 1.000x | - | - | 38.790 | - | - | - |
| 32k | DFlash | 7.058 | 26.58 | 1.366x | - | - | 40.540 | - | 32/32 | - |
| 32k | refine w=1 | 7.127 | 26.67 | 1.362x | 0.487 | 0.25% | 40.540 | 6.35 | 30/32 | 0 / 0 |
| 32k | refine w=2 | 7.058 | 27.06 | 1.342x | 0.486 | 0.24% | 40.540 | 6.35 | 31/32 | 0 / 0 |
| 32k | refine w=4 | 7.058 | 26.85 | 1.353x | 0.486 | 0.25% | 40.540 | 6.35 | 31/32 | 0 / 0 |

### 7.3 Acceptance on identical-output samples only

| ctx | window | matched samples | accept len DFlash -> refine (matched only) | accepted tokens | verify steps | headline accept len delta (all 32) |
|---|---|---|---|---|---|---|
| 4k | 1 | 19/32 | 11.258 -> 11.209 (-0.048) | 2392 -> 2392 | 233 -> 234 | +0.418 |
| 4k | 2 | 19/32 | 11.173 -> 11.057 (-0.116) | 1945 -> 1943 | 191 -> 193 | -0.071 |
| 4k | 4 | 21/32 | 11.473 -> 11.473 (+0.000) | 2883 -> 2883 | 275 -> 275 | +0.849 |
| 16k | 1 | 17/32 | 4.694 -> 4.694 (+0.000) | 133 -> 133 | 36 -> 36 | -0.326 |
| 16k | 2 | 16/32 | 4.692 -> 4.692 (+0.000) | 96 -> 96 | 26 -> 26 | +0.138 |
| 16k | 4 | 17/32 | 4.694 -> 4.694 (+0.000) | 133 -> 133 | 36 -> 36 | +0.111 |
| 32k | 1 | 30/32 | 3.618 -> 3.618 (+0.000) | 144 -> 144 | 55 -> 55 | +0.069 |
| 32k | 2 | 31/32 | 3.375 -> 3.375 (+0.000) | 152 -> 152 | 64 -> 64 | +0.000 |
| 32k | 4 | 31/32 | 3.375 -> 3.375 (+0.000) | 152 -> 152 | 64 -> 64 | +0.000 |

- **On the matched subset the refinement never helps.** Deltas are 0.000 at six
  of nine points and -0.048 to -0.116 at the other three. The headline column
  next to it swings from -0.326 to +0.849 on the same runs, which is the size
  of the artefact.
- The matched subset is biased toward short generations, since a long output
  has more chances to diverge. That is why its acceptance lengths (3.4-11.6)
  differ from the headline ones; it is still a like-for-like DFlash comparison
  within the subset.
- Refinement changes far fewer proposals on 9B than on 8B: 0.45-1.49% against
  2.8-4.2%. The 9B drafter is much more confident, so its top-1 rarely loses to
  a rerank of 16 candidates.

### 7.4 4k alpha sweep

| alpha | window | matched samples | accept len, matched only (DFlash -> refine) | headline accept len (DFlash -> refine, all 32) | changed proposals | helped / hurt | TPOT ms (DFlash) |
|---|---|---|---|---|---|---|---|
| 0.25 | 1 | 21/32 | 11.390 -> 11.431 (+0.041) | 8.976 -> 8.657 (-0.319) | 57 (0.45%) | 3 / 0 | 17.43 (16.57) |
| 0.25 | 2 | 19/32 | 11.468 -> 11.346 (-0.122) | 8.976 -> 9.109 (+0.134) | 74 (0.62%) | 1 / 0 | 16.54 (16.57) |
| 0.25 | 4 | 21/32 | 11.599 -> 11.515 (-0.085) | 8.976 -> 8.599 (-0.377) | 106 (0.78%) | 2 / 2 | 17.52 (16.57) |
| 0.5 | 1 | 20/32 | 11.299 -> 11.251 (-0.048) | 8.976 -> 7.667 (-1.309) | 146 (0.97%) | 3 / 2 | 19.16 (16.19) |
| 0.5 | 2 | 20/32 | 11.224 -> 11.224 (+0.000) | 8.976 -> 8.505 (-0.471) | 140 (1.10%) | 5 / 4 | 17.26 (16.19) |
| 0.5 | 4 | 20/32 | 11.496 -> 11.446 (-0.050) | 8.976 -> 9.240 (+0.264) | 121 (0.91%) | 4 / 4 | 15.89 (16.19) |
| 1 | 1 | 19/32 | 11.258 -> 11.209 (-0.048) | 8.976 -> 9.394 (+0.418) | 94 (0.76%) | 2 / 1 | 16.08 (16.58) |
| 1 | 2 | 19/32 | 11.173 -> 11.057 (-0.116) | 8.976 -> 8.905 (-0.071) | 134 (1.02%) | 0 / 2 | 16.95 (16.58) |
| 1 | 4 | 21/32 | 11.473 -> 11.473 (+0.000) | 8.976 -> 9.825 (+0.849) | 106 (0.89%) | 2 / 3 | 15.41 (16.58) |
| 2 | 1 | 20/32 | 10.792 -> 10.792 (+0.000) | 8.976 -> 8.347 (-0.629) | 130 (1.00%) | 3 / 3 | 17.83 (16.31) |
| 2 | 2 | 18/32 | 10.397 -> 10.201 (-0.196) | 8.976 -> 8.414 (-0.562) | 123 (0.95%) | 2 / 4 | 17.66 (16.31) |
| 2 | 4 | 21/32 | 11.188 -> 11.109 (-0.079) | 8.976 -> 8.186 (-0.790) | 187 (1.42%) | 2 / 7 | 18.17 (16.31) |
| 4 | 1 | 20/32 | 10.971 -> 10.971 (+0.000) | 8.976 -> 8.587 (-0.389) | 130 (0.96%) | 1 / 2 | 17.55 (16.58) |
| 4 | 2 | 16/32 | 10.686 -> 10.686 (+0.000) | 8.976 -> 8.371 (-0.605) | 139 (1.00%) | 2 / 3 | 18.03 (16.58) |
| 4 | 4 | 19/32 | 10.888 -> 10.778 (-0.110) | 8.976 -> 8.506 (-0.470) | 132 (0.91%) | 2 / 2 | 17.72 (16.58) |

Averaged over the three windows, on matched samples only:

| alpha | mean matched accept-len delta | matched samples | helped / hurt |
|---|---|---|---|
| 0.25 | -0.055 | 61/96 | 6 / 2 |
| 0.5 | -0.033 | 60/96 | 12 / 10 |
| 1 | -0.055 | 59/96 | 4 / 6 |
| 2 | -0.092 | 59/96 | 7 / 14 |
| 4 | -0.037 | 55/96 | 5 / 7 |

No alpha helps. Every value is negative on the matched subset, and larger
alphas both change more proposals and diverge more often, so there is no
setting where the extra conditioning pays for itself.

### 7.5 Cost on 9B

| ctx | window | calls | soft embedding ms | K/V projection ms | attention ms | rerank ms | total ms / call | % of DFlash steady draft fwd |
|---|---|---|---|---|---|---|---|---|
| 4k | 1 | 830 | 0.129 | 0.113 | 0.140 | 0.083 | 0.473 | 5.8 |
| 4k | 2 | 881 | 0.129 | 0.113 | 0.140 | 0.082 | 0.471 | 5.8 |
| 4k | 4 | 798 | 0.129 | 0.113 | 0.140 | 0.082 | 0.472 | 5.8 |
| 16k | 1 | 921 | 0.130 | 0.114 | 0.141 | 0.083 | 0.475 | 5.6 |
| 16k | 2 | 824 | 0.129 | 0.114 | 0.141 | 0.083 | 0.475 | 5.6 |
| 16k | 4 | 876 | 0.129 | 0.114 | 0.141 | 0.083 | 0.475 | 5.6 |
| 32k | 1 | 102 | 0.133 | 0.117 | 0.144 | 0.085 | 0.487 | 5.5 |
| 32k | 2 | 103 | 0.133 | 0.117 | 0.144 | 0.085 | 0.486 | 5.5 |
| 32k | 4 | 103 | 0.133 | 0.117 | 0.144 | 0.085 | 0.486 | 5.5 |

- The refinement costs 0.47-0.49 ms per draft call, the same as on 8B, and
  0.24-0.31% of decode. It is a smaller share than on 8B because 9B acceptance
  is 7-9, so there are far fewer draft calls per output token.
- Peak memory is unchanged to the byte at every context, and the transient is
  6.4-7.4 MiB, as on 8B.
- The structural checks hold: 0 drafter forwards and 0 full-vocabulary head
  forwards inside `decode: local refine` at every context and window.
- TPOT comparisons on 9B are not meaningful per configuration, because the
  configurations do not generate the same tokens. The refinement's own share of
  decode, under 0.31%, is the sound cost figure.

### 7.6 What a clean 9B answer would need

The batched engine (`dflash/batch.py`) already implements the replay-based GDN
rollback that makes 9B verification exact. Running this comparison there, or
porting that rollback into `dflash_generate`, would give a 9B acceptance
comparison as trustworthy as the 8B one. Until then the only 9B conclusions
that stand are the cost and structural ones in §7.5, plus the matched-subset
result that acceptance does not improve.

Records: `record_refine/qwen3-8b_4096_20260917-174906.json`,
`record_refine/qwen3-8b_16384_20260917-180932.json`,
`record_refine/qwen3-8b_32768_20260917-180252.json`,
`record_refine/timing_control/` (isolated 4k), `record_refine/alpha_sweep/`,
`record_refine/best_alpha/`, `record_refine_9b/` and
`record_refine_9b/alpha_sweep/`. Logs in `logs/refine/`.

