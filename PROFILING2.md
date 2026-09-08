# Phase-Local Memory and the Drafter's Own Prefill

A second pass over the `--hidden-states selective` sweep. Nothing about
inference changed; what changed is where the measurements are taken. Two
questions the first sweep could not answer:

1. **What is actually live when the run peaks?** The old split subtracted a sum
   of per-component maxima from the peak and called the remainder `transient`.
   Those maxima are not simultaneous, so the remainder was not a quantity.
2. **What does the drafter cost at long context?** The drafter's first call of a
   request projects the entire prompt into its KV cache — O(S) work that happens
   once — while every later call handles only the tokens the last verify
   accepted. Averaged together they described neither.

Both are now measured directly. `record_selective/` holds the new records
alongside the old ones; nothing was overwritten.

## 1. Why the old transient accounting was wrong

`record.py:_budget` computed

```
resident  = target_weight + target_KV_max + draft_weight + draft_KV_max
          + target_hidden_max + context_feature_max
transient = peak_allocated − resident
```

Every term on the first line is a **maximum over the whole request**, and the
maxima land at different moments:

| Term | When it is largest |
| --- | --- |
| `target_KV` | the last decode step, after every accepted token is cached |
| `target_hidden_states` | during prefill, before the first token exists |
| `context_feature` | immediately after prefill, at prompt length |
| `draft_KV` | the first draft call, once the prompt is projected |
| `peak_allocated` | inside the target's last prefill layer |

Subtracting a sum of five different instants from a sixth is not a
decomposition. Three consequences showed up in the numbers:

* **It over-subtracts what is not there yet.** At the prefill peak the target KV
  holds only the prompt; the figure subtracted at that point is the KV at the
  *end* of decode. The drafter's KV cache does not exist at all at the prefill
  peak, and neither does the context feature — yet both were subtracted.
* **It under-subtracts what is there twice.** During the first decode step the
  prompt-length context feature and the freshly captured residual streams are
  live at the same time, plus the target's verify activation. Nothing in the
  resident line represents that coincidence.
* **It could not name a site.** `peak_site` said *which operation* peaked;
  `transient_gb` could not say what the peak consisted of.

The 9B column of the old table shows the failure plainly: `transient_gb` runs
from 1.94 GB at 4k to 31.79 GB at 64k, roughly half the peak. Read literally
that says half the memory is unexplained activation. It is not, and the phase
records now close the arithmetic exactly.

**The largest single error was in `max_target_cache_gb`.** `_cache_bytes` is
called once, when a sample ends. On Qwen3.5-9B, 24 of 32 layers are gated
delta-rule: during prefill their cache holds a sequence-length buffer that the
first decode-step forward releases. So the recorded target KV is the *post*-
release figure, while the peak lands *before* it. At 4k:

| | GB |
| --- | --- |
| target KV live at the prefill peak (`prefill: target forward`, from the phase probe) | 1.672 |
| `max_target_cache_gb`, read at end of sample | 0.195 |
| under-counted | **1.477** |

The same 1.67 → 0.18 collapse appears in the baseline's phase table, at the same
phase, so it is the target's own behaviour and nothing to do with the drafter.

With that in hand the old number decomposes exactly, and the identity is
general — every run in the sweep peaks in `prefill: target forward`, so

```
transient_gb = real prefill activation
             + (target KV live at the peak − max_target_cache_gb)
             − (max_context_feature_gb + max_draft_cache_gb)
```

The last line is the drafter's own state, which does not exist yet when the
target's prefill peaks, and was subtracted anyway. Both presets, at opposite
ends of the sweep:

| | Qwen3.5-9B 4k | Qwen3-8B 64k |
| --- | --- | --- |
| real prefill activation | 0.8111 | 6.5417 |
| target KV live at peak − recorded max | +1.4771 | −0.0703 |
| context feature + draft KV, subtracted but not live | −0.3463 | −3.7598 |
| **= `transient_gb`** | **1.9419** | **2.7116** |
| `transient_gb` as recorded in the old sweep | 1.9419 | 2.7116 |

Agreement to four decimals, on numbers measured two entirely different ways —
and it holds for **all ten** configurations in the sweep, not just these two:

| model | ctx | old `transient_gb` | real prefill activation | KV live − recorded | not-yet-live |
| --- | --- | --- | --- | --- | --- |
| qwen3-8b | 4k | 0.1019 | 0.4163 | −0.0703 | −0.2441 |
| qwen3-8b | 8k | 0.2759 | 0.8247 | −0.0703 | −0.4785 |
| qwen3-8b | 16k | 0.6240 | 1.6415 | −0.0702 | −0.9473 |
| qwen3-8b | 32k | 1.3903 | 3.2749 | −0.0084 | −1.8762 |
| qwen3-8b | 64k | 2.7116 | 6.5417 | −0.0703 | −3.7598 |
| qwen3.5-9b | 4k | 1.9419 | 0.8111 | +1.4771 | −0.3463 |
| qwen3.5-9b | 8k | 3.9007 | 1.6108 | +2.9778 | −0.6879 |
| qwen3.5-9b | 16k | 7.8120 | 3.2103 | +5.9771 | −1.3754 |
| qwen3.5-9b | 32k | 15.6365 | 6.4091 | +11.9778 | −2.7504 |
| qwen3.5-9b | 64k | 31.7937 | 13.3152 | +23.9789 | −5.5004 |

The two presets fail in opposite directions, which is the point. On the 8B the
old figure is consistently ~2.4x too *small* — 2.71 GB reported at 64k against
6.54 GB of real target prefill activation, because drafter state that does not
yet exist when the target peaks was subtracted from it. On the 9B it is ~2.4x
too *large* in the other direction — 31.79 GB at 64k where 13.32 was real —
because the GDN cache collapse hides a target-KV term that reaches **23.98 GB**.
No sign convention rescues a number that can be wrong either way, and both
errors grow linearly with context, so the fault is worst exactly where the
measurement matters.

`max_target_cache_gb` is still recorded unchanged;
`phase_memory[...]["peak_components_gb"]["target_kv_bytes"]` is the figure to
read for what the cache actually costs at the peak.

`transient_gb` is likewise still written to the records, unchanged, so old and
new records stay comparable. It is no longer the figure to read.

## 2. Phase-local memory accounting

`PhaseMemory` (`model.py:479`) records a **phase**: a named span of one request.
`dflash_generate` declares twelve of them, via the `phase()` helper
(`model.py:847`) that closes the interval `PeakTracker` had open and opens the
next:

| Phase | Covers |
| --- | --- |
| `prefill: target forward` | one prefill chunk through the target |
| `prefill: context-feature build` | the per-chunk `cat` of the injected layers |
| `prefill: first token` | sampling the prompt's continuation |
| `prefill: context-feature concat` | joining the chunks into one feature |
| `prefill: rollback/crop` | cropping the target KV back to the prompt |
| `decode: first draft forward` | the drafter's own prefill — see §3 |
| `decode: draft forward` | every later draft call |
| `decode: draft rollback/crop` | cropping the draft KV back after a call |
| `decode: draft logits` | the drafter's head and candidate selection |
| `decode: target verify` | the target's verify forward |
| `decode: verify rollback/crop` | cropping the target KV to the accepted length |
| `decode: context-feature build` | rebuilding the feature from accepted tokens |

For each phase the record carries, per occurrence:

* `allocated_before_bytes` / `allocated_after_bytes` — live bytes on entry, on exit
* `interval_peak_bytes` — the largest allocation inside it
* `components` at the peaking occurrence: `target_kv`, `draft_kv`,
  `selected_hidden`, `context_feature`, `draft_weight` (plus `target_weight`,
  added at aggregation time), **all read at one instant**
* `peak_interval_label` — which sub-interval peaked, which on a sharded target
  names the decoder layer

The aggregate (`record.py:aggregate_phases`, :233) keeps the largest occurrence
across samples and its component split, plus per-sample means, and adds

```
peak_unattributed = interval_peak − Σ components
```

Unlike `transient_gb` this is one instant minus the components live *at that
same instant*, so it is a real remainder: attention workspaces, logits, and the
target's own layer activations. It cannot go negative by mixing moments.
`record.py:peak_phase` (:301) names the phase whose largest occurrence set the
run's high-water mark.

**Three honest limits.** (a) The component reading is taken when the peaking
occurrence *closes*, not at the allocator's high-water mark inside it, which no
CUDA API exposes. (b) On a sharded target a phase spans one interval per decoder
layer, so the reading is the phase's end rather than the peaking layer's.
(c) Where a component *shrinks* inside a phase, the close reading catches its
lower value and the difference lands in `peak_unattributed`. The one place that
happens is `decode: target verify` on Qwen3.5-9B, where the first verify
releases the gated delta-rule prefill buffers: the row enters at 21.12 GB,
leaves at 19.63, reports `target_kv` 0.18 rather than the 1.67 that was live on
the way in, and carries 1.51 GB of unattributed as a result. The bracket is
visible in the record — `allocated_before_bytes` on that phase and the
`target_kv` probe on the phase before it — and the phase that actually sets the
peak (`prefill: target forward`) has no shrinking component, so its split is
exact.

**Cost.** The probe walks both KV caches, which is the only reading here
expensive enough to reach a per-token latency, so it is taken only when an
occurrence is the largest that phase has seen — true for the first decode step
and then almost never. `allocated_before/after` use the raw allocator binding
(`_device_bytes`, ~10 µs) rather than `torch.cuda.memory_allocated` (~81 µs).

**The extra boundaries do not move the peak.** `peak_memory_bytes` is the
maximum over intervals of the summed per-device peak. On one device a maximum
over time equals the maximum over any partition of it, so finer intervals leave
it exactly unchanged — confirmed to the byte in §5. On a sharded target a finer
partition can only tighten it.

## 3. First draft vs steady draft

The drafter is fed the target's context feature and keeps its own KV cache. On
the **first** call of a request that feature is the whole prompt: `fc` and
`hidden_norm` run over S rows, `k_proj`/`v_proj` run over S rows, and S entries
land in the draft KV cache. On **every later** call the feature is only the
tokens the last verify accepted — typically 2 to 9 rows. The first call is the
drafter's own prefill; folding it into a mean over draft calls is what made the
drafter look cheap at 4k and unexplained at 64k.

They are now separate phases, separate timers and separate records:

| Metric | Meaning |
| --- | --- |
| `mean_first_draft_forward_s` | the first call, per request |
| `max_first_draft_peak_memory_gb` | the allocation peak inside it |
| `max_first_draft_transient_gb` | that peak less what stood on both sides — the part it borrowed rather than kept |
| `max_first_draft_cache_gb` | the draft KV it leaves behind |
| `mean_first_draft_fraction_of_decode` | its share of decode latency |
| `first_draft_stage_s` | inside it: `context_projection` (fc + norm), `context_kv_projection`, `cache_update`, `attention`, `output_head` |
| `mean_steady_draft_forward_s` | every later call, per call |
| `mean_steady_attention_s`, `mean_steady_cache_update_s`, … | the same stages, steady state |
| `max_steady_draft_cache_gb` | the draft KV in steady state |

`draft_forward_s` still spans both calls and keeps the meaning it had in every
earlier record; the two new timers split it.

The stage timers (`_DraftStages`, `model.py:684`) are CUDA events reached
through a module-level handle (`_stage`, :720) rather than an argument: every
kwarg the draft decoder layer takes is forwarded into the attention kernel, so
anything threaded through the call signature would end up in SDPA. They are on
by default and can be turned off with `--no-draft-stage-profiling`.

## 4. Separating DFlash overhead from target overhead

`summary["overhead_split"]` keeps the two apart:

**DFlash-specific** — none of it exists in a baseline run:

* `dflash_draft_weight_gb` — the drafter's parameters
* `dflash_draft_kv_gb` — the draft KV cache
* `dflash_selected_hidden_gb` — the target residual streams DFlash reads
* `dflash_context_feature_gb` — the feature built from them
* `dflash_first_draft_transient_gb` / `dflash_steady_draft_transient_gb` — borrowed inside a draft call
* `dflash_persistent_resident_gb` — what survives between decode steps: draft weights + steady draft KV + steady context feature

**Target-architecture** — paid whether or not a drafter exists:

* `target_weight_gb`, `target_kv_gb`
* `target_prefill_transient_gb` — the target's prefill activation, measured
  *inside* the `prefill: target forward` phase rather than inferred from the
  run's peak. Recorded for the baseline too, and the two agree, which is the
  check that it is not drafter overhead.

## 5. Correctness

Greedy decoding (`temperature=0`), `--hidden-states selective`, both presets,
four LongBench prompts each at 4k, DFlash and baseline, run once on this commit
and once on `e5abae2` (the commit before it). Compared field by field:

`output_ids`, `num_output_tokens`, `acceptance_lengths`, `accepted_lengths`,
`proposed_lengths`, `num_accepted_tokens`, `num_proposed_tokens`,
`num_verify_steps`, `num_draft_calls`, `target_cache_bytes`,
`draft_cache_bytes`.

| preset | samples | result |
| --- | --- | --- |
| qwen3-8b | 4 | identical on every field |
| qwen3.5-9b | 4 | identical on every field |

`peak_memory_bytes` also matched to the byte (18.345 GB on 8B, 21.817 GB on 9B),
which is the check that the extra phase boundaries did not repartition the peak.

A second, stronger check falls out of the sweep itself: the new qwen3-8b 4k run
and the `--hidden-states selective` record it supersedes are the same 32
LongBench prompts under the same seed, and they agree exactly.

| | new | old (`20260908-185243`) |
| --- | --- | --- |
| `mean_acceptance_length` | 3.1101321585903086 | 3.1101321585903086 |
| `total_accepted_tokens` | 3861 | 3861 |
| `total_proposed_tokens` | 27009 | 27009 |
| `total_verify_steps` | 1816 | 1816 |
| `total_output_tokens` (dflash / baseline) | 5680 / 5640 | 5680 / 5640 |
| `peak_memory_gb` (dflash / baseline) | 18.344727039337158 / 18.188477039337158 | same, to the byte |
| `max_target_cache_gb` / `max_draft_cache_gb` | 0.6328125 / 0.087890625 | same |

Only latency moved, by 0.1% on the speedup — run-to-run timing noise.

The instrumentation cannot change decoding by construction — no probe touches a
tensor, and the stage timers only record CUDA events — but the run is the
evidence, not the argument.

## 6. `--draft-window-size`: not implemented, and what it would take

A drafter that keeps only the last `W` tokens of injected context would bound
the one DFlash-specific term that grows linearly with the prompt. It is not in
this change: every way of getting it either breaks an invariant the rollback
path depends on, or changes the thing the sweep is trying to isolate. What it
needs, precisely:

1. **Front-eviction of the draft KV cache.** The drafter's cache holds one entry
   per absolute token position. `_crop_to` (`model.py:286`) removes from the
   *end* only — `DynamicCache.crop` has no other mode — so a window needs the
   per-layer `keys`/`values` resliced by hand each step. On the Qwen3.5-9B
   drafter five of six layers are `sliding_attention`, whose `crop()` under
   `activate_past_recording` already re-points `keys` at a view of the last
   `sliding_window - 1` positions (PROFILING.md, "How the memory numbers are
   computed"); front-slicing on top of that view has to not break the rollback
   contract that flag exists for.
2. **Relative cache bookkeeping.** Today `cache.get_seq_length() == start`, the
   absolute position, and every crop site relies on it. With a window the crop
   target becomes `min(W, start)` and the absolute-to-relative offset has to be
   carried by `dflash_generate` explicitly.
3. **Rollback under eviction.** A rejected block is undone by cropping the draft
   cache back to the accepted length. Today the crop only ever removes the noise
   block, so nothing evicted is ever needed again. A window makes that no longer
   true in general, and the invariant has to be re-established rather than
   assumed.
4. **Mask and RoPE re-check.** `_attention_mask` (`model.py:741`) derives query
   and key positions from `q.shape[-2]` and `k.shape[-2]`, so it stays
   internally consistent under a shorter cache, and cached K already carries
   RoPE at its absolute position. Both need a test rather than an argument,
   especially where a sliding window is applied on top of the draft window.

**The cheap alternative, and why it is not the same experiment.** Running the
drafter cacheless over a rolling `W`-row buffer (`past_key_values=None`,
`target_hidden = feature[-W:]`) is correct by construction — it is exactly the
first-draft code path with `W` in place of `S` — and touches no cache
internals. But it re-projects `W` rows every step, so it changes the compute
profile as well as the memory one, and a sweep over it would not isolate the
memory variable.

**What the sweep would measure.** Under greedy verification a windowed drafter
still emits exactly the target's tokens — speculative decoding is exact, the
window only changes what the drafter guesses. So `output_ids` must stay
identical to full context at every `W`, and the quantities that move are
acceptance length, `max_steady_draft_cache_gb`, and the first draft call's
latency and peak. Planned points: `2k, 4k, 8k, 16k, 32k, full`.

## 7. Experiment

Four NVIDIA RTX A6000 (49140 MiB each), driver 565.57.01, torch 2.13.0+cu129,
transformers 5.16.1. All four were idle and unclaimed before the sweep started
(`nvidia-smi`: 2 MiB used, no compute processes); no other user's GPU was
touched.

`queue/run_selective2_sweep.sh 0 1 2 3` runs it. One job per card:

| GPU | jobs |
| --- | --- |
| 0 | qwen3-8b 4k, 8k, 16k, 32k |
| 1 | qwen3.5-9b 4k, 8k, 16k, 32k |
| 2 | qwen3-8b 64k, single card |
| 2 + 3 | qwen3.5-9b 64k, sharded, after the 8B 64k run frees the card |

Every run:

```
python -m dflash.cli benchmark transformers \
  --model-preset {qwen3-8b|qwen3.5-9b} --context-length {4096..65536} \
  --max-samples 32 --max-new-tokens 512 --reasoning off \
  --hidden-states selective --profile-draft-memory --record-dir record_selective
```

plus `--rope-scaling yarn` for qwen3-8b at 64k (target and draft are both
trained to 40960 positions, so 64k has to be interpolated rather than
extrapolated — the same flag the earlier 64k records used), and
`--device-map balanced` for qwen3.5-9b at 64k.

Dataset, sample count, seed, block size, `max_new_tokens` and reasoning setting
are unchanged from the selective sweep this supersedes: LongBench through
`dflash/context.py` with the same auto split and auto extend rules, 32 samples,
`torch.manual_seed(0)`, greedy. `--profile-draft-memory` is new to this sweep;
it costs two allocator reads per draft step and adds a `draft_activation_gb`
term to the (superseded) `memory_budget` block.

**Single GPU wherever it fit.** Nine of the ten configurations ran on one card.
Only qwen3.5-9b at 64k needed two, and its latency is therefore
pipeline-parallel: compare its acceptance and memory against the single-GPU
rows, not its timings.

## 8. Results

`--hidden-states selective`, 32 LongBench prompts per point, greedy, 512 max new
tokens. Every row reproduces the acceptance and peak of the selective sweep it
supersedes; only the instrumentation is new.

### 8.1 Headline

| model | ctx | GPUs | accept | peak GB | base peak GB | tpot ms | base tpot ms | speedup |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 4k | 1 | 3.11 | 18.34 | 18.19 | 16.9 | 30.8 | 1.82x |
| qwen3-8b | 8k | 1 | 3.09 | 19.47 | 19.16 | 21.8 | 33.4 | 1.53x |
| qwen3-8b | 16k | 1 | 2.50 | 21.73 | 21.10 | 42.8 | 38.5 | 0.90x |
| qwen3-8b | 32k | 1 | 1.48 | 26.23 | 24.98 | 117.5 | 44.7 | 0.38x |
| qwen3-8b | 64k | 1 | 2.55 | 35.25 | 32.75 | 128.3 | 68.7 | 0.54x |
| qwen3.5-9b | 4k | 1 | 8.98 | 21.82 | 21.60 | 16.9 | 39.8 | 2.35x |
| qwen3.5-9b | 8k | 1 | 7.07 | 24.49 | 24.05 | 21.5 | 39.7 | 1.84x |
| qwen3.5-9b | 16k | 1 | 8.83 | 29.84 | 28.97 | 17.2 | 38.6 | 2.25x |
| qwen3.5-9b | 32k | 1 | 7.06 | 40.54 | 38.79 | 26.4 | 36.6 | 1.38x |
| qwen3.5-9b | 64k | 2 | 6.91 | 62.45 | 58.45 | 35.4 | 46.4 | 1.31x |

`tpot` is aggregate: total decode time over total output tokens. The qwen3.5-9b
64k row is sharded over two GPUs, so its latency is pipeline-parallel and is not
comparable with the single-GPU rows; its acceptance and memory are.

### 8.2 Peak phase, and what is live there

**`prefill: target forward` sets the peak in all ten configurations** — both
presets, every length, sharded and not. The drafter never sets it. The split
below is read at one instant inside that phase:

| model | ctx | peak phase | peak GB | tgt W | tgt KV | drf W | drf KV | sel hid | ctx feat | unattributed |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 4k | prefill: target forward | 18.34 | 15.26 | 0.56 | 1.95 | 0.00 | 0.16 | 0.00 | 0.42 |
| qwen3-8b | 8k | prefill: target forward | 19.47 | 15.26 | 1.12 | 1.95 | 0.00 | 0.31 | 0.00 | 0.82 |
| qwen3-8b | 16k | prefill: target forward | 21.73 | 15.26 | 2.25 | 1.95 | 0.00 | 0.63 | 0.00 | 1.64 |
| qwen3-8b | 32k | prefill: target forward | 26.23 | 15.26 | 4.50 | 1.95 | 0.00 | 1.25 | 0.00 | 3.27 |
| qwen3-8b | 64k | prefill: target forward | 35.25 | 15.26 | 9.00 | 1.95 | 0.00 | 2.50 | 0.00 | 6.54 |
| qwen3.5-9b | 4k | prefill: target forward | 21.82 | 16.68 | 1.67 | 2.41 | 0.00 | 0.25 | 0.00 | 0.81 |
| qwen3.5-9b | 8k | prefill: target forward | 24.49 | 16.68 | 3.30 | 2.41 | 0.00 | 0.50 | 0.00 | 1.61 |
| qwen3.5-9b | 16k | prefill: target forward | 29.84 | 16.68 | 6.55 | 2.41 | 0.00 | 1.00 | 0.00 | 3.21 |
| qwen3.5-9b | 32k | prefill: target forward | 40.54 | 16.68 | 13.05 | 2.41 | 0.00 | 2.00 | 0.00 | 6.41 |
| qwen3.5-9b | 64k | prefill: target forward | 62.45 | 16.68 | 26.05 | 2.41 | 0.00 | 4.00 | 0.00 | 13.32 |

Two things to read off this table. **The draft KV cache and the injected context
feature are 0.00 in every row** — neither exists yet when the target's prefill
peaks, so the drafter's contribution to the *peak* is exactly its weights plus
the residual streams the tap captured: 2.11 GB at 4k rising to 4.45 GB at 64k on
the 8B, 2.66 to 6.41 on the 9B. Everything else in the row is the target's.

And `unattributed` — the target's own prefill activation — is the term that
grows fastest of anything measured here: **0.42 → 6.54 GB** on the 8B and
**0.81 → 13.32 GB** on the 9B, doubling with each doubling of context.

### 8.3 Phase profile

Peak allocation of each phase's largest occurrence.

| phase (peak GB) | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| prefill: target forward | 18.34 | 19.47 | 21.73 | 26.23 | 35.25 |
| prefill: context-feature build | 18.09 | 18.97 | 20.72 | 24.22 | 31.22 |
| prefill: first token | 17.94 | 18.66 | 20.09 | 22.97 | 28.72 |
| prefill: context-feature concat | 17.94 | 18.66 | 20.09 | 22.97 | 28.72 |
| prefill: rollback/crop | 17.94 | 18.66 | 20.09 | 22.97 | 28.72 |
| decode: first draft forward | 18.10 | 18.97 | 20.73 | 24.24 | 31.26 |
| decode: draft rollback/crop | 18.02 | 18.81 | 20.41 | 23.59 | 29.97 |
| decode: draft logits | 18.02 | 18.82 | 20.42 | 23.60 | 29.98 |
| decode: target verify | 18.09 | 18.95 | 20.67 | 24.11 | 30.98 |
| decode: verify rollback/crop | 18.03 | 18.82 | 20.42 | 23.61 | 29.98 |
| decode: context-feature build | 18.03 | 18.82 | 20.42 | 23.61 | 29.98 |
| decode: draft forward | 17.96 | 18.61 | 19.90 | 22.43 | 27.69 |

| phase (peak GB) | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| prefill: target forward | 21.82 | 24.49 | 29.84 | 40.54 | 62.45 |
| prefill: context-feature build | 21.26 | 23.39 | 27.64 | 36.14 | 53.15 |
| prefill: first token | 21.01 | 22.89 | 26.64 | 34.14 | 49.15 |
| prefill: context-feature concat | 21.01 | 22.89 | 26.64 | 34.14 | 49.15 |
| prefill: rollback/crop | 21.01 | 22.89 | 26.64 | 34.14 | 49.15 |
| decode: first draft forward | 21.21 | 23.27 | 27.40 | 35.66 | 52.19 |
| decode: draft rollback/crop | 21.11 | 23.08 | 27.02 | 34.89 | 50.65 |
| decode: draft logits | 21.12 | 23.09 | 27.03 | 34.91 | 50.67 |
| decode: target verify | 21.12 | 23.09 | 27.02 | 34.90 | 50.66 |
| decode: verify rollback/crop | 19.63 | 20.10 | 21.04 | 22.91 | 26.67 |
| decode: context-feature build | 19.63 | 20.10 | 21.04 | 22.91 | 26.67 |
| decode: draft forward | 19.46 | 19.65 | 20.06 | 20.93 | 22.69 |

The 9B table shows the gated delta-rule collapse directly: every phase from
`decode: verify rollback/crop` onward sits ~24 GB below the ones before it at
64k, because the first verify releases the prefill's recurrent buffers. Steady
decode on the 9B at 64k runs at 22.7 GB against a 62.5 GB peak — a 2.8x gap
between what the run must be provisioned for and what it spends its time using.

### 8.4 First draft call: the drafter's own prefill

| model | ctx | first ms | of decode | steady ms | first/steady | first borrowed GB | draft KV GB | fc+norm ms | ctx K/V ms | KV append ms | attn ms | head ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 4k | 17 | 7.4% | 5.87 | 3x | 0.08 | 0.08 | 6.7 | 3.5 | 0.5 | 0.3 | 1.7 |
| qwen3-8b | 8k | 29 | 8.7% | 6.07 | 5x | 0.16 | 0.16 | 12.8 | 6.9 | 1.0 | 0.5 | 1.7 |
| qwen3-8b | 16k | 52 | 8.7% | 7.16 | 7x | 0.32 | 0.31 | 24.9 | 11.8 | 2.1 | 1.0 | 1.7 |
| qwen3-8b | 32k | 99 | 13.7% | 9.26 | 11x | 0.64 | 0.63 | 49.0 | 23.5 | 4.2 | 1.9 | 1.7 |
| qwen3-8b | 64k | 202 | 13.2% | 14.46 | 14x | 1.29 | 1.25 | 103.7 | 47.5 | 8.7 | 4.0 | 1.7 |
| qwen3.5-9b | 4k | 25 | 6.1% | 8.19 | 3x | 0.10 | 0.09 | 10.3 | 4.2 | 0.7 | 2.2 | 2.8 |
| qwen3.5-9b | 8k | 43 | 12.0% | 8.36 | 5x | 0.19 | 0.19 | 19.8 | 8.2 | 1.4 | 4.0 | 2.8 |
| qwen3.5-9b | 16k | 81 | 16.2% | 8.47 | 10x | 0.38 | 0.38 | 39.6 | 14.3 | 2.6 | 9.3 | 2.8 |
| qwen3.5-9b | 32k | 148 | 37.9% | 8.98 | 16x | 0.77 | 0.75 | 73.8 | 26.9 | 4.9 | 17.9 | 2.8 |
| qwen3.5-9b | 64k | 291 | 49.7% | 10.70 | 27x | 1.54 | 1.50 | 140.4 | 56.1 | 10.6 | 37.9 | 0.1 |

Linear in context, as it must be: the call projects `S` rows through `fc`, then
`k_proj`/`v_proj`, then appends `S` entries to the draft KV cache. **`fc +
hidden_norm` is ~50% of it at every length on both presets** — the single
largest stage, ahead of the K/V projections it feeds. The output head is flat
(it only ever sees the block), and the drafter's own attention is flat on the 8B
but reaches 37.9 ms on the 9B, whose draft layers attend over the injected
context. (The 9B 64k `head ms` of 0.1 is a sharding artifact — see finding 9.)

`of decode` above is the mean over requests. The decode-weighted aggregate is a
different number, and the gap is itself informative:

| model | ctx | mean over requests | decode-weighted | mean output tokens | steady draft calls |
| --- | --- | --- | --- | --- | --- |
| qwen3-8b | 4k | 7.4% | 0.6% | 177.5 | 55.8 |
| qwen3-8b | 8k | 8.7% | 0.7% | 183.0 | 58.0 |
| qwen3-8b | 16k | 8.7% | 0.6% | 217.8 | 85.5 |
| qwen3-8b | 32k | 13.7% | 6.9% | 12.3 | 6.6 |
| qwen3-8b | 64k | 13.2% | 3.2% | 49.0 | 17.8 |
| qwen3.5-9b | 4k | 6.1% | 0.6% | 245.9 | 26.3 |
| qwen3.5-9b | 8k | 12.0% | 1.1% | 184.9 | 25.0 |
| qwen3.5-9b | 16k | 16.2% | 1.9% | 245.8 | 26.7 |
| qwen3.5-9b | 32k | 37.9% | 23.6% | 23.7 | 2.2 |
| qwen3.5-9b | 64k | 49.7% | 33.8% | 24.3 | 2.4 |

At 4k–16k the one-time cost amortises away: under 2% of decode however it is
weighted. Past that it stops amortising, because the requests that survive to
32k and 64k generate almost nothing — 12 to 24 tokens over **2.2 to 6.6 steady
draft calls per request**. On qwen3.5-9b at 64k the drafter's own prefill is a
third of all decode time by aggregate and half of it per request, not because
the call got slower relative to context but because there is nothing left to
amortise it over.

### 8.5 Steady state

| model | ctx | calls | forward ms | attn ms | KV append ms | ctx K/V ms | fc+norm ms | head ms | draft KV GB | borrowed GB |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 4k | 55.8 | 5.87 | 0.30 | 0.43 | 0.25 | 0.30 | 1.74 | 0.09 | 0.01 |
| qwen3-8b | 8k | 58.0 | 6.07 | 0.47 | 0.75 | 0.25 | 0.30 | 1.71 | 0.17 | 0.02 |
| qwen3-8b | 16k | 85.5 | 7.16 | 0.85 | 1.43 | 0.24 | 0.30 | 1.70 | 0.32 | 0.03 |
| qwen3-8b | 32k | 6.6 | 9.26 | 1.62 | 2.79 | 0.23 | 0.31 | 1.70 | 0.63 | 0.06 |
| qwen3-8b | 64k | 17.8 | 14.46 | 3.28 | 5.68 | 0.34 | 0.31 | 1.72 | 1.26 | 0.13 |
| qwen3.5-9b | 4k | 26.3 | 8.19 | 1.89 | 0.48 | 0.30 | 0.44 | 2.82 | 0.10 | 0.06 |
| qwen3.5-9b | 8k | 25.0 | 8.36 | 1.93 | 0.55 | 0.30 | 0.44 | 2.79 | 0.11 | 0.05 |
| qwen3.5-9b | 16k | 26.7 | 8.47 | 1.98 | 0.68 | 0.30 | 0.43 | 2.78 | 0.14 | 0.02 |
| qwen3.5-9b | 32k | 2.2 | 8.98 | 2.14 | 0.95 | 0.30 | 0.45 | 2.78 | 0.21 | 0.02 |
| qwen3.5-9b | 64k | 2.4 | 10.70 | 2.64 | 1.62 | 0.39 | 0.46 | 0.04 | 0.33 | 0.02 |

The steady call grows 5.9 → 14.4 ms on the 8B and 8.2 → 10.7 ms on the 9B, and
**all of the growth is attention plus KV append** — the two stages whose cost is
the length of the draft KV cache. `ctx K/V` and `fc+norm` stay flat at ~0.3 ms
because they only ever process the tokens the last verify accepted, and the
output head is constant. So the drafter's steady per-call cost is not "the
drafter is expensive at long context"; it is the drafter reading a KV cache that
is long because the prompt was.

### 8.6 Draft KV: what the first call builds vs what the run keeps

| model | ctx | first-call draft KV GB | steady draft KV GB | ratio |
|---|---|---|---|---|
| qwen3-8b | 4k | 0.078 | 0.088 | 0.89x |
| qwen3-8b | 8k | 0.157 | 0.166 | 0.94x |
| qwen3-8b | 16k | 0.313 | 0.322 | 0.97x |
| qwen3-8b | 32k | 0.625 | 0.626 | 1.00x |
| qwen3-8b | 64k | 1.250 | 1.260 | 0.99x |
| qwen3.5-9b | 4k | 0.094 | 0.096 | 0.98x |
| qwen3.5-9b | 8k | 0.188 | 0.112 | 1.68x |
| qwen3.5-9b | 16k | 0.375 | 0.143 | 2.62x |
| qwen3.5-9b | 32k | 0.750 | 0.206 | 3.65x |
| qwen3.5-9b | 64k | 1.500 | 0.331 | 4.54x |

On the 8B the two agree — its five draft layers are all full attention, so the
cache the first call builds is the cache the run keeps. On the 9B they diverge
by **4.5x at 64k**: five of six draft layers are `sliding_attention` with a 4096
window, so the first call's `S`-deep storage is released once the run reaches
steady state. PROFILING.md inferred that from the spread across samples; these
are the two quantities measured separately.

The consequence for the older records: `max_draft_cache_gb` is the *first-call*
figure, because every sample passes through it. It is a real peak, but it is not
what the drafter keeps.

### 8.7 DFlash-specific vs target-architecture memory

| model | ctx | draft W | draft KV | sel hidden | ctx feature | DFlash total | target KV | target prefill act | same, baseline |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 4k | 1.95 | 0.09 | 0.16 | 0.16 | 2.35 | 0.63 | 0.41 | 0.41 |
| qwen3-8b | 8k | 1.95 | 0.17 | 0.31 | 0.31 | 2.74 | 1.20 | 0.82 | 0.82 |
| qwen3-8b | 16k | 1.95 | 0.32 | 0.63 | 0.63 | 3.53 | 2.32 | 1.63 | 1.63 |
| qwen3-8b | 32k | 1.95 | 0.63 | 1.25 | 1.25 | 5.08 | 4.51 | 3.27 | 3.27 |
| qwen3-8b | 64k | 1.95 | 1.26 | 2.50 | 2.50 | 8.21 | 9.07 | 6.53 | 6.53 |
| qwen3.5-9b | 4k | 2.41 | 0.10 | 0.25 | 0.25 | 3.00 | 0.19 | 0.80 | 0.83 |
| qwen3.5-9b | 8k | 2.41 | 0.19 | 0.50 | 0.50 | 3.59 | 0.32 | 1.60 | 1.66 |
| qwen3.5-9b | 16k | 2.41 | 0.38 | 1.00 | 1.00 | 4.78 | 0.57 | 3.20 | 3.33 |
| qwen3.5-9b | 32k | 2.41 | 0.75 | 2.00 | 2.00 | 7.16 | 1.07 | 6.40 | 6.65 |
| qwen3.5-9b | 64k | 2.41 | 1.50 | 4.00 | 4.00 | 11.91 | 2.07 | 13.30 | 13.30 |

The last two columns are the same measurement taken with and without a drafter,
and they agree — which is the check that target prefill activation is the
target's cost, not DFlash's.

Read across: DFlash's own memory grows 2.35 → 8.21 GB (8B) and 3.00 → 11.91 GB
(9B), while the target's KV plus prefill activation grows 1.04 → 15.60 GB (8B)
and 0.99 → 15.37 GB (9B). At 4k the drafter is the larger of the two; by 64k it
is roughly half. And of DFlash's total, only the weights and the selected hidden
states are live when the run actually peaks.

## 9. What this found

**1. DFlash never sets the peak.** `prefill: target forward` is the peak phase in
all ten configurations. At that instant the drafter's KV cache and its injected
context feature do not exist yet, so DFlash contributes only its weights plus
the residual streams the tap captured — 4.45 GB of a 35.25 GB peak at 64k on the
8B, 6.41 of 62.45 on the 9B. Any statement of the form "DFlash costs X GB at the
peak" that includes the draft KV cache or the context feature is wrong.

**2. The old `transient_gb` was wrong in both directions, and the error grows
linearly with context.** ~2.4x too small on the 8B, ~2.4x too large on the 9B,
and §1 closes the identity to four decimals on all ten points. The 9B error
reaches 24 GB at 64k.

**3. `max_target_cache_gb` under-reports the target's KV on any hybrid model.**
It reads the cache once, when a sample ends. On Qwen3.5-9B the gated delta-rule
layers release a sequence-length prefill buffer on the first decode forward, so
the recorded figure is the post-release one — 2.07 GB at 64k against **26.05 GB
live at the peak**. The same 12x gap appears in the baseline, so it is not a
DFlash artifact. `phase_memory[peak_phase]["peak_components_gb"]["target_kv_bytes"]`
is the figure to use.

**4. The drafter's own prefill is the DFlash-specific term that scales, and
`fc + hidden_norm` is half of it.** The first draft call runs 17 → 202 ms (8B)
and 25 → 291 ms (9B) across 4k → 64k, against a steady call of 5.9 → 14.4 and
8.2 → 10.7 ms. Consistently ~50% of it is the drafter's `fc` and `hidden_norm`
over the full context feature — a single `(n_inj·d) x d` GEMM over `S` rows,
larger than the K/V projections it feeds. If one thing is worth optimising in
the drafter at long context, the stage timers say it is that projection.

**5. The one-time cost stops amortising exactly where context gets long.** At
4k-16k the first draft call is under 2% of decode however it is weighted. At 32k
and 64k the surviving requests generate 12-24 tokens over 2.2-6.6 steady draft
calls, and the same one-time call becomes 6.9% (8B 32k), 23.6% and **33.8%**
(9B 32k, 64k) of aggregate decode time. This is not the drafter getting slower
per unit of context; it is the denominator collapsing. Any long-context speedup
claim that assumes the drafter's prefill amortises needs to state the generation
length it is amortising over.

**6. The draft KV cache the 9B keeps is 4.5x smaller than the one it builds.**
First-call 1.50 GB against 0.33 GB steady at 64k, because five of six draft
layers are sliding-window. On the 8B, whose draft layers are all full attention,
the two agree. `max_draft_cache_gb` in every earlier record is the first-call
figure.

**7. The prompt-length context feature outlives its last use.** `target_hidden`
is only read by the first draft call, but it stays referenced through the draft
head, the first verify and the first context-feature rebuild, where reassignment
finally frees it: 2.50 GB on the 8B at 64k, 4.00 GB on the 9B. It does *not* set
the run's peak (the target prefill is 4 GB higher on the 8B, 10 GB on the 9B),
so freeing it would not lower the provisioning requirement — but it would cut
the first decode step's residency by that much. Left unchanged here so the
measurement stays comparable with the records it supersedes; it is a real,
uncontroversial saving whenever the code is next touched.

**8. Qwen3.5-9B's bottleneck is its own architecture, not DFlash.** Its prefill
activation is 0.81 → 13.32 GB against the 8B's 0.42 → 6.54 for a model of
comparable size — roughly 2x at every length — and its target KV peaks at 26 GB
where the 8B's peaks at 9. Both are measured identically in the baseline, both
come from the gated delta-rule layers and their fp32 fallback, and together they
are what forces a second GPU at 64k. DFlash's own footprint on the 9B (11.91 GB
at 64k, of which 6.41 is live at the peak) is the smaller half of that run's
non-weight memory. **The 9B's long-context memory behaviour must not be read as
DFlash intrinsic overhead.**

**9. Caveat: stage timings are not valid across a shard boundary.** The drafter
runs on one device; on the sharded 9B 64k run the target's `lm_head` runs on the
other, and the CUDA events bracketing it are recorded on the drafter's stream.
That row reports an output head of 0.06 ms against 2.8 ms on every single-GPU
9B row. Read `output_head` — and, more weakly, any stage timing — only from
single-GPU records. The memory figures on that row are unaffected.
