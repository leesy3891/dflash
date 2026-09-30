# PROFILING3 — B=1 architecture profiling on `main`

This document defines what the `main` branch measures, how each number is
defined, and what has actually been run so far. It is the contract the
`batch` branch's B=1 bridge point must match.

`PROFILING.md` and `PROFILING2.md` describe the earlier single-request and
selective-hidden sweeps. Nothing here is backward compatible with those
records. Existing records under `record/`, `record_batch_fla/`,
`record_refine*/` and `record_selective/` are untouched; everything new is
written under `record_arch_main/`.

---

## 1. Scope

Three research questions, all at batch size 1.

1. **Where the decode state goes when attention KV shrinks.** Qwen3-8B (full
   attention) against Qwen3.5-9B (GDN + attention): as the target's attention
   KV gets smaller, what share is taken by the DFlash draft KV, the target's
   selected hidden states, and the context-feature projection?
2. **Per-layer verify cost of a MoE target.** Qwen3.5-35B-A3B: route,
   dispatch, expert GEMM, GDN vs attention mixer, and the draft/target
   boundary.
3. **How the split moves with S and block width.** First-draft setup, steady
   draft, and verify — memory, compute and host-bound behaviour — plus speedup
   against AR.

### Explicitly out of scope on this branch

- `B > 1` static batching, per-row KV, continuous scheduling. Those are the
  `batch` branch's.
- Tensor or expert parallelism, vLLM/SGLang, cross-request prefix caching.
- **Four GPUs are not four requests.** The 35B target is layer-sharded across
  several cards; that is one request whose layers execute in sequence on
  different devices.

---

## 2. Environment, as found

Captured by `python -m dflash.arch manifest` into
`record_arch_main/manifest.json`. The facts below are what the machine
actually reported, not what was specified.

### 2.1 GPUs — one card is unusable

The requirement fixes **four** ~49 GB RTX A6000s across every model, AR and
DFlash run. The machine currently exposes **three**.

| nvidia-smi index | PCI | UUID | CUDA-openable |
|---|---|---|---|
| 0 | `0000:17:00.0` | `GPU-34cd448f-…` | **no** |
| 1 | `0000:65:00.0` | `ef8b9800-…` | yes → torch `cuda:0` |
| 2 | `0000:CA:00.0` | `8911bb88-…` | yes → torch `cuda:1` |
| 3 | `0000:E3:00.0` | `8c8a823b-…` | yes → torch `cuda:2` |

`torch.cuda.device_count()` reports 4, but `get_device_properties` on the
fourth raises `AssertionError: Invalid device id`. `nvidia-smi` shows that
same card with `ERR!` against fan, power and ECC. It is a driver-level
failure, not a configuration one, and recovering it needs a GPU reset this
process cannot and should not perform on a shared machine.

Consequences, all recorded rather than worked around:

- Total usable memory is 3 × 47.4 GiB ≈ 142 GiB, not ~190 GiB.
- Layer sharding is **3-way**, not 4-way. `placement.split_layers` takes the
  device list from `env.usable_devices()`, so the map follows the hardware.
- Comparisons stay internally valid: every model, AR and DFlash alike, uses
  the same three cards and the same contiguous-block map. What is lost is
  comparability with any earlier or later 4-GPU run.
- `env.device_health()` sets `degraded: true`, and every record carries it.

All three usable cards have mutual peer access. `nvidia-smi topo -m` is not
available on this driver build (565.57.01); the topology field is `NA` with
that reason, and no NVLink claim is made from it.

**This deviation needs a decision.** Either the fourth card is recovered and
the sweep is rerun on four, or the sweep is declared 3-GPU. It is not
something the code can resolve.

### 2.2 Kernel backends — GDN runs the pure-torch reference

`python -c "from dflash.arch import backend; backend.probe()"` resolves each
entry point through its `use_kernel_func_from_hub_with_fallback` wrapper by
walking the closure, rather than asking whether a package is importable.

| entry point | resolved |
|---|---|
| `chunk_gated_delta_rule` | `torch_reference` |
| `recurrent_gated_delta_rule` | `torch_reference` |
| `causal_conv1d_fn` | `torch_reference` |
| `causal_conv1d_update` | `torch_reference` |

**Updated 2026-09-29: FLA 0.5.2 was installed mid-session.** The table above
is the stack §5.5–§5.7 were measured on. The current stack resolves the two
delta-rule entry points to `fla` and leaves the two conv entry points on
`torch_reference`, because `causal_conv1d` is still absent — so the backend
is **mixed** and `homogeneous` is `False`. See §5.9 for the measured effect,
which includes a change in acceptance, and for how the axis should be handled.

`flash_attn` is not installed; attention runs SDPA.

Numbers taken under `torch_reference` and under `fla` are different
populations. `backend.probe()["homogeneous"]` is checked before any
aggregation, and `distinct_backends` is recorded on every run.

### 2.3 Software

torch 2.13.0+cu129 · transformers 5.16.1 · accelerate 1.14.0 · triton 3.7.1 ·
CUDA 12.9 · driver 565.57.01 · python 3.11.16. BF16, greedy, SDPA attention.

---

## 3. Measurement definitions

These are the definitions the `batch` branch must match at its B=1 point.

### 3.1 Block width

`block_size` is the **target verify token width including the anchor**. Block
4 / 8 / 16 propose 3 / 7 / 15 draft tokens. Records keep four separate
counts — nominal width, actual verify tokens, active tokens, padding tokens —
because at the end of a request the last block is narrower than nominal.

### 3.2 The AR baseline

AR is measured in a **process that never constructs the drafter**
(`dflash.arch.ar`). A `block_size=1` run through `dflash_generate` is not an
AR memory baseline: the drafter's weights and cache object are on the card.

AR runs under **two cache policies**, because the choice is not neutral on a
hybrid target:

- **`native`** — plain `DynamicCache`. What AR decoding actually costs, and
  the denominator for a reported speedup.
- **`recording`** — `activate_past_recording()`, matching DFlash exactly.

The distinction matters because `Qwen3_5GatedDeltaNet.forward` takes the
fused single-token path `causal_conv1d_update` only when
`seq_len == 1 and not record_past`. With recording on, every decode step goes
through the general `causal_conv1d_fn` path instead.

**The recording arm must also crop every step**, as `dflash_generate` does.
Without the crop the conv state grows by a column per token and the arm
measures unbounded buffer growth rather than a cache policy — which is the
error §5.8 corrects. `ar.generate` ties `crop_each_step` to the policy by
default, and "native + crop" is not offered because `crop` raises on a cache
without `record_past`: the crop and the recording are one feature.

Measured correctly, the gap between the two policies is **~2%** (§5.8), not
the ~50% an uncropped baseline reports. On a full-attention target it is nil.

### 3.3 Latency

- **TTFT** — from prepared input IDs to the first token being available. The
  drafter's first prefill sits *after* the first token in the B=1
  implementation and is **not** added to TTFT.
- **First speculative commit latency** — to the first draft+verify result.
- **TPOT** — decode wall time over produced tokens, from each request's own
  first and last output timestamps. Reported separately from E2E, decode wall
  time and speedup.
- **Acceptance** — `accepted draft tokens / proposed draft tokens`. Committed
  per step (accepted + the correction or bonus token) is recorded separately
  and clipped to what the output actually took.

### 3.4 Output policy

Two policies, never averaged together:

- **`shape_controlled`** — EOS ignored, exactly N tokens. Token count is a
  controlled variable.
- **`natural`** — stops at EOS.

Acceptance measured past a natural EOS is **not** mixed into natural-output
acceptance: past EOS the drafter is proposing continuations of text the model
considers finished.

### 3.5 Memory components

`_cache_bytes` in `dflash/model.py` sums every CUDA storage a cache reaches.
On a hybrid target that number is **not** an attention KV figure — it also
contains GDN recurrent states, conv states, and the conv recording buffer.
`dflash/arch/statemem.py` splits the same walk:

| component | scales with | what it is |
|---|---|---|
| `attention_kv` | O(S) | keys/values of full- and sliding-attention layers |
| `gdn_recurrent` | O(1) | GDN recurrent states |
| `gdn_conv_working` | O(1) | last `conv_kernel_size` columns — what the next forward reads |
| `gdn_conv_recording` | O(S) during prefill | history kept only so `crop` can roll back; DFlash's machinery, not the target's decode state |

Two sizes per component:

- **`logical_bytes`** — `numel × element_size`.
- **`storage_bytes`** — the allocation actually kept alive.

They differ whenever a tensor is a view into a larger buffer, which is exactly
what `crop` produces: it slices rather than copies, so a cropped cache still
pins the uncropped allocation until the next `cat` replaces it. **The real
high-water mark follows `storage_bytes`.** Each distinct allocation is charged
to exactly one component; anything reached twice is listed under `aliases`
rather than double-counted.

### 3.6 Peaks

Total GPU peak is the largest **simultaneous** residency, resolved by
`PeakTracker` cutting the run into short intervals. The sum of per-device
maxima is reported beside it as a separate field, never as the total. Per-GPU
peak, simultaneous aggregate and naive sum-of-maxima are three distinct
columns.

### 3.7 Time attribution

- `duration_s` is **exclusive** (children removed); `inclusive_s` is the full
  span. They are never added together.
- Each kernel is attributed to the most specific operation, once.
- Concurrent GPU kernel time does not sum to wall time. Kernel interval
  **union** is stored separately, and unattributed wall residual is reported
  rather than distributed.
- Per-GPU kernel busy, copy activity and overlap, and global no-work gaps are
  separate fields.
- Elapsed time is never computed from two CUDA events on different devices.

### 3.8 Gap and bottleneck classification

A gap is classified only with evidence, into `structural_shard_wait`,
`target_draft_dependency`, `host_launch_or_sync`, `transfer`, `profiler`, or
`unknown`.

**A device with no kernels while another device runs its shard is
`structural_shard_wait`** — the structure of layer sharding at batch 1. It is
not a memory stall and not DFlash overhead.

A gap with no kernel on the GPU and a DRAM stall *inside* a kernel are
different phenomena and are never merged. Memory capacity (OOM), memory
bandwidth and compute throughput are three separate verdicts.

Bottleneck values: `bandwidth`, `compute`, `host_launch`, `transfer`,
`dependency`, `capacity`, `mixed`, `undetermined`. `undetermined` is a real
answer and the default. `nvidia-smi` utilisation is never used as evidence of
compute-bound behaviour. Where an Nsight counter is unavailable, the field is
`NA` with a reason and no bottleneck is asserted from scaling alone.

### 3.9 Passes

| pass | what it may report | notes |
|---|---|---|
| `perf` | latency, TPOT, speedup | no probe, no per-layer hooks; ≥3 repeats after warmup, median + spread |
| `memory` | component splits, phase peaks | probe installed; latency recorded but marked, never mixed into `perf` |
| `trace` | Nsight Systems timeline | NVTX `phase → layer → operator`; diagnostic only |
| `counters` | Nsight Compute counters | replay serialises and perturbs cache state; diagnostic only |

The perturbation between `perf` and `memory` is **measured** at the
representative condition, not assumed. `run.compare_passes` checks that
acceptance and output token counts are identical across passes; if they are
not, the instrumentation is not passive and nothing downstream is trusted.

No CUDA synchronize is inserted per layer or per phase on the `perf` path.

---

## 4. Sweep plan

Axes are swept **one at a time** around a representative point. No Cartesian
product. Ten conditions per model.

| sweep | varies | held at |
|---|---|---|
| `sequence` | S ∈ {4k, 8k, 16k, 32k, 64k} | block 16, 256 out |
| `block` | block ∈ {4, 8, 16} | S = 32k, 256 out |
| `output` | out ∈ {256, 1024, 4096} | S = 32k, block 16 |
| `output_policy` | natural vs shape-controlled | S = 32k, block 16, 4096 out |

`python -m dflash.arch plan` prints it and writes `record_arch_main/plan.json`
without loading weights.

### Context-limit flags

Recorded per condition, not enforced. Running past a stated range produces
numbers; what it must not do is have them read as comparable.

| model | target RoPE range | drafter trained context | 64k condition |
|---|---|---|---|
| qwen3-8b | 40 960 | not stated | `beyond_target_rope_range`, `beyond_draft_rope_range` |
| qwen3.5-9b | 262 144 | not stated | within all stated ranges |
| qwen3.5-35b-a3b | 262 144 | 40 960 (model card) | `beyond_draft_trained_context` |

`max_position_embeddings` is a RoPE range, not a trained length; both are
recorded.

---

## 5. What has been measured

Everything in this section was run on this machine. Section 6 lists what is
implemented but not yet executed.

### 5.1 Model pairings — all three validate

`models.check_pairing` cross-checks vocabulary, `num_target_layers`,
`target_layer_ids` against the layer count, and the mask token against the
vocabulary, from the two configs alone.

| pair | vocab | layers | tapped layer IDs | context-feature dim | compatible |
|---|---|---|---|---|---|
| Qwen3-8B ↔ Qwen3-8B-DFlash-b16 | 151 936 | 36 | 1,9,17,25,33 | 20 480 | ✅ |
| Qwen3.5-9B ↔ Qwen3.5-9B-DFlash | 248 320 | 32 | 1,5,9,13,17,21,25,29 | 32 768 | ✅ |
| Qwen3.5-35B-A3B ↔ Qwen3.5-35B-A3B-DFlash | 248 320 | 40 | 1,6,11,16,22,27,32,37 | 16 384 | ✅ |

The 35B drafter (candidate) declares `architectures: ["DFlashDraftModel"]`,
`block_size: 16`, six layers (5 sliding + 1 full attention), hidden size 2048.
It loads through the existing `DFlashDraftModel` path with no loader change.

### 5.2 35B target shard/index integrity — intact

`~/models/Qwen3.5-35B-A3B`: 1811 tensors across 14 shards, every file named in
`model.safetensors.index.json` present on disk, zero missing. Index
`total_size` 71 903 655 008 B against 71 903 878 016 B on disk — the 223 KB
excess is safetensors header overhead, as expected.

Note the shard filenames are non-standard
(`model.safetensors-00001-of-00014.safetensors` rather than
`model-00001-of-00014.safetensors`), but the index and the files agree, so
loading resolves correctly.

### 5.3 Layer taxonomy — read from instantiated modules

Derived on the `meta` device, so no weights are loaded.
`declared_vs_built_mismatch` is empty for all three: `layer_types` in the
config agrees with the modules that were actually built.

| model | layers | hidden | mixer | FFN | stack path |
|---|---|---|---|---|---|
| qwen3-8b | 36 | 4096 | 36 full attention | 36 dense | `model.layers` |
| qwen3.5-9b | 32 | 4096 | 24 GDN + 8 full | 32 dense | `model.language_model.layers` |
| qwen3.5-35b-a3b | 40 | 2048 | 30 GDN + 10 full | 40 MoE+shared | `model.language_model.layers` |

35B MoE facts, read off the modules: 256 experts, top-8, expert intermediate
512, shared expert present, expert weights stored fused as 3-D parameters.

### 5.4 Analytic decode state — the setup for research question 1

Closed form from the instantiated shapes, in
`record_arch_main/taxonomy_analytic.json`. The measured component split is
checked against this.

| model | attention KV / token | at S=32k | GDN state (fixed) |
|---|---|---|---|
| qwen3-8b | 147 456 B | **4.50 GiB** | 0 |
| qwen3.5-9b | 32 768 B | **1.00 GiB** | 49.5 MiB |
| qwen3.5-35b-a3b | 20 480 B | **0.62 GiB** | 61.9 MiB |

The hybrid target's attention KV is **4.5×** smaller than the dense one's per
token; the MoE hybrid's is **7.2×** smaller. The GDN state is constant in S —
about 50–62 MiB — so it is negligible at 32k but is roughly 28% of the 9B's
decode state at S=4k. That crossover is what the sequence sweep measures the
draft KV, selected hidden and context feature against.

Verified against the running 9B: `attention_kv_bytes_per_token = 32768`,
`gdn_recurrent = 50 331 648`, `gdn_conv = 1 572 864` — the analytic and
instantiated figures agree exactly.

### 5.5 GDN rollback is **not** lossless — the recurrent state never rolls back

`python -m dflash.arch rollback qwen3.5-9b` — Qwen3.5-9B, 512-token prompt,
block 16, greedy, single device. Record:
`record_arch_main/rollback/qwen3.5-9b.json`.

The check runs one continuation two ways and compares both the next-token
logits and every cache tensor, per layer and per mixer. Two references are
used because a hybrid target separates two different effects:

- **`segmented`** — prompt prefilled, then the accepted tokens fed as their
  own forward: the same call segmentation as the rollback path, minus the
  rejected suffix. A difference here **is** the rollback.
- **`one_shot`** — prompt and accepted tokens prefilled together. This differs
  from `segmented` on a GDN target even with no rollback involved, because the
  chunked delta-rule kernel does not reproduce across a call boundary — the
  same effect `prefill_chunking_safe()` refuses prefill chunking for.
  Comparing only against this would charge that to the rollback.

**Result at accepted = 0** (whole block rejected), against `segmented`:

```
full_attention   tensors=16   equal=16   differs=0    max|diff|=0
gdn              tensors=48   equal=24   differs=24   max|diff|=12.1085
differs: ['recurrent'] | matches: ['conv', 'keys', 'values']
next-token argmax match : False
```

The isolation is complete:

- **Attention keys and values roll back exactly** — 16/16 equal, bitwise.
- **GDN conv states roll back exactly** — 24/24 equal.
- **GDN recurrent states never roll back** — 24/24 differ, one per GDN layer.
- The very first fully-rejected block **already changes the next-token
  argmax**.

This is the behaviour the source predicts.
`LinearAttentionCacheLayerMixin.crop` trims `conv_states` and returns; it
never touches `recurrent_states`. `update_recurrent_state` writes in place
with `copy_`, so the pre-verify value is gone by the time the rejection is
known. `_crop_to` in `dflash/model.py` calls exactly that `crop`.

At accepted ∈ {1, 4, 15} the attention tensors also begin to differ, but by
`max|diff|` 0.22–0.5 against `segmented` versus 22.6 against `one_shot` —
i.e. the large attention divergence belongs to prefill segmentation, not to
the rollback, which is precisely what the dual reference was built to
separate. The recurrent state differs at every acceptance level.

**Consequence.** DFlash on a GDN target is not lossless. Verified output can
diverge from what the target would have produced, independent of acceptance
rate, and the error compounds across verify steps because the corrupted state
is carried forward. Reported acceptance on 9B and 35B is measured against a
target whose state is already wrong.

*(This is the same defect recorded earlier for Qwen3.5-9B; this run isolates
it to the recurrent state specifically, exonerates the conv state, and shows
it changes output at zero acceptance.)*

### 5.5.1 Qwen3-8B control — full attention **is** lossless

`python -m dflash.arch rollback qwen3-8b --single-device --prompt-tokens 512`.
Record: `record_arch_main/qwen3-8b/rollback.json`.

**At accepted = 0**, against `segmented`:

```
full_attention   tensors=72   equal=72   differs=0    max|diff|=0
differs: none | matches: ['keys', 'values']
next-token argmax match : True
VERDICT (vs segmented): lossless
```

All 72 tensors (keys and values across 36 layers) are **bitwise identical**.
Rolling back a fully rejected block on a full-attention target restores the
state exactly, as slicing a KV cache should.

At accepted ∈ {1, 4, 15} the keys and values differ by at most 0.5 absolute
against `segmented` — about 4 ULP in bf16 at the ~25 magnitude these tensors
carry. This is SDPA kernel dispatch: a 16-query attention call and a 4-query
one select different tilings and round differently, and the difference
compounds across 36 layers. It is not a state error. The next-token argmax
matches at every acceptance level and the logits differ by 0.08–0.14.

**The contrast is the result:**

| | Qwen3-8B (full attention) | Qwen3.5-9B (GDN + attention) |
|---|---|---|
| state at accepted=0 | bitwise identical | recurrent state differs, `max\|diff\|` 12.1 |
| logits `max\|diff\|` at accepted=0 | 0 | 4.07 |
| next-token argmax at accepted=0 | matches | **differs** |
| verdict | lossless | **not lossless** |

Speculative decoding is exact on the dense target and lossy on the hybrid one,
and the loss is attributable to one tensor kind that `crop` does not touch.

### 5.5.2 Why the recurrent state cannot be cropped

This is a structural property of the layer, not a missing line in
`cache_utils.py`, so it is worth stating precisely.

#### The dispatch path, exactly

`dflash/model.py::_crop_to` calls `cache.crop(-remove)`, which fans out to
each cache layer's own `crop`. For Qwen3.5-9B the cache holds two layer
classes, chosen from `layer_types`:

| `layer_types` entry | cache layer class | tensors held |
|---|---|---|
| `full_attention` (8×) | `DynamicLayer` | `keys`, `values` |
| `linear_attention` (24×) | `LinearAttentionLayer` | `conv_states`, `recurrent_states` |

`DynamicLayer.crop` slices `keys` and `values` on the sequence axis.
`LinearAttentionCacheLayerMixin.crop` slices `conv_states` on the column axis
and **returns**. It contains no reference to `recurrent_states` at all. The
hybrid class that carries both halves,
`LinearAttentionAndFullAttentionLayer.crop`, calls both parents — and so also
never touches the recurrent state.

Meanwhile `update_recurrent_state` writes in place:

```python
self.recurrent_states[state_idx].copy_(recurrent_states)
```

so by the time the verify's logits come back and the rejection is known, the
pre-verify value has already been overwritten. There is nothing left to
restore from. That is why §5.5 finds `recurrent` as the **only** differing
tensor kind at zero acceptance: `keys`, `values` and `conv` are all sliced
back correctly, and the recurrent state simply is not.

**So yes — the original code, run as-is, is affected.** No flag, no
configuration and no environment changes this; `_crop_to` is on the only
rollback path `dflash_generate` has, and every rejected token on a Qwen3.5
target stays folded into all 24 (9B) or 30 (35B) GDN layers.

#### Why slicing cannot work in principle

The gated delta rule's per-token update, read off
`torch_recurrent_gated_delta_rule`, is

```
S_t = g_t · (I − β_t k_t k_tᵀ) · S_{t−1}  +  β_t k_t v_tᵀ
```

with `k_t` L2-normalised, `β_t = sigmoid(·) ∈ (0,1)` and
`g_t = exp(−A·softplus(·)) ∈ (0,1]`.

Compare the three kinds of state a decoder layer can carry:

| state | update | rollback |
|---|---|---|
| attention KV | `S_t = S_{t−1} ⊕ (k_t, v_t)` — an **append**, indexed by token | drop the last *n* entries. Exact, and the cost does not depend on *n* |
| conv window | last `k` columns of `[x_1 … x_t]` — a **windowed append** | slice to an earlier window — exact **only if the history was kept**, which is exactly what `activate_past_recording` is for |
| GDN recurrent | the equation above — a **fold** | *not recoverable from `S_t` alone* |

The first two are token-indexed accumulations: token *t*'s contribution
occupies its own slot, so removing it is addressing. The recurrent state is a
fixed-size summary in which every token is **superposed**. There is no slot
for token *t* to delete.

Inverting the recurrence is algebraically possible but numerically unsafe.
Since `‖k_t‖ = 1`, Sherman–Morrison gives

```
S_{t−1} = (1/g_t) · (I + β_t/(1−β_t) · k_t k_tᵀ) · (S_t − β_t k_t v_tᵀ)
```

Both correction factors diverge exactly where the layer is doing its job:
`1/(1−β_t) → ∞` as the delta rule writes hard (`β_t → 1`), and `1/g_t > 1`
compounds once per undone token, so undoing a 15-token rejected suffix
multiplies the error by `∏ 1/g_t`. The gate exists **to forget**; running it
backwards amplifies whatever it forgot.

#### The generalisation

The discriminating property is not "uses a single hidden state". It is
**whether the state is a token-indexed accumulation or a fixed-size fold**.

Any layer whose state is a *fixed-size* function of an unbounded history has
this problem — Mamba/Mamba2 SSM states, RWKV, RetNet, linear attention in
general, and GDN here. Any layer whose state grows with the sequence and
keeps tokens addressable — softmax attention, and a convolution window with
its history retained — does not.

Note the trade this implies. §5.4 measures the GDN state at a constant
49.5 MiB on 9B against 4.5 GiB of attention KV at 32k. That constancy is the
architecture's whole memory advantage, and it is *the same property* that
makes the state unrollbackable: a summary small enough not to grow with *S*
is a summary with no per-token slot to remove. **Cheap state and cheap
rollback are in direct tension**, and any speculative scheme on a hybrid
target has to pay for one of them.

#### What a correct rollback costs

Three options, and the cheapest is cheap:

1. **Snapshot and replay.** Save `S` before the verify, restore it on
   rejection, re-run the accepted prefix through the GDN layers only. Memory
   is one state copy — 49.5 MiB on 9B, **independent of S** — and the
   recompute is at most `block_size − 1` tokens. This is what the `batch`
   branch implemented in `TargetCache.commit`.
2. **Emit per-position intermediate states** from the kernel, then select the
   accepted one. Bounded by block width rather than by *S*: 16 × 2 MiB × 24
   layers ≈ 790 MiB on 9B at block 16. No recompute, more memory.
3. **Invert the recurrence.** Closed form above; rejected on the numerical
   grounds above, not on cost.

Option 1 is the one that fits this branch's B=1 path. It is **not implemented
in `dflash_generate`** — only in the batch branch — so every 9B and 35B
acceptance number on `main` remains measured against a corrupted target
state.

### 5.6 End-to-end sweep point — Qwen3.5-9B, S=4096

`python -m dflash.arch sweep qwen3.5-9b --single-device --sweeps sequence
--input-tokens 4096 --repeats 2 --no-natural`. Record:
`record_arch_main/qwen3.5-9b/sweep_20260929-060246.json`.

One condition, run to validate the whole pipeline. Single device (`cuda:0`),
not sharded. Prompt: LongBench `multi_news`, split `e`, **not** composed, **not**
truncated, 4096 tokens exactly, `token_hash c02da7d463080bf6`. Block 16, 256
output tokens, shape-controlled, greedy, `torch_reference` GDN throughout.

#### Measured under `torch_reference` GDN, with a flawed AR baseline

**Superseded — see §5.8.** This point was taken before FLA was installed, and
its AR `recording` arm never called `crop`, so it measured unbounded conv
buffer growth rather than DFlash's cache policy. The component and
pass-perturbation results below stand; the speedup numbers do not.

| run | TPOT | vs DFlash |
|---|---|---|
| AR `native` | 38.73 ms/token | 2.119× |
| AR `recording` (uncropped — invalid) | 58.52 ms/token | 3.202× |
| DFlash `perf` | 18.28 ms/token | — |

Acceptance 0.485 (225 accepted of 464 proposed), 256 tokens committed.

#### The memory probe is passive

| | `perf` | `memory` |
|---|---|---|
| TPOT | 18.278 ms | 18.511 ms |
| accepted tokens | 225 | 225 |
| output tokens | 256 | 256 |

TPOT perturbation **+1.27%**; acceptance and output length **identical**. The
probe changes the clock slightly and does not change the run, which is the
condition §3.9 requires before any component figure is trusted.

#### Component split — the answer to research question 1

All figures MiB, `storage_bytes`, read at the peak interval of each phase.

| moment | attn KV | GDN state | conv recording | draft KV | sel. hidden | ctx feature | draftKV / attnKV | draftKV / decode state |
|---|---|---|---|---|---|---|---|---|
| prefill end | 128.0 | 49.5 | **1534.5** | 0.0 | 0.0 | 256.0 | — | — |
| first draft | 128.0 | 49.5 | **1534.5** | 96.4 | 0.0 | 256.0 | 0.753 | 0.543 |
| steady decode | 135.5 | 49.5 | 6.0 | 97.6 | 1.0 | 1.0 | 0.720 | 0.527 |

Measured against analytic (§5.4): attention KV 134 217 728 B and GDN state
51 904 512 B are **exact** matches. The component walk and the closed form
agree to the byte.

Three things this says:

1. **The draft KV is not a rounding term.** At 96–98 MiB it is **75% of the
   target's entire attention KV** and **54% of everything the target carries
   between decode steps**. On a dense target with 4.5× more KV per token the
   same drafter would be a much smaller fraction — which is exactly the
   comparison the 8B arm of the sequence sweep is for.
2. **The context feature is the largest DFlash term during setup** — 256 MiB,
   **twice** the target's attention KV — and it is held from the end of
   prefill through the entire first draft call before collapsing to 1 MiB in
   steady state. The selected hidden states are another 256 MiB, live during
   the prefill target forward and released as soon as the feature is built.
3. **The conv recording buffer dominates everything.** See below.

#### The conv recording buffer is 1.5 GiB, and `crop` does not free it

At S=4096 the GDN conv recording buffer is **1534.5 MiB — twelve times the
target's attention KV and thirty-one times the GDN state it belongs to.** It
is `8192 conv_dim × 4096 columns × 2 B × 24 layers = 1 610 612 736 B`, i.e.
the full prompt held column by column.

It is also held far longer than it needs to be. Tracing it across phases:

```
prefill: target forward          1534.5 MiB
prefill: context-feature build   1534.5
prefill: first token             1534.5
prefill: rollback/crop           1534.5   <- crop() runs here
decode: first draft forward      1534.5
decode: draft rollback/crop      1534.5
decode: draft logits             1534.5
decode: target verify               6.0   <- actually freed here
```

`_crop_to` calls `crop(0)`, and
`LinearAttentionCacheLayerMixin.crop` does
`self.conv_states[i] = self.conv_states[i][..., -conv_kernel_size:]` — a
**view**, not a copy. The 4-column view keeps the 4096-column allocation
pinned. It is released only when the next `update_conv_state` runs
`torch.cat` and rebinds the slot, which is the first target verify — after the
drafter's whole O(S) setup has already run alongside it.

This is precisely what the `logical_bytes` / `storage_bytes` split in §3.5
exists to catch. A logical-size accounting reports 6 MiB at
`prefill: rollback/crop` and misses 1.5 GiB.

**Scaling.** The buffer is O(S). At 64k it would be roughly **24 GiB** on this
target, on top of a 2 GiB attention KV — which would make the rollback
bookkeeping, not the KV cache, the term that decides whether a 64k request
fits. The sequence sweep is what will confirm or refute that; it has not been
run.

### 5.7 `DynamicCache` copies the whole KV cache every step

Read from `transformers/cache_utils.py`, `DynamicLayer.update`:

```python
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
```

Every verify reallocates and copies the entire attention KV cache. At S=64k on
Qwen3-8B that is a 9 GiB read plus a 9 GiB write per verify step, and a
transient peak of roughly twice the cache while both allocations are live.
`crop` then slices rather than copies, so a cropped cache keeps the uncropped
allocation pinned until the next `cat` replaces it — which is why
`statemem` reports `storage_bytes` alongside `logical_bytes`.

Quantified measurement of this is in section 6, not here.

---

### 5.8 The AR baseline, corrected — and where its time actually goes

`python queue/arch_ar_bench.py qwen3.5-9b --input-tokens 4096 --new-tokens 128
--repeats 3 [--no-fla]`. Records:
`record_arch_main/qwen3.5-9b/ar_bench_{fla,torch}.json`.

Every variant below runs on the same loaded model and the same prompt
(LongBench `multi_news`, 4096 tokens, hash `c02da7d463080bf6`), and **all of
them emit byte-identical tokens**, so the rows differ only in overhead.
ms/token over 128 greedy tokens, median of 3.

| variant | FLA | torch ref | what it adds |
|---|---|---|---|
| `ar_native_syncfree` | **31.02** | **34.84** | nothing — the floor |
| `ar_recording_crop` | 31.63 | 35.44 | DFlash's cache policy, cropped each step |
| `ar_native_perstep_sync` | 33.96 | 37.82 | one D2H copy of the token per step |
| `hf_generate` | 40.06 | 47.66 | HuggingFace's own generation loop |
| `dflash_b1_nostats` | 40.26 | 48.12 | `dflash_generate` at `block_size=1` |
| `dflash_b1_stats` | 40.97 | 48.89 | + `PeakTracker` / `PhaseMemory` |
| `ar_recording_nocrop` | 55.73 | 60.10 | recording **without** the crop — see below |

#### Correction: the recording policy costs ~2%, not 51%

§5.6 reported that `activate_past_recording()` made AR **51% slower**. That
was wrong, and the error was in the baseline, not the model.

The arch AR `recording` arm never called `crop`. With `record_past` on,
`update_conv_state` keeps the concatenated history instead of the last
`conv_kernel_size` columns, so an uncropped loop grows its conv state by one
column per token and gets steadily slower — visible above as
`ar_recording_nocrop` at +80%, and in that row's enormous run-to-run spread
(stdev 1.57 s on a 7.08 s median) as the cost climbs within each run.

`dflash_generate` crops after every verify, which bounds the buffer at
`conv_kernel_size`. Measured with the crop included, the recording policy
costs **+2.0% (FLA) / +1.7% (torch)** — `causal_conv1d_fn` over 5 columns
instead of the fused single-token update, which is nearly free.

`ar.generate` now crops whenever the policy is `recording`
(`crop_each_step`, defaulting to the policy), and "native + crop" is
deliberately not offered: `crop` raises on a cache without `record_past`, so
the crop cannot be priced apart from the recording it requires.

#### What the old `block_size=1` baseline was actually paying

`dflash_b1_nostats` is **+29.8% (FLA) / +38.1% (torch)** above a tight AR
loop. Attributing it:

- **Not the recording policy** — that is the 2% above.
- **~9.5%** is the per-step device-to-host sync
  (`ar_native_perstep_sync` − `ar_native_syncfree`).
- **The rest is generic per-step loop overhead**, not DFlash's: HuggingFace's
  own `generate` lands at 40.06 ms against `dflash_b1`'s 40.26, a difference
  of **0.5%**. Python dispatch, cache bookkeeping and kernel-launch latency
  per decode step dominate, and `dflash_generate` at width 1 is no worse than
  the reference implementation of the same loop.
- **`PeakTracker` + `PhaseMemory` cost 1.8%** (`dflash_b1_stats` vs
  `nostats`), which is why they live on the memory pass and not the perf one.

The per-step sync mattered for a second reason: DFlash synchronises roughly
once per *verify step*, i.e. about once per 7.7 committed tokens at the
measured acceptance. An AR loop that synced once per *token* paid that cost
~8× more often than the thing it was the denominator for, and all of it
landed in the reported speedup. `ar.generate` is now sync-free under
`ignore_eos`; under a natural stop it checks EOS every `eos_check_interval`
(default 8) steps, trims any overrun and reports it as `eos_overrun_tokens`,
with latency divided by tokens *executed* so the overrun is charged rather
than hidden.

#### Corrected speedup at the §5.6 condition

Re-run with FLA and the fixed baseline
(`record_arch_main/qwen3.5-9b/sweep_20260929-081633.json`), S=4096, block 16,
256 tokens, 3 repeats:

| | TPOT | speedup |
|---|---|---|
| AR `native` | 31.74 ms | 4.038× |
| AR `recording` | 32.14 ms | 4.089× |
| DFlash | **7.86 ms** | — |

E2E 3.24× / 3.28×. The two AR policies now agree to **1.3%**, so the choice of
baseline no longer moves the headline — which is the point of the fix. Pass
perturbation +1.81% with acceptance identical across passes.

### 5.9 FLA changes latency **and** acceptance

FLA 0.5.2 is installed. `backend.probe()` shows the resolution is **mixed**:

| entry point | resolves to |
|---|---|
| `chunk_gated_delta_rule` | **`fla`** |
| `recurrent_gated_delta_rule` | **`fla`** |
| `causal_conv1d_fn` | `torch_reference` |
| `causal_conv1d_update` | `torch_reference` |

`causal_conv1d` is not installed, so the delta rule is FLA while the
convolution beside it in the same layer is still the inline torch reference.
`homogeneous` is `False` and the record says so.

**Latency.** FLA is worth 1.11–1.12× on the tight AR loops and 1.19× on
`hf_generate`. On the full DFlash decode it is worth considerably more: the
same sweep condition went from 18.28 ms/token under `torch_reference` to
7.86 ms/token, though those two runs also differ in the AR fix, so the clean
kernel-only comparison is the benchmark table above.

**Acceptance.** This is the part that matters for how the backend is treated.
Same prompt, same block width, same greedy decode, 256 shape-controlled
tokens:

| GDN backend | acceptance |
|---|---|
| `torch_reference` | 0.4849 |
| `fla` (+ torch conv) | 0.4542 |

**A 6.3% relative change in acceptance from a kernel swap alone.** FLA and the
torch reference compute the same function in different chunk orders, so they
round differently; the target's logits shift slightly, and different draft
tokens survive verification. The first 8 greedy tokens are identical between
backends, so the trajectory does not diverge immediately — but over 256 tokens
the acceptance does.

#### How to treat this in the profiling

The kernel choice is **a configuration axis, not measurement error**. Three
consequences:

1. **The original code does not pin a kernel.** `dflash_generate` never names
   an implementation; `use_kernel_func_from_hub_with_fallback` resolves at
   import time in the order Hub kernel → original package → torch reference.
   So "what the original code costs" on a Qwen3.5 target is a property of the
   *environment*, not of the repository. Any published number has to name the
   backend or it is not reproducible.
2. **Do not average or interpolate across backends.** They are two
   populations. `backend.probe()["homogeneous"]` is checked before any
   aggregation, and `distinct_backends` goes into every record.
3. **Acceptance is backend-specific too**, so a backend change invalidates
   previously measured acceptance, not just previously measured latency. The
   0.485 figure in §5.6 belongs to `torch_reference` and must not be quoted
   beside an FLA latency.

**Recommendation.** Install `causal_conv1d` so the GDN layer is coherently
FLA end to end, then run the sweep on that stack and report it as the primary
configuration — it is what the model card assumes and what a deployment would
use. Keep a `--no-fla` arm at the representative condition only, reported as
a separate labelled row, to show what the fallback costs. The benchmark
already supports forcing it (`sys.modules["fla"] = None` before the modeling
module imports).

## 6. Implemented, not yet run

Code is in place and imports cleanly; these have not been executed on this
machine in this session. Commands are given in section 8. **Nothing in this
section should be read as a result.**

| capability | module | status |
|---|---|---|
| Full sequence/block/output sweep, all three models | `arch/sweep.py`, `arch/run.py`, `arch/cli.py` | **one point measured** (§5.6: 9B, S=4k, single device). The other 29 conditions and all 3-GPU sharded runs are not run |
| AR baseline, both cache policies | `arch/ar.py` | **measured** at the §5.6 point only |
| Phase-resolved component memory during a DFlash run | `arch/probe.py` | **measured** at the §5.6 point only |
| Verify width q ∈ {1,4,8,16} from a restored state | `arch/verifyq.py` | not run |
| MoE routing capture on 35B | `arch/moe.py` | not run |
| NVTX ranges / Nsight Systems timeline | `arch/nvtxr.py` | wired, no trace captured |
| Nsight Compute counters, roofline | — | **not implemented**; `ncu` availability unverified |
| Phase×layer heatmap, 3-GPU timeline, roofline plots | — | **not implemented** |
| 35B end-to-end load on 3 GPUs | `arch/loader.py` | not attempted |

---

## 7. Record schema

### 7.1 Layout

```
record_arch_main/
  manifest.json              environment, GPUs, versions, device health
  plan.json                  conditions a sweep would run
  taxonomy_analytic.json     per-layer taxonomy + closed-form decode state
  <model_key>/
    taxonomy.json            instantiated taxonomy + placement
    rollback.json            rollback correctness
    verify_width.json        verify cost vs q
    sweep_<UTC>.json         the sweep record
    events_<UTC>.jsonl       per-event diagnostic log
    csv/                     exported tables
  rollback/
    <model_key>.json
```

### 7.2 Event identity

Every diagnostic event carries: `run_id`, `request_id`, `step`, `phase`,
`parent_event`, `model_role`, `layer`, `mixer`, `ffn`, `module`, `device`,
`stream`, `shape`, `backend`. Fields left absent were **not measured**; they
are not zero. `na_reason` carries why.

`mixer ∈ {full_attention, sliding_attention, gdn, other}` and
`ffn ∈ {dense, moe, moe+shared, other}` are **independent axes**, read from
the instantiated modules and cross-checked against `layer_types`. No layer
numbering rule is hardcoded anywhere.

### 7.3 Bridge to the `batch` branch

The `batch` branch's B=1 point is comparable if it matches: the block-width
definition (§3.1), the drafter-free AR baseline and its two cache policies
(§3.2), TTFT excluding the drafter's first prefill (§3.3), the component names
and the `logical`/`storage` distinction (§3.5), simultaneous-peak rather than
sum-of-maxima (§3.6), and exclusive/inclusive time kept apart (§3.7).

`schema_version` is `arch-main/1`; events are `arch-main-events/1`. Batch
records should carry `batch_size` and a populated `request_id` and otherwise
keep these field names.

---

## 8. Commands

Commands that were **run** in producing section 5:

```bash
python -m dflash.arch manifest
python -m dflash.arch plan
python -m dflash.arch rollback qwen3.5-9b --single-device --prompt-tokens 512
python -m dflash.arch rollback qwen3-8b  --single-device --prompt-tokens 512
python -m dflash.arch sweep qwen3.5-9b --single-device --sweeps sequence \
    --input-tokens 4096 --repeats 3 --no-natural
python queue/arch_ar_bench.py qwen3.5-9b --input-tokens 4096 --new-tokens 128
python queue/arch_ar_bench.py qwen3.5-9b --input-tokens 4096 --new-tokens 128 --no-fla
```

Commands **written but not yet run** (section 6):

```bash
python -m dflash.arch taxonomy qwen3.5-35b-a3b
python -m dflash.arch verify-width qwen3.5-9b --input-tokens 4096 32768
python -m dflash.arch sweep qwen3.5-9b --sweeps sequence block output
python -m dflash.arch sweep qwen3-8b --sweeps sequence block output
python -m dflash.arch sweep qwen3.5-35b-a3b --sweeps sequence
DFLASH_NVTX=1 nsys profile -o record_arch_main/trace/qwen3.5-9b \
    python -m dflash.arch verify-width qwen3.5-9b --input-tokens 32768
```

`sweep` runs, per condition: the AR baseline under both cache policies, then
DFlash under the `perf` and `memory` passes, then the pass comparison and the
speedup against each AR policy. It writes `sweep_<UTC>.json` plus the CSVs.

---

## 8.1 Regression tests

`tests/test_arch.py` — tiny random models, fp32, CPU, no GPU required.
Fifteen tests, all passing.

The module pins the torch reference GDN kernel
(`sys.modules.setdefault("fla", None)` before transformers imports). Once FLA
is installed its delta-rule kernels are Triton and CUDA-only, so every GDN
layer becomes unrunnable on CPU. What these tests check — cache layout, crop
semantics, which tensors roll back — belongs to `cache_utils`, not to the
kernel, so pinning the reference keeps them deterministic and hardware-free. They cover the taxonomy reading mixers from modules rather
than layer numbers, contiguous layer splitting, the storage ledger charging a
shared allocation once, union-vs-sum of overlapping intervals, and pairing
rejection.

Three pin the §5.8 AR corrections: that the recording policy crops and the
native one does not, that the decode loop performs no synchronisation when
EOS is ignored, and that an EOS overrun is trimmed from the output while
still being charged to the per-token latency.

Two pin section 5.5 without a GPU:

- `test_full_attention_rollback_is_exact` — a 4-layer dense model rolls back
  bitwise.
- `test_gdn_recurrent_state_does_not_roll_back` — a 4-layer hybrid model shows
  `differing_tensor_kinds == ["recurrent"]` and
  `matching_tensor_kinds == {"conv", "keys", "values"}`.

Because these run in fp32 on a toy model, the divergence is exact arithmetic,
not accumulated rounding. If the second test ever passes as lossless, `crop`
has learned to restore the recurrent state and section 5.5 is stale.

`pytest` is **not installed** in the `dflash-vllm` environment. Install it, or
drive the module functions directly.

```bash
CUDA_VISIBLE_DEVICES= python -m pytest tests/test_arch.py -q
```

---

## 9. Limitations

1. **Three GPUs, not four.** §2.1. Accepted by the user 2026-09-29; the
   sweep is a 3-GPU sweep and is not comparable with any 4-GPU run.
2. **The GDN backend is mixed and was changed mid-session.** §2.2, §5.9.
   Delta rule on FLA, conv on the torch reference. §5.5–§5.7 predate the FLA
   install. Acceptance as well as latency depends on this axis.
3. **DFlash is not lossless on GDN targets.** §5.5. Acceptance and quality
   numbers on 9B and 35B are measured against a corrupted target state.
4. **One sweep condition of thirty has been measured.** §5.6 is a single
   point on a single GPU. Every claim about how quantities scale with S, block
   width or output length is a prediction until the rest is run.
5. **No Nsight pass has been run.** No kernel-level DRAM traffic, achieved
   bandwidth, SM/Tensor pipe utilisation, occupancy, stall reasons or roofline
   exists yet. No bottleneck is claimed from scaling behaviour alone.
6. **Structural idle is reported, never inferred.** A device waiting for
   another device's shard is `structural_shard_wait` and is not a stall.
7. **None of this is serving performance.** Batch 1, Transformers, explicit
   layer sharding, no continuous batching, no prefix cache reuse. It
   characterises layer and phase costs; it does not predict a served system.
