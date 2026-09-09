# VISUALIZATION.md

How to read the `_v2` figures in `visualization_selective/` — what each panel
plots, which number on it carries the finding, and where a panel can mislead.

These are the figures for the **phase-instrumented** sweep: the ten records in
`record_selective/` that carry `summary.*.phase_memory`. The v1 scripts
(`plot_summary.py`, `plot_vs_baseline.py`) still read the same directory and
still work; the v2 scripts add everything that only exists because the run is
now cut into named phases and each phase reports its own peak with the live
component split probed at that instant. `PROFILING2.md` is the prose write-up
of the same sweep — this file is the map from a claim there to the panel that
shows it.

```bash
python visualization_selective/plot_summary_v2.py        # 15 panels, both presets
python visualization_selective/plot_vs_baseline_v2.py    # 12 panels, one figure per preset
python visualization_selective/plot_phase_profile_v2.py  # 9 panels, phase + draft-stage detail
```

| File | Draws | Outputs |
| --- | --- | --- |
| `_style_v2.py` | Phase order, component palettes, `load_v2()`, the shard caveat text. Re-exports `_style.py`, so v1 and v2 panels share colours. | — |
| `plot_summary_v2.py` | Benchmark-wide averages, both presets on one figure. Panels 1–11 are the v1 panels; 12–15 are new. | `dflash_summary_overview_selective_v2.{png,pdf}`, `dflash_summary_metrics_selective_v2.csv` |
| `plot_vs_baseline_v2.py` | DFlash against the `block_size=1` baseline, one figure per preset. Panels 1–9 are the v1 panels; 6 is re-measured; 10–12 are new. | `dflash_vs_baseline_<model>_selective_v2.{png,pdf}`, `dflash_vs_baseline_metrics_selective_v2.csv` |
| `plot_phase_profile_v2.py` | Phase-by-phase memory and the draft forward's five stages. No v1 counterpart. | `dflash_phase_profile_selective_v2.{png,pdf}`, `dflash_phase_metrics_selective_v2.csv` |

## Conventions shared by all three

* **Colour** is one per (model, context length) — blues for Qwen3-8B, oranges
  for Qwen3.5-9B, darker with length — and greys for the baseline. Identical to
  `visualization/` and to the v1 selective figures, so any two panels in this
  directory can be laid side by side without re-reading a legend.
* **A pale green title** means the panel is new or re-measured relative to its
  v1 counterpart. The marker after the title (`[1]`, `[2]`, …) points into the
  boxed block under the figure. `ADDED` = a field the pre-instrumentation
  records did not carry; `REVISED` = the same quantity read at a different
  instant. Untouched panels get a plain title, so the highlight carries
  information.
* **`*` on an x tick** marks a run that was sharded over more than one card.
  Memory stays comparable across a device change; latency does not, and neither
  do the per-stage draft timings (see the trap list at the end).
* **`DF` / `BL` under a bar** is DFlash / baseline; **`8B` / `9B`** is the
  preset. Both are printed under the bar rather than in a legend so a column
  can be identified without leaving the panel.
* **A `—` tick over a bar** is a second measurement of the same thing: reserved
  against allocated, aggregate against per-request mean, at-peak against
  end-of-sample. Where the bar and the tick disagree, the disagreement is the
  point of the panel.

## Vocabulary

| Term | Meaning |
| --- | --- |
| **phase** | One of twelve named regions of a request — five in prefill, seven in the decode loop. The baseline runs only the five it has. `PHASE_ORDER` in `_style_v2.py` puts them in execution order; the short labels `P1…P5`, `D1…D6` on the phase axes follow it. |
| **interval peak** | The largest allocated-bytes reading inside one phase, summed across devices. `max_interval_peak_gb` is the largest such over the 32 samples. |
| **peak phase** | The phase holding the run's largest interval peak. Equal to the run's peak memory. |
| **components at the peak** | `target_weight` / `target_kv` / `draft_weight` / `draft_kv` / `selected_hidden` / `context_feature`, probed live at the close of the occurrence that peaked. Storage-deduplicated, so a view and its base are counted once. |
| **unattributed** | `interval peak − Σ components`. Real activation, plus anything that shrank between the peak instant and the probe. |
| **borrowed** (`peak_transient_gb`) | `interval peak − max(allocated before, allocated after)`. What the phase needed on top of what it kept. |
| **first vs steady draft** | The drafter's first forward of a request projects the whole context into its KV; every later one appends a block. They are recorded separately. |

---

## Figure 1 — `dflash_summary_overview_selective_v2.png`

Benchmark-wide averages, both presets, 5×3.

| # | Panel | What to read |
| --- | --- | --- |
| 1 | Mean acceptance length | Unchanged from v1 and from the full-mode sweep. It is the control: **3.11 / 3.09 / 2.50 / 1.48 / 2.55** on 8B and **8.98 / 7.07 / 8.83 / 7.06 / 6.91** on 9B, byte-identical to the pre-instrumentation records. If a number here ever moved, the instrumentation would have changed decoding. |
| 2 | Acceptance length distribution | Unchanged. The 9B's mass at 16 (full γ block accepted) is why its acceptance mean is 2–3× the 8B's. |
| 3 | Peak GPU memory | Bar = DFlash, `—` = the baseline on the same run. **The gap, not the bar, is DFlash's cost**: 0.16 → 2.50 GB on 8B and 0.22 → 4.00 GB on 9B across 4K→64K. Panel 12 of Figure 2 says what that gap is made of. |
| 4 | Draft-side memory overhead breakdown | The v1 method, kept for comparison: component-wise maxima summed. Useful as a resident budget, **not** as a description of any one instant. |
| 5 | Whole-process memory budget (v1 method) | Also the v1 method. The `Transient (activation)` band here is the term panel 15 shows to be wrong. |
| 6 | **Peak, decomposed at the instant it happened** | The replacement for panel 5. Every segment was live simultaneously. Read the hatched top band as activation, and note that **`Draft KV` and `Context feature` are 0.00 GB in all ten columns** — neither exists yet when the peak happens. |
| 7–9 | Latency, per-token latency, throughput | Unchanged from v1. 9B/64K is starred: pipeline-parallel. |
| 10 | Decode wall-clock breakdown | Unchanged. Target verify dominates everywhere. |
| 11 | Drafter share of decode time | Total height is the v1 number; **the hatched top is the one-time first draft forward**. On 9B/64K the hatch is 33.8 of 38.2 points — almost the entire drafter cost is a single call per request. |
| 12 | Which phase set the peak | `prefill: target forward`, **unanimously, in all ten runs**, at target layer 35 (8B) / 30 (9B). `DFlash@pk` is the DFlash-specific sum live at that instant: 2.11 → 4.45 GB on 8B, 2.66 → 6.41 GB on 9B, of which all but the draft weights is the injected hidden. |
| 13 | **First vs steady draft forward** | Log axis. Solid = first call, **17 → 202 ms** (8B) and **25 → 291 ms** (9B) across 4K→64K, linear in context. Dashed = steady mean, **5.9 → 14.5 ms** and **8.2 → 10.7 ms**. The ratio is 3× at 4K and 14–27× at 64K. |
| 14 | **First draft as a share of decode** | Bars are aggregate share; the dotted line on the right axis is the number of steady calls it amortises over. The two move together: at 32K/64K generations collapse to 12–24 tokens, steady calls fall to 2–18, and the one-time build stops amortising — **0.6% at 4K, 33.8% at 9B/64K**. |
| 15 | **v1 transient vs phase-local transient** | Pale = v1 `transient_gb`, solid = `peak_transient_gb`. The bold label is their ratio. **8B under-reports (0.05–0.23×), 9B over-reports (2.28–2.32×)**, and both errors grow with context. Symlog, because the two differ by more than a decade at 4K. |

## Figure 2 — `dflash_vs_baseline_<model>_selective_v2.png`

DFlash against the `block_size=1` baseline, one figure per preset, 4×3. Only
metrics both configurations report; draft-only metrics stay in Figure 1.

| # | Panel | What to read |
| --- | --- | --- |
| 1–5 | Latency, TTFT, per-token latency, throughput, peak memory | Unchanged from v1. TTFT is identical in both columns by construction — same target, same prefill. |
| 6 | **Target KV: recorded vs live at the peak** | Bar = `max_target_cache_gb`, the end-of-sample reading v1 plotted. `—` with an arrow = the same cache probed at the peak. **On 8B they agree exactly** (0.63 → 9.07 GB). **On 9B they do not**: 0.19 vs 1.67 GB at 4K, **2.07 vs 26.05 GB at 64K**. The arrow appears in the baseline column too, which is what makes it the target's behaviour and not DFlash's. |
| 7–8 | Target forwards per request, DFlash relative to baseline | Unchanged. |
| 9 | Memory budget, v1 method | Kept so the v1 figure can be diffed against this one. |
| 10 | **Peak of every phase** | Solid = DFlash, dashed = baseline, same shade. Two things to read: the curves **never cross** — DFlash is a constant offset above the baseline at every phase — and the **9B's cliff at `D1' steady draft`**, from ~49 GB to ~23 GB at 64K, present in the baseline curve too. |
| 11 | **What each phase borrows** | `interval peak − what the phase kept`, symlog. Only two phases borrow anything: `P1 target fwd` (6.5 GB on 8B, 13.3 GB on 9B at 64K) and `D1 first draft` (1.3 / 1.6 GB). Every rollback and logits phase is flat at ~0.02 GB. |
| 12 | **What the peak delta is made of** | Bar = `peak(DFlash) − peak(baseline)`, line = the selected target hidden live at the peak, ratio printed above. **8B: 1.000× at every length.** **9B: 0.875 = 7/8** — one of the eight injected layers' residual streams is a tensor the target's own forward allocated anyway, so the tap retains it rather than adding it. |

## Figure 3 — `dflash_phase_profile_selective_v2.png`

The phase and draft-stage detail. No v1 counterpart — every panel is new. 3×3.

| # | Panel | What to read |
| --- | --- | --- |
| 1–2 | **What is live at each phase** (8B @ 64K, 9B @ 64K) | Stack = probed components, hatched = unattributed, black line = the interval peak. The 8B is nearly flat: target weights + a 9 GB KV that never moves. The 9B is not — the target KV band **collapses at `D1' steady draft`** and the hatched band balloons at `D4 target verify`, which is the same 24 GB seen from the other side (see trap 1). |
| 3 | **Context feature live at each phase's largest occurrence** | Built at `P4 ctx-feat concat` over the whole prompt, read once by `D1 first draft`, and still allocated through `D5 verify rollback` — **2.50 GB (8B) and 4.00 GB (9B) at 64K held across four phases that never touch it**. The dip at `D1'` is not a free-and-rebuild; see trap 2. |
| 4 | **Split at `D4 target verify`** | Left bar = DFlash, right dotted = baseline, both stacked the same way. The hatched remainder is ~24 GB on 9B/64K in **both** columns. |
| 5 | **First draft forward, by stage** | Stacked ms; `Σ` is the stage sum and `—` the measured call, so the gap is the un-staged remainder (q/o projections, MLP, norms) — 18% on 8B/64K. **`fc + hidden_norm` alone is 51–63% of the first call**, and with the context K/V projection 72–92%. |
| 6 | **Steady-state draft forward, by stage** | Same five stages over one block. The mix inverts: `lm_head` is 38–58% on 8B up to 16K, and by 64K `KV append` + `attention` are 79%. At 4K every stage here is sub-millisecond except the head; by 64K `KV append` (5.7 ms) and `attention` (3.3 ms) are the two that grew. |
| 7 | **First-call stage scaling** | Log-log; slope 1 = linear in context. `fc + hidden_norm`, context K/V, KV append and attention all sit at slope ≈1. `lm_head` is **flat** — it runs over the proposal block, not the context. The dashed red line diving at 64K is the shard artifact, not a speedup (trap 3). |
| 8 | **Draft KV: first call vs steady** | Solid = after the first call, pale = steady state, `—` = borrowed by that call, ratio above. **8B: ≈1.0× at every length — the drafter keeps what it builds.** **9B: 1.0× at 4K rising to 4.5× at 64K**, because five of its six draft layers are sliding-window (4096) and drop what falls outside. |
| 9 | **DFlash's footprint vs the target's** | Left bar = target weights + target KV + target prefill activation, right bar = draft weights + draft KV + selected hidden + context feature. DFlash is **12.6% → 21.0%** of the total on 8B and **14.5% → 27.1%** on 9B across 4K→64K. Of the DFlash bar, only the draft weights (1.95 / 2.41 GB) are there for the whole benchmark. |

## The CSVs

| File | Grain | Use it for |
| --- | --- | --- |
| `dflash_summary_metrics_selective_v2.csv` | one row per (model, context) | The peak phase and its full component split, both transient numbers and their ratio, the first/steady draft pair, and the headline latency metrics. |
| `dflash_vs_baseline_metrics_selective_v2.csv` | two rows per (model, context) — DFlash and baseline | Anything that needs the baseline as a control, including `target_kv_gap_gb` (at-peak minus recorded) and `peak_delta_vs_baseline_gb`. Ratio columns are filled on the DFlash row only. |
| `dflash_phase_metrics_selective_v2.csv` | one row per (model, context, mode, phase) | The tidy form of Figure 3's panels 1–4: every phase's before/after/peak/borrowed/unattributed and its six components. 170 rows. |

---

## What the data says

Five findings, each with the panel that carries it. All ten configurations agree
unless stated.

1. **The peak is never DFlash's.** `prefill: target forward` holds it in every
   run, at the last-injected target layer, and draft KV and the context feature
   are both 0.00 GB at that instant — they do not exist yet.
   *Figure 1 panels 6, 12.*

2. **What DFlash adds to the peak is the injected hidden states, and nothing
   else.** The DFlash-minus-baseline peak gap equals the selected target hidden
   live at the peak: exactly 1.000× on 8B, 7/8 on 9B. Draft weights are in both
   columns — the baseline loads the drafter it never calls.
   *Figure 2 panel 12.*

3. **`max_target_cache_gb` under-reports hybrid targets by up to 24 GB.** It is
   read once at end of sample, after Qwen3.5-9B's 24 gated delta-rule layers
   have released a sequence-length prefill buffer on the first decode forward.
   The same collapse is in the baseline's phase table, so it is a target
   architecture cost and **not** DFlash overhead. It is also why the v1
   `transient_gb` over-reports 2.3× on the 9B and under-reports on the 8B.
   *Figure 2 panels 6, 10; Figure 3 panels 2, 4; Figure 1 panel 15.*

4. **The drafter's one-time prefill stops amortising at long context.** The
   first draft forward is linear in context (17→202 ms on 8B, 25→291 ms on 9B)
   while the steady call is nearly flat. That is affordable while a request
   emits 180–250 tokens; at 32K/64K LongBench-E generations collapse to 12–24
   tokens, the call count falls with them, and the single call reaches **33.8%
   of aggregate decode time** on 9B/64K. `fc + hidden_norm` over the injected
   feature is 51–63% of it.
   *Figure 1 panels 11, 13, 14; Figure 3 panels 5, 7.*

5. **The prompt-length context feature outlives its last read by four phases.**
   Built at the prefill concat, consumed by the first draft forward, and still
   resident through the draft rollback, the draft logits, the target verify and
   the verify rollback — 2.50 GB (8B) / 4.00 GB (9B) at 64K. It is under the
   peak, so it costs nothing at 64K today; it is the first thing that would
   matter if the peak ever moved into the decode loop. Left unchanged: freeing
   it early is a semantics change, and this sweep was instrumentation only.
   *Figure 3 panel 3.*

A sixth, smaller one: the 9B drafter's sliding-window layers mean the first
call builds a KV **4.5× larger at 64K** (1.0× at 4K) than what steady state keeps, so its
draft-KV footprint is a first-call transient rather than a resident cost.
*Figure 3 panel 8.*

## Traps

1. **A component that shrinks inside a phase lands in `unattributed`.** The
   split is probed when the interval closes, so at 9B's `D4 target verify` the
   GDN buffer is live when the interval opens and gone when it closes: target
   KV reads 2.05 GB against a 24 GB remainder. Read the two together at that
   phase — the memory is accounted for, just not attributed.
   *Affects Figure 3 panels 1, 2, 4.*

2. **A curve across phases is an envelope, not a timeline.** Each phase reports
   its own largest occurrence, and different phases peak in different decode
   iterations. The context feature's dip at `D1' steady draft` in Figure 3
   panel 3 means the steady-state draft forward peaked *later*, when only the
   block-sized feature was live. It does not mean the feature was freed and
   rebuilt between `D1` and `D2`.

3. **Per-stage draft timings are invalid across a shard boundary.** The stages
   are timed with CUDA events on the drafter's stream, so a stage whose weights
   live on the other card is timed as the launch rather than the work. On
   9B/64K `lm_head` reads ~0.05 ms against ~2.8 ms single-GPU. Read a starred
   column's stage split qualitatively only, and never subtract it from a
   single-GPU column. *Figure 3 panels 5, 6, 7.*

4. **`Σ` is not the call.** The five stages do not tile a draft forward — the
   q/o projections, the MLP and the norms are not instrumented. Figure 3
   panel 5 prints both, and the gap is 15–25%.

5. **Do not compare a sharded latency with a single-GPU one.** Only 9B/64K is
   sharded here (two cards). Its memory is comparable; its wall clock is not.
   Every affected panel stars the tick.

6. **Panels 4, 5 and 9 of Figure 1 are the v1 method on purpose.** They sum
   component-wise maxima taken at different instants. They are kept so the two
   figures can be diffed, not because they describe a moment in the run.
