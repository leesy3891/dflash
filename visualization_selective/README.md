# visualization_selective

Figures for the `--hidden-states selective` sweep in `record_selective/`. They
are the counterparts of `visualization/`, which plots the `--hidden-states full`
sweep in `record/`, and the two are meant to be read side by side.

There are two generations of figure here. The **v1** scripts below plot the
fields the first selective sweep produced. The **`_v2`** scripts plot the
phase-instrumented re-run — see [VISUALIZATION.md](VISUALIZATION.md), which is
the panel-by-panel guide to them and the place to start if you want the
findings rather than the history.

```bash
python visualization_selective/plot_summary.py            # v1
python visualization_selective/plot_vs_baseline.py        # v1

python visualization_selective/plot_summary_v2.py         # v2
python visualization_selective/plot_vs_baseline_v2.py     # v2
python visualization_selective/plot_phase_profile_v2.py   # v2, no v1 counterpart
```

| File | What it draws |
| --- | --- |
| `_style.py` | Palette, record loading, and the change-annotation machinery below. Shared by both scripts. |
| `plot_summary.py` | Twelve panels of benchmark-wide averages, both presets on one figure. → `dflash_summary_overview_selective.{png,pdf}`, `dflash_summary_metrics_selective.csv` |
| `plot_vs_baseline.py` | Nine panels of DFlash against the `block_size=1` baseline, one figure per preset. → `dflash_vs_baseline_<model>_selective.{png,pdf}`, `dflash_vs_baseline_metrics_selective.csv` |
| `_style_v2.py` | v2 additions: phase order and short labels, the component and stage palettes, `load_v2()` (records carrying `phase_memory` only), and the sharded-stage caveat. Re-exports `_style.py`. |
| `plot_summary_v2.py` | Fifteen panels — the eleven above plus the peak decomposed where it happens, the phase that set it, first-vs-steady draft forward, and the v1 transient against the phase-local one. → `dflash_summary_overview_selective_v2.{png,pdf}`, `dflash_summary_metrics_selective_v2.csv` |
| `plot_vs_baseline_v2.py` | Twelve panels — the nine above with target KV re-measured, plus every phase's peak in both configurations, what each phase borrows, and what the peak delta is made of. → `dflash_vs_baseline_<model>_selective_v2.{png,pdf}`, `dflash_vs_baseline_metrics_selective_v2.csv` |
| `plot_phase_profile_v2.py` | Nine panels of phase-local memory and the draft forward's five stages. → `dflash_phase_profile_selective_v2.{png,pdf}`, `dflash_phase_metrics_selective_v2.csv` |
| `phase_tables.py` | The markdown tables PROFILING2.md quotes, from the same records. Prints to stdout. |

The colours are the same as `visualization/` — one per (model, context length),
greys for the baseline — so a panel here can be compared with its counterpart
without re-reading the legend.

## Reading the green highlights

A **pale green title** means the panel is *not* a like-for-like redraw of the
one in `visualization/`: either the quantity did not exist in the full-mode
records, or the same quantity was measured differently. The marker after the
title (`[1]`, `[2]`, …) points at the boxed block under the figure, which says
which. Untouched panels get a plain title, so the highlight carries information.

Two kinds of note:

* **ADDED** — a record field the full-mode sweep does not carry:
  `memory_budget`, `target_weight_gb`, `peak_site_histogram`.
* **REVISED** — the same quantity, measured differently: `hidden_states=selective`,
  the `peak_site` interval-attribution fix, and a changed device count.

The notes are generated from the records rather than hard-coded, so a re-run
after a new sweep re-derives them. In particular `device_note()` diffs
`num_devices` against `record/` and names each run that moved, and a `*` on an
x tick marks a run that is *still* sharded, whose timings are pipeline-parallel
and must not be read against a single-GPU column.

## Panels that are not highlighted

Acceptance length, its distribution, the decode wall-clock breakdown, the
drafter's share of decode, the target KV cache, and target forwards per request
are unmarked on purpose: they are identical to the full-mode figures, which is
the point. Selective changes what the run costs, not what it measures — mean
acceptance length and mean output tokens match at all ten points.

## v1 and v2 side by side

The v1 and v2 scripts read the same directory, so both run against whatever is
in `record_selective/` — but only `load_v2()` filters to the records that carry
`phase_memory`. The v1 panels that survive into v2 unchanged (acceptance,
throughput, the wall-clock breakdown) are the check that the instrumentation
changed nothing: `mean_acceptance_length` and `mean_output_tokens` match the
pre-instrumentation records at all ten points.

Where a v2 panel deliberately keeps the v1 *method* — the draft-overhead
breakdown, the whole-process budget — the title says so, because those sum
component-wise maxima taken at different instants and are kept for
comparability rather than because they describe any one moment. `VISUALIZATION.md`
lists that and the other four traps.
