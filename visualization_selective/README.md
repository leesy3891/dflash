# visualization_selective

Figures for the `--hidden-states selective` sweep in `record_selective/`. They
are the counterparts of `visualization/`, which plots the `--hidden-states full`
sweep in `record/`, and the two are meant to be read side by side.

```bash
python visualization_selective/plot_summary.py
python visualization_selective/plot_vs_baseline.py
```

| File | What it draws |
| --- | --- |
| `_style.py` | Palette, record loading, and the change-annotation machinery below. Shared by both scripts. |
| `plot_summary.py` | Twelve panels of benchmark-wide averages, both presets on one figure. → `dflash_summary_overview_selective.{png,pdf}`, `dflash_summary_metrics_selective.csv` |
| `plot_vs_baseline.py` | Nine panels of DFlash against the `block_size=1` baseline, one figure per preset. → `dflash_vs_baseline_<model>_selective.{png,pdf}`, `dflash_vs_baseline_metrics_selective.csv` |

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
