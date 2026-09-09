"""DFlash LongBench-E summary metrics, phase-instrumented sweep (v2).

The counterpart of ``plot_summary.py``, reading the same
``record_selective/*.json`` but the *re-run* sweep, which carries the
phase-local memory instrumentation. The first eleven panels are the v1 panels
unchanged, so the two figures can be laid side by side; the last four are new
and carry the reason the sweep was re-run:

* where the peak actually happens, phase by phase, with the component split
  read at that instant rather than assembled from separate maxima;
* the drafter's first forward -- which builds its KV over the whole context --
  measured apart from the steady-state forwards it is otherwise averaged into;
* what the v1 ``transient_gb`` got wrong, and by how much, at each point.

Writes one multi-panel figure plus a tidy CSV to visualization_selective/.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _style_v2 import (  # noqa: E402
    BUDGET_COMPONENTS, CONFIG_COLORS, CTX_LABELS, CTX_LENGTHS, MEM_COMPONENTS,
    MODEL_COLORS, MODEL_MARKERS, MODELS, Notes, OUT_DIR, PEAK_COMPONENTS,
    TIME_COMPONENTS, UNATTRIBUTED_COLOR, bar_positions, configs, ctx_axis,
    device_note, footnote_box, load_v2, shard_ticks, sharded_note, short_model,
    titled,
)

V2_HEADER = (
    "Panels highlighted in green are new or re-measured relative to plot_summary.py (v1, same records "
    "directory, pre-instrumentation sweep) — ADDED = a quantity the v1 records did not carry, "
    "REVISED = same quantity, measured at a different instant"
)


def build_notes(recs):
    n = Notes()
    n.add(
        "phase", "added",
        "summary.*.phase_memory, a new record field. The run is cut into twelve named phases and each one "
        "records allocated-before, allocated-after and its own interval peak, plus the live component split "
        "probed at the occurrence that peaked. peak_phase names the phase holding the largest interval peak.",
    )
    n.add(
        "transient", "revised",
        "Transient is now peak_transient_gb -- the peak phase's interval peak minus what that phase kept "
        "resident, read at one instant. The v1 transient_gb was peak minus a sum of component-wise maxima "
        "taken at different instants, so it was not a decomposition: it under-reports on Qwen3-8B and "
        "over-reports on Qwen3.5-9B, and both errors grow linearly with context. Panel 15 plots the gap.",
    )
    n.add(
        "firststeady", "added",
        "mean_first_draft_forward_s / mean_steady_draft_forward_s and the matching cache and stage fields, all "
        "new. The drafter's first forward projects the whole context into its KV; every later one appends a "
        "block. v1 averaged the two together, which charged the one-time build to every call.",
    )
    n.add(
        "peakphase", "revised",
        "Where the peak landed is now read from phase_memory rather than peak_site alone. peak_site names the "
        "operation; peak_phase names the phase and comes with the component split live at that instant, which "
        "is what separates DFlash's contribution to the peak from the target's.",
    )
    dev = device_note(recs)
    if dev:
        n.add("devices", "revised", dev)
    shard = sharded_note(recs)
    if shard:
        n.add("shard", "revised", shard)
    return n


# --------------------------------------------------------------------------
# Panel helpers (same shapes as plot_summary.py)
# --------------------------------------------------------------------------

def grouped_bar(ax, recs, key, scale=1.0, fmt="{:.2f}", source="dflash"):
    _x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            val = recs[(model, ctx)][source][key] * scale
            ax.bar(offsets[mi][ci], val, width,
                   color=CONFIG_COLORS[(model, ctx)], edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], val, fmt.format(val), ha="center", va="bottom",
                    fontsize=6.8)
            ax.text(offsets[mi][ci], 0, short_model(model),
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)


def stacked_bar(ax, recs, components, source="dflash", sub=None, total_fmt="{:.2f}"):
    _x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)][source]
            if sub:
                s = s[sub]
            bottom = 0.0
            for key, _label, color in components:
                val = s.get(key) or 0.0
                ax.bar(offsets[mi][ci], val, width, bottom=bottom,
                       color=color, edgecolor="white", linewidth=0.5)
                bottom += val
            ax.text(offsets[mi][ci], bottom, total_fmt.format(bottom),
                    ha="center", va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, short_model(model),
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)


def line_plot(ax, recs, key, scale=1.0, fmt="{:.2f}", source="dflash", sub=None):
    for model in MODELS:
        xs, ys = [], []
        for ctx in CTX_LENGTHS:
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)][source]
            if sub:
                s = s[sub]
            xs.append(ctx)
            ys.append((s[key] or 0.0) * scale)
        ax.plot(xs, ys, color=MODEL_COLORS[model], linewidth=1.8, zorder=1)
        for xv, yv in zip(xs, ys):
            ax.scatter(xv, yv, s=70, zorder=3, marker=MODEL_MARKERS[model],
                       color=CONFIG_COLORS[(model, xv)],
                       edgecolor=MODEL_COLORS[model], linewidth=1.2)
            ax.annotate(fmt.format(yv), (xv, yv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=6.8)
    ctx_log_axis(ax)


def ctx_log_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(CTX_LENGTHS)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.margins(y=0.22)


def sample_footnote(recs):
    parts = []
    for ctx in CTX_LENGTHS:
        got = [recs[(m, ctx)] for m in MODELS if (m, ctx) in recs]
        if not got:
            continue
        ns = {g["n"] for g in got}
        n = " / ".join(str(g["n"]) for g in got) if len(ns) > 1 else str(got[0]["n"])
        ds = {g["n_datasets"] for g in got}
        d = " / ".join(str(g["n_datasets"]) for g in got) if len(ds) > 1 else str(got[0]["n_datasets"])
        comp = max(g["n_composed"] for g in got)
        parts.append(f"{CTX_LABELS[ctx]}: n={n} ({d} datasets{f', {comp} composed' if comp else ''})")
    tok = "  ·  ".join(
        f"{CTX_LABELS[c]}: " + " / ".join(f"{recs[(m, c)]['dflash']['mean_output_tokens']:.0f}"
                                          for m in MODELS if (m, c) in recs)
        for c in CTX_LENGTHS if any((m, c) in recs for m in MODELS))
    return [
        "Samples per context length [8B / 9B where they differ] — " + "  ·  ".join(parts),
        "Mean output tokens per request [8B / 9B] — " + tok +
        "   (long-context runs emit far fewer tokens; compare per-token metrics, not per-request ones)",
    ]


def main():
    recs = load_v2()
    if not recs:
        raise SystemExit("no phase-instrumented records in record_selective/")
    cfgs = configs(recs)
    notes = build_notes(recs)
    dev = ["devices"] if "devices" in notes._text else []

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.dpi": 130,
    })

    fig, axes = plt.subplots(5, 3, figsize=(19.5, 26.0))
    fig.suptitle(
        "DFlash speculative decoding — LongBench-E averages, hidden_states=selective, phase-instrumented sweep\n"
        "Qwen3-8B vs Qwen3.5-9B @ 4K / 8K / 16K / 32K / 64K · block=16 · gamma=15 · greedy",
        fontsize=12.5, fontweight="bold", y=0.992,
    )
    _x, offsets, width = bar_positions()

    # (1) acceptance length -- unchanged from v1, and from the full-mode sweep
    ax = axes[0, 0]
    line_plot(ax, recs, "mean_acceptance_length")
    ax.set_ylabel("tokens / verify step")
    titled(ax, "Mean acceptance length\n(accepted tokens per verify step)")

    # (2) acceptance length distribution -- unchanged
    ax = axes[0, 1]
    for model, ctx in cfgs:
        hist = np.asarray(recs[(model, ctx)]["dflash"]["acceptance_length_histogram"], float)
        ax.plot(np.arange(len(hist)), hist * 100, marker=MODEL_MARKERS[model], markersize=3.5,
                color=CONFIG_COLORS[(model, ctx)], linewidth=1.5,
                label=f"{model} {CTX_LABELS[ctx]}")
    ax.set_xlabel("accepted tokens in a verify step")
    ax.set_ylabel("share of verify steps (%)")
    titled(ax, "Acceptance length distribution\n(16 = full γ block accepted)")
    ax.legend(fontsize=6.2, ncol=2)

    # (3) peak memory -- unchanged measurement; the phase panels explain it
    ax = axes[0, 2]
    grouped_bar(ax, recs, "peak_memory_gb", fmt="{:.1f}")
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            base = recs[(model, ctx)]["baseline"]["peak_memory_gb"]
            ax.scatter(offsets[mi][ci], base, marker="_", s=200, linewidth=2.0,
                       color="#222222", zorder=4)
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "Peak GPU memory\n(bar = DFlash, — = baseline on the same run)", notes, dev)

    # (4) drafter overhead breakdown -- unchanged
    ax = axes[1, 0]
    stacked_bar(ax, recs, MEM_COMPONENTS)
    ax.set_ylabel("GB")
    titled(ax, "Draft-side memory overhead breakdown\n(total = draft_overhead_gb, component-wise maxima)")
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in MEM_COMPONENTS],
              fontsize=7, loc="upper left")

    # (5) whole-process budget -- unchanged
    ax = axes[1, 1]
    stacked_bar(ax, recs, BUDGET_COMPONENTS, sub="memory_budget")
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "Whole-process memory budget (v1 method)\n(resident terms + transient = peak)", notes, dev)
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in BUDGET_COMPONENTS],
              fontsize=6.2, loc="upper left")

    # (6) the same peak, decomposed where it happens
    ax = axes[1, 2]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            phase = recs[(model, ctx)]["dflash"]["peak_phase_entry"]
            parts = phase["peak_components_gb"]
            bottom = 0.0
            for key, _label, color in PEAK_COMPONENTS:
                val = parts.get(key) or 0.0
                ax.bar(offsets[mi][ci], val, width, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                bottom += val
            ax.bar(offsets[mi][ci], phase["peak_unattributed_gb"], width, bottom=bottom,
                   color=UNATTRIBUTED_COLOR, edgecolor="white", linewidth=0.5,
                   hatch="///")
            bottom += phase["peak_unattributed_gb"]
            ax.text(offsets[mi][ci], bottom, f"{bottom:.1f}", ha="center",
                    va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "Peak, decomposed at the instant it happened\n(hatched = live but unattributed)",
           notes, ["phase"] + dev)
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in PEAK_COMPONENTS]
              + [Patch(facecolor=UNATTRIBUTED_COLOR, hatch="///", label="Unattributed (activation)")],
              fontsize=5.8, loc="upper left")

    # (7) end-to-end latency
    ax = axes[2, 0]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]["dflash"]
            col = CONFIG_COLORS[(model, ctx)]
            ax.bar(offsets[mi][ci], s["mean_ttft_s"], width, color=col, alpha=0.45,
                   edgecolor="white", linewidth=0.6)
            ax.bar(offsets[mi][ci], s["mean_decode_s"], width, bottom=s["mean_ttft_s"],
                   color=col, edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], s["mean_latency_s"], f"{s['mean_latency_s']:.1f}",
                    ha="center", va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    shard_ticks(ax, recs)
    ax.set_ylabel("seconds / request")
    titled(ax, "Mean end-to-end latency\n(light = TTFT, solid = decode)", notes, dev)

    # (8) per-token decode latency
    ax = axes[2, 1]
    grouped_bar(ax, recs, "mean_time_per_output_token_s", scale=1000.0, fmt="{:.1f}")
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            agg = recs[(model, ctx)]["dflash"]["aggregate_time_per_output_token_s"] * 1000
            ax.scatter(offsets[mi][ci], agg, marker="_", s=260, linewidth=2.0,
                       color="#222222", zorder=4)
    shard_ticks(ax, recs)
    ax.set_ylabel("ms / output token")
    titled(ax, "Per-token decode latency\n(bar = per-request mean, — = aggregate)", notes, dev)

    # (9) decode throughput
    ax = axes[2, 2]
    line_plot(ax, recs, "decode_throughput_tok_s", fmt="{:.1f}")
    ax.set_ylabel("tokens / s")
    titled(ax, "Decode throughput\n(output tokens / s)", notes, dev)

    # (10) decode wall-clock breakdown -- unchanged
    ax = axes[3, 0]
    stacked_bar(ax, recs, TIME_COMPONENTS, total_fmt="{:.0f}")
    ax.set_ylabel("seconds (32 requests total)")
    titled(ax, "Decode wall-clock breakdown\n(sum over 32 requests)")
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in TIME_COMPONENTS],
              fontsize=7, loc="upper left")

    # (11) drafter share of decode, now split into its two halves
    ax = axes[3, 1]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]["dflash"]
            total = s["drafter_share_of_decode"] * 100
            first = s["first_draft_share_of_decode"] * 100
            col = CONFIG_COLORS[(model, ctx)]
            ax.bar(offsets[mi][ci], total - first, width, color=col,
                   edgecolor="white", linewidth=0.6)
            ax.bar(offsets[mi][ci], first, width, bottom=total - first, color=col,
                   edgecolor="#7a2b2b", linewidth=0.9, hatch="xxx")
            ax.text(offsets[mi][ci], total, f"{total:.1f}", ha="center", va="bottom",
                    fontsize=6.8)
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    ax.set_ylabel("% of decode time")
    titled(ax, "Drafter share of decode time\n(hatched = the one-time first draft forward)",
           notes, ["firststeady"])
    ax.legend(handles=[Patch(facecolor="#bbbbbb", label="steady-state draft + context feature"),
                       Patch(facecolor="#bbbbbb", hatch="xxx", edgecolor="#7a2b2b",
                             label="first draft forward (once per request)")],
              fontsize=6.2, loc="upper left")

    # (12) where the peak landed -- now by phase, with the split at that instant
    ax = axes[3, 2]
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.968,
            f"{'run':<8}{'phase holding the peak':<26}{'interval':>9}{'DFlash@pk':>10}",
            fontsize=7.4, family="DejaVu Sans Mono", va="top", color="#555555")
    for i, (model, ctx) in enumerate(cfgs):
        s = recs[(model, ctx)]["dflash"]
        phase = s["peak_phase_entry"]
        parts = phase["peak_components_gb"]
        dflash_at_peak = sum(parts.get(k) or 0.0 for k in
                             ("draft_weight_bytes", "draft_kv_bytes",
                              "selected_hidden_bytes", "context_feature_bytes"))
        ax.text(0.02, 0.912 - i * 0.049,
                f"{short_model(model) + ' ' + CTX_LABELS[ctx]:<8}"
                f"{s['peak_phase']:<26}{phase['max_interval_peak_gb']:>8.1f}G"
                f"{dflash_at_peak:>9.2f}G",
                fontsize=7.4, family="DejaVu Sans Mono", va="top",
                color=MODEL_COLORS[model])
    ax.text(0.02, 0.912 - len(cfgs) * 0.049 - 0.055,
            "Unanimous across all ten runs: the peak is a target\n"
            "prefill layer, never a DFlash allocation. Of the DFlash\n"
            "column, draft KV and the context feature are 0.00 GB at\n"
            "that instant — both are built after prefill ends. What is\n"
            "live is the draft weights plus the selected target hidden,\n"
            "and the latter is exactly the DFlash-minus-baseline peak\n"
            "gap (panel 3): 0.16 GB at 4K rising to 2.50 / 4.00 GB at 64K.",
            fontsize=7.0, va="top", color="#333333", style="italic")
    titled(ax, "Which phase set the peak\n(peak_phase, and DFlash's live share of it)",
           notes, ["peakphase", "phase"])

    # (13) first vs steady draft forward -- the metric v1 could not separate
    ax = axes[4, 0]
    for model in MODELS:
        xs = [c for c in CTX_LENGTHS if (model, c) in recs]
        first = [recs[(model, c)]["dflash"]["mean_first_draft_forward_s"] * 1000 for c in xs]
        steady = [recs[(model, c)]["dflash"]["mean_steady_draft_forward_s"] * 1000 for c in xs]
        ax.plot(xs, first, color=MODEL_COLORS[model], linewidth=1.9, zorder=1)
        ax.plot(xs, steady, color=MODEL_COLORS[model], linewidth=1.5, linestyle="--",
                zorder=1, alpha=0.85)
        for xv, fv, sv in zip(xs, first, steady):
            ax.scatter(xv, fv, s=70, zorder=3, marker=MODEL_MARKERS[model],
                       color=CONFIG_COLORS[(model, xv)], edgecolor=MODEL_COLORS[model],
                       linewidth=1.2)
            ax.scatter(xv, sv, s=42, zorder=3, marker=MODEL_MARKERS[model],
                       color="white", edgecolor=MODEL_COLORS[model], linewidth=1.2)
            ax.annotate(f"{fv:.0f}", (xv, fv), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6.8)
            ax.annotate(f"{sv:.1f}", (xv, sv), textcoords="offset points",
                        xytext=(-14 if model == MODELS[0] else 14, -12),
                        ha="center", fontsize=6.8, color=MODEL_COLORS[model])
    ax.set_yscale("log")
    ctx_log_axis(ax)
    ax.set_ylabel("ms per draft forward (log)")
    titled(ax, "First vs steady-state draft forward\n(solid = first call, dashed = steady mean)",
           notes, ["firststeady"])
    ax.legend(handles=[Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                              label=f"{m} first") for m in MODELS]
              + [Line2D([0], [0], color=MODEL_COLORS[m], linestyle="--",
                        marker=MODEL_MARKERS[m], markerfacecolor="white",
                        label=f"{m} steady") for m in MODELS],
              fontsize=6.0, loc="upper left")

    # (14) does the one-time build amortise? -- against tokens emitted
    ax = axes[4, 1]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]["dflash"]
            val = s["first_draft_share_of_decode"] * 100
            ax.bar(offsets[mi][ci], val, width, color=CONFIG_COLORS[(model, ctx)],
                   edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], val, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=6.8)
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    ax.set_ylabel("% of aggregate decode time")
    twin = ax.twinx()
    twin.grid(False)
    for model in MODELS:
        xs = [ci for ci, c in enumerate(CTX_LENGTHS) if (model, c) in recs]
        ys = [recs[(model, CTX_LENGTHS[ci])]["dflash"]["mean_steady_draft_calls"] for ci in xs]
        twin.plot(xs, ys, color=MODEL_COLORS[model], linewidth=1.6, linestyle=":",
                  marker=MODEL_MARKERS[model], markersize=5, zorder=5)
        for xv, yv in zip(xs, ys):
            twin.annotate(f"{yv:.0f}", (xv, yv), textcoords="offset points",
                          xytext=(0, 7 if model == MODELS[0] else -13), ha="center",
                          fontsize=6.4, color=MODEL_COLORS[model], zorder=6,
                          bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                                    pad=1.2))
    twin.set_ylabel("steady draft calls per request (dotted)", fontsize=8)
    twin.set_yscale("log")
    twin.margins(y=0.30)
    titled(ax, "First draft forward as a share of decode\n(dotted = steady calls it amortises over)",
           notes, ["firststeady"])

    # (15) the correction: v1 transient against the phase-local one
    ax = axes[4, 2]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]["dflash"]
            ax.bar(offsets[mi][ci] - width / 4, s["old_transient_gb"], width / 2,
                   color=CONFIG_COLORS[(model, ctx)], alpha=0.45,
                   edgecolor="white", linewidth=0.5)
            ax.bar(offsets[mi][ci] + width / 4, s["real_transient_gb"], width / 2,
                   color=CONFIG_COLORS[(model, ctx)], edgecolor="white", linewidth=0.5)
            top = max(s["old_transient_gb"], s["real_transient_gb"])
            ratio = (s["old_transient_gb"] / s["real_transient_gb"]
                     if s["real_transient_gb"] else 0.0)
            ax.text(offsets[mi][ci], top, f"{ratio:.2f}x", ha="center", va="bottom",
                    fontsize=6.6, fontweight="bold",
                    color="#c0504d" if ratio > 1.15 or ratio < 0.85 else "#333333")
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("GB (symlog)")
    titled(ax, "v1 transient vs phase-local transient\n(pale = v1 peak−Σmaxima, solid = measured at the peak)",
           notes, ["transient", "phase"])
    ax.legend(handles=[Patch(facecolor="#999999", alpha=0.45, label="v1 transient_gb"),
                       Patch(facecolor="#999999", label="peak_transient_gb (this sweep)")],
              fontsize=6.2, loc="upper left")
    ax.text(0.02, 0.60,
            "label = v1 / phase-local\n8B under-reports (0.05–0.23x)\n"
            "9B over-reports (2.28–2.32x)\nboth errors grow with context",
            transform=ax.transAxes, fontsize=6.6, va="top", ha="left",
            color="#333333", style="italic", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.6))

    # shared config legend
    handles = [Patch(facecolor=CONFIG_COLORS[(m, c)], label=f"{m} · {CTX_LABELS[c]}")
               for m in MODELS for c in CTX_LENGTHS]
    handles += [Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                       label=f"{m} (trend)") for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.208))
    for i, line in enumerate(sample_footnote(recs)):
        fig.text(0.5, 0.196 - i * 0.007, line, ha="center", fontsize=7.6, color="#333333")
    footnote_box(fig, notes, y=0.006, title=V2_HEADER)

    fig.tight_layout(rect=[0, 0.228, 1, 0.972])
    png = os.path.join(OUT_DIR, "dflash_summary_overview_selective_v2.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)

    cols = ["model", "context_length", "hidden_states", "num_devices",
            "mean_acceptance_length", "peak_memory_gb", "baseline_peak_memory_gb",
            "peak_phase", "peak_interval_label", "peak_interval_gb",
            "peak_target_weight_gb", "peak_target_kv_gb", "peak_draft_weight_gb",
            "peak_draft_kv_gb", "peak_selected_hidden_gb", "peak_context_feature_gb",
            "peak_unattributed_gb", "peak_transient_gb", "v1_transient_gb",
            "v1_over_phase_local", "target_kv_recorded_gb", "target_kv_at_peak_gb",
            "first_draft_forward_ms", "steady_draft_forward_ms", "first_over_steady",
            "first_draft_share_of_decode_pct", "mean_first_draft_fraction_pct",
            "steady_draft_calls", "first_draft_cache_gb", "steady_draft_cache_gb",
            "first_draft_transient_gb", "steady_draft_transient_gb",
            "mean_latency_s", "mean_ttft_s", "mean_decode_s",
            "aggregate_time_per_output_token_ms", "decode_throughput_tok_s",
            "drafter_share_of_decode_pct", "mean_output_tokens"]
    csv_path = os.path.join(OUT_DIR, "dflash_summary_metrics_selective_v2.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model, ctx in cfgs:
            r = recs[(model, ctx)]
            s = r["dflash"]
            phase = s["peak_phase_entry"]
            p = phase["peak_components_gb"]
            w.writerow([
                model, ctx, r["hidden_states"], r["num_devices"],
                round(s["mean_acceptance_length"], 4),
                round(s["peak_memory_gb"], 4),
                round(r["baseline"]["peak_memory_gb"], 4),
                s["peak_phase"], phase["peak_interval_label"],
                round(phase["max_interval_peak_gb"], 4),
                round(p.get("target_weight_bytes", 0), 4),
                round(p.get("target_kv_bytes", 0), 4),
                round(p.get("draft_weight_bytes", 0), 4),
                round(p.get("draft_kv_bytes", 0), 4),
                round(p.get("selected_hidden_bytes", 0), 4),
                round(p.get("context_feature_bytes", 0), 4),
                round(phase["peak_unattributed_gb"], 4),
                round(s["real_transient_gb"], 4), round(s["old_transient_gb"], 4),
                round(s["old_transient_gb"] / s["real_transient_gb"], 4)
                if s["real_transient_gb"] else "",
                round(s["target_kv_recorded_gb"], 4), round(s["target_kv_at_peak_gb"], 4),
                round(s["mean_first_draft_forward_s"] * 1000, 3),
                round(s["mean_steady_draft_forward_s"] * 1000, 3),
                round(s["first_over_steady"], 2),
                round(s["first_draft_share_of_decode"] * 100, 3),
                round(s["mean_first_draft_fraction_of_decode"] * 100, 3),
                round(s["mean_steady_draft_calls"], 2),
                round(s["max_first_draft_cache_gb"], 4),
                round(s["max_steady_draft_cache_gb"], 4),
                round(s["max_first_draft_transient_gb"], 4),
                round(s["max_steady_draft_transient_gb"], 4),
                round(s["mean_latency_s"], 4), round(s["mean_ttft_s"], 4),
                round(s["mean_decode_s"], 4),
                round(s["aggregate_time_per_output_token_s"] * 1000, 3),
                round(s["decode_throughput_tok_s"], 3),
                round(s["drafter_share_of_decode"] * 100, 3),
                round(s["mean_output_tokens"], 2),
            ])
    print("wrote", png)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
