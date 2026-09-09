"""DFlash vs. baseline per model, phase-instrumented sweep (v2).

The counterpart of ``plot_vs_baseline.py``. Same nine panels, plus three that
only exist because the baseline is now phase-instrumented too:

* the peak of every phase, both configurations on one axis, which shows that
  the two curves separate by a constant and never cross;
* the target KV cache as recorded at end of sample against the target KV that
  was actually live when the peak happened -- the gap is the whole reason the
  v1 budget mis-attributed memory on Qwen3.5-9B;
* the DFlash-minus-baseline peak delta against the one DFlash term that is
  live at the peak, which is what the delta is made of.

The third is the panel to read before calling any of this "DFlash overhead":
the baseline runs the same target and pays the same prefill activation, so a
term that appears in both columns is the target's, not the drafter's.

Writes one figure per model plus a tidy CSV to visualization_selective/.
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
    BASE_COLORS, BASE_LINE_COLOR, BASELINE_BUDGET_COMPONENTS, BUDGET_COMPONENTS,
    CONFIG_COLORS, CTX_LABELS, CTX_LENGTHS, MODEL_COLORS, MODEL_MARKERS, MODELS,
    Notes, OUT_DIR, PHASE_ORDER, PHASE_SHORT, bar_positions, ctx_axis,
    device_note, footnote_box, load_v2, phases_of, shard_ticks, sharded_note,
    titled, value_text_color,
)

V2_HEADER = (
    "Panels highlighted in green are new or re-measured relative to plot_vs_baseline.py (v1, same records "
    "directory, pre-instrumentation sweep) — ADDED = a quantity the v1 records did not carry, "
    "REVISED = same quantity, measured at a different instant"
)

RATIO_METRICS = [
    ("speedup_tpot", "Decode speedup\n(aggregate s/token)", "#3b4d8f"),
    ("ratio_throughput", "Decode throughput\nratio", "#6aa84f"),
    ("ratio_target_forwards", "Target forward\nreduction", "#c0504d"),
    ("ratio_memory", "Peak memory\nratio", "#e8a33d"),
]


def build_notes(recs, model):
    n = Notes()
    n.add(
        "phase", "added",
        "summary.*.phase_memory, a new record field, for both configurations. The run is cut into named phases "
        "-- twelve for DFlash, the five the baseline actually runs -- and each records its own interval peak "
        "plus the component split probed at the occurrence that peaked.",
    )
    n.add(
        "kvpeak", "revised",
        "Target KV now has two numbers. max_target_cache_gb is _cache_bytes() called once at end of sample, "
        "which is what v1 plotted; peak_components_gb['target_kv_bytes'] is the same cache measured when the "
        "peak happened. They agree on Qwen3-8B and do not on Qwen3.5-9B, whose gated delta-rule layers hold a "
        "sequence-length prefill buffer that is released on the first decode forward -- so the end-of-sample "
        "reading misses up to 24 GB that was resident through the whole peak.",
    )
    n.add(
        "delta", "added",
        "The DFlash-minus-baseline peak delta, against selected_hidden at the peak. Both configurations load the "
        "drafter's weights and both run the same target prefill, so the delta is neither of those; it is the "
        "injected target hidden states, the one DFlash term alive while the target is still prefilling. The "
        "ratio is 1.000 on Qwen3-8B and 7/8 on Qwen3.5-9B, where one of the eight injected layers' residual "
        "streams is a tensor the target's own forward had allocated regardless -- the tap retains it rather "
        "than adding it.",
    )
    dev = device_note(recs, model)
    if dev:
        n.add("devices", "revised", dev)
    shard = sharded_note(recs, model)
    if shard:
        n.add("shard", "revised", shard)
    return n


def grouped_bar(ax, recs, model, key, scale=1.0, fmt="{:.2f}", tag=True,
                label_inside=False, top_key=None):
    _, offsets, width = bar_positions()
    for ci, ctx in enumerate(CTX_LENGTHS):
        if (model, ctx) not in recs:
            continue
        r = recs[(model, ctx)]
        for mi, (mode, color) in enumerate((("dflash", CONFIG_COLORS[(model, ctx)]),
                                            ("baseline", BASE_COLORS[ctx]))):
            val = r[mode][key] * scale
            ax.bar(offsets[mi][ci], val, width, color=color,
                   edgecolor="white", linewidth=0.6)
            if label_inside:
                ax.text(offsets[mi][ci], val * 0.97, fmt.format(val), ha="center",
                        va="top", fontsize=7.5, fontweight="bold",
                        color=value_text_color(color))
            else:
                y = val if top_key is None else max(val, r[mode][top_key] * scale)
                ax.text(offsets[mi][ci], y, fmt.format(val), ha="center",
                        va="bottom", fontsize=6.8)
            if tag:
                ax.text(offsets[mi][ci], 0, "DF" if mode == "dflash" else "BL",
                        ha="center", va="top", fontsize=6.0,
                        color=MODEL_COLORS[model] if mode == "dflash" else BASE_LINE_COLOR)
    ctx_axis(ax)


def budget_bars(ax, recs, model):
    _, offsets, width = bar_positions()
    for ci, ctx in enumerate(CTX_LENGTHS):
        if (model, ctx) not in recs:
            continue
        r = recs[(model, ctx)]
        for mi, (mode, comps) in enumerate((("dflash", BUDGET_COMPONENTS),
                                            ("baseline", BASELINE_BUDGET_COMPONENTS))):
            budget = r[mode]["memory_budget"]
            bottom = 0.0
            for key, _label, color in comps:
                val = budget.get(key) or 0.0
                ax.bar(offsets[mi][ci], val, width, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                bottom += val
            ax.text(offsets[mi][ci], bottom, f"{bottom:.1f}", ha="center",
                    va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, "DF" if mode == "dflash" else "BL",
                    ha="center", va="top", fontsize=6.0,
                    color=MODEL_COLORS[model] if mode == "dflash" else BASE_LINE_COLOR)
    ctx_axis(ax)


def line_pair(ax, recs, model, key, scale=1.0, fmt="{:.1f}"):
    for mode in ("dflash", "baseline"):
        xs = [c for c in CTX_LENGTHS if (model, c) in recs]
        ys = [recs[(model, c)][mode][key] * scale for c in xs]
        line_c = MODEL_COLORS[model] if mode == "dflash" else BASE_LINE_COLOR
        ax.plot(xs, ys, color=line_c, linewidth=1.8,
                linestyle="-" if mode == "dflash" else "--", zorder=1)
        for xv, yv in zip(xs, ys):
            face = CONFIG_COLORS[(model, xv)] if mode == "dflash" else BASE_COLORS[xv]
            ax.scatter(xv, yv, s=70, zorder=3, marker=MODEL_MARKERS[model],
                       color=face, edgecolor=line_c, linewidth=1.2)
            ax.annotate(fmt.format(yv), (xv, yv), textcoords="offset points",
                        xytext=(0, 10 if mode == "dflash" else -18), ha="center",
                        fontsize=6.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CTX_LENGTHS)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.margins(y=0.25)


def ratios(r):
    df, bl = r["dflash"], r["baseline"]
    return {
        "speedup_tpot": r["speedup"],
        "ratio_throughput": df["decode_throughput_tok_s"] / bl["decode_throughput_tok_s"],
        "ratio_target_forwards": bl["target_forwards_per_req"] / df["target_forwards_per_req"],
        "ratio_memory": df["peak_memory_gb"] / bl["peak_memory_gb"],
    }


def sample_footnote(recs, model):
    parts, toks = [], []
    for ctx in CTX_LENGTHS:
        r = recs.get((model, ctx))
        if r is None:
            continue
        extra = f", {r['n_composed']} composed" if r["n_composed"] else ""
        parts.append(f"{CTX_LABELS[ctx]}: n={r['n']} ({r['n_datasets']} datasets{extra})")
        toks.append(f"{CTX_LABELS[ctx]}: {r['dflash']['mean_output_tokens']:.0f}"
                    f" / {r['baseline']['mean_output_tokens']:.0f}")
    return [
        f"Samples per context length ({model}) — " + "  ·  ".join(parts),
        "Mean output tokens per request [DFlash / baseline] — " + "  ·  ".join(toks) +
        "   (long-context runs emit far fewer tokens; compare per-token metrics, not per-request ones)",
    ]


def make_figure(recs, model):
    notes = build_notes(recs, model)
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "figure.dpi": 130,
    })
    fig, axes = plt.subplots(4, 3, figsize=(18.5, 24.0))
    fig.suptitle(
        f"DFlash vs. baseline — {model} on LongBench-E, hidden_states=selective, phase-instrumented sweep\n"
        f"4K / 8K / 16K / 32K / 64K · draft-only metrics stay in plot_summary_v2.py",
        fontsize=12.5, fontweight="bold", y=0.992,
    )
    _, offsets, width = bar_positions()
    dev = ["devices"] if "devices" in notes._text else []

    # (1) end-to-end latency
    ax = axes[0, 0]
    for ci, ctx in enumerate(CTX_LENGTHS):
        if (model, ctx) not in recs:
            continue
        r = recs[(model, ctx)]
        for mi, (mode, color) in enumerate((("dflash", CONFIG_COLORS[(model, ctx)]),
                                            ("baseline", BASE_COLORS[ctx]))):
            s = r[mode]
            ax.bar(offsets[mi][ci], s["mean_ttft_s"], width, color=color, alpha=0.45,
                   edgecolor="white", linewidth=0.6)
            ax.bar(offsets[mi][ci], s["mean_decode_s"], width, bottom=s["mean_ttft_s"],
                   color=color, edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], s["mean_latency_s"], f"{s['mean_latency_s']:.1f}",
                    ha="center", va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, "DF" if mode == "dflash" else "BL",
                    ha="center", va="top", fontsize=6.0,
                    color=MODEL_COLORS[model] if mode == "dflash" else BASE_LINE_COLOR)
    ctx_axis(ax)
    shard_ticks(ax, recs, model)
    ax.set_ylabel("seconds / request")
    titled(ax, "Mean end-to-end latency\n(light = TTFT, solid = decode)", notes, dev)

    # (2) TTFT
    ax = axes[0, 1]
    grouped_bar(ax, recs, model, "mean_ttft_s", fmt="{:.2f}")
    shard_ticks(ax, recs, model)
    ax.set_ylabel("seconds")
    titled(ax, "Mean TTFT\n(prefill, identical work in both modes)", notes, dev)

    # (3) per-token decode latency
    ax = axes[0, 2]
    grouped_bar(ax, recs, model, "mean_time_per_output_token_s", scale=1000.0,
                fmt="{:.1f}", top_key="aggregate_time_per_output_token_s")
    for ci, ctx in enumerate(CTX_LENGTHS):
        if (model, ctx) not in recs:
            continue
        for mi, mode in enumerate(("dflash", "baseline")):
            agg = recs[(model, ctx)][mode]["aggregate_time_per_output_token_s"] * 1000
            ax.scatter(offsets[mi][ci], agg, marker="_", s=260, linewidth=2.0,
                       color="#222222", zorder=4)
    shard_ticks(ax, recs, model)
    ax.set_ylabel("ms / output token")
    titled(ax, "Per-token decode latency\n(bar = per-request mean, — = aggregate)", notes, dev)

    # (4) decode throughput
    ax = axes[1, 0]
    line_pair(ax, recs, model, "decode_throughput_tok_s")
    ax.set_ylabel("tokens / s")
    titled(ax, "Decode throughput\n(output tokens / s)", notes, dev)

    # (5) peak memory
    ax = axes[1, 1]
    grouped_bar(ax, recs, model, "peak_memory_gb", fmt="{:.1f}", label_inside=True)
    for ci, ctx in enumerate(CTX_LENGTHS):
        if (model, ctx) not in recs:
            continue
        for mi, mode in enumerate(("dflash", "baseline")):
            res = recs[(model, ctx)][mode]["peak_memory_reserved_gb"]
            ax.scatter(offsets[mi][ci], res, marker="_", s=260, linewidth=2.0,
                       color="#222222", zorder=4)
            ax.text(offsets[mi][ci], res, f"{res:.1f}", ha="center", va="bottom",
                    fontsize=6.8)
    shard_ticks(ax, recs, model)
    ax.set_ylabel("GB")
    titled(ax, "Peak GPU memory\n(bar = allocated, — = reserved)", notes, dev)

    # (6) target KV cache -- and the same cache read where the peak happened
    ax = axes[1, 2]
    grouped_bar(ax, recs, model, "target_kv_recorded_gb", fmt="{:.2f}")
    for ci, ctx in enumerate(CTX_LENGTHS):
        if (model, ctx) not in recs:
            continue
        for mi, mode in enumerate(("dflash", "baseline")):
            at_peak = recs[(model, ctx)][mode]["target_kv_at_peak_gb"]
            ax.scatter(offsets[mi][ci], at_peak, marker="_", s=260, linewidth=2.2,
                       color="#7a1f1f", zorder=4)
            rec = recs[(model, ctx)][mode]["target_kv_recorded_gb"]
            if at_peak > rec * 1.05:
                ax.annotate("", xy=(offsets[mi][ci], at_peak),
                            xytext=(offsets[mi][ci], rec), zorder=5,
                            arrowprops=dict(arrowstyle="->", color="#7a1f1f", lw=1.0))
                ax.annotate(f"{at_peak:.1f}", (offsets[mi][ci], at_peak),
                            textcoords="offset points", xytext=(0, 5), ha="center",
                            fontsize=6.6, color="#7a1f1f", fontweight="bold")
    ax.set_ylabel("GB")
    titled(ax, "Target KV cache: recorded vs live at the peak\n(bar = end of sample, — = at peak_phase)",
           notes, ["kvpeak", "phase"])

    # (7) target forward passes per request -- untouched
    ax = axes[2, 0]
    grouped_bar(ax, recs, model, "target_forwards_per_req", fmt="{:.0f}")
    ax.set_ylabel("forward passes / request")
    titled(ax, "Target forward passes per request\n(DF = verify steps, BL = decode steps)")

    # (8) DFlash relative to baseline
    ax = axes[2, 1]
    x = np.arange(len(CTX_LENGTHS), dtype=float)
    w = 0.19
    for ki, (key, label, color) in enumerate(RATIO_METRICS):
        vals = [ratios(recs[(model, c)])[key] for c in CTX_LENGTHS if (model, c) in recs]
        pos = x[:len(vals)] + (ki - (len(RATIO_METRICS) - 1) / 2) * w
        ax.bar(pos, vals, w, color=color, edgecolor="white", linewidth=0.6, label=label)
        for pv, vv in zip(pos, vals):
            ax.annotate(f"{vv:.2f}", (pv, vv), textcoords="offset points",
                        xytext=(0, 2 + 8 * (ki % 2)), ha="center", fontsize=6.4)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":")
    ax.set_xlabel("context length")
    shard_ticks(ax, recs, model)
    ax.set_ylabel("x baseline  (>1 = DFlash better,\nexcept memory ratio)")
    titled(ax, "DFlash relative to baseline", notes, dev)
    ax.legend(fontsize=6.0, ncol=2, loc="upper left")
    ax.margins(y=0.30)

    # (9) the v1 budget, both configurations
    ax = axes[2, 2]
    budget_bars(ax, recs, model)
    shard_ticks(ax, recs, model)
    ax.set_ylabel("GB")
    titled(ax, "Memory budget, v1 method\n(stack = resident terms + transient = peak)", notes, dev)
    seen, handles = set(), []
    for key, label, color in BUDGET_COMPONENTS + BASELINE_BUDGET_COMPONENTS:
        if label in seen:
            continue
        seen.add(label)
        handles.append(Patch(facecolor=color, label=label))
    ax.legend(handles=handles, fontsize=5.8, loc="upper left")

    # (10) every phase's peak, both configurations
    ax = axes[3, 0]
    names = [n for n in PHASE_ORDER]
    idx = {n: i for i, n in enumerate(names)}
    for ctx in CTX_LENGTHS:
        if (model, ctx) not in recs:
            continue
        entry = recs[(model, ctx)]
        for mode, color, style in (("dflash", CONFIG_COLORS[(model, ctx)], "-"),
                                   ("baseline", BASE_COLORS[ctx], "--")):
            pairs = phases_of(entry, mode)
            xs = [idx[n] for n, _ in pairs if n in idx]
            ys = [p["max_interval_peak_gb"] for n, p in pairs if n in idx]
            ax.plot(xs, ys, style, color=color, linewidth=1.7, marker="o",
                    markersize=3.6, zorder=2 if mode == "dflash" else 1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PHASE_SHORT[n] for n in names], rotation=55, ha="right",
                       fontsize=6.6)
    ax.set_ylabel("phase interval peak (GB)")
    titled(ax, "Peak of every phase\n(solid = DFlash, dashed = baseline)", notes, ["phase"])
    ax.legend(handles=[Line2D([0], [0], color=CONFIG_COLORS[(model, c)],
                              label=f"DFlash {CTX_LABELS[c]}") for c in CTX_LENGTHS]
              + [Line2D([0], [0], color=BASE_LINE_COLOR, linestyle="--",
                        label="baseline (same shade, dashed)")],
              fontsize=5.8, loc="upper right")

    # (11) how much each phase borrows on top of what it keeps
    ax = axes[3, 1]
    for ctx in CTX_LENGTHS:
        if (model, ctx) not in recs:
            continue
        pairs = phases_of(recs[(model, ctx)], "dflash")
        xs = [idx[n] for n, _ in pairs if n in idx]
        ys = [p["peak_transient_gb"] for n, p in pairs if n in idx]
        ax.plot(xs, ys, "-", color=CONFIG_COLORS[(model, ctx)], linewidth=1.7,
                marker="o", markersize=3.6, label=f"DFlash {CTX_LABELS[ctx]}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PHASE_SHORT[n] for n in names], rotation=55, ha="right",
                       fontsize=6.6)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("borrowed at that phase (GB, symlog)")
    titled(ax, "What each phase borrows\n(interval peak − what the phase kept)",
           notes, ["phase"])
    ax.legend(fontsize=5.8, loc="upper right")

    # (12) what the peak delta is actually made of
    ax = axes[3, 2]
    xs = [ci for ci, c in enumerate(CTX_LENGTHS) if (model, c) in recs]
    delta = [recs[(model, CTX_LENGTHS[ci])]["dflash"]["peak_memory_gb"]
             - recs[(model, CTX_LENGTHS[ci])]["baseline"]["peak_memory_gb"] for ci in xs]
    hidden = [recs[(model, CTX_LENGTHS[ci])]["dflash"]["peak_phase_entry"]
              ["peak_components_gb"].get("selected_hidden_bytes", 0.0) for ci in xs]
    ax.bar(xs, delta, 0.5, color=CONFIG_COLORS[(model, CTX_LENGTHS[-1])], alpha=0.55,
           edgecolor="white", linewidth=0.6, label="peak(DFlash) − peak(baseline)")
    ax.plot(xs, hidden, color="#e8a33d", linewidth=2.0, marker="D", markersize=6,
            zorder=4, label="selected target hidden, live at the peak")
    for xv, dv, hv in zip(xs, delta, hidden):
        ax.annotate(f"{dv:.2f}", (xv, dv), textcoords="offset points", xytext=(-17, 3),
                    ha="center", fontsize=6.8)
        ax.annotate(f"{hv:.2f}", (xv, hv), textcoords="offset points", xytext=(17, 3),
                    ha="center", fontsize=6.8, color="#8a5c12")
        if hv:
            ax.annotate(f"{dv / hv:.3f}x", (xv, max(dv, hv)), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=6.6, fontweight="bold",
                        color="#555555")
    ax.set_xticks(range(len(CTX_LENGTHS)))
    shard_ticks(ax, recs, model)
    ax.set_ylabel("GB")
    ax.set_xlabel("context length")
    ax.margins(y=0.28)
    titled(ax, "What the peak delta is made of\n(bar = measured delta, line = the term live at the peak)",
           notes, ["delta", "phase"])
    ax.legend(fontsize=6.2, loc="upper left")
    ax.text(0.02, 0.62,
            "ratio above each column.\n"
            "8B: 1.000 exactly, at every length.\n"
            "9B: 0.875 = 7/8 -- the baseline's own\n"
            "peak-phase activation is larger by one\n"
            "layer's hidden (8 are injected), so the\n"
            "tap retains a tensor the target had\n"
            "allocated anyway and only 7 are new.",
            transform=ax.transAxes, fontsize=6.4, va="top", color="#333333",
            style="italic", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.8))

    handles = [Patch(facecolor=CONFIG_COLORS[(model, c)], label=f"DFlash · {CTX_LABELS[c]}")
               for c in CTX_LENGTHS]
    handles += [Patch(facecolor=BASE_COLORS[c], label=f"Baseline · {CTX_LABELS[c]}")
                for c in CTX_LENGTHS]
    handles += [
        Line2D([0], [0], color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
               label="DFlash (trend)"),
        Line2D([0], [0], color=BASE_LINE_COLOR, linestyle="--",
               marker=MODEL_MARKERS[model], label="Baseline (trend)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.250))
    for i, line in enumerate(sample_footnote(recs, model)):
        fig.text(0.5, 0.232 - i * 0.007, line, ha="center", fontsize=7.6, color="#333333")
    footnote_box(fig, notes, y=0.006, title=V2_HEADER)

    fig.tight_layout(rect=[0, 0.272, 1, 0.974])
    out = os.path.join(OUT_DIR, f"dflash_vs_baseline_{model}_selective_v2.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("wrote", out)


def main():
    recs = load_v2()
    if not recs:
        raise SystemExit("no phase-instrumented records in record_selective/")
    for model in MODELS:
        if any((model, c) in recs for c in CTX_LENGTHS):
            make_figure(recs, model)

    cols = ["model", "context_length", "mode", "num_devices", "mean_latency_s",
            "mean_ttft_s", "mean_decode_s", "aggregate_time_per_output_token_ms",
            "decode_throughput_tok_s", "peak_memory_gb", "peak_memory_reserved_gb",
            "peak_phase", "peak_interval_label", "peak_interval_gb",
            "peak_transient_gb", "v1_transient_gb",
            "target_kv_recorded_gb", "target_kv_at_peak_gb", "target_kv_gap_gb",
            "target_weight_gb", "target_forwards_per_req", "mean_output_tokens",
            "speedup_tpot", "ratio_throughput", "ratio_target_forwards", "ratio_memory",
            "peak_delta_vs_baseline_gb", "selected_hidden_at_peak_gb"]
    path = os.path.join(OUT_DIR, "dflash_vs_baseline_metrics_selective_v2.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model in MODELS:
            for ctx in CTX_LENGTHS:
                if (model, ctx) not in recs:
                    continue
                r = recs[(model, ctx)]
                rt = ratios(r)
                delta = r["dflash"]["peak_memory_gb"] - r["baseline"]["peak_memory_gb"]
                for mode in ("dflash", "baseline"):
                    s = r[mode]
                    phase = s["peak_phase_entry"]
                    w.writerow([
                        model, ctx, mode, r["num_devices"],
                        round(s["mean_latency_s"], 4), round(s["mean_ttft_s"], 4),
                        round(s["mean_decode_s"], 4),
                        round(s["aggregate_time_per_output_token_s"] * 1000, 3),
                        round(s["decode_throughput_tok_s"], 3),
                        round(s["peak_memory_gb"], 4),
                        round(s["peak_memory_reserved_gb"], 4),
                        s["peak_phase"], phase["peak_interval_label"],
                        round(phase["max_interval_peak_gb"], 4),
                        round(s["real_transient_gb"], 4), round(s["old_transient_gb"], 4),
                        round(s["target_kv_recorded_gb"], 4),
                        round(s["target_kv_at_peak_gb"], 4),
                        round(s["target_kv_at_peak_gb"] - s["target_kv_recorded_gb"], 4),
                        round(s["target_weight_gb"], 4),
                        round(s["target_forwards_per_req"], 2),
                        round(s["mean_output_tokens"], 2),
                    ] + ([round(rt["speedup_tpot"], 4), round(rt["ratio_throughput"], 4),
                          round(rt["ratio_target_forwards"], 4), round(rt["ratio_memory"], 4),
                          round(delta, 4),
                          round(phase["peak_components_gb"].get("selected_hidden_bytes", 0), 4)]
                         if mode == "dflash" else ["", "", "", "", "", ""]))
    print("wrote", path)


if __name__ == "__main__":
    main()
