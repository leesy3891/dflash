"""DFlash vs. baseline per model, from the selective sweep.

The counterpart of ``visualization/plot_vs_baseline.py``, reading
``record_selective/*.json`` instead of ``record/``. Only metrics both modes
report are plotted; draft-only metrics stay in plot_summary.py.

The slot that used to be blank now carries the baseline's own memory budget,
which the full-mode records could not produce: it is the panel that shows the
drafter's weights sitting on the card during a run that never calls the
drafter, and therefore why a DFlash-minus-baseline peak delta understates the
drafter's cost.

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

from _style import (  # noqa: E402
    BASE_COLORS, BASE_LINE_COLOR, BASELINE_BUDGET_COMPONENTS, BUDGET_COMPONENTS,
    CONFIG_COLORS, CTX_LABELS, CTX_LENGTHS, MODEL_COLORS, MODEL_MARKERS, MODELS,
    Notes, OUT_DIR, bar_positions, ctx_axis, device_note, footnote_box,
    load_records, shard_ticks, titled, value_text_color,
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
        "selective", "revised",
        "hidden_states=selective. The drafter reads the target's residual streams through forward hooks on the "
        "injected layers rather than output_hidden_states=True, so the tuple the target materialises drops from "
        "L_t+1 layers to n_inj. Only DFlash pays that term, so only the DFlash bars move; the baseline's peak is "
        "unchanged from the full-mode records. Acceptance and output tokens are identical in both modes.",
    )
    n.add(
        "budget", "added",
        "summary.*.memory_budget and target_weight_gb, new record fields, for both configurations. The baseline "
        "panel lists the draft weights it never calls, because the drafter is resident on the same card for the "
        "whole benchmark -- which is why the peak delta between the two bars is not the drafter's cost.",
    )
    note = device_note(recs, model)
    if note:
        n.add("devices", "revised", note)
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
    """DFlash and baseline budgets side by side, each stacked by term."""
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
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 16.5))
    fig.suptitle(
        f"DFlash vs. baseline — {model} on LongBench-E, benchmark-wide averages, "
        f"hidden_states=selective (4K / 8K / 16K / 32K / 64K; draft-only metrics excluded)",
        fontsize=13, fontweight="bold", y=0.988,
    )
    dev = ["devices"] if "devices" in notes._text else []

    # (1) end-to-end latency
    ax = axes[0, 0]
    _, offsets, width = bar_positions()
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
    titled(ax, "Peak GPU memory\n(bar = allocated, — = reserved)", notes,
           ["selective"] + dev)

    # (6) target KV cache -- same target, same context, untouched
    ax = axes[1, 2]
    grouped_bar(ax, recs, model, "max_target_cache_gb", fmt="{:.2f}")
    ax.set_ylabel("GB")
    titled(ax, "Max target KV cache\n(same target model, same context)")

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
            # Speedup and throughput ratio are near-identical by construction,
            # so alternate the label height rather than let them overprint.
            ax.annotate(f"{vv:.2f}", (pv, vv), textcoords="offset points",
                        xytext=(0, 2 + 8 * (ki % 2)), ha="center", fontsize=6.4)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":")
    ax.set_xlabel("context length")
    shard_ticks(ax, recs, model)
    ax.set_ylabel("x baseline  (>1 = DFlash better,\nexcept memory ratio)")
    titled(ax, "DFlash relative to baseline", notes, ["selective"] + dev)
    ax.legend(fontsize=6.0, ncol=2, loc="upper left")
    ax.margins(y=0.30)

    # (9) the budget, both configurations -- new; this slot used to be blank
    ax = axes[2, 2]
    budget_bars(ax, recs, model)
    shard_ticks(ax, recs, model)
    ax.set_ylabel("GB")
    titled(ax, "Memory budget, both configurations\n(stack = resident terms + transient = peak)",
           notes, ["budget"])
    seen, handles = set(), []
    for key, label, color in BUDGET_COMPONENTS + BASELINE_BUDGET_COMPONENTS:
        if label in seen:
            continue
        seen.add(label)
        handles.append(Patch(facecolor=color, label=label))
    ax.legend(handles=handles, fontsize=5.8, loc="upper left")

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
               frameon=False, bbox_to_anchor=(0.5, 0.196))
    for i, line in enumerate(sample_footnote(recs, model)):
        fig.text(0.5, 0.182 - i * 0.009, line, ha="center", fontsize=7.6, color="#333333")
    footnote_box(fig, notes, y=0.006)

    fig.tight_layout(rect=[0, 0.238, 1, 0.972])
    out = os.path.join(OUT_DIR, f"dflash_vs_baseline_{model}_selective.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("wrote", out)


def main():
    recs = load_records()
    if not recs:
        raise SystemExit("no records in record_selective/ -- run queue/run_selective_sweep.sh first")
    for model in MODELS:
        if any((model, c) in recs for c in CTX_LENGTHS):
            make_figure(recs, model)

    cols = ["model", "context_length", "mode", "num_devices", "mean_latency_s",
            "mean_ttft_s", "mean_decode_s", "mean_time_per_output_token_ms",
            "aggregate_time_per_output_token_ms", "decode_throughput_tok_s",
            "peak_memory_gb", "peak_memory_reserved_gb", "max_target_cache_gb",
            "target_weight_gb", "resident_total_gb", "transient_gb",
            "peak_site_operation", "target_forwards_per_req", "mean_output_tokens",
            "speedup_tpot", "ratio_throughput", "ratio_target_forwards", "ratio_memory"]
    path = os.path.join(OUT_DIR, "dflash_vs_baseline_metrics_selective.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model in MODELS:
            for ctx in CTX_LENGTHS:
                if (model, ctx) not in recs:
                    continue
                r = recs[(model, ctx)]
                rt = ratios(r)
                for mode in ("dflash", "baseline"):
                    s, mb = r[mode], r[mode]["memory_budget"]
                    w.writerow([
                        model, ctx, mode, r["num_devices"],
                        round(s["mean_latency_s"], 4), round(s["mean_ttft_s"], 4),
                        round(s["mean_decode_s"], 4),
                        round(s["mean_time_per_output_token_s"] * 1000, 3),
                        round(s["aggregate_time_per_output_token_s"] * 1000, 3),
                        round(s["decode_throughput_tok_s"], 3),
                        round(s["peak_memory_gb"], 4),
                        round(s["peak_memory_reserved_gb"], 4),
                        round(s["max_target_cache_gb"], 4),
                        round(s["target_weight_gb"], 4),
                        round(mb["resident_total_gb"], 4),
                        round(mb["transient_gb"], 4),
                        s["peak_site"]["operation"],
                        round(s["target_forwards_per_req"], 2),
                        round(s["mean_output_tokens"], 2),
                    ] + ([round(rt["speedup_tpot"], 4), round(rt["ratio_throughput"], 4),
                          round(rt["ratio_target_forwards"], 4), round(rt["ratio_memory"], 4)]
                         if mode == "dflash" else ["", "", "", ""]))
    print("wrote", path)


if __name__ == "__main__":
    main()
