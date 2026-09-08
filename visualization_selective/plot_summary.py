"""DFlash LongBench-E summary metrics from the selective sweep.

The counterpart of ``visualization/plot_summary.py``, reading
``record_selective/*.json`` (``--hidden-states selective``) instead of
``record/``. Same nine panels, plus three that the earlier records could not
produce, and a marker on every panel whose numbers or measurement moved.

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

from _style import (  # noqa: E402
    BUDGET_COMPONENTS, CONFIG_COLORS, CTX_LABELS, CTX_LENGTHS, MEM_COMPONENTS,
    MODEL_COLORS, MODEL_MARKERS, MODELS, Notes, OUT_DIR, TIME_COMPONENTS,
    bar_positions, configs, ctx_axis, device_note, footnote_box, load_records,
    shard_ticks, titled,
)


def build_notes(recs):
    """Register every way this figure differs from the full-mode one."""
    n = Notes()
    n.add(
        "selective", "revised",
        "hidden_states=selective. The target's residual streams reach the drafter through forward hooks on the "
        "injected layers (HiddenStateTap) instead of output_hidden_states=True, so max_target_hidden_states_gb "
        "counts n_inj layers (5 on 8B, 8 on 9B) rather than L_t+1 (37, 33). The context feature is bit-identical: "
        "mean acceptance length and mean output tokens match the full-mode records at all 10 points.",
    )
    n.add(
        "budget", "added",
        "summary.*.memory_budget, a new record field. Splits the peak into resident terms and the remainder "
        "(transient_gb = peak - resident). Needs target_weight_gb = module_bytes(target), also new, which the "
        "full-mode records do not carry -- they accounted for the drafter's share only.",
    )
    n.add(
        "peaksite", "revised",
        "peak_site now stamps the decode position an interval *opened* at, not the one current when it closed. "
        "Previously an interval that opened at the last prefill layer and was still open when decoding began was "
        "reported as 'target layer N, decode token 0'. A boundary is also closed at the end of prefill for every "
        "configuration, not only block_size>1. peak_site_histogram (added) counts the setting operation per sample.",
    )
    n.add("devices", "revised", device_note(recs))
    return n


# --------------------------------------------------------------------------
# Panel helpers (same shapes as visualization/plot_summary.py)
# --------------------------------------------------------------------------

def grouped_bar(ax, recs, key, scale=1.0, fmt="{:.2f}", source="dflash"):
    x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            val = recs[(model, ctx)][source][key] * scale
            ax.bar(offsets[mi][ci], val, width,
                   color=CONFIG_COLORS[(model, ctx)], edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], val, fmt.format(val), ha="center", va="bottom",
                    fontsize=6.8)
            ax.text(offsets[mi][ci], 0, "8B" if model == "qwen3-8b" else "9B",
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)


def stacked_bar(ax, recs, components, source="dflash", sub=None, total_fmt="{:.2f}"):
    """Stack ``components`` per configuration; ``sub`` selects a nested dict."""
    x, offsets, width = bar_positions()
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
            ax.text(offsets[mi][ci], 0, "8B" if model == "qwen3-8b" else "9B",
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
            ys.append(s[key] * scale)
        ax.plot(xs, ys, color=MODEL_COLORS[model], linewidth=1.8, zorder=1)
        for xv, yv in zip(xs, ys):
            ax.scatter(xv, yv, s=70, zorder=3, marker=MODEL_MARKERS[model],
                       color=CONFIG_COLORS[(model, xv)],
                       edgecolor=MODEL_COLORS[model], linewidth=1.2)
            ax.annotate(fmt.format(yv), (xv, yv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=6.8)
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
    recs = load_records()
    if not recs:
        raise SystemExit("no records in record_selective/ -- run queue/run_selective_sweep.sh first")
    cfgs = configs(recs)
    notes = build_notes(recs)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.dpi": 130,
    })

    fig, axes = plt.subplots(4, 3, figsize=(19.5, 19.0))
    fig.suptitle(
        "DFlash speculative decoding — LongBench-E benchmark-wide averages, hidden_states=selective "
        "(Qwen3-8B vs Qwen3.5-9B @ 4K / 8K / 16K / 32K / 64K, block=16, gamma=15, greedy)",
        fontsize=13, fontweight="bold", y=0.988,
    )

    # (1) acceptance length -- unchanged, identical to the full-mode records
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

    # (3) peak memory -- smaller under selective, and fewer cards
    ax = axes[0, 2]
    grouped_bar(ax, recs, "peak_memory_gb", fmt="{:.1f}")
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "Peak GPU memory\n(allocated, incl. draft overhead)", notes,
           ["selective", "devices"])

    # (4) drafter overhead breakdown -- the hidden-state term is the one that moved
    ax = axes[1, 0]
    stacked_bar(ax, recs, MEM_COMPONENTS)
    ax.set_ylabel("GB")
    titled(ax, "Draft-side memory overhead breakdown\n(total = draft_overhead_gb)",
           notes, ["selective"])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in MEM_COMPONENTS],
              fontsize=7, loc="upper left")

    # (5) whole-process budget -- new field
    ax = axes[1, 1]
    stacked_bar(ax, recs, BUDGET_COMPONENTS, sub="memory_budget")
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "Whole-process memory budget\n(resident terms + transient = peak)",
           notes, ["budget", "devices"])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in BUDGET_COMPONENTS],
              fontsize=6.2, loc="upper left")

    # (6) transient -- new field, and the whole story of the two presets
    ax = axes[1, 2]
    line_plot(ax, recs, "transient_gb", sub="memory_budget")
    for model in MODELS:
        ys = [recs[(model, c)]["dflash"]["memory_budget"]["resident_total_gb"]
              for c in CTX_LENGTHS if (model, c) in recs]
        xs = [c for c in CTX_LENGTHS if (model, c) in recs]
        ax.plot(xs, ys, color=MODEL_COLORS[model], linewidth=1.2, linestyle=":",
                alpha=0.75, zorder=0)
    ax.set_ylabel("GB")
    titled(ax, "Transient (activation) vs resident\n(solid = transient, dotted = resident subtotal)",
           notes, ["budget"])

    # (7) end-to-end latency
    ax = axes[2, 0]
    x, offsets, width = bar_positions()
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
            ax.text(offsets[mi][ci], 0, "8B" if model == "qwen3-8b" else "9B",
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    shard_ticks(ax, recs)
    ax.set_ylabel("seconds / request")
    titled(ax, "Mean end-to-end latency\n(light = TTFT, solid = decode)", notes, ["devices"])

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
    titled(ax, "Per-token decode latency\n(bar = per-request mean, — = aggregate)",
           notes, ["devices"])

    # (9) decode throughput
    ax = axes[2, 2]
    line_plot(ax, recs, "decode_throughput_tok_s", fmt="{:.1f}")
    ax.set_ylabel("tokens / s")
    titled(ax, "Decode throughput\n(output tokens / s)", notes, ["devices"])

    # (10) decode wall-clock breakdown -- unchanged method
    ax = axes[3, 0]
    stacked_bar(ax, recs, TIME_COMPONENTS, total_fmt="{:.0f}")
    ax.set_ylabel("seconds (32 requests total)")
    titled(ax, "Decode wall-clock breakdown\n(sum over 32 requests)")
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in TIME_COMPONENTS],
              fontsize=7, loc="upper left")

    # (11) drafter share of decode -- unchanged method
    ax = axes[3, 1]
    grouped_bar(ax, recs, "drafter_share_of_decode", scale=100.0, fmt="{:.1f}")
    ax.set_ylabel("% of decode time")
    titled(ax, "Drafter share of decode time\n(draft forward + context feature)")

    # (12) where the peak landed -- new field
    ax = axes[3, 2]
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    rows = []
    for model, ctx in cfgs:
        s = recs[(model, ctx)]["dflash"]
        hist = s["peak_site_histogram"]
        op, count = max(hist.items(), key=lambda kv: kv[1])
        total = sum(hist.values())
        rows.append((f"{'8B' if model == 'qwen3-8b' else '9B'} {CTX_LABELS[ctx]:>3}",
                     op, f"{count}/{total}", MODEL_COLORS[model]))
    ax.text(0.02, 0.960, f"{'run':<8}{'operation that set the peak':<26}{'samples':>8}",
            fontsize=7.6, family="DejaVu Sans Mono", va="top", color="#555555")
    for i, (run, op, frac, color) in enumerate(rows):
        ax.text(0.02, 0.905 - i * 0.052, f"{run:<8}{op:<26}{frac:>8}",
                fontsize=7.6, family="DejaVu Sans Mono", va="top", color=color)
    # Keep the closing note inside the axes: ax.text does not clip, so an
    # overflow here is drawn straight across the bottom spine.
    ax.text(0.02, 0.905 - len(rows) * 0.052 - 0.070,
            "Unanimous: the peak is the last prefill layer,\n"
            "not a decode-time allocation. Under\n"
            "hidden_states=full it lands at 'decode: draft\n"
            "forward, decode token 0' instead, where the\n"
            "tuple-holding output object is still live.",
            fontsize=7.2, va="top", color="#333333", style="italic")
    titled(ax, "Where the peak landed\n(peak_site_histogram over 32 samples)",
           notes, ["peaksite", "selective"])

    # shared config legend
    handles = [Patch(facecolor=CONFIG_COLORS[(m, c)], label=f"{m} · {CTX_LABELS[c]}")
               for m in MODELS for c in CTX_LENGTHS]
    handles += [Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                       label=f"{m} (trend)") for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.197))
    for i, line in enumerate(sample_footnote(recs)):
        fig.text(0.5, 0.186 - i * 0.008, line, ha="center", fontsize=7.6, color="#333333")
    footnote_box(fig, notes, y=0.006)

    fig.tight_layout(rect=[0, 0.232, 1, 0.972])
    png = os.path.join(OUT_DIR, "dflash_summary_overview_selective.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)

    cols = ["model", "context_length", "hidden_states", "num_devices",
            "mean_acceptance_length", "peak_memory_gb", "peak_memory_reserved_gb",
            "draft_overhead_gb", "draft_weight_gb", "max_draft_cache_gb",
            "max_target_hidden_states_gb", "max_context_feature_gb", "max_target_cache_gb",
            "target_weight_gb", "resident_total_gb", "transient_gb", "peak_site_operation",
            "peak_site_unanimity", "mean_latency_s", "mean_ttft_s", "mean_decode_s",
            "mean_time_per_output_token_ms", "aggregate_time_per_output_token_ms",
            "decode_throughput_tok_s", "drafter_share_of_decode_pct",
            "draft_forward_s", "context_feature_s", "target_forward_s", "other_s",
            "mean_output_tokens"]
    csv_path = os.path.join(OUT_DIR, "dflash_summary_metrics_selective.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model, ctx in cfgs:
            r = recs[(model, ctx)]
            s, mb = r["dflash"], r["dflash"]["memory_budget"]
            hist = s["peak_site_histogram"]
            op, count = max(hist.items(), key=lambda kv: kv[1])
            w.writerow([
                model, ctx, r["hidden_states"], r["num_devices"],
                round(s["mean_acceptance_length"], 4),
                round(s["peak_memory_gb"], 4), round(s["peak_memory_reserved_gb"], 4),
                round(s["draft_overhead_gb"], 4), round(s["draft_weight_gb"], 4),
                round(s["max_draft_cache_gb"], 4),
                round(s["max_target_hidden_states_gb"], 4),
                round(s["max_context_feature_gb"], 4),
                round(s["max_target_cache_gb"], 4),
                round(s["target_weight_gb"], 4),
                round(mb["resident_total_gb"], 4), round(mb["transient_gb"], 4),
                op, f"{count}/{sum(hist.values())}",
                round(s["mean_latency_s"], 4), round(s["mean_ttft_s"], 4),
                round(s["mean_decode_s"], 4),
                round(s["mean_time_per_output_token_s"] * 1000, 3),
                round(s["aggregate_time_per_output_token_s"] * 1000, 3),
                round(s["decode_throughput_tok_s"], 3),
                round(s["drafter_share_of_decode"] * 100, 3),
                round(s["draft_forward_s"], 3), round(s["context_feature_s"], 3),
                round(s["target_forward_s"], 3), round(s["other_s"], 3),
                round(s["mean_output_tokens"], 2),
            ])
    print("wrote", png)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
