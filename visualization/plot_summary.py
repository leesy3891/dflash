"""Plot DFlash LongBench-E summary metrics (benchmark-wide averages, `summary` only).

Reads record/*.json, uses only summary.dflash (baseline and acceptance_rate are
excluded from the comparison), and writes a single multi-panel figure plus a
tidy CSV to visualization/.
"""

import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "visualization")

CTX_LENGTHS = [4096, 8192, 16384, 32768, 65536]
CTX_LABELS = {4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K"}
MODELS = ["qwen3-8b", "qwen3.5-9b"]

# One color per (model, context_length) -- reused in every panel.
CONFIG_COLORS = {
    ("qwen3-8b", 4096): "#c6dbef",
    ("qwen3-8b", 8192): "#6baed6",
    ("qwen3-8b", 16384): "#2171b5",
    ("qwen3-8b", 32768): "#08519c",
    ("qwen3-8b", 65536): "#08306b",
    ("qwen3.5-9b", 4096): "#fdd0a2",
    ("qwen3.5-9b", 8192): "#fdae6b",
    ("qwen3.5-9b", 16384): "#f16913",
    ("qwen3.5-9b", 32768): "#d94801",
    ("qwen3.5-9b", 65536): "#8c2d04",
}
MODEL_COLORS = {"qwen3-8b": "#2171b5", "qwen3.5-9b": "#e6550d"}
MODEL_MARKERS = {"qwen3-8b": "o", "qwen3.5-9b": "s"}

# One color per memory-overhead component.
MEM_COMPONENTS = [
    ("draft_weight_gb", "Draft weights", "#3b4d8f"),
    ("max_draft_cache_gb", "Draft KV cache", "#7fb3d5"),
    ("max_target_hidden_states_gb", "Target hidden states", "#e8a33d"),
    ("max_context_feature_gb", "Context feature", "#6aa84f"),
]
# One color per decode-time component.
TIME_COMPONENTS = [
    ("draft_forward_s", "Draft forward", "#3b4d8f"),
    ("context_feature_s", "Context feature", "#6aa84f"),
    ("target_forward_s", "Target forward (verify)", "#c0504d"),
    ("other_s", "Other (sampling/overhead)", "#b0b0b0"),
]


def load_records():
    recs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "record", "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        s = d["summary"]["dflash"]  # benchmark-wide average, ignore per-sample entries
        decode_total = s["drafter_latency_s"] / s["drafter_share_of_decode"]
        s = dict(s)
        s["decode_total_s"] = decode_total
        s["other_s"] = decode_total - s["drafter_latency_s"] - s["target_forward_s"]
        s["mean_decode_s"] = s["mean_latency_s"] - s["mean_ttft_s"]
        s["num_samples"] = d["num_samples"]
        s["n_datasets"] = len({x["task"] for x in d["samples"]})
        s["n_composed"] = sum(1 for x in d["samples"] if x.get("composed"))
        recs[(d["model_name"], d["context_length"])] = s
    return recs


def configs(recs):
    return [(m, c) for m in MODELS for c in CTX_LENGTHS if (m, c) in recs]


def bar_positions(n_ctx=len(CTX_LENGTHS), n_model=2, width=0.38):
    x = np.arange(n_ctx, dtype=float)
    return x, [x + (i - (n_model - 1) / 2) * width for i in range(n_model)], width


def grouped_bar(ax, recs, key, scale=1.0, fmt="{:.2f}"):
    x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            val = recs[(model, ctx)][key] * scale
            ax.bar(offsets[mi][ci], val, width,
                   color=CONFIG_COLORS[(model, ctx)], edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], val, fmt.format(val), ha="center", va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, "8B" if model == "qwen3-8b" else "9B",
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.margins(y=0.18)


def stacked_bar(ax, recs, components, scale=1.0, total_fmt="{:.2f}"):
    x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]
            bottom = 0.0
            for key, _label, color in components:
                val = (s.get(key) or 0.0) * scale
                ax.bar(offsets[mi][ci], val, width, bottom=bottom,
                       color=color, edgecolor="white", linewidth=0.5)
                bottom += val
            ax.text(offsets[mi][ci], bottom, total_fmt.format(bottom),
                    ha="center", va="bottom", fontsize=6.8)
            # model tag under each stack
            ax.text(offsets[mi][ci], 0, "8B" if model == "qwen3-8b" else "9B",
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.margins(y=0.20)


def line_plot(ax, recs, key, scale=1.0, fmt="{:.2f}"):
    for model in MODELS:
        xs, ys = [], []
        for ctx in CTX_LENGTHS:
            if (model, ctx) not in recs:
                continue
            xs.append(ctx)
            ys.append(recs[(model, ctx)][key] * scale)
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
    """Per-context-length sample bookkeeping, shown under the color legend."""
    parts = []
    for ctx in CTX_LENGTHS:
        got = [recs[(m, ctx)] for m in MODELS if (m, ctx) in recs]
        if not got:
            continue
        n = " / ".join(str(g["num_samples"]) for g in got) if len({g["num_samples"] for g in got}) > 1 \
            else str(got[0]["num_samples"])
        ds = " / ".join(str(g["n_datasets"]) for g in got) if len({g["n_datasets"] for g in got}) > 1 \
            else str(got[0]["n_datasets"])
        comp = max(g["n_composed"] for g in got)
        extra = f", {comp} composed" if comp else ""
        parts.append(f"{CTX_LABELS[ctx]}: n={n} ({ds} datasets{extra})")
    line1 = "Samples per context length [8B / 9B where they differ] — " + "  ·  ".join(parts)
    tok = "  ·  ".join(
        f"{CTX_LABELS[c]}: " + " / ".join(f"{recs[(m, c)]['mean_output_tokens']:.0f}"
                                          for m in MODELS if (m, c) in recs)
        for c in CTX_LENGTHS if any((m, c) in recs for m in MODELS))
    line2 = ("Mean output tokens per request [8B / 9B] — " + tok +
             "   (long-context runs emit far fewer tokens; compare per-token metrics, not per-request ones)")
    return [line1, line2]


def main():
    recs = load_records()
    cfgs = configs(recs)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.dpi": 130,
    })

    fig, axes = plt.subplots(3, 3, figsize=(19.5, 14.0))
    fig.suptitle(
        "DFlash speculative decoding — LongBench-E benchmark-wide averages "
        "(Qwen3-8B vs Qwen3.5-9B @ 4K / 8K / 16K / 32K / 64K, block=16, gamma=15, greedy)",
        fontsize=13, fontweight="bold", y=0.985,
    )

    # (1) acceptance length
    ax = axes[0, 0]
    line_plot(ax, recs, "mean_acceptance_length")
    ax.set_ylabel("tokens / verify step")
    ax.set_title("Mean acceptance length\n(accepted tokens per verify step)")

    # (2) acceptance length distribution
    ax = axes[0, 1]
    for model, ctx in cfgs:
        hist = np.asarray(recs[(model, ctx)]["acceptance_length_histogram"], dtype=float)
        ax.plot(np.arange(len(hist)), hist * 100, marker=MODEL_MARKERS[model], markersize=3.5,
                color=CONFIG_COLORS[(model, ctx)], linewidth=1.5,
                label=f"{model} {CTX_LABELS[ctx]}")
    ax.set_xlabel("accepted tokens in a verify step")
    ax.set_ylabel("share of verify steps (%)")
    ax.set_title("Acceptance length distribution\n(16 = full \u03b3 block accepted)")
    ax.legend(fontsize=6.2, ncol=2)

    # (3) peak memory
    ax = axes[0, 2]
    grouped_bar(ax, recs, "peak_memory_gb", fmt="{:.1f}")
    ax.set_ylabel("GB")
    ax.set_title("Peak GPU memory\n(allocated, incl. draft overhead)")

    # (4) draft memory overhead breakdown
    ax = axes[1, 0]
    stacked_bar(ax, recs, MEM_COMPONENTS)
    ax.set_ylabel("GB")
    ax.set_title("Draft-side memory overhead breakdown\n(total = draft_overhead_gb)")
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in MEM_COMPONENTS],
              fontsize=7, loc="upper left")

    # (5) end-to-end latency (TTFT + decode)
    ax = axes[1, 1]
    x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]
            col = CONFIG_COLORS[(model, ctx)]
            ax.bar(offsets[mi][ci], s["mean_ttft_s"], width, color=col,
                   alpha=0.45, edgecolor="white", linewidth=0.6)
            ax.bar(offsets[mi][ci], s["mean_decode_s"], width, bottom=s["mean_ttft_s"],
                   color=col, edgecolor="white", linewidth=0.6)
            ax.text(offsets[mi][ci], s["mean_latency_s"], f"{s['mean_latency_s']:.1f}",
                    ha="center", va="bottom", fontsize=6.8)
            ax.text(offsets[mi][ci], 0, "8B" if model == "qwen3-8b" else "9B",
                    ha="center", va="top", fontsize=6.0, color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.set_ylabel("seconds / request")
    ax.set_title("Mean end-to-end latency\n(light = TTFT, solid = decode)")
    ax.margins(y=0.18)

    # (6) per-token decode latency
    ax = axes[1, 2]
    grouped_bar(ax, recs, "mean_time_per_output_token_s", scale=1000.0, fmt="{:.1f}")
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            agg = recs[(model, ctx)]["aggregate_time_per_output_token_s"] * 1000
            ax.scatter(offsets[mi][ci], agg, marker="_", s=260, linewidth=2.0,
                       color="#222222", zorder=4)
    ax.set_ylabel("ms / output token")
    ax.set_title("Per-token decode latency\n(bar = per-request mean, \u2014 = aggregate over all tokens)")

    # (7) decode throughput
    ax = axes[2, 0]
    line_plot(ax, recs, "decode_throughput_tok_s", fmt="{:.1f}")
    ax.set_ylabel("tokens / s")
    ax.set_title("Decode throughput\n(output tokens / s)")

    # (8) decode time breakdown
    ax = axes[2, 1]
    stacked_bar(ax, recs, TIME_COMPONENTS, total_fmt="{:.0f}")
    ax.set_ylabel("seconds (32 requests total)")
    ax.set_title("Decode wall-clock breakdown\n(sum over 32 requests)")
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in TIME_COMPONENTS],
              fontsize=7, loc="upper left")

    # (9) drafter share of decode
    ax = axes[2, 2]
    grouped_bar(ax, recs, "drafter_share_of_decode", scale=100.0, fmt="{:.1f}")
    ax.set_ylabel("% of decode time")
    ax.set_title("Drafter share of decode time\n(draft forward + context feature)")

    # shared config legend
    handles = [Patch(facecolor=CONFIG_COLORS[(m, c)], label=f"{m} · {CTX_LABELS[c]}")
               for m in MODELS for c in CTX_LENGTHS]
    handles += [Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                       label=f"{m} (trend)") for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.038))
    for i, line in enumerate(sample_footnote(recs)):
        fig.text(0.5, 0.024 - i * 0.013, line, ha="center", fontsize=7.6,
                 color="#333333")

    fig.tight_layout(rect=[0, 0.062, 1, 0.965])
    png = os.path.join(OUT_DIR, "dflash_summary_overview.png")
    fig.savefig(png)
    fig.savefig(os.path.join(OUT_DIR, "dflash_summary_overview.pdf"))
    plt.close(fig)

    # tidy CSV of everything plotted
    cols = ["model", "context_length", "mean_acceptance_length", "peak_memory_gb",
            "draft_overhead_gb", "draft_weight_gb", "max_draft_cache_gb",
            "max_target_hidden_states_gb", "max_context_feature_gb", "max_target_cache_gb",
            "mean_latency_s", "mean_ttft_s", "mean_decode_s",
            "mean_time_per_output_token_ms", "aggregate_time_per_output_token_ms",
            "decode_throughput_tok_s", "drafter_share_of_decode_pct",
            "draft_forward_s", "context_feature_s", "target_forward_s", "other_s",
            "mean_output_tokens"]
    csv_path = os.path.join(OUT_DIR, "dflash_summary_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model, ctx in cfgs:
            s = recs[(model, ctx)]
            w.writerow([model, ctx,
                        round(s["mean_acceptance_length"], 4),
                        round(s["peak_memory_gb"], 4),
                        round(s["draft_overhead_gb"], 4),
                        round(s["draft_weight_gb"], 4),
                        round(s["max_draft_cache_gb"], 4),
                        round(s["max_target_hidden_states_gb"], 4),
                        round(s["max_context_feature_gb"], 4),
                        round(s["max_target_cache_gb"], 4),
                        round(s["mean_latency_s"], 4),
                        round(s["mean_ttft_s"], 4),
                        round(s["mean_decode_s"], 4),
                        round(s["mean_time_per_output_token_s"] * 1000, 3),
                        round(s["aggregate_time_per_output_token_s"] * 1000, 3),
                        round(s["decode_throughput_tok_s"], 3),
                        round(s["drafter_share_of_decode"] * 100, 3),
                        round(s["draft_forward_s"], 3),
                        round(s["context_feature_s"], 3),
                        round(s["target_forward_s"], 3),
                        round(s["other_s"], 3),
                        round(s["mean_output_tokens"], 2)])
    print("wrote", png)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
