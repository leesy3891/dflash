"""DFlash vs. baseline (vanilla AR decoding) per model, across 4K / 8K / 16K.

Only metrics both modes report are plotted; draft-only metrics (acceptance
length/histogram, draft memory overhead, drafter share, decode wall-clock
breakdown) are omitted because the baseline has no counterpart.

Reads record/*.json summary blocks (benchmark-wide averages, `samples` unused)
and writes one figure per model to visualization/.
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

# DFlash colors: identical to dflash_summary_overview.png (one per model+context).
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
# Baseline: neutral grays, same light -> dark progression over context length.
BASE_COLORS = {4096: "#dcdcdc", 8192: "#b3b3b3", 16384: "#8a8a8a",
               32768: "#5c5c5c", 65536: "#303030"}
MODEL_COLORS = {"qwen3-8b": "#2171b5", "qwen3.5-9b": "#e6550d"}
MODEL_MARKERS = {"qwen3-8b": "o", "qwen3.5-9b": "s"}
BASE_LINE_COLOR = "#6e6e6e"

# One color per ratio metric (shared by both model figures).
RATIO_METRICS = [
    ("speedup_tpot", "Decode speedup\n(aggregate s/token)", "#3b4d8f"),
    ("ratio_throughput", "Decode throughput\nratio", "#6aa84f"),
    ("ratio_target_forwards", "Target forward\nreduction", "#c0504d"),
    ("ratio_memory", "Peak memory\nratio", "#e8a33d"),
]


def load_records():
    recs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "record", "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        s, b = dict(d["summary"]["dflash"]), dict(d["summary"]["baseline"])
        n = d["num_samples"]
        s["mean_decode_s"] = s["mean_latency_s"] - s["mean_ttft_s"]
        b["mean_decode_s"] = b["mean_latency_s"] - b["mean_ttft_s"]
        # target forward passes per request: verify steps vs. one step per token
        s["target_forwards_per_req"] = s["total_verify_steps"] / n
        b["target_forwards_per_req"] = b["mean_decode_steps"]
        recs[(d["model_name"], d["context_length"])] = {
            "dflash": s, "baseline": b, "speedup": d["decoding_speedup"], "n": n,
            "n_datasets": len({x["task"] for x in d["samples"]}),
            "n_composed": sum(1 for x in d["samples"] if x.get("composed")),
        }
    return recs


def bar_positions(width=0.38):
    x = np.arange(len(CTX_LENGTHS), dtype=float)
    return x, [x - width / 2, x + width / 2], width


def ctx_axis(ax):
    x, _, _ = bar_positions()
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.margins(y=0.20)


def grouped_bar(ax, recs, model, key, scale=1.0, fmt="{:.2f}", tag=True,
                label_inside=False, top_key=None):
    _, offsets, width = bar_positions()
    for ci, ctx in enumerate(CTX_LENGTHS):
        r = recs[(model, ctx)]
        for mi, (mode, color) in enumerate((("dflash", CONFIG_COLORS[(model, ctx)]),
                                            ("baseline", BASE_COLORS[ctx]))):
            val = r[mode][key] * scale
            ax.bar(offsets[mi][ci], val, width, color=color,
                   edgecolor="white", linewidth=0.6)
            if label_inside:
                r_, g_, b_ = matplotlib.colors.to_rgb(color)
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                ax.text(offsets[mi][ci], val * 0.97, fmt.format(val), ha="center",
                        va="top", fontsize=7.5, fontweight="bold",
                        color="white" if lum < 0.6 else "#222222")
            else:
                y = val
                if top_key is not None:
                    y = max(y, r[mode][top_key] * scale)
                ax.text(offsets[mi][ci], y, fmt.format(val), ha="center",
                        va="bottom", fontsize=6.8)
            if tag:
                ax.text(offsets[mi][ci], 0, "DF" if mode == "dflash" else "BL",
                        ha="center", va="top", fontsize=6.0,
                        color=MODEL_COLORS[model] if mode == "dflash" else BASE_LINE_COLOR)
    ctx_axis(ax)


def line_pair(ax, recs, model, key, scale=1.0, fmt="{:.1f}"):
    for mode in ("dflash", "baseline"):
        xs = CTX_LENGTHS
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
    """Per-context-length sample bookkeeping, shown under the color legend."""
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
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "figure.dpi": 130,
    })
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 14.5))
    axes[2, 2].axis("off")
    fig.suptitle(
        f"DFlash vs. baseline — {model} on LongBench-E, benchmark-wide averages "
        f"(4K / 8K / 16K / 32K / 64K; draft-only metrics excluded)",
        fontsize=13, fontweight="bold", y=0.985,
    )

    # (1) end-to-end latency, TTFT + decode stacked
    ax = axes[0, 0]
    _, offsets, width = bar_positions()
    for ci, ctx in enumerate(CTX_LENGTHS):
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
    ax.set_ylabel("seconds / request")
    ax.set_title("Mean end-to-end latency\n(light = TTFT, solid = decode)")

    # (2) TTFT
    ax = axes[0, 1]
    grouped_bar(ax, recs, model, "mean_ttft_s", fmt="{:.2f}")
    ax.set_ylabel("seconds")
    ax.set_title("Mean TTFT\n(prefill, identical work in both modes)")

    # (3) per-token decode latency
    ax = axes[0, 2]
    grouped_bar(ax, recs, model, "mean_time_per_output_token_s", scale=1000.0, fmt="{:.1f}",
                top_key="aggregate_time_per_output_token_s")
    _, offsets, _ = bar_positions()
    for ci, ctx in enumerate(CTX_LENGTHS):
        r = recs[(model, ctx)]
        for mi, mode in enumerate(("dflash", "baseline")):
            agg = r[mode]["aggregate_time_per_output_token_s"] * 1000
            ax.scatter(offsets[mi][ci], agg, marker="_", s=260, linewidth=2.0,
                       color="#222222", zorder=4)
    ax.set_ylabel("ms / output token")
    ax.set_title("Per-token decode latency\n(bar = per-request mean, — = aggregate)")

    # (4) decode throughput
    ax = axes[1, 0]
    line_pair(ax, recs, model, "decode_throughput_tok_s")
    ax.set_ylabel("tokens / s")
    ax.set_title("Decode throughput\n(output tokens / s)")

    # (5) peak memory
    ax = axes[1, 1]
    grouped_bar(ax, recs, model, "peak_memory_gb", fmt="{:.1f}", label_inside=True)
    for ci, ctx in enumerate(CTX_LENGTHS):
        r = recs[(model, ctx)]
        for mi, mode in enumerate(("dflash", "baseline")):
            res = r[mode]["peak_memory_reserved_gb"]
            ax.scatter(offsets[mi][ci], res, marker="_", s=260, linewidth=2.0,
                       color="#222222", zorder=4)
            ax.text(offsets[mi][ci], res, f"{res:.1f}", ha="center", va="bottom",
                    fontsize=6.8)
    ax.set_ylabel("GB")
    ax.set_title("Peak GPU memory\n(bar = allocated, — = reserved)")

    # (6) target KV cache
    ax = axes[1, 2]
    grouped_bar(ax, recs, model, "max_target_cache_gb", fmt="{:.2f}")
    ax.set_ylabel("GB")
    ax.set_title("Max target KV cache\n(same target model, same context)")

    # (7) target forward passes per request
    ax = axes[2, 0]
    grouped_bar(ax, recs, model, "target_forwards_per_req", fmt="{:.0f}")
    ax.set_ylabel("forward passes / request")
    ax.set_title("Target forward passes per request\n(DF = verify steps, BL = decode steps)")

    # (8) DFlash relative to baseline
    ax = axes[2, 1]
    x = np.arange(len(CTX_LENGTHS), dtype=float)
    w = 0.19
    for ki, (key, label, color) in enumerate(RATIO_METRICS):
        vals = [ratios(recs[(model, c)])[key] for c in CTX_LENGTHS]
        pos = x + (ki - (len(RATIO_METRICS) - 1) / 2) * w
        ax.bar(pos, vals, w, color=color, edgecolor="white", linewidth=0.6, label=label)
        for pv, vv in zip(pos, vals):
            ax.text(pv, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.set_ylabel("x baseline  (>1 = DFlash better,\nexcept memory ratio)")
    ax.set_title("DFlash relative to baseline")
    ax.legend(fontsize=6.0, ncol=2, loc="upper left")
    ax.margins(y=0.30)

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
               frameon=False, bbox_to_anchor=(0.5, 0.040))
    for i, line in enumerate(sample_footnote(recs, model)):
        fig.text(0.5, 0.023 - i * 0.013, line, ha="center", fontsize=7.6,
                 color="#333333")
    fig.tight_layout(rect=[0, 0.068, 1, 0.958])
    out = os.path.join(OUT_DIR, f"dflash_vs_baseline_{model}.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("wrote", out)


def main():
    recs = load_records()
    for model in MODELS:
        make_figure(recs, model)

    cols = ["model", "context_length", "mode", "mean_latency_s", "mean_ttft_s",
            "mean_decode_s", "mean_time_per_output_token_ms",
            "aggregate_time_per_output_token_ms", "decode_throughput_tok_s",
            "peak_memory_gb", "peak_memory_reserved_gb", "max_target_cache_gb",
            "target_forwards_per_req", "mean_output_tokens",
            "speedup_tpot", "ratio_throughput", "ratio_target_forwards", "ratio_memory"]
    path = os.path.join(OUT_DIR, "dflash_vs_baseline_metrics.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model in MODELS:
            for ctx in CTX_LENGTHS:
                r = recs[(model, ctx)]
                rt = ratios(r)
                for mode in ("dflash", "baseline"):
                    s = r[mode]
                    w.writerow([
                        model, ctx, mode,
                        round(s["mean_latency_s"], 4), round(s["mean_ttft_s"], 4),
                        round(s["mean_decode_s"], 4),
                        round(s["mean_time_per_output_token_s"] * 1000, 3),
                        round(s["aggregate_time_per_output_token_s"] * 1000, 3),
                        round(s["decode_throughput_tok_s"], 3),
                        round(s["peak_memory_gb"], 4),
                        round(s["peak_memory_reserved_gb"], 4),
                        round(s["max_target_cache_gb"], 4),
                        round(s["target_forwards_per_req"], 2),
                        round(s["mean_output_tokens"], 2),
                    ] + ([round(rt["speedup_tpot"], 4), round(rt["ratio_throughput"], 4),
                          round(rt["ratio_target_forwards"], 4), round(rt["ratio_memory"], 4)]
                         if mode == "dflash" else ["", "", "", ""]))
    print("wrote", path)


if __name__ == "__main__":
    main()
