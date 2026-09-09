"""Phase-by-phase memory and draft-stage profile (v2, new figure).

This figure has no v1 counterpart: none of what it plots existed in the
pre-instrumentation records. It answers three questions the aggregate panels
in plot_summary_v2.py can only summarise.

**What is resident at each point of a request?** Panels 1-3 walk the twelve
phases in execution order and stack the live components at each one. Two things
show up that a peak number cannot: on Qwen3.5-9B the target KV collapses by an
order of magnitude the moment decoding starts -- its gated delta-rule layers
release a sequence-length prefill buffer -- and the prompt-length context
feature stays allocated from the prefill concat until the next context-feature
build, several phases after its last read.

**Where does the drafter's time go, and does it scale?** Panels 5-7 split a
draft forward into its five stages, separately for the first call (which
projects the whole context into the draft KV) and the steady-state calls (which
append a block). Only the first call scales with context.

**What of this is DFlash's, and what would the target have cost anyway?**
Panels 8-9. The baseline column is the control: a term that appears there is
the target's, whatever it does to the DFlash peak.

Writes one figure plus a tidy CSV to visualization_selective/.
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
    CONFIG_COLORS, CTX_LABELS, CTX_LENGTHS, DFLASH_OVERHEAD, MODEL_COLORS,
    MODEL_MARKERS, MODELS, Notes, OUT_DIR, PEAK_COMPONENTS, PHASE_ORDER,
    PHASE_SHORT, STAGE_COMPONENTS, TARGET_OVERHEAD, UNATTRIBUTED_COLOR,
    bar_positions, configs, ctx_axis, footnote_box, load_v2, phases_of,
    shard_ticks, sharded_note, short_model, titled,
)

V2_HEADER = (
    "Every panel here is new — the pre-instrumentation records carried none of these fields, so there is no v1 "
    "figure to compare against. The notes say what each quantity is and where it can mislead"
)


def build_notes(recs):
    n = Notes()
    n.add(
        "probe", "added",
        "Each phase records the live component split probed at the *close* of its largest occurrence. A "
        "component that shrinks during the phase is therefore read at its post-shrink size, and the difference "
        "lands in 'unattributed'. That is exactly what happens to Qwen3.5-9B at 'D4 target verify': the GDN "
        "prefill buffer is live when the interval opens and gone when it closes, so target KV reads 2.05 GB "
        "against a 24 GB unattributed remainder. Read the two together at that phase, not separately.",
    )
    n.add(
        "gdn", "added",
        "Qwen3.5-9B's target KV drops from 26.05 GB to 2.05 GB at 64K on the first decode forward: 24 of its 32 "
        "layers are gated delta-rule, and their cache holds a sequence-length prefill buffer that is released "
        "once decoding starts. It is a target-architecture cost -- the same collapse is in the baseline's phase "
        "table, which never calls the drafter -- and must not be read as DFlash overhead.",
    )
    n.add(
        "ctxfeat", "added",
        "The context feature is built at prefill over the whole prompt and read once, by the first draft "
        "forward. It then stays allocated through the draft rollback, the draft logits, the target verify and "
        "the verify rollback, until the next context-feature build replaces it -- 2.50 GB on 8B and 4.00 GB on "
        "9B at 64K, held across phases that do not use it. Inference semantics are unchanged here; the panel "
        "records the lifetime rather than shortening it.",
    )
    n.add(
        "envelope", "added",
        "A phase reports its single largest occurrence, and different phases peak in different decode "
        "iterations, so a curve across phases is an envelope over the request rather than one timeline. The "
        "prompt-length context feature plateau (P4 through D5) is the first decode iteration; the dip at "
        "D1' is real and means the steady-state draft forward peaked later, when only the block-sized feature "
        "was live -- not that the feature was freed and rebuilt between D1 and D2.",
    )
    n.add(
        "stages", "added",
        "first_draft_stage_s / mean_steady_*_s, new fields. A draft forward is timed in five stages with CUDA "
        "events: fc+hidden_norm over the injected feature, the context K/V projection, the cache append, the "
        "drafter's own SDPA, and the target's lm_head. The first call runs them over the whole context, every "
        "later call over one block.",
    )
    shard = sharded_note(recs)
    if shard:
        n.add("shard", "revised", shard)
    return n


def phase_axis(ax, names):
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PHASE_SHORT[n] for n in names], rotation=55, ha="right",
                       fontsize=6.6)
    ax.set_xlim(-0.4, len(names) - 0.6)


def live_stack(ax, entry, names, model, ctx):
    """Stack the live components across phases, with the interval peak on top."""
    idx = {n: i for i, n in enumerate(names)}
    pairs = [(n, p) for n, p in phases_of(entry, "dflash") if n in idx]
    xs = [idx[n] for n, _ in pairs]
    series = []
    for key, _label, _color in PEAK_COMPONENTS:
        series.append([(p["peak_components_gb"].get(key) or 0.0) for _n, p in pairs])
    ax.stackplot(xs, *series, colors=[c for _k, _l, c in PEAK_COMPONENTS], alpha=0.92,
                 edgecolor="white", linewidth=0.4)
    resident = np.sum(series, axis=0)
    peak = np.array([p["max_interval_peak_gb"] for _n, p in pairs])
    ax.fill_between(xs, resident, peak, color=UNATTRIBUTED_COLOR, alpha=0.55,
                    hatch="///", edgecolor="white", linewidth=0.0)
    ax.plot(xs, peak, color="#222222", linewidth=1.6, marker="o", markersize=3.4,
            zorder=5)
    for xv, pv in zip(xs, peak):
        ax.annotate(f"{pv:.0f}", (xv, pv), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=6.0, color="#222222")
    phase_axis(ax, names)
    ax.set_ylabel("GB live at that phase")
    ax.set_ylim(0, float(peak.max()) * 1.45)


def stage_bars(ax, recs, bucket):
    """Stacked per-stage draft-forward time, one column per configuration."""
    _x, offsets, width = bar_positions()
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]["dflash"]
            bottom = 0.0
            for key, _label, color in STAGE_COMPONENTS:
                val = (s["first_draft_stage_s"][key] if bucket == "first"
                       else (s[f"mean_steady_{key}_s"] or 0.0)) * 1000.0
                ax.bar(offsets[mi][ci], val, width, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                bottom += val
            total = (s["mean_first_draft_forward_s"] if bucket == "first"
                     else s["mean_steady_draft_forward_s"]) * 1000.0
            ax.text(offsets[mi][ci], bottom, f"Σ{bottom:.0f}" if bucket == "first"
                    else f"Σ{bottom:.1f}", ha="center", va="bottom", fontsize=6.4)
            ax.scatter(offsets[mi][ci], total, marker="_", s=200, linewidth=1.8,
                       color="#222222", zorder=4)
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)


def main():
    recs = load_v2()
    if not recs:
        raise SystemExit("no phase-instrumented records in record_selective/")
    cfgs = configs(recs)
    notes = build_notes(recs)
    names = [n for n in PHASE_ORDER]
    idx = {n: i for i, n in enumerate(names)}

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "figure.dpi": 130,
    })
    fig, axes = plt.subplots(3, 3, figsize=(19.5, 21.5))
    fig.suptitle(
        "DFlash phase-local memory and draft-stage profile — LongBench-E, hidden_states=selective\n"
        "Qwen3-8B vs Qwen3.5-9B @ 4K / 8K / 16K / 32K / 64K · phases in execution order · block=16 · gamma=15",
        fontsize=12.5, fontweight="bold", y=0.992,
    )
    _x, offsets, width = bar_positions()

    # (1)(2) what is live at each phase, at the longest context of each model
    for col, model in enumerate(MODELS):
        ctx = max(c for c in CTX_LENGTHS if (model, c) in recs)
        ax = axes[0, col]
        live_stack(ax, recs[(model, ctx)], names, model, ctx)
        titled(ax, f"What is live at each phase — {model} @ {CTX_LABELS[ctx]}\n"
                   f"(stack = probed components, line = interval peak)",
               notes, ["probe", "gdn"] if model == "qwen3.5-9b" else ["probe"])
        ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in PEAK_COMPONENTS]
                  + [Patch(facecolor=UNATTRIBUTED_COLOR, hatch="///", alpha=0.55,
                           label="unattributed (activation, or a\ncomponent that shrank in-phase)")],
                  fontsize=5.6, loc="upper right", ncol=1)

    # (3) the context feature's lifetime, every configuration
    ax = axes[0, 2]
    for model, ctx in cfgs:
        pairs = [(n, p) for n, p in phases_of(recs[(model, ctx)], "dflash") if n in idx]
        xs = [idx[n] for n, _ in pairs]
        ys = [(p["peak_components_gb"].get("context_feature_bytes") or 0.0)
              for _n, p in pairs]
        ax.plot(xs, ys, color=CONFIG_COLORS[(model, ctx)], linewidth=1.7,
                marker=MODEL_MARKERS[model], markersize=3.6,
                label=f"{short_model(model)} {CTX_LABELS[ctx]}")
    phase_axis(ax, names)
    ax.set_ylabel("context feature live (GB)")
    titled(ax, "Context feature live at each phase's largest occurrence\n"
               "(plateau = the prompt-length feature; dip = a phase that peaked later)",
           notes, ["ctxfeat", "envelope"])
    ax.legend(fontsize=5.8, ncol=2, loc="upper left")
    ax.margins(y=0.30)
    ax.set_ylim(bottom=-0.12)

    # (4) the phase where the target's own cache collapses
    ax = axes[1, 0]
    verify = "decode: target verify"
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            entry = recs[(model, ctx)]
            for pos, (mode, hatch) in enumerate((("dflash", None), ("baseline", ".."))):
                phase = entry[mode]["phase_memory"].get(verify)
                if phase is None:
                    continue
                x = offsets[mi][ci] + (pos - 0.5) * width / 2
                parts = phase["peak_components_gb"]
                bottom = 0.0
                for key, _label, color in PEAK_COMPONENTS:
                    val = parts.get(key) or 0.0
                    ax.bar(x, val, width / 2, bottom=bottom, color=color,
                           edgecolor="white", linewidth=0.4, hatch=hatch)
                    bottom += val
                ax.bar(x, phase["peak_unattributed_gb"], width / 2, bottom=bottom,
                       color=UNATTRIBUTED_COLOR, edgecolor="white", linewidth=0.4,
                       hatch="///")
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "Split at 'D4 target verify' (left = DFlash, right dotted = baseline)\n"
               "(hatched remainder is the released GDN buffer on 9B)", notes, ["gdn", "probe"])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in PEAK_COMPONENTS]
              + [Patch(facecolor=UNATTRIBUTED_COLOR, hatch="///", label="unattributed")],
              fontsize=5.6, loc="upper left")

    # (5) first draft forward, by stage
    ax = axes[1, 1]
    stage_bars(ax, recs, "first")
    shard_ticks(ax, recs)
    ax.set_ylabel("ms (one call per request)")
    titled(ax, "First draft forward, by stage\n(Σ = stage sum, — = measured call)",
           notes, ["stages", "shard"])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in STAGE_COMPONENTS],
              fontsize=6.2, loc="upper left")

    # (6) steady-state draft forward, by stage
    ax = axes[1, 2]
    stage_bars(ax, recs, "steady")
    shard_ticks(ax, recs)
    ax.set_ylabel("ms (mean over steady calls)")
    titled(ax, "Steady-state draft forward, by stage\n(same five stages, one block instead of S)",
           notes, ["stages", "shard"])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in STAGE_COMPONENTS],
              fontsize=6.2, loc="upper left")

    # (7) which stages scale with the context, and which do not
    ax = axes[2, 0]
    for key, label, color in STAGE_COMPONENTS:
        for model in MODELS:
            xs = [c for c in CTX_LENGTHS if (model, c) in recs]
            ys = [recs[(model, c)]["dflash"]["first_draft_stage_s"][key] * 1000 for c in xs]
            ax.plot(xs, ys, color=color, linewidth=1.7,
                    linestyle="-" if model == MODELS[0] else "--",
                    marker=MODEL_MARKERS[model], markersize=4.5,
                    label=label if model == MODELS[0] else None)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(CTX_LENGTHS)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.set_ylabel("ms in the first draft forward (log)")
    titled(ax, "First-call stage scaling\n(solid = 8B, dashed = 9B; slope 1 = linear in S)",
           notes, ["stages", "shard"])
    ax.legend(fontsize=6.2, ncol=2, loc="upper left")
    ax.margins(y=0.28)

    # (8) what the first call builds against what steady state keeps
    ax = axes[2, 1]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            s = recs[(model, ctx)]["dflash"]
            col = CONFIG_COLORS[(model, ctx)]
            ax.bar(offsets[mi][ci] - width / 4, s["max_first_draft_cache_gb"], width / 2,
                   color=col, edgecolor="white", linewidth=0.5)
            ax.bar(offsets[mi][ci] + width / 4, s["max_steady_draft_cache_gb"], width / 2,
                   color=col, alpha=0.42, edgecolor="white", linewidth=0.5)
            ax.scatter(offsets[mi][ci] - width / 4, s["max_first_draft_transient_gb"],
                       marker="_", s=140, linewidth=1.8, color="#7a1f1f", zorder=4)
            ax.scatter(offsets[mi][ci] + width / 4, s["max_steady_draft_transient_gb"],
                       marker="_", s=140, linewidth=1.8, color="#7a1f1f", zorder=4)
            ratio = s["draft_kv_first_over_steady"]
            ax.text(offsets[mi][ci],
                    max(s["max_first_draft_cache_gb"], s["max_first_draft_transient_gb"]),
                    f"{ratio:.1f}x", ha="center", va="bottom", fontsize=6.4,
                    fontweight="bold",
                    color="#c0504d" if ratio > 1.5 else "#333333")
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    ax.set_ylabel("GB")
    titled(ax, "Draft KV: what the first call builds vs what steady state keeps\n"
               "(solid = first, pale = steady, — = borrowed at that call)", notes, ["stages"])
    ax.legend(handles=[Patch(facecolor="#888888", label="draft KV after the first call"),
                       Patch(facecolor="#888888", alpha=0.42, label="draft KV in steady state"),
                       Line2D([0], [0], color="#7a1f1f", marker="_", linestyle="None",
                              markersize=10, label="transient borrowed by that call")],
              fontsize=6.0, loc="upper left")

    # (9) DFlash's own footprint against the target's
    ax = axes[2, 2]
    for mi, model in enumerate(MODELS):
        for ci, ctx in enumerate(CTX_LENGTHS):
            if (model, ctx) not in recs:
                continue
            split = recs[(model, ctx)]["dflash"]["overhead_split"]
            for pos, comps in enumerate((TARGET_OVERHEAD, DFLASH_OVERHEAD)):
                x = offsets[mi][ci] + (pos - 0.5) * width / 2
                bottom = 0.0
                for key, _label, color in comps:
                    val = split.get(key) or 0.0
                    ax.bar(x, val, width / 2, bottom=bottom, color=color,
                           edgecolor="white", linewidth=0.4)
                    bottom += val
                ax.text(x, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=6.2)
            ax.text(offsets[mi][ci], 0, short_model(model), ha="center", va="top",
                    fontsize=6.0, color=MODEL_COLORS[model])
    ctx_axis(ax)
    shard_ticks(ax, recs)
    ax.set_ylabel("GB")
    titled(ax, "DFlash's footprint vs the target's\n(left = target terms, right = DFlash-specific)",
           notes, ["gdn"])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in TARGET_OVERHEAD]
              + [Patch(facecolor=c, label=l) for _k, l, c in DFLASH_OVERHEAD],
              fontsize=5.8, loc="upper left")

    handles = [Patch(facecolor=CONFIG_COLORS[(m, c)], label=f"{m} · {CTX_LABELS[c]}")
               for m in MODELS for c in CTX_LENGTHS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.0,
               frameon=False, bbox_to_anchor=(0.5, 0.278))
    footnote_box(fig, notes, y=0.006, title=V2_HEADER)

    fig.tight_layout(rect=[0, 0.300, 1, 0.976])
    png = os.path.join(OUT_DIR, "dflash_phase_profile_selective_v2.png")
    fig.savefig(png)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)

    # One row per (configuration, mode, phase): the tidy form of panels 1-4.
    cols = ["model", "context_length", "mode", "num_devices", "phase", "phase_index",
            "is_peak_phase", "mean_occurrences_per_sample", "peak_interval_label",
            "mean_interval_peak_gb", "max_interval_peak_gb",
            "peak_allocated_before_gb", "peak_allocated_after_gb", "peak_transient_gb",
            "peak_unattributed_gb", "target_weight_gb", "target_kv_gb",
            "draft_weight_gb", "draft_kv_gb", "selected_hidden_gb", "context_feature_gb"]
    csv_path = os.path.join(OUT_DIR, "dflash_phase_metrics_selective_v2.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for model, ctx in cfgs:
            entry = recs[(model, ctx)]
            for mode in ("dflash", "baseline"):
                for name, phase in phases_of(entry, mode):
                    p = phase["peak_components_gb"]
                    w.writerow([
                        model, ctx, mode, entry["num_devices"], name,
                        idx.get(name, ""), name == entry[mode]["peak_phase"],
                        round(phase["mean_occurrences_per_sample"], 3),
                        phase["peak_interval_label"],
                        round(phase["mean_interval_peak_gb"], 4),
                        round(phase["max_interval_peak_gb"], 4),
                        round(phase["peak_allocated_before_gb"], 4),
                        round(phase["peak_allocated_after_gb"], 4),
                        round(phase["peak_transient_gb"], 4),
                        round(phase["peak_unattributed_gb"], 4),
                        round(p.get("target_weight_bytes", 0), 4),
                        round(p.get("target_kv_bytes", 0), 4),
                        round(p.get("draft_weight_bytes", 0), 4),
                        round(p.get("draft_kv_bytes", 0), 4),
                        round(p.get("selected_hidden_bytes", 0), 4),
                        round(p.get("context_feature_bytes", 0), 4),
                    ])
    print("wrote", png)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
