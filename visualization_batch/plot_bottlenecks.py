"""The rest of the batch sweep's bottlenecks (plan §9, F1–F3, F5, F6, F8–F10).

    python visualization_batch/plot_bottlenecks.py

One figure per model for each of:

* ``batch_throughput_<model>`` — F1 decode throughput, F2 same-B speedup, F9
  acceptance and output agreement, F10 how much of the decode ran full.
* ``batch_step_<model>`` — F3 where a step's time goes, per step and per token,
  and the break-even between step-cost growth and τ.
* ``batch_memory_<model>`` — F5 the peak by component, F6 throughput against
  peak memory (iso-memory), and every DFlash phase's peak above the static
  allocations -- which is where the drafter's own prefill would show up.

and ``batch_verify_breakdown`` (F8) for both models, from the profiled verify
forwards in ``record_batch/verify_breakdown_*.json``. F4 and F7 are in
``plot_bound.py``.
"""

from __future__ import annotations

import argparse

import json

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

from _batch_style import (
    add_record_arguments, select_records,
    AQUA, AXIS, BATCHES, BLUE, CARD_GIB, CONTEXTS, CTX_LABEL, INK, INK2, MAGENTA, MEM_PARTS,
    MODE_COLOR, MODE_LABEL, MODELS, MODEL_LABEL, MODES, MUTED, NEUTRAL, NEUTRAL_FILL, ORANGE,
    RECORD_DIR, STEP_PARTS, gdn_backend, TRANSIENT_FILL, VIOLET, YELLOW, footnote, log2_batch_axis, plt,
    records, save, setup_matplotlib, status, summary, write_csv,
)


def ok_batches(recs, model, context, mode):
    return [b for b in BATCHES if summary(recs, model, context, b, mode)]


def mark_oom(ax, recs, model, context, y, modes=MODES, fontsize=6.5):
    """'OOM' at each batch a mode ran out of memory, in the mode's colour."""
    trans = ax.get_xaxis_transform()
    for i, mode in enumerate(modes):
        for batch in BATCHES:
            if status(recs, model, context, batch, mode) == "oom":
                ax.text(batch, y + 0.06 * i, f"{'BL' if mode == 'baseline' else 'DF'} OOM",
                        transform=trans, ha="center", fontsize=fontsize, color=MODE_COLOR[mode],
                        fontweight="bold")


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=6.6,
                color=INK2, va="center")


# ============================================================ F1, F2, F9, F10

def throughput_figure(model, recs):
    fig, axes = plt.subplots(5, 4, figsize=(18.5, 21))
    for col, context in enumerate(CONTEXTS):
        ax_tp, ax_sp, ax_tau, ax_occ, ax_eq = axes[:, col]

        # (a) throughput, with linear scaling from B=1 as the reference.
        for mode in MODES:
            bs = ok_batches(recs, model, context, mode)
            if not bs:
                continue
            color = MODE_COLOR[mode]
            mk = [summary(recs, model, context, b, mode)["decode_tok_s_makespan"] for b in bs]
            full = [summary(recs, model, context, b, mode)["decode_tok_s_full_occupancy"] for b in bs]
            ax_tp.plot(bs, [mk[0] * b for b in bs], color=color, lw=0.9, ls=":", zorder=2)
            ax_tp.plot(bs, full, color=color, lw=1.3, ls=(0, (5, 2)), marker="^", mfc="white", zorder=3)
            ax_tp.plot(bs, mk, color=color, marker="o", zorder=4)
            label_end(ax_tp, bs[-1], mk[-1], f"{mk[-1]:.0f}", color)
        mark_oom(ax_tp, recs, model, context, 0.03)
        ax_tp.set_yscale("log", base=2)
        ax_tp.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax_tp.set_ylabel("decode tokens / s (all rows)")
        ax_tp.set_title(f"(a) decode throughput · S = {CTX_LABEL[context]}", loc="left")

        # (b) same-B speedup, both definitions.
        bs = [b for b in BATCHES if (rec := recs.get((model, context, b))) and rec.get("speedup")
              and rec["speedup"].get("same_b_makespan")]
        ax_sp.axhline(1.0, color=INK, lw=1.0, zorder=2)
        if bs:
            sp = [recs[(model, context, b)]["speedup"] for b in bs]
            ax_sp.plot(bs, [s["same_b_full_occupancy"] for s in sp], color=BLUE, lw=1.3,
                       ls=(0, (5, 2)), marker="^", mfc="white", zorder=3)
            ax_sp.plot(bs, [s["same_b_makespan"] for s in sp], color=BLUE, marker="o", zorder=4)
            for b, s in zip(bs, sp):
                ax_sp.annotate(f"{s['same_b_makespan']:.2f}", (b, s["same_b_makespan"]), xytext=(0, -11),
                               textcoords="offset points", ha="center", fontsize=6.6, color=INK2)
        best = {m: max((summary(recs, model, context, b, m)["decode_tok_s_makespan"], b)
                       for b in ok_batches(recs, model, context, m)) for m in MODES
                if ok_batches(recs, model, context, m)}
        if len(best) == 2:
            ax_sp.text(0.98, 0.95, f"iso-memory (best on one card):\nDF {best['dflash'][0]:.0f} tok/s @B={best['dflash'][1]}"
                       f" / BL {best['baseline'][0]:.0f} @B={best['baseline'][1]}\n= {best['dflash'][0] / best['baseline'][0]:.2f}×",
                       transform=ax_sp.transAxes, ha="right", va="top", fontsize=6.8, color=INK)
        mark_oom(ax_sp, recs, model, context, 0.03, modes=["dflash"])
        ax_sp.set_ylim(0, 3.0)
        ax_sp.set_ylabel("DFlash / baseline at the same B")
        ax_sp.set_title(f"(b) same-B speedup · S = {CTX_LABEL[context]}", loc="left")

        # (c) acceptance length.
        bs = ok_batches(recs, model, context, "dflash")
        for key, color, label in (("mean_acceptance_length", BLUE, "τ (all)"),
                                  ("mean_acceptance_length_pre_eos", ORANGE, "τ before natural EOS"),
                                  ("mean_acceptance_length_post_eos", AQUA, "τ after (EOS suppressed)")):
            ys = [summary(recs, model, context, b, "dflash").get(key) for b in bs]
            ax_tau.plot(bs, ys, color=color, marker="o", lw=1.8)
        ax_tau.set_ylim(0, 6)
        ax_tau.set_ylabel("mean tokens kept per verify")
        ax_tau.set_title(f"(c) acceptance τ · S = {CTX_LABEL[context]}", loc="left")

        # (d) straggler tail: share of decode time with every row still running.
        width = 0.36
        for i, mode in enumerate(MODES):
            bs = ok_batches(recs, model, context, mode)
            xs = np.log2(bs) + (i - 0.5) * width
            ys = [100 * summary(recs, model, context, b, mode)["full_occupancy_fraction_of_decode"] for b in bs]
            ax_occ.bar(xs, ys, width=width * 0.92, color=MODE_COLOR[mode], edgecolor="white", linewidth=0.6)
        ax_occ.set_xticks(np.log2(BATCHES))
        ax_occ.set_xticklabels([str(b) for b in BATCHES])
        ax_occ.set_xlabel("batch size B")
        ax_occ.set_ylim(0, 105)
        ax_occ.set_ylabel("% of decode time with all B rows active")
        ax_occ.set_title(f"(d) static-batch straggler tail · S = {CTX_LABEL[context]}", loc="left")

        # (e) output agreement (greedy, bf16).
        ref = recs.get((model, context, 1))
        ref_rows = {r["prompt"]: r for r in ref["rows"]} if ref else {}
        series = {"DF == BL (same B)": [], "BL == BL at B=1": [], "DF == DF at B=1": []}
        bs = [b for b in BATCHES if (rec := recs.get((model, context, b))) and rec["rows"]
              and "dflash" in rec["rows"][0] and "baseline" in rec["rows"][0]]
        for b in bs:
            rows = recs[(model, context, b)]["rows"]
            series["DF == BL (same B)"].append(100 * np.mean([r["dflash_matches_baseline"] for r in rows]))
            for mode, name in (("baseline", "BL == BL at B=1"), ("dflash", "DF == DF at B=1")):
                same = [r[mode]["tokens"] == ref_rows[r["prompt"]][mode]["tokens"]
                        for r in rows if r["prompt"] in ref_rows]
                series[name].append(100 * np.mean(same))
        for (name, ys), color in zip(series.items(), (BLUE, NEUTRAL, ORANGE)):
            ax_eq.plot(bs, ys, color=color, marker="o", lw=1.8, label=name)
        ax_eq.set_ylim(0, 105)
        ax_eq.set_ylabel("% of 32 rows with identical 256 tokens")
        ax_eq.set_title(f"(e) output agreement · S = {CTX_LABEL[context]}", loc="left")

        for ax in (ax_tp, ax_sp, ax_tau, ax_eq):
            log2_batch_axis(ax)

    axes[0, 0].legend(handles=[
        Line2D([], [], color=MODE_COLOR[m], marker="o", label=f"{MODE_LABEL[m]} — makespan") for m in MODES
    ] + [
        Line2D([], [], color=MODE_COLOR[m], marker="^", mfc="white", ls=(0, (5, 2)), lw=1.3,
               label=f"{MODE_LABEL[m]} — all rows active") for m in MODES
    ] + [Line2D([], [], color=MUTED, ls=":", lw=0.9, label="B × (B=1 makespan): linear scaling")],
        loc="upper left", fontsize=6.6)
    axes[1, 0].legend(handles=[
        Line2D([], [], color=BLUE, marker="o", label="makespan (whole batch until its last row)"),
        Line2D([], [], color=BLUE, marker="^", mfc="white", ls=(0, (5, 2)), lw=1.3,
               label="steps with all rows active only"),
        Line2D([], [], color=INK, lw=1.0, label="break-even"),
    ], loc="upper left", fontsize=6.6)
    axes[2, 0].legend(handles=[Line2D([], [], color=c, marker="o", label=l) for c, l in (
        (BLUE, "τ (all)"), (ORANGE, "τ before natural EOS"), (AQUA, "τ after natural EOS (EOS suppressed)"))],
        loc="lower left", fontsize=6.6)
    axes[3, 0].legend(handles=[Patch(color=MODE_COLOR[m], label=MODE_LABEL[m]) for m in MODES],
                      loc="upper right", fontsize=6.6)
    axes[4, 0].legend(loc="lower left", fontsize=6.6)
    fig.suptitle(f"{MODEL_LABEL[model]} — throughput, speedup, acceptance and the static-batch tail "
                 f"(32 prompts, N = 256 per row)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    footnote(fig, (
        "Makespan = all 32×256 tokens over the decode wall time; a static batch runs until its slowest row, and a "
        "DFlash row finishes after 256/τ steps that vary by prompt, so the tail is DFlash's. 'All rows active' uses "
        "only the steps before the first row of a batch finished. τ counts the bonus token (τ = 1 + accepted). "
        "Output agreement is exact greedy equality in bf16: baseline against itself at B=1 disagrees about as often "
        "as DFlash against baseline, so the gap is batch-shape numerics, not a lossless-ness bug (fp32 CPU tests match "
        "exactly). The prompt mix differs by S, so τ across S mixes length with task."
    ), y=0.01)
    return fig


# ============================================================ F3

def step_figure(model, recs):
    fig, axes = plt.subplots(3, 4, figsize=(18.5, 13.5))
    width = 0.38
    for col, context in enumerate(CONTEXTS):
        ax_step, ax_tok, ax_ratio = axes[:, col]
        for i, mode in enumerate(MODES):
            bs = ok_batches(recs, model, context, mode)
            xs = np.log2(bs) + (i - 0.5) * width
            for ax, per_token in ((ax_step, False), (ax_tok, True)):
                bottom = np.zeros(len(bs))
                for key, _label, color in STEP_PARTS:
                    ys = []
                    for b in bs:
                        s = summary(recs, model, context, b, mode)
                        v = max(s["step_time_s"].get(key) or 0.0, 0.0) * 1e3
                        ys.append(v / s["mean_tokens_per_step"] if per_token else v)
                    ys = np.array(ys)
                    ax.bar(xs, ys, bottom=bottom, width=width * 0.92, color=color,
                           edgecolor="white", linewidth=0.5, alpha=1.0 if mode == "dflash" else 0.55)
                    bottom += ys
                for x, top, b in zip(xs, bottom, bs):
                    s = summary(recs, model, context, b, mode)
                    text = "BL" if mode == "baseline" else f"DF\n{s['mean_tokens_per_step']:.0f}t"
                    ax.text(x, top, text, ha="center", va="bottom", fontsize=5.8, color=INK2, linespacing=0.9)
        for ax in (ax_step, ax_tok):
            ax.set_xticks(np.log2(BATCHES))
            ax.set_xticklabels([str(b) for b in BATCHES])
            ax.set_xlabel("batch size B")
            ax.margins(y=0.12)
        ax_step.set_ylabel("ms per decode step")
        ax_tok.set_ylabel("ms per generated token (makespan)")
        ax_step.set_title(f"(a) step time by phase · S = {CTX_LABEL[context]}", loc="left")
        ax_tok.set_title(f"(b) time per token by phase · S = {CTX_LABEL[context]}", loc="left")

        # (c) speedup at full occupancy is τ over the step-cost ratio: where
        # the step ratio climbs past τ, DFlash loses.
        bs = [b for b in ok_batches(recs, model, context, "dflash")
              if summary(recs, model, context, b, "baseline")]
        df = [summary(recs, model, context, b, "dflash") for b in bs]
        bl = [summary(recs, model, context, b, "baseline") for b in bs]
        tau = [d["mean_acceptance_length"] for d in df]
        step_ratio = [d["mean_step_wall_s"] / b_["mean_step_wall_s"] for d, b_ in zip(df, bl)]
        verify_ratio = [d["step_time_s"]["verify"] / b_["step_time_s"]["verify"] for d, b_ in zip(df, bl)]
        draft_ratio = [d["step_time_s"]["draft"] / b_["step_time_s"]["verify"] for d, b_ in zip(df, bl)]
        if not bs:                       # a sweep that did not cover this model
            ax_ratio.set_axis_off()
            continue
        ax_ratio.axhline(1.0, color=AXIS, lw=0.9)
        # Tokens per row-step counting the rows that already finished: with the
        # baseline at exactly 1, makespan speedup = this / step ratio.
        effective = [d["mean_tokens_per_step"] / b for d, b in zip(df, bs)]
        ax_ratio.plot(bs, tau, color=INK, lw=1.4, ls=(0, (5, 2)), marker="D", ms=4, mfc="white")
        ax_ratio.plot(bs, effective, color=INK, lw=1.4, ls=":", marker="D", ms=4)
        ax_ratio.plot(bs, step_ratio, color=MAGENTA, marker="o")
        ax_ratio.plot(bs, verify_ratio, color=BLUE, marker="o")
        ax_ratio.plot(bs, draft_ratio, color=ORANGE, marker="s", mfc="white", lw=1.3)
        cross = [b for b, t, r in zip(bs, effective, step_ratio) if r > t]
        if cross:
            ax_ratio.axvspan(cross[0] / 1.25, 45, color="#f3f2ee", zorder=0)
            ax_ratio.text(cross[0] / 1.2, 0.03, "step ratio > effective\ntokens per row-step:\nslower than baseline",
                          transform=ax_ratio.get_xaxis_transform(), fontsize=6.5, color=INK, va="bottom")
        log2_batch_axis(ax_ratio)
        ax_ratio.set_xlim(0.7, 45)
        ax_ratio.set_ylim(0, max(max(step_ratio + tau) * 1.5, 3))
        ax_ratio.set_ylabel("× baseline step (same B)  /  tokens per row-step")
        ax_ratio.set_title(f"(c) cost ratio vs τ · S = {CTX_LABEL[context]}", loc="left")

    axes[0, 0].legend(handles=[Patch(color=c, label=l) for _k, l, c in STEP_PARTS] + [
        Patch(color="none", label="left bar baseline (faded), right bar DFlash;"),
        Patch(color="none", label="'DF 24t' = tokens produced per DFlash step")], loc="upper left", fontsize=6.6)
    axes[2, 0].legend(handles=[
        Line2D([], [], color=INK, lw=1.4, ls=(0, (5, 2)), marker="D", ms=4, mfc="white", label="τ (per active row)"),
        Line2D([], [], color=INK, lw=1.4, ls=":", marker="D", ms=4, label="tokens per row-step incl. finished rows"),
        Line2D([], [], color=MAGENTA, marker="o", label="DF step / BL step"),
        Line2D([], [], color=BLUE, marker="o", label="DF verify / BL verify  (c_verify)"),
        Line2D([], [], color=ORANGE, marker="s", mfc="white", lw=1.3, label="DF draft / BL verify"),
    ], loc="upper left", fontsize=6.6, ncol=2)
    fig.suptitle(f"{MODEL_LABEL[model]} — where a decode step's time goes as B grows",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    notes = ("GPU phases are CUDA-event means per step; CPU + sync is wall − Σ GPU (can be slightly negative, drawn as 0). "
             "(c) Makespan speedup = (tokens per row-step incl. finished rows) / (DF step / BL step) exactly — the magenta "
             "line crossing the dotted line is the break-even. The gap between the dashed τ and the dotted line is the "
             "static-batch tail (finished rows still verified). c_verify near 1 means 15 extra tokens per row are nearly "
             "free (memory-bound); it rises once the verify is compute-bound. ")
    if model == "qwen3.5-9b":
        notes += (f"9B, {gdn_backend(recs)}: accept/commit is the GDN rollback (replaying the kept prefix through the "
                  "chunk kernel) for DFlash, and for the baseline a torch.where over every recurrent state it does "
                  "not need — a baseline cost that grows with B and flatters DFlash at large B.")
    footnote(fig, notes, y=0.01)
    return fig


# ============================================================ F5, F6, phase peaks

PHASE_SERIES = [
    ("prefill: target forward", "prefill: target forward (1 row)", BLUE),
    ("prefill: drafter prefill", "prefill: drafter prefill (1 row)", ORANGE),
    ("prefill: context-feature build", "prefill: context-feature build", AQUA),
    ("decode: accept/rollback", "decode: accept / rollback", YELLOW),
    ("decode: draft forward", "decode: draft forward", MAGENTA),
    ("decode: target verify", "decode: target verify", VIOLET),
]


def memory_figure(model, recs):
    fig, axes = plt.subplots(3, 4, figsize=(18.5, 14))
    width = 0.38
    for col, context in enumerate(CONTEXTS):
        ax_peak, ax_pareto, ax_phase = axes[:, col]

        # (a) the peak, split at the instant it happened.
        for i, mode in enumerate(MODES):
            bs = ok_batches(recs, model, context, mode)
            xs = np.log2(bs) + (i - 0.5) * width
            bottom = np.zeros(len(bs))
            for key, _label, color, hatch in MEM_PARTS:
                ys = []
                for b in bs:
                    s = summary(recs, model, context, b, mode)
                    entry = s["phase_memory"][s["peak_phase"]]
                    ys.append(entry["peak_unattributed_gb"] if key == "unattributed"
                              else entry["peak_components_gb"].get(key, 0.0))
                ys = np.array(ys)
                ax_peak.bar(xs, ys, bottom=bottom, width=width * 0.92, color=color, hatch=hatch,
                            edgecolor="white" if not hatch else "#b9b7ae", linewidth=0.4)
                bottom += ys
            for x, top, b in zip(xs, bottom, bs):
                s = summary(recs, model, context, b, mode)
                where = "P" if s["peak_phase"].startswith("prefill") else "D"
                ax_peak.text(x, top + 0.4, f"{'BL' if mode == 'baseline' else 'DF'}\n{where}", ha="center",
                             va="bottom", fontsize=5.6, color=INK2, linespacing=0.9)
            for b in BATCHES:
                if status(recs, model, context, b, mode) == "oom":
                    ax_peak.text(np.log2(b) + (i - 0.5) * width, 2, "OOM", rotation=90, ha="center",
                                 va="bottom", fontsize=6.5, color=MODE_COLOR[mode], fontweight="bold")
        ax_peak.axhline(CARD_GIB, color=INK, lw=1.0, ls=(0, (5, 3)))
        ax_peak.text(5.55, CARD_GIB + 0.6, f"card {CARD_GIB:.1f} GiB", fontsize=6.5, color=INK, ha="right")
        ax_peak.set_xticks(np.log2(BATCHES))
        ax_peak.set_xticklabels([str(b) for b in BATCHES])
        ax_peak.set_xlim(-0.6, 5.6)
        ax_peak.set_ylim(0, 54)
        ax_peak.set_xlabel("batch size B")
        ax_peak.set_ylabel("allocated GiB at the peak")
        ax_peak.set_title(f"(a) peak memory by component · S = {CTX_LABEL[context]}", loc="left")

        # (b) iso-memory: throughput against the memory it took.
        for mode in MODES:
            bs = ok_batches(recs, model, context, mode)
            xs = [summary(recs, model, context, b, mode)["peak_memory_gb"] for b in bs]
            ys = [summary(recs, model, context, b, mode)["decode_tok_s_makespan"] for b in bs]
            ax_pareto.plot(xs, ys, color=MODE_COLOR[mode], marker="o", zorder=4)
            for x, y, b in zip(xs, ys, bs):
                ax_pareto.annotate(f"B={b}", (x, y), xytext=(4 if mode == "dflash" else -4, 5 if mode == "dflash" else -10),
                                   textcoords="offset points", fontsize=6.3, color=INK2,
                                   ha="left" if mode == "dflash" else "right")
        ax_pareto.axvline(CARD_GIB, color=INK, lw=1.0, ls=(0, (5, 3)))
        ax_pareto.set_xlim(15, 50)
        ax_pareto.set_ylim(0, None)
        ax_pareto.set_xlabel("peak allocated GiB")
        ax_pareto.set_ylabel("decode tokens / s (makespan)")
        ax_pareto.set_title(f"(b) throughput vs memory (iso-memory) · S = {CTX_LABEL[context]}", loc="left")

        # (c) each DFlash phase's own peak, above what setup allocated.
        bs = ok_batches(recs, model, context, "dflash")
        for phase, _label, color in PHASE_SERIES:
            ys = []
            for b in bs:
                pm = summary(recs, model, context, b, "dflash")["phase_memory"]
                base = pm["setup: allocate caches"]["mean_allocated_after_gb"]
                ys.append(pm[phase]["max_interval_peak_gb"] - base if phase in pm else np.nan)
            hollow = phase == "prefill: drafter prefill"
            ax_phase.plot(bs, ys, color=color, marker="s" if hollow else "o", mfc="white" if hollow else color,
                          ms=7 if hollow else 5, lw=1.7, ls=(0, (4, 2)) if hollow else "-", zorder=5 if hollow else 4)
        log2_batch_axis(ax_phase)
        ax_phase.set_xlim(0.7, 45)
        ax_phase.set_ylim(0, None)
        ax_phase.set_ylabel("phase peak − static allocations (GiB)")
        ax_phase.set_title(f"(c) DFlash phase peaks above the static caches · S = {CTX_LABEL[context]}", loc="left")

    fig.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor="#b9b7ae" if h else c, label=l)
                        for _k, l, c, h in MEM_PARTS] + [
        Patch(color="none", label="(a): left bar baseline, right bar DFlash; P / D = peak in prefill / decode")],
        loc="upper left", bbox_to_anchor=(0.01, 0.963), ncol=5, fontsize=7)
    axes[1, 0].legend(handles=[Line2D([], [], color=MODE_COLOR[m], marker="o", label=MODE_LABEL[m]) for m in MODES],
                      loc="upper left", fontsize=6.6)
    axes[2, 0].legend(handles=[Line2D([], [], color=c, marker="s" if "drafter" in p else "o",
                                      mfc="white" if "drafter" in p else c,
                                      ls=(0, (4, 2)) if "drafter" in p else "-", label=l)
                               for p, l, c in PHASE_SERIES], loc="upper left", bbox_to_anchor=(0.0, 0.9),
                      fontsize=6.3)
    fig.suptitle(f"{MODEL_LABEL[model]} — capacity: what sets the peak, and what the drafter adds",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 0.935))
    footnote(fig, (
        "Components are read at the one instant the run peaked. Prefill is per row (one request at a time) into "
        "caches allocated for all B rows at setup, so the peak is the static caches plus ONE row's transient; every "
        "OOM happened at 'setup: allocate caches'. Both modes load the drafter's weights (the same process runs "
        "both), so a baseline-only process would sit that much lower. (c) is the peak inside each phase minus "
        "what setup allocated: the drafter's own prefill never exceeds the target's, because its context tokens "
        "only pass through fc and the K/V projections, not through its attention queries or MLP."
    ), y=0.01)
    return fig


# ============================================================ F8

MODULE_PARTS = [("mlp", "MLP", BLUE), ("attention block", "full-attention block (QKV/O proj + attention)", ORANGE),
                ("gdn", "GDN block (torch fallback)", AQUA), ("lm_head", "LM head", YELLOW),
                ("_rest", "rest (norms, embed, glue)", NEUTRAL_FILL)]
KERNEL_PARTS = [("gemm", "GEMM", BLUE), ("attention", "attention kernels", ORANGE),
                ("elementwise/other", "elementwise / other", AQUA), ("copy/index", "copy / index", YELLOW),
                ("_idle", "GPU idle inside the forward (launch-bound)", TRANSIENT_FILL)]


def verify_breakdown_figure():
    data = {}
    for model in MODELS:
        path = RECORD_DIR / f"verify_breakdown_{model}.json"
        if path.exists():
            data[model] = json.loads(path.read_text())["rows"]
    fig, axes = plt.subplots(max(len(data), 1), 2, figsize=(18.5, 5.6 * max(len(data), 1)), squeeze=False)
    table = []
    for r, (model, rows) in enumerate(data.items()):
        rows = sorted(rows, key=lambda x: (x["context"], x["batch"], x["q"]))
        xs, labels, x = [], [], 0.0
        for i, row in enumerate(rows):
            if i and (row["context"], row["batch"]) != (rows[i - 1]["context"], rows[i - 1]["batch"]):
                x += 0.6
            if i and row["context"] != rows[i - 1]["context"]:
                x += 0.8
            xs.append(x)
            labels.append(f"q={row['q']}")
            x += 1.0
        for c, (parts, source, title) in enumerate((
            (MODULE_PARTS, "module_ms", "by module (CUDA events around each block)"),
            (KERNEL_PARTS, "kernel_ms", "by kernel category (torch.profiler)"),
        )):
            ax = axes[r, c]
            bottom = np.zeros(len(rows))
            for key, _label, color in parts:
                ys = []
                for row in rows:
                    if key == "_rest":
                        ys.append(max(row["wall_ms"] - sum(row["module_ms"].values()), 0.0))
                    elif key == "_idle":
                        ys.append(max(row["wall_ms"] - row["kernel_total_ms"], 0.0))
                    else:
                        ys.append(row[source].get(key, 0.0))
                ys = np.array(ys)
                ax.bar(xs, ys, bottom=bottom, width=0.9, color=color, edgecolor="white" if key != "_idle" else "#b9b7ae",
                       hatch="////" if key == "_idle" else None, linewidth=0.5)
                bottom += ys
            for xx, row in zip(xs, rows):
                ax.text(xx, row["wall_ms"] + 1.2, f"{row['wall_ms']:.0f}", ha="center", fontsize=6.2, color=INK2)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=6.5)
            groups = {}
            for xx, row in zip(xs, rows):
                groups.setdefault((row["context"], row["batch"]), []).append(xx)
            for (context, batch), gx in groups.items():
                ax.text(np.mean(gx), -0.13, f"{CTX_LABEL[context]} · B={batch}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=7, color=INK)
            ax.set_ylabel("ms per verify forward")
            ax.set_title(f"({'abcd'[2 * r + c]}) {MODEL_LABEL[model]} — {title}", loc="left")
            ax.legend(handles=[Patch(facecolor=col, hatch="////" if k == "_idle" else None,
                                     edgecolor="#b9b7ae" if k == "_idle" else col, label=l)
                               for k, l, col in parts if k != "gdn" or model == "qwen3.5-9b"],
                      loc="upper left", fontsize=6.6)
            ax.margins(y=0.1)
        for row in rows:
            table.append(dict(model=model, context=row["context"], batch=row["batch"], q=row["q"],
                              wall_ms=row["wall_ms"], kernel_total_ms=row["kernel_total_ms"],
                              gpu_idle_ms=row["wall_ms"] - row["kernel_total_ms"],
                              **{f"module_{k}": v for k, v in row["module_ms"].items()},
                              **{f"kernel_{k}": v for k, v in row["kernel_ms"].items()}))
    fig.suptitle("Verify forward, profiled at fixed shapes: q = 1 is the baseline's forward, q = 16 DFlash's",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=3.0)
    footnote(fig, (
        "Measured outside the timed sweep (queue/verify_breakdown.py): one row prefilled and copied into the others, "
        "every forward rewriting the same slots. The hatched block is wall time the GPU spent without a kernel "
        "running — Python and launch overhead the kernels could not hide. For the 9B GDN torch fallback at q = 16 it "
        "is over half the forward; for 8B at 32K the attention kernel at q = 16 costs ~4× its q = 1 time at B = 1 "
        "because it does not share K/V across the query heads of a GQA group."
    ), y=0.015)
    return fig, table


# ============================================================ tables

def metrics_rows(recs):
    out = []
    for (model, context, batch), rec in sorted(recs.items()):
        for mode in MODES:
            s = rec["summary"].get(mode)
            row = dict(model=model, context=context, batch=batch, mode=mode, status=rec["status"].get(mode))
            if s:
                t = s["step_time_s"]
                row.update(
                    decode_tok_s_makespan=s["decode_tok_s_makespan"],
                    decode_tok_s_full_occupancy=s["decode_tok_s_full_occupancy"],
                    full_occupancy_fraction=s["full_occupancy_fraction_of_decode"],
                    e2e_tok_s=s["e2e_tok_s"], step_ms=s["mean_step_wall_s"] * 1e3,
                    tokens_per_step=s["mean_tokens_per_step"],
                    **{f"{k}_ms": (t.get(k) or 0.0) * 1e3 for k, _l, _c in STEP_PARTS},
                    peak_gib=s["peak_memory_gb"], peak_phase=s["peak_phase"],
                    target_kv_gib=s["target_kv_alloc_gb"], draft_kv_gib=s["draft_kv_alloc_gb"],
                    gdn_state_gib=s["gdn_state_gb"],
                    tau=s.get("mean_acceptance_length"), tau_pre_eos=s.get("mean_acceptance_length_pre_eos"),
                    tau_post_eos=s.get("mean_acceptance_length_post_eos"),
                )
                if mode == "dflash" and rec.get("speedup"):
                    row.update({f"speedup_{k}": v for k, v in rec["speedup"].items()})
            out.append(row)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_record_arguments(parser)
    args = parser.parse_args(argv)
    select_records(args.records, args.suffix)
    setup_matplotlib()
    recs = records()
    for model in [m for m in MODELS if any(key[0] == m for key in recs)]:
        for stem, build in ((f"batch_throughput_{model}", throughput_figure),
                            (f"batch_step_{model}", step_figure),
                            (f"batch_memory_{model}", memory_figure)):
            for path in save(build(model, recs), stem):
                print("wrote", path)
    fig, table = verify_breakdown_figure()
    if table:                      # a sweep without a profiling pass has none
        for path in save(fig, "batch_verify_breakdown"):
            print("wrote", path)
        print("wrote", write_csv("batch_verify_breakdown", table))
    print("wrote", write_csv("batch_metrics", metrics_rows(recs)))


if __name__ == "__main__":
    main()
