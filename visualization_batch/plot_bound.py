"""Where the verify forward crosses from memory-bound to compute-bound as B grows.

One figure per model, one column per context length, three rows:

(a) Roofline. Achieved TFLOP/s against arithmetic intensity for the baseline's
    1-token verify, DFlash's 16-token verify and the draft forward, one point
    per batch size. Roofs are the *measured* copy bandwidth and bf16 GEMM peak
    (record_batch/ceiling.json); the dotted roof is the data sheet, the dashed
    curve the measured GEMM at M tokens, i.e. what a weight-streaming matmul
    actually reaches at that batch.
(b) Arithmetic intensity against B, with the ridge as the line between the two
    regimes and the two asymptotes a verify lives between: all-weights
    (≈ B·q FLOP/B, climbs with B) and all-KV (q·H_q/H_kv = 4q, flat in B).
(c) Achieved fraction of each ceiling against B, with plan §5.4's rules as
    lines: ≥ 70 % of copy bandwidth is HBM-bound, ≥ 60 % of GEMM peak is
    compute-bound. Neither is the third verdict: kernel or launch bound.

    python visualization_batch/plot_bound.py
"""

from __future__ import annotations

import argparse

import numpy as np

from _batch_style import (
    add_record_arguments, select_records,
    ARCH, AXIS, BATCHES, BLUE, CONTEXTS, CTX_LABEL, DRAFT_COLOR, INK, INK2, MODE_COLOR,
    MODE_LABEL, MODELS, MODEL_LABEL, MODES, MUTED, NEUTRAL, ceiling, draft_work, footnote,
    gdn_backend, log2_batch_axis, plt, records, save, setup_matplotlib, summary, verify_work, write_csv,
)

DATASHEET_GBPS, DATASHEET_TFLOPS = 768.0, 154.8   # RTX A6000
HBM_RULE, COMPUTE_RULE = 0.70, 0.60                # plan §5.4


def gemm_curve(ceil: dict):
    """Measured bf16 GEMM, (M×4096)·(4096×12288), as (intensity, TFLOP/s)."""
    k, n = 4096, 12288
    points = []
    for m, v in sorted(ceil["gemm"].items(), key=lambda item: int(item[0])):
        m = int(m)
        intensity = 2 * m * k * n / (2 * (k * n + m * k + m * n))
        points.append((intensity, v["tflops"]))
    return np.array(points)


def collect(recs: dict, ceil: dict) -> list[dict]:
    bw, peak = ceil["copy_gbps"] * 1e9, ceil["gemm"]["4096"]["tflops"] * 1e12
    rows = []
    for (model, context, batch), rec in sorted(recs.items()):
        for mode in MODES:
            s = summary(recs, model, context, batch, mode)
            if not s:
                continue
            work = verify_work(model, rec, mode)
            seconds = s["step_time_s"]["verify"]
            rows.append(dict(
                model=model, context=context, batch=batch, phase=f"{mode} verify",
                seconds=seconds, bytes=work["bytes"], flops=work["flops"],
                intensity=work["flops"] / work["bytes"],
                gbps=work["bytes"] / seconds / 1e9, tflops=work["flops"] / seconds / 1e12,
                bw_fraction=work["bytes"] / seconds / bw, compute_fraction=work["flops"] / seconds / peak,
                weight_bytes=work["weight_bytes"], kv_bytes=work["kv_bytes"], gdn_bytes=work["gdn_bytes"],
                gemm_flops=work["gemm_flops"], attention_flops=work["attention_flops"],
            ))
            if mode == "dflash" and s["step_time_s"].get("draft"):
                work = draft_work(model, rec)
                seconds = s["step_time_s"]["draft"]
                rows.append(dict(
                    model=model, context=context, batch=batch, phase="dflash draft",
                    seconds=seconds, bytes=work["bytes"], flops=work["flops"],
                    intensity=work["flops"] / work["bytes"],
                    gbps=work["bytes"] / seconds / 1e9, tflops=work["flops"] / seconds / 1e12,
                    bw_fraction=work["bytes"] / seconds / bw, compute_fraction=work["flops"] / seconds / peak,
                    draft_context_columns=work["widest"],
                ))
    for row in rows:
        if row["bw_fraction"] >= HBM_RULE:
            row["verdict"] = "HBM-bound"
        elif row["compute_fraction"] >= COMPUTE_RULE:
            row["verdict"] = "compute-bound"
        else:
            row["verdict"] = ("past ridge, below compute roof" if row["intensity"] >= peak / bw
                              else "below both roofs (kernel/launch)")
    return rows


SERIES = [
    ("baseline verify", MODE_COLOR["baseline"], "o", "-", "baseline verify (q=1)"),
    ("dflash verify", MODE_COLOR["dflash"], "o", "-", "DFlash verify (q=16)"),
    ("dflash draft", DRAFT_COLOR, "s", "--", "DFlash draft fwd + logits"),
]


def pick(rows, model, context, phase):
    got = [r for r in rows if r["model"] == model and r["context"] == context and r["phase"] == phase]
    return sorted(got, key=lambda r: r["batch"])


def oom_batches(recs, model, context, mode):
    return [b for b in BATCHES
            if (rec := recs.get((model, context, b))) and rec["status"].get(mode) == "oom"]


def roofline_panel(ax, rows, model, context, ceil, curve):
    bw, peak = ceil["copy_gbps"], ceil["gemm"]["4096"]["tflops"]
    x = np.logspace(-0.3, 3.2, 200)
    ax.plot(x, np.minimum(x * bw / 1e3, peak), color=INK, lw=1.4, zorder=2)
    ax.plot(x, np.minimum(x * DATASHEET_GBPS / 1e3, DATASHEET_TFLOPS), color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.plot(curve[:, 0], curve[:, 1], color=MUTED, lw=1.0, ls="--", marker=".", ms=4, zorder=2)
    ridge = peak * 1e3 / bw
    ax.axvline(ridge, color=AXIS, lw=0.9, zorder=1)
    ax.text(ridge * 1.08, 0.13, f"ridge\n{ridge:.0f}", fontsize=6.8, color=INK2, va="bottom")
    ax.text(0.6, 2.2, f"copy {bw:.0f} GB/s ↗", fontsize=6.8, color=INK2, ha="left")
    ax.text(900, peak * 1.12, f"GEMM peak {peak:.0f}", ha="right", fontsize=6.8, color=INK2)

    for phase, color, marker, ls, _label in SERIES:
        pts = pick(rows, model, context, phase)
        if not pts:
            continue
        xs = [p["intensity"] for p in pts]
        ys = [p["tflops"] for p in pts]
        filled = phase != "dflash draft"
        ax.plot(xs, ys, color=color, ls=ls, lw=1.6 if filled else 1.2, zorder=4)
        ax.scatter(xs, ys, s=30 if filled else 22, marker=marker, zorder=5,
                   facecolor=color if filled else "white", edgecolor=color if not filled else "white",
                   linewidth=1.2 if not filled else 0.8)
        if phase != "dflash draft":
            for p in pts:
                ax.annotate(str(p["batch"]), (p["intensity"], p["tflops"]),
                            xytext=(-3, 5) if phase == "dflash verify" else (3, -9),
                            textcoords="offset points", fontsize=6.6, color=INK2,
                            ha="right" if phase == "dflash verify" else "left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1500)
    ax.set_ylim(0.12, 320)
    ax.set_xlabel("arithmetic intensity (FLOP / byte, analytic)")
    ax.set_ylabel("achieved TFLOP/s")


def intensity_panel(ax, rows, model, context, ceil, recs):
    a = ARCH[model]
    bw, peak = ceil["copy_gbps"], ceil["gemm"]["4096"]["tflops"]
    ridge = peak * 1e3 / bw
    group = a["q_heads"] / a["kv_heads"]
    ax.axhspan(ridge, 5000, color="#f3f2ee", zorder=0)
    ax.axhline(ridge, color=INK, lw=1.1, zorder=2)
    ax.axhline(DATASHEET_TFLOPS * 1e3 / DATASHEET_GBPS, color=MUTED, lw=0.9, ls=":", zorder=2)
    ax.text(1.15, ridge * 1.12, f"compute-bound  (ridge {ridge:.0f}, measured)", fontsize=6.8, color=INK)
    ax.text(1.15, ridge / 1.45, "memory-bound", fontsize=6.8, color=INK)
    b = np.array(BATCHES, dtype=float)
    # Asymptotes: a verify made only of weights streams at ~B·q FLOP/B; one made
    # only of KV reads is attention at q·H_q/H_kv, whatever B is.
    for q, color in ((16, BLUE), (1, NEUTRAL)):
        ax.plot(b, b * q, color=color, lw=0.9, ls=(0, (4, 3)), zorder=2, alpha=0.9)
        ax.axhline(q * group, color=color, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.text(b[-1] * 1.05, b[-1] * 16, "weights only\n≈16B", fontsize=6.3, color=BLUE, va="center")
    ax.text(b[-1] * 1.05, 16 * group, f"KV only\n16·{group:.0f}={16 * group:.0f}", fontsize=6.3, color=BLUE, va="center")
    ax.text(b[-1] * 1.05, b[-1], "≈B", fontsize=6.3, color=NEUTRAL, va="center")
    ax.text(b[-1] * 1.05, group, f"{group:.0f}", fontsize=6.3, color=NEUTRAL, va="center")

    for phase, color, marker, ls, _label in SERIES:
        pts = pick(rows, model, context, phase)
        if not pts:
            continue
        filled = phase != "dflash draft"
        ax.plot([p["batch"] for p in pts], [p["intensity"] for p in pts], color=color, ls=ls,
                marker=marker, mfc=color if filled else "white", mec=color if not filled else "white",
                lw=1.8 if filled else 1.2, zorder=4)
    for mode in MODES:
        for batch in oom_batches(recs, model, context, mode):
            ax.text(batch, 0.62 if mode == "baseline" else 0.9, "OOM", ha="center", fontsize=6.3,
                    color=MODE_COLOR[mode], fontweight="bold")
    log2_batch_axis(ax)
    ax.set_xlim(0.7, 70)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2000)
    ax.set_ylabel("arithmetic intensity (FLOP / byte)")


def utilization_panel(ax, rows, model, context, recs):
    ax.axhline(HBM_RULE * 100, color=INK, lw=0.9, ls="-", zorder=2)
    ax.axhline(COMPUTE_RULE * 100, color=INK, lw=0.9, ls=(0, (5, 3)), zorder=2)
    ax.text(0.72, HBM_RULE * 100 + 1.5, "HBM-bound ≥ 70 % of copy BW", fontsize=6.5, color=INK)
    ax.text(0.72, COMPUTE_RULE * 100 + 1.5, "compute-bound ≥ 60 % of GEMM peak", fontsize=6.5, color=INK)
    for mode in MODES:
        pts = pick(rows, model, context, f"{mode} verify")
        if not pts:
            continue
        bs = [p["batch"] for p in pts]
        color = MODE_COLOR[mode]
        ax.plot(bs, [100 * p["bw_fraction"] for p in pts], color=color, marker="o", lw=1.8, zorder=4)
        ax.plot(bs, [100 * p["compute_fraction"] for p in pts], color=color, marker="^", lw=1.4,
                ls=(0, (5, 2)), mfc="white", zorder=4)
    log2_batch_axis(ax)
    ax.set_xlim(0.7, 45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("verify: achieved % of ceiling")


def figure(model, rows, recs, ceil):  # noqa: C901
    curve = gemm_curve(ceil)
    fig, axes = plt.subplots(3, 4, figsize=(18.5, 13.2))
    for col, context in enumerate(CONTEXTS):
        roofline_panel(axes[0, col], rows, model, context, ceil, curve)
        intensity_panel(axes[1, col], rows, model, context, ceil, recs)
        utilization_panel(axes[2, col], rows, model, context, recs)
        axes[0, col].set_title(f"(a) roofline · S = {CTX_LABEL[context]}", loc="left")
        axes[1, col].set_title(f"(b) intensity vs B · S = {CTX_LABEL[context]}", loc="left")
        axes[2, col].set_title(f"(c) ceiling utilisation · S = {CTX_LABEL[context]}", loc="left")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=m, ls=ls, mfc=c if "draft" not in p else "white",
                      mec=c if "draft" in p else "white", lw=1.8, label=lab)
               for p, c, m, ls, lab in SERIES]
    handles += [
        Line2D([], [], color=INK, lw=1.4, label="measured roof (copy BW / GEMM peak)"),
        Line2D([], [], color=MUTED, lw=1.0, ls=":", label="data-sheet roof (768 GB/s, 154.8 TFLOP/s)"),
        Line2D([], [], color=MUTED, lw=1.0, ls="--", marker=".", label="measured bf16 GEMM at M tokens"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=6.8)
    util_handles = [
        Line2D([], [], color=MODE_COLOR[m], marker="o", lw=1.8, label=f"{MODE_LABEL[m]}: % copy BW")
        for m in MODES
    ] + [
        Line2D([], [], color=MODE_COLOR[m], marker="^", mfc="white", ls=(0, (5, 2)), lw=1.4,
               label=f"{MODE_LABEL[m]}: % GEMM peak") for m in MODES
    ]
    axes[2, 0].legend(handles=util_handles, loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=6.6)
    fig.suptitle(f"{MODEL_LABEL[model]} — verify forward: memory-bound → compute-bound as the batch grows "
                 f"(RTX A6000, N = 256, EOS suppressed)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    notes = (
        "Bytes and FLOPs are analytic (plan §5.4): executed target weights once (no embedding table), KV at the mean "
        "context S + N/2 once per row, GDN state in and out, logits written and read; FLOPs = 2·params·tokens + "
        "attention 4·q·L·H_q·d per layer + the delta rule. The draft adds fc and K/V projection of the kept context "
        "columns and 15 LM-head rows. Time is the CUDA-event mean per step, over all steps of the run (a finished row "
        "still computes). "
    )
    if model == "qwen3.5-9b":
        notes += (f"9B ran on {gdn_backend(recs)}. A DFlash verify sitting below both roofs is the GDN block, whose "
                  "per-layer kernels leave the card idle between launches — see batch_verify_breakdown, not the batch.")
    else:
        notes += ("8B at 16K–32K: KV dominates the bytes, so intensity stays near the KV-only line (64) and the "
                  "q=16 attention kernel, not HBM, sets the time (see batch_verify_breakdown).")
    footnote(fig, notes, y=0.012)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_record_arguments(parser)
    args = parser.parse_args(argv)
    select_records(args.records, args.suffix)
    setup_matplotlib()
    recs, ceil = records(), ceiling()
    rows = collect(recs, ceil)
    for model in [m for m in MODELS if any(key[0] == m for key in recs)]:
        for path in save(figure(model, rows, recs, ceil), f"batch_bound_{model}"):
            print("wrote", path)
    print("wrote", write_csv("batch_bound_metrics", rows))


if __name__ == "__main__":
    main()
