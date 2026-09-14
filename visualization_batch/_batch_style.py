"""Loading, palette and the analytic byte/FLOP model shared by the batch figures.

Every figure here reads a sweep directory -- ``record_batch/`` unless ``--records``
says otherwise -- through :func:`dflash.batch_report.load` (latest record per model,
context and batch), so a figure and the tables that ``python -m dflash.batch_report``
prints always come from the same files. The
``stale_*`` subdirectories are never read: they hold the 9B records measured
with the two engine bugs (BATCH_PROFILING_PLAN.md, Appendix B).

The byte and FLOP counts are analytic (plan §5.4). They count what a forward
*has* to move and compute -- weights once, the KV each row attends over once,
the GDN state once in and once out -- not what the kernels happen to move. A
point sitting far below the roof is therefore a kernel or launch inefficiency,
not a miscount: the verify breakdown figure says which.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dflash.batch_report import EMBED_BYTES, load  # noqa: E402

RECORD_DIR = ROOT / "record_batch"
OUT_DIR = Path(__file__).resolve().parent
SUFFIX = ""


def select_records(directory=None, suffix=None) -> None:
    """Point the figures at another sweep, e.g. the fla rerun of the 9B grid.

    ``suffix`` goes on every file the run writes, so two kernel sets can live
    side by side instead of one silently overwriting the other.
    """
    global RECORD_DIR, SUFFIX
    if directory:
        RECORD_DIR = Path(directory).resolve()
    if suffix is not None:
        SUFFIX = suffix


def add_record_arguments(parser) -> None:
    parser.add_argument("--records", help="record directory (default: record_batch)")
    parser.add_argument("--suffix", default=None, help="appended to every output filename")


MODELS = ["qwen3-8b", "qwen3.5-9b"]
MODEL_LABEL = {"qwen3-8b": "Qwen3-8B", "qwen3.5-9b": "Qwen3.5-9B"}
CONTEXTS = [4096, 8192, 16384, 32768]
CTX_LABEL = {4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K"}
BATCHES = [1, 2, 4, 8, 16, 32]
MODES = ["baseline", "dflash"]

GIB = float(1 << 30)
# torch's usable capacity on the A6000, as the OOM messages report it.
CARD_GIB = 47.43

# ---------------------------------------------------------------- palette
# Reference data-viz palette, validated (adjacent CVD ΔE >= 9.1, normal >= 16.3).
# Three of the hues sit below 3:1 on white, so every figure ships a legend and
# its numbers go to a CSV next to it.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NEUTRAL = "#8c8a84"          # the baseline: a reference, not a competitor
NEUTRAL_FILL = "#c3c2b7"     # target weights: the floor both modes pay
TRANSIENT_FILL = "#e6e4dc"   # activation transient, hatched

MODE_COLOR = {"baseline": NEUTRAL, "dflash": BLUE}
MODE_LABEL = {"baseline": "Baseline (B×1 tokens)", "dflash": "DFlash (B×16 tokens)"}
DRAFT_COLOR = ORANGE

# Step components, stacked in palette order so every adjacent pair is one the
# validator passed. Colour follows the entity: verify is blue and draft orange
# wherever they appear.
STEP_PARTS = [
    ("verify", "target verify", BLUE),
    ("draft", "draft forward + logits", ORANGE),
    ("accept", "accept / commit / GDN rollback", AQUA),
    ("ctx", "context-feature build", YELLOW),
    ("cpu_and_sync", "CPU + sync (wall − GPU)", MAGENTA),
]

# Memory at the peak, bottom to top: what both modes pay, then what only DFlash adds.
MEM_PARTS = [
    ("target_weight_bytes", "target weights", NEUTRAL_FILL, None),
    ("draft_weight_bytes", "draft weights (loaded in both modes)", "#a9a79f", None),
    ("target_kv_bytes", "target KV (static, all rows)", BLUE, None),
    ("gdn_state_bytes", "GDN state (all rows)", VIOLET, None),
    ("unattributed", "activation transient", TRANSIENT_FILL, "////"),
    ("draft_kv_bytes", "draft KV (static, all rows)", ORANGE, None),
    ("selected_hidden_bytes", "tapped target hidden (1 row)", AQUA, None),
    ("context_feature_bytes", "context feature", YELLOW, None),
]


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
    })


def save(fig, stem: str) -> list[Path]:
    paths = [OUT_DIR / f"{stem}{SUFFIX}.png", OUT_DIR / f"{stem}{SUFFIX}.pdf"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def write_csv(stem: str, rows: list[dict]) -> Path:
    path = OUT_DIR / f"{stem}{SUFFIX}.csv"
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    return path


def log2_batch_axis(ax, batches=BATCHES) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlim(batches[0] / 1.4, batches[-1] * 1.4)
    ax.minorticks_off()
    ax.set_xlabel("batch size B")


def footnote(fig, text: str, y: float = -0.01) -> None:
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7.2, color=INK2, wrap=True)


# ---------------------------------------------------------------- records

def records() -> dict:
    return load(RECORD_DIR)


def ceiling() -> dict:
    path = RECORD_DIR / "ceiling.json"
    if not path.exists():          # same card, so the main sweep's measurement holds
        path = ROOT / "record_batch" / "ceiling.json"
    return json.loads(path.read_text())


def gdn_backend(recs: dict) -> str | None:
    """How the GDN layers were run in this sweep, from the records themselves.

    Records written before the field existed do not say; they all predate the
    fla install, so an absent field means the torch fallback.
    """
    kernels = {rec.get("gdn_kernels", {}).get("torch_chunk_gated_delta_rule")
               for key, rec in recs.items() if key[0] == "qwen3.5-9b"}
    if not kernels:
        return None
    if len(kernels) > 1:
        return "mixed GDN kernels -- do not compare these points"
    kernel = kernels.pop()
    if kernel and kernel.startswith("fla"):
        version = next(rec["gdn_kernels"].get("fla_version") for key, rec in recs.items()
                       if key[0] == "qwen3.5-9b")
        return f"fla {version} kernels"
    return "torch fallback (no fla)"


def summary(recs: dict, model: str, context: int, batch: int, mode: str) -> dict | None:
    rec = recs.get((model, context, batch))
    if rec is None:
        return None
    return rec["summary"].get(mode)


def status(recs: dict, model: str, context: int, batch: int, mode: str) -> str | None:
    """'ok', 'oom', or None when the point was never attempted."""
    rec = recs.get((model, context, batch))
    if rec is None:
        return None
    return rec["status"].get(mode)


# ---------------------------------------------------------------- analytic model
# Only what the records do not carry. KV per token, weights and GDN state come
# from the records themselves.
ARCH = {
    "qwen3-8b": dict(
        vocab=151936, hidden=4096, attn_layers=36, q_heads=32, kv_heads=8, head_dim=128,
        gdn_layers=0, gdn_heads=0, gdn_dk=0, gdn_dv=0,
        draft=dict(full_layers=5, sliding_layers=0, window=None, taps=5,
                   q_heads=32, kv_heads=8, head_dim=128),
    ),
    "qwen3.5-9b": dict(
        vocab=248320, hidden=4096, attn_layers=8, q_heads=16, kv_heads=4, head_dim=256,
        gdn_layers=24, gdn_heads=32, gdn_dk=128, gdn_dv=128,
        draft=dict(full_layers=1, sliding_layers=5, window=4096, taps=8,
                   q_heads=32, kv_heads=8, head_dim=128),
    ),
}


def mean_context(rec: dict) -> float:
    """Mean keys a decode step attends over: the prompt plus half the output."""
    return rec["context_length"] + rec["fixed_output_tokens"] / 2


def verify_work(model: str, rec: dict, mode: str) -> dict:
    """Bytes and FLOPs one verify forward of the whole batch must move and do."""
    a = ARCH[model]
    s = rec["summary"][mode]
    batch = rec["batch_size"]
    q = 1 if mode == "baseline" else rec["block_size"]
    keys = mean_context(rec)
    weights = s["target_weight_gb"] * GIB - EMBED_BYTES[model]   # executed: no embedding table
    kv_token = a["attn_layers"] * 2 * a["kv_heads"] * a["head_dim"] * 2
    kv = batch * (keys + q) * kv_token
    gdn = 2 * s["gdn_state_gb"] * GIB                             # recurrent + conv state, in and out
    logits = 2 * batch * q * a["vocab"] * 2                       # written, then read by the argmax
    gemm = 2 * (weights / 2) * batch * q
    attention = 4 * batch * q * keys * a["q_heads"] * a["head_dim"] * a["attn_layers"]
    # Recurrent form, per token and head: decay, S^T k, the rank-1 update, q^T S.
    delta_rule = 7 * batch * q * a["gdn_layers"] * a["gdn_heads"] * a["gdn_dk"] * a["gdn_dv"]
    return {
        "bytes": weights + kv + gdn + logits,
        "flops": gemm + attention + delta_rule,
        "weight_bytes": weights, "kv_bytes": kv, "gdn_bytes": gdn,
        "gemm_flops": gemm, "attention_flops": attention,
    }


def mean_widest(rec: dict) -> float:
    """Mean over steps of the most tokens any active row kept: the context
    columns every row's draft forward projects that step (plan §4.1)."""
    rows = {r["prompt"]: r for r in rec["rows"]}
    widths = []
    for batch in rec["batches"]["dflash"]:
        kept = [rows[p]["dflash"]["kept"] for p in batch["prompts"]]
        for step in range(max(len(k) for k in kept)):
            widths.append(max(k[step] for k in kept if len(k) > step))
    return float(np.mean(widths)) if widths else 0.0


def draft_work(model: str, rec: dict) -> dict:
    """One steady draft forward plus its logits, for the whole batch.

    Context tokens only pass through ``fc`` and each layer's K/V projection;
    the 16 block positions go through every layer; 15 of them reach the
    target's LM head.
    """
    a, d = ARCH[model], ARCH[model]["draft"]
    s = rec["summary"]["dflash"]
    batch, q = rec["batch_size"], rec["block_size"]
    hidden, vocab = a["hidden"], a["vocab"]
    keys = mean_context(rec) + q
    layers = d["full_layers"] + d["sliding_layers"]
    window_keys = min(keys, d["window"]) if d["window"] else keys
    attended = d["full_layers"] * keys + d["sliding_layers"] * window_keys
    kv_layer_token = 2 * d["kv_heads"] * d["head_dim"] * 2
    widest = mean_widest(rec)

    draft_weights = s["draft_weight_gb"] * GIB
    params = draft_weights / 2
    fc = d["taps"] * hidden * hidden
    kv_proj = 2 * hidden * d["kv_heads"] * d["head_dim"]
    head = vocab * hidden
    gemm = 2 * batch * (q * (params - fc) + widest * (fc + layers * kv_proj) + (q - 1) * head)
    attention = 4 * batch * q * attended * d["q_heads"] * d["head_dim"]
    moved = (draft_weights + 2 * head
             + batch * attended * kv_layer_token
             + batch * widest * d["taps"] * hidden * 2
             + 2 * batch * (q - 1) * vocab * 2)
    return {"bytes": moved, "flops": gemm + attention, "widest": widest}
