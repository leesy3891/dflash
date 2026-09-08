"""Shared palette, record loading, and change-annotation machinery.

These plots are the `--hidden-states selective` counterparts of the ones in
`visualization/`, and are meant to be read against them. Some panels carry the
same quantity measured the same way and only differ because the run is cheaper;
others carry a quantity that did not exist before, or one whose *measurement*
changed. Those are the ones a reader has to be told about, so this module
carries the mechanism for saying so:

* :func:`titled` puts a pale-green wash behind a panel title and appends a
  footnote marker, for any panel whose numbers or method moved.
* :func:`footnote_box` renders the collected markers as a boxed block under the
  figure, in footnote order.

Panels that are untouched get a plain title and no marker, so the highlight
means something.
"""

import glob
import json
import os
import textwrap

import matplotlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_DIR = os.path.join(ROOT, "record_selective")
OUT_DIR = os.path.join(ROOT, "visualization_selective")

CTX_LENGTHS = [4096, 8192, 16384, 32768, 65536]
CTX_LABELS = {4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K"}
MODELS = ["qwen3-8b", "qwen3.5-9b"]

# Identical to visualization/ so the two sets of figures can be read together.
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
BASE_COLORS = {4096: "#dcdcdc", 8192: "#b3b3b3", 16384: "#8a8a8a",
               32768: "#5c5c5c", 65536: "#303030"}
MODEL_COLORS = {"qwen3-8b": "#2171b5", "qwen3.5-9b": "#e6550d"}
MODEL_MARKERS = {"qwen3-8b": "o", "qwen3.5-9b": "s"}
BASE_LINE_COLOR = "#6e6e6e"

# The wash behind a changed panel's title, and the frame of the footnote box.
HILITE = "#dcf0dd"
HILITE_EDGE = "#7bbf86"

# Drafter-overhead components, as in visualization/plot_summary.py.
MEM_COMPONENTS = [
    ("draft_weight_gb", "Draft weights", "#3b4d8f"),
    ("max_draft_cache_gb", "Draft KV cache", "#7fb3d5"),
    ("max_target_hidden_states_gb", "Target hidden states", "#e8a33d"),
    ("max_context_feature_gb", "Context feature", "#6aa84f"),
]
TIME_COMPONENTS = [
    ("draft_forward_s", "Draft forward", "#3b4d8f"),
    ("context_feature_s", "Context feature", "#6aa84f"),
    ("target_forward_s", "Target forward (verify)", "#c0504d"),
    ("other_s", "Other (sampling/overhead)", "#b0b0b0"),
]
# Whole-process budget: every resident term, then what the run borrows on top.
# Target weights lead because they are the floor both configurations pay.
BUDGET_COMPONENTS = [
    ("target_weight_gb", "Target weights", "#8c8c8c"),
    ("target_cache_gb", "Target KV cache", "#c0504d"),
    ("draft_weight_gb", "Draft weights", "#3b4d8f"),
    ("draft_cache_gb", "Draft KV cache", "#7fb3d5"),
    ("target_hidden_states_gb", "Target hidden states", "#e8a33d"),
    ("context_feature_gb", "Context feature", "#6aa84f"),
    ("transient_gb", "Transient (activation)", "#d9b3d9"),
]
# The baseline never calls the drafter but keeps its weights on the same card.
BASELINE_BUDGET_COMPONENTS = [
    ("target_weight_gb", "Target weights", "#8c8c8c"),
    ("target_cache_gb", "Target KV cache", "#c0504d"),
    ("draft_weight_resident_gb", "Draft weights (loaded, unused)", "#b8bedd"),
    ("transient_gb", "Transient (activation)", "#d9b3d9"),
]


def load_records(record_dir=RECORD_DIR):
    """One entry per (model, context length), both decoding configurations."""
    recs = {}
    for path in sorted(glob.glob(os.path.join(record_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        s, b = dict(d["summary"]["dflash"]), dict(d["summary"]["baseline"])
        n = d["num_samples"]
        decode_total = s["drafter_latency_s"] / s["drafter_share_of_decode"]
        s["decode_total_s"] = decode_total
        s["other_s"] = decode_total - s["drafter_latency_s"] - s["target_forward_s"]
        for m in (s, b):
            m["mean_decode_s"] = m["mean_latency_s"] - m["mean_ttft_s"]
        # Target forwards per request: verify steps against one step per token.
        s["target_forwards_per_req"] = s["total_verify_steps"] / n
        b["target_forwards_per_req"] = b["mean_decode_steps"]
        recs[(d["model_name"], d["context_length"])] = {
            "dflash": s,
            "baseline": b,
            "speedup": d["decoding_speedup"],
            "n": n,
            "hidden_states": d["hidden_states"],
            "num_devices": d.get("num_devices") or 1,
            "rope_scaling": d.get("rope_scaling"),
            "n_datasets": len({x["task"] for x in d["samples"]}),
            "n_composed": sum(1 for x in d["samples"] if x.get("composed")),
        }
    return recs


def configs(recs):
    return [(m, c) for m in MODELS for c in CTX_LENGTHS if (m, c) in recs]


def sharded(recs):
    """(model, context) pairs whose run was split over more than one card."""
    return [k for k, r in recs.items() if r["num_devices"] > 1]


def device_count_changes(recs, full_dir=None, model=None):
    """Runs whose device count differs from the full-mode sweep.

    Selective frees enough memory that some runs need fewer cards than they did
    in ``record/``. That matters twice over: it is why a peak dropped further
    than the hidden-state term alone explains, and it is why a timing must not
    be read across the two sweeps. Returns ``(model, ctx, full_n, sel_n)``.
    """
    full_dir = full_dir or os.path.join(ROOT, "record")
    full = {}
    for path in sorted(glob.glob(os.path.join(full_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        full[(d["model_name"], d["context_length"])] = d.get("num_devices") or 1
    out = []
    for (m, c), r in sorted(recs.items()):
        if model and m != model:
            continue
        before = full.get((m, c))
        if before is not None and before != r["num_devices"]:
            out.append((m, c, before, r["num_devices"]))
    return out


def short_model(model):
    return "8B" if model == "qwen3-8b" else "9B"


def device_note(recs, model=None):
    """The footnote text for a device-count change, or None when there is none."""
    changes = device_count_changes(recs, model=model)
    if not changes:
        return None
    moved = ", ".join(f"{short_model(m)}/{CTX_LABELS[c]} {a}->{b}" for m, c, a, b in changes)
    still = [f"{short_model(m)}/{CTX_LABELS[c]}" for (m, c), r in sorted(recs.items())
             if r["num_devices"] > 1 and (model is None or m == model)]
    star = (", ".join(still) + " is still sharded and starred on the x axis, so its timings are "
            "pipeline-parallel.") if still else \
        "No run here is sharded any more, so every column is single-GPU."
    return (
        f"Device count differs from the full-mode records: {moved}. Selective frees enough memory to fit on "
        f"fewer cards. {star} Memory stays comparable across a device change; timings do not -- going from "
        f"sharded to single-GPU speeds the *baseline* up more than it speeds DFlash, so a speedup can read "
        f"lower without DFlash having got slower."
    )


# --------------------------------------------------------------------------
# Change annotation
# --------------------------------------------------------------------------

class Notes:
    """Collects footnotes and hands out markers in the order panels are drawn.

    A note is registered once under a key and may be attached to several
    panels; the marker stays the same, so two panels that changed for the same
    reason point at one explanation rather than repeating it.
    """

    def __init__(self):
        self._order = []
        self._text = {}

    WIDTH = 118
    INDENT = " " * 16

    def add(self, key: str, kind: str, text: str) -> None:
        """Register a note. ``kind`` is 'added' or 'revised'.

        ``text`` is one paragraph; wrapping is done here so every note in the
        box breaks at the same column however it was written.
        """
        if key in self._text:
            raise KeyError(f"note {key!r} registered twice")
        self._order.append(key)
        self._text[key] = (kind, " ".join(text.split()))

    def marker(self, key: str) -> str:
        if key not in self._text:
            raise KeyError(f"unknown note {key!r}")
        return f"[{self._order.index(key) + 1}]"

    def lines(self) -> list[str]:
        out = []
        for i, key in enumerate(self._order, 1):
            kind, text = self._text[key]
            out.extend(textwrap.wrap(
                f"[{i}] {kind.upper():<7} {text}",
                width=self.WIDTH,
                subsequent_indent=self.INDENT,
                break_long_words=False,
                break_on_hyphens=False,
            ))
        return out


def titled(ax, title: str, notes: Notes | None = None, keys=()):
    """Set a panel title, washed pale green when the panel carries a note.

    ``keys`` are note keys; their markers are appended to the title so the box
    under the figure can be read against the panels without hunting.
    """
    if not keys:
        ax.set_title(title)
        return
    marks = " ".join(notes.marker(k) for k in keys)
    ax.set_title(
        f"{title}  {marks}",
        bbox=dict(facecolor=HILITE, edgecolor=HILITE_EDGE, linewidth=0.8,
                  boxstyle="round,pad=0.34", alpha=0.95),
    )


def footnote_box(fig, notes: Notes, y=0.012, fontsize=7.4, title=None):
    """Render the collected notes as one boxed block under the figure."""
    header = title or (
        "Panels highlighted in green differ from visualization/ (hidden_states=full) — "
        "ADDED = a quantity that did not exist in those records, "
        "REVISED = same quantity, changed measurement"
    )
    body = "\n".join([header, ""] + notes.lines())
    fig.text(0.5, y, body, ha="center", va="bottom", fontsize=fontsize,
             family="DejaVu Sans Mono", color="#243024", linespacing=1.55,
             bbox=dict(facecolor=HILITE, edgecolor=HILITE_EDGE, linewidth=1.0,
                       boxstyle="round,pad=0.7", alpha=0.55))


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------

def bar_positions(n_ctx=len(CTX_LENGTHS), n_model=2, width=0.38):
    x = np.arange(n_ctx, dtype=float)
    return x, [x + (i - (n_model - 1) / 2) * width for i in range(n_model)], width


def ctx_axis(ax):
    x, _, _ = bar_positions()
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in CTX_LENGTHS])
    ax.set_xlabel("context length")
    ax.margins(y=0.20)


def shard_ticks(ax, recs, model=None):
    """Star the context lengths whose run was sharded.

    Acceptance and memory stay comparable across a device count, but timings do
    not, so the reader needs to see which columns are pipeline-parallel without
    consulting the record.
    """
    labels = []
    for c in CTX_LENGTHS:
        keys = [(model, c)] if model else [(m, c) for m in MODELS]
        star = any(recs[k]["num_devices"] > 1 for k in keys if k in recs)
        labels.append(CTX_LABELS[c] + (" *" if star else ""))
    ax.set_xticks(np.arange(len(CTX_LENGTHS), dtype=float))
    ax.set_xticklabels(labels)


def value_text_color(color):
    r, g, b = matplotlib.colors.to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else "#222222"
