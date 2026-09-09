"""Palette and loading for the v2 (phase-instrumented) selective figures.

Extends :mod:`_style` rather than replacing it: colours, the shard marking, the
footnote machinery and the bar geometry are all shared, so a v2 panel can be
laid next to its v1 counterpart and read without re-learning the legend.

What is new here is everything that came out of the phase instrumentation --
``summary.*.phase_memory``, ``overhead_split``, and the first-vs-steady draft
fields. The v1 records did not carry any of it, so a v2 figure is *not* a
redraw of the v1 one; the green wash on a panel title says which.

The one thing worth reading twice is :data:`PHASE_ORDER`. The records store
phases in first-occurrence order, which puts the steady-state draft forward
last (it does not happen until the second decode iteration). That is the wrong
order to plot: it separates the two draft forwards, which are the pair the
whole figure exists to compare. PHASE_ORDER restores execution order instead.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _style import (  # noqa: F401,E402  (re-exported for the v2 scripts)
    BASE_COLORS, BASE_LINE_COLOR, BASELINE_BUDGET_COMPONENTS, BUDGET_COMPONENTS,
    CONFIG_COLORS, CTX_LABELS, CTX_LENGTHS, HILITE, HILITE_EDGE, MEM_COMPONENTS,
    MODEL_COLORS, MODEL_MARKERS, MODELS, Notes, OUT_DIR, RECORD_DIR, ROOT,
    TIME_COMPONENTS, bar_positions, configs, ctx_axis, device_note,
    footnote_box, load_records, shard_ticks, short_model, titled,
    value_text_color,
)

# Execution order, not record order. See the module docstring.
PHASE_ORDER = [
    "prefill: target forward",
    "prefill: context-feature build",
    "prefill: first token",
    "prefill: context-feature concat",
    "prefill: rollback/crop",
    "decode: first draft forward",
    "decode: draft forward",
    "decode: draft rollback/crop",
    "decode: draft logits",
    "decode: target verify",
    "decode: verify rollback/crop",
    "decode: context-feature build",
]
PHASE_SHORT = {
    "prefill: target forward": "P1 target fwd",
    "prefill: context-feature build": "P2 ctx-feat build",
    "prefill: first token": "P3 first token",
    "prefill: context-feature concat": "P4 ctx-feat concat",
    "prefill: rollback/crop": "P5 rollback",
    "decode: first draft forward": "D1 first draft",
    "decode: draft forward": "D1' steady draft",
    "decode: draft rollback/crop": "D2 draft rollback",
    "decode: draft logits": "D3 draft logits",
    "decode: target verify": "D4 target verify",
    "decode: verify rollback/crop": "D5 verify rollback",
    "decode: context-feature build": "D6 ctx-feat build",
}

# The component split recorded at the instant a phase peaked. Ordered so the
# terms both configurations pay sit at the bottom and the DFlash-specific ones
# stack on top: the height of the upper block is the drafter's contribution to
# that peak, readable without subtracting two bars.
PEAK_COMPONENTS = [
    ("target_weight_bytes", "Target weights", "#8c8c8c"),
    ("target_kv_bytes", "Target KV (live at peak)", "#c0504d"),
    ("draft_weight_bytes", "Draft weights", "#3b4d8f"),
    ("draft_kv_bytes", "Draft KV", "#7fb3d5"),
    ("selected_hidden_bytes", "Selected target hidden", "#e8a33d"),
    ("context_feature_bytes", "Context feature", "#6aa84f"),
]
UNATTRIBUTED_COLOR = "#d9b3d9"

# The drafter's five internal stages, in the order a draft forward runs them.
STAGE_COMPONENTS = [
    ("context_projection", "fc + hidden_norm", "#3b4d8f"),
    ("context_kv_projection", "context K/V proj", "#7fb3d5"),
    ("cache_update", "KV append", "#6aa84f"),
    ("attention", "attention (SDPA)", "#e8a33d"),
    ("output_head", "lm_head", "#c0504d"),
]

# overhead_split: what only DFlash pays, against what the target costs anyway.
DFLASH_OVERHEAD = [
    ("dflash_draft_weight_gb", "Draft weights", "#3b4d8f"),
    ("dflash_draft_kv_gb", "Draft KV (steady)", "#7fb3d5"),
    ("dflash_selected_hidden_gb", "Selected target hidden", "#e8a33d"),
    ("dflash_context_feature_gb", "Context feature", "#6aa84f"),
]
TARGET_OVERHEAD = [
    ("target_weight_gb", "Target weights", "#8c8c8c"),
    ("target_kv_gb", "Target KV (recorded)", "#c0504d"),
    ("target_prefill_transient_gb", "Target prefill activation", "#d9b3d9"),
]


def load_v2(record_dir=RECORD_DIR):
    """Records from the phase-instrumented sweep only.

    A directory may still hold pre-instrumentation records from an earlier
    sweep; mixing the two would put a v1 row in a v2 table with silent zeroes,
    so anything without ``phase_memory`` is dropped rather than defaulted. When
    two records exist for one configuration the newer filename wins.
    """
    best = {}
    for path in sorted(glob.glob(os.path.join(record_dir, "*.json"))):
        with open(path) as fh:
            payload = json.load(fh)
        if not payload["summary"]["dflash"].get("phase_memory"):
            continue
        key = (payload["model_name"], payload["context_length"])
        name = os.path.basename(path)
        if key not in best or name > best[key][0]:
            best[key] = (name, payload)

    recs = {}
    for key, (_name, payload) in best.items():
        entry = load_one(payload)
        recs[key] = entry
    return recs


def load_one(payload):
    """One record, with the derived quantities the v2 panels plot."""
    s, b = dict(payload["summary"]["dflash"]), dict(payload["summary"]["baseline"])
    n = payload["num_samples"]
    decode_total = s["drafter_latency_s"] / s["drafter_share_of_decode"]
    s["decode_total_s"] = decode_total
    s["other_s"] = decode_total - s["drafter_latency_s"] - s["target_forward_s"]
    for m in (s, b):
        m["mean_decode_s"] = m["mean_latency_s"] - m["mean_ttft_s"]
    s["target_forwards_per_req"] = s["total_verify_steps"] / n
    b["target_forwards_per_req"] = b["mean_decode_steps"]

    # The correction this whole sweep exists to make. transient_gb subtracted
    # component-wise maxima taken at different instants; peak_transient_gb is
    # the same quantity read at the one instant the peak actually happened.
    for m in (s, b):
        peak_phase = m["phase_memory"][m["peak_phase"]]
        m["peak_phase_entry"] = peak_phase
        m["real_transient_gb"] = peak_phase["peak_transient_gb"]
        m["old_transient_gb"] = m["memory_budget"]["transient_gb"]
        m["target_kv_at_peak_gb"] = peak_phase["peak_components_gb"].get(
            "target_kv_bytes", 0.0)
        m["target_kv_recorded_gb"] = m["max_target_cache_gb"]

    # Draft ratios only mean something when both numbers exist and are nonzero.
    steady = s.get("mean_steady_draft_forward_s") or 0.0
    first = s.get("mean_first_draft_forward_s") or 0.0
    s["first_over_steady"] = (first / steady) if steady else 0.0
    s["draft_kv_first_over_steady"] = (
        s["max_first_draft_cache_gb"] / s["max_steady_draft_cache_gb"]
        if s["max_steady_draft_cache_gb"] else 0.0)

    return {
        "dflash": s,
        "baseline": b,
        "speedup": payload["decoding_speedup"],
        "n": n,
        "hidden_states": payload["hidden_states"],
        "num_devices": payload.get("num_devices") or 1,
        "rope_scaling": payload.get("rope_scaling"),
        "n_datasets": len({x["task"] for x in payload["samples"]}),
        "n_composed": sum(1 for x in payload["samples"] if x.get("composed")),
    }


def phases_of(entry, source="dflash"):
    """(name, phase) in execution order, skipping phases this run never hit.

    The baseline runs five of the twelve -- it has no drafter -- so a caller
    that wants both configurations on one axis must not assume the full list.
    """
    stored = entry[source]["phase_memory"]
    known = [n for n in PHASE_ORDER if n in stored]
    # Anything the records grew that PHASE_ORDER has not been told about is
    # appended rather than dropped, so a new phase shows up as an unplaced
    # column instead of vanishing.
    known += [n for n in stored if n not in PHASE_ORDER]
    return [(n, stored[n]) for n in known]


def sharded_note(recs, model=None):
    """Why a sharded run's *stage* timings cannot be read as latency.

    Distinct from device_note(): that one is about comparing across the two
    sweeps. This one is about a within-figure trap -- on a sharded run lm_head
    sits on a different card from the drafter, and the CUDA events that time
    the stage are recorded on the drafter's stream, so the head reads as free.
    """
    starred = [f"{short_model(m)}/{CTX_LABELS[c]}"
               for (m, c), r in sorted(recs.items())
               if r["num_devices"] > 1 and (model is None or m == model)]
    if not starred:
        return None
    return (
        f"{', '.join(starred)} is sharded over more than one card (starred on the x axis). Per-stage draft "
        f"timings are recorded with CUDA events on the drafter's stream, so a stage whose weights live on the "
        f"other card -- lm_head, in practice -- is timed as the launch, not the work: it reads as ~0.05 ms "
        f"against ~2.8 ms on a single card. Read the stage split of a starred column qualitatively only, and "
        f"never subtract it from a single-GPU column."
    )
