"""The condition sets, and the driver that walks them.

The three axes -- input length, block width, output length -- are swept one at
a time around a representative point, never as a Cartesian product. Crossing
them would be 45 conditions per model where 11 answer the questions, and at
64k a single condition is minutes of prefill.

``sequence``
    S in {4k, 8k, 16k, 32k, 64k} at block 16, 256 output tokens. This is the
    axis research questions 1 and 3 are asked along.
``block``
    Block width in {4, 8, 16} at the representative S. Width 4 proposes 3
    draft tokens, 8 proposes 7, 16 proposes 15 -- the anchor is part of the
    verify but is not a proposal.
``output``
    Output length in {256, 1024, 4096} at the representative S and block 16.
    Separates the first-draft setup cost, which is paid once, from the steady
    state that amortises it.

Every condition is run under both output policies where it matters:
shape-controlled (EOS ignored, exactly N tokens, so token count is a
controlled variable) and natural (stops at EOS). Acceptance from a
shape-controlled run past the natural stopping point is not mixed into the
natural figure -- past EOS the model is being asked to continue text it
considers finished, and the drafter's acceptance there is not the acceptance a
request would see.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import torch

from . import env, models, placement, run, taxonomy
from .run import (
    Condition,
    OUTPUT_NATURAL,
    OUTPUT_SHAPE_CONTROLLED,
    PASS_MEMORY,
    PASS_PERF,
)

SEQUENCE_LENGTHS = (4096, 8192, 16384, 32768, 65536)
BLOCK_WIDTHS = (4, 8, 16)
OUTPUT_LENGTHS = (256, 1024, 4096)

REPRESENTATIVE_S = 32768
DEFAULT_BLOCK = 16
DEFAULT_OUTPUT = 256

RECORD_ROOT = "record_arch_main"


def conditions(
    model_key: str,
    *,
    sweeps: tuple[str, ...] = ("sequence", "block", "output"),
    representative_s: int = REPRESENTATIVE_S,
    repeats: int = 3,
    natural_output: bool = True,
) -> list[Condition]:
    """The condition list for one model. No axis is crossed with another."""
    out: list[Condition] = []
    if "sequence" in sweeps:
        for length in SEQUENCE_LENGTHS:
            out.append(Condition(
                model_key, length, DEFAULT_BLOCK, DEFAULT_OUTPUT,
                OUTPUT_SHAPE_CONTROLLED, "sequence", repeats,
            ))
    if "block" in sweeps:
        for width in BLOCK_WIDTHS:
            if width == DEFAULT_BLOCK and "sequence" in sweeps:
                continue  # already covered at the representative S
            out.append(Condition(
                model_key, representative_s, width, DEFAULT_OUTPUT,
                OUTPUT_SHAPE_CONTROLLED, "block", repeats,
            ))
    if "output" in sweeps:
        for length in OUTPUT_LENGTHS:
            if length == DEFAULT_OUTPUT and "sequence" in sweeps:
                continue
            out.append(Condition(
                model_key, representative_s, DEFAULT_BLOCK, length,
                OUTPUT_SHAPE_CONTROLLED, "output", repeats,
            ))
    if natural_output:
        # One natural-output point, at the representative condition, so the
        # shape-controlled numbers can be checked against a run that stopped
        # where the model wanted to.
        out.append(Condition(
            model_key, representative_s, DEFAULT_BLOCK, OUTPUT_LENGTHS[-1],
            OUTPUT_NATURAL, "output_policy", repeats,
        ))
    return out


def context_limits(pair: models.ModelPair, target_config, draft_config) -> dict:
    """Trained-context and RoPE facts, per condition length.

    Recorded rather than enforced. A run past a trained window still produces
    numbers; what it must not do is have those numbers read as comparable with
    the rest of the sweep.
    """
    text = getattr(target_config, "text_config", None) or target_config
    target_max = getattr(text, "max_position_embeddings", None)
    draft_max = getattr(draft_config, "max_position_embeddings", None)
    rope = getattr(text, "rope_parameters", None) or getattr(
        text, "rope_scaling", None
    )
    return {
        "target_max_position_embeddings": target_max,
        "draft_max_position_embeddings": draft_max,
        "draft_trained_context": pair.draft_trained_context,
        "target_rope": rope if isinstance(rope, dict) else str(rope),
        "note": (
            "max_position_embeddings is a RoPE range, not a trained length. "
            "draft_trained_context is what the drafter's model card states."
        ),
    }


def flag_condition(condition: Condition, limits: dict) -> dict:
    """Whether this condition sits outside a stated range, and which one."""
    total = condition.input_tokens + condition.max_new_tokens
    flags = []
    target_max = limits.get("target_max_position_embeddings")
    draft_max = limits.get("draft_max_position_embeddings")
    trained = limits.get("draft_trained_context")
    if target_max is not None and total > target_max:
        flags.append("beyond_target_rope_range")
    if draft_max is not None and total > draft_max:
        flags.append("beyond_draft_rope_range")
    if trained is not None and total > trained:
        flags.append("beyond_draft_trained_context")
    return {
        "positions_needed": total,
        "flags": flags,
        "within_all_stated_ranges": not flags,
    }


def record_dir(model_key: str, root: str = RECORD_ROOT) -> str:
    path = os.path.join(root, model_key)
    os.makedirs(path, exist_ok=True)
    return path


def write_record(model_key: str, payload: dict, *, root: str = RECORD_ROOT,
                 label: str = "sweep", stamp: str | None = None) -> str:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(record_dir(model_key, root), f"{label}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=str)
    return path


def plan(model_keys: list[str], **kwargs) -> dict:
    """The conditions a sweep would run, without loading anything.

    Written as its own entry point because the plan is part of the record: a
    reader needs to distinguish conditions that were measured from conditions
    that were only ever scheduled.
    """
    from transformers import AutoConfig

    out: dict = {"models": {}, "generated_utc": datetime.now(timezone.utc).isoformat()}
    for key in model_keys:
        pair = models.resolve(key)
        entry: dict = {"pair": pair.__dict__.copy()}
        try:
            target_config = AutoConfig.from_pretrained(pair.target)
            draft_config = AutoConfig.from_pretrained(pair.draft)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            out["models"][key] = entry
            continue
        limits = context_limits(pair, target_config, draft_config)
        entry["limits"] = limits
        entry["pairing"] = models.check_pairing(target_config, draft_config)
        entry["conditions"] = [
            {**c.as_dict(), "key": c.key(), **flag_condition(c, limits)}
            for c in conditions(key, **kwargs)
        ]
        out["models"][key] = entry
    out["device_health"] = env.device_health()
    return out
