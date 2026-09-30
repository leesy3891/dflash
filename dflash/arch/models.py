"""The three targets, their drafters, and what each pairing is allowed to do.

A drafter is trained against one target's residual streams and its vocabulary;
pairing it with a different target produces a model that runs and proposes
nonsense. So the pairing is declared here and checked against both configs
before either is loaded onto a GPU.

``trained_context`` is the length the drafter's model card states it was
trained at, separately from the target's ``max_position_embeddings``. They are
different limits and a run can be past one but not the other -- the 64k point
of this sweep is past both for Qwen3-8B and past the drafter's alone for
Qwen3.5-35B-A3B -- so both are recorded and the condition is flagged rather
than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPair:
    key: str
    target: str
    draft: str
    family: str
    # What the drafter's model card states it was trained to, in tokens.
    draft_trained_context: int | None = None
    note: str | None = None


REGISTRY: dict[str, ModelPair] = {
    "qwen3-8b": ModelPair(
        key="qwen3-8b",
        target="Qwen/Qwen3-8B",
        draft="z-lab/Qwen3-8B-DFlash-b16",
        family="dense/full-attention",
        note="36 full-attention layers, dense MLP. The full-attention control.",
    ),
    "qwen3.5-9b": ModelPair(
        key="qwen3.5-9b",
        target="/home/seoyounglee/models/Qwen3.5-9B",
        draft="z-lab/Qwen3.5-9B-DFlash",
        family="hybrid/gdn+attention",
        note="24 GDN + 8 full-attention layers, dense MLP.",
    ),
    "qwen3.5-35b-a3b": ModelPair(
        key="qwen3.5-35b-a3b",
        target="/home/seoyounglee/models/Qwen3.5-35B-A3B",
        draft="z-lab/Qwen3.5-35B-A3B-DFlash",
        family="hybrid/gdn+moe",
        draft_trained_context=40960,
        note=(
            "30 GDN + 10 full-attention layers, MoE (256 experts, top-8, "
            "shared expert) on every layer."
        ),
    ),
}


def resolve(key: str) -> ModelPair:
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; have {sorted(REGISTRY)}")
    return REGISTRY[key]


def check_pairing(target_config, draft_config) -> dict:
    """Whether this drafter can legitimately serve this target.

    Three things have to line up. The vocabularies must match, or the
    drafter's proposals index a different token set. The drafter's
    ``num_target_layers`` must equal the target's layer count, or its
    ``target_layer_ids`` point at layers that do not exist. And the context
    feature it projects is a concatenation of the target's residual streams,
    so the target's hidden size times the number of tapped layers has to be
    what the drafter's context projection expects.
    """
    text = getattr(target_config, "text_config", None) or target_config
    dflash_config = getattr(draft_config, "dflash_config", {}) or {}
    layer_ids = dflash_config.get("target_layer_ids") or []
    declared_layers = getattr(draft_config, "num_target_layers", None)

    checks = {
        "vocab_size": {
            "target": getattr(text, "vocab_size", None),
            "draft": getattr(draft_config, "vocab_size", None),
        },
        "num_target_layers": {
            "target": getattr(text, "num_hidden_layers", None),
            "draft": declared_layers,
        },
        "target_layer_ids": layer_ids,
        "max_target_layer_id": max(layer_ids) if layer_ids else None,
        "target_hidden_size": getattr(text, "hidden_size", None),
        "draft_hidden_size": getattr(draft_config, "hidden_size", None),
        "context_feature_dim": (
            getattr(text, "hidden_size", 0) * len(layer_ids) if layer_ids else None
        ),
        "block_size": dflash_config.get("block_size"),
        "mask_token_id": dflash_config.get("mask_token_id"),
    }

    problems = []
    if checks["vocab_size"]["target"] != checks["vocab_size"]["draft"]:
        problems.append("vocab_size mismatch")
    if (
        declared_layers is not None
        and checks["num_target_layers"]["target"] != declared_layers
    ):
        problems.append("num_target_layers mismatch")
    if layer_ids and checks["num_target_layers"]["target"] is not None:
        # HiddenStateTap cannot serve the final layer, whose reported hidden
        # state is taken after the final norm.
        if max(layer_ids) >= checks["num_target_layers"]["target"] - 1:
            problems.append("target_layer_ids include the final layer")
    mask = dflash_config.get("mask_token_id")
    if mask is not None and checks["vocab_size"]["target"] is not None:
        if mask >= checks["vocab_size"]["target"]:
            problems.append("mask_token_id outside the target vocabulary")

    return {"checks": checks, "problems": problems, "compatible": not problems}
