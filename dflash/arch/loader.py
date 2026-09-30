"""Loading a target and a drafter onto a fixed, recorded placement.

Two properties matter more than convenience here. The placement must be the
same for the AR run and the DFlash run of a given model, and it must be
written down. :mod:`dflash.arch.placement` builds the map; this module applies
it and reports what actually happened, because accelerate may not honour a map
it considers infeasible and a silently different placement invalidates the
comparison it was built for.

The drafter is loaded separately and can be skipped entirely. ``cmd_sweep``
runs every AR condition before it calls :func:`load_draft`, which is what makes
the AR memory figures a drafter-free baseline rather than a block-size-1 run
with the drafter sitting on the card. (Earlier records loaded both up front
through :func:`load_pair`; their AR peaks include the drafter's weights.)
"""

from __future__ import annotations

import torch

from . import env, models, placement, taxonomy


def _target_class(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_id)
    architectures = config.architectures or []
    if any("ConditionalGeneration" in name for name in architectures):
        return AutoModelForImageTextToText, config
    return AutoModelForCausalLM, config


def _stack_path_from_config(config) -> str:
    """Where the decoder stack will sit, before the model exists.

    The device map has to be built before ``from_pretrained`` is called, so the
    path cannot be discovered from the instantiated module tree the way
    :func:`taxonomy.describe` does it. A conditional-generation wrapper nests
    the text tower one level deeper; everything else keeps it at
    ``model.layers``. The instantiated path is checked against this afterwards.
    """
    architectures = config.architectures or []
    if any("ConditionalGeneration" in name for name in architectures):
        return "model.language_model.layers"
    return "model.layers"


def load_target(
    model_id: str,
    *,
    devices: list[int] | None = None,
    shard: bool = True,
    dtype=torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> tuple[torch.nn.Module, dict]:
    """Load one target onto an explicit layer-sharded placement."""
    cls, config = _target_class(model_id)
    text = taxonomy.text_config(config)
    devices = devices or env.usable_devices()
    stack_path = _stack_path_from_config(config)

    kwargs = {"dtype": dtype, "attn_implementation": attn_implementation}
    requested = None
    if shard and len(devices) > 1:
        requested = placement.build_device_map(
            type(config).__name__, text.num_hidden_layers, stack_path, devices
        )
        kwargs["device_map"] = requested
    model = cls.from_pretrained(model_id, **kwargs)
    if requested is None:
        model = model.to(f"cuda:{devices[0]}")
    model = model.eval()

    spec = taxonomy.describe(model, role="target")
    report = {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "dtype": str(dtype),
        "attn_implementation": attn_implementation,
        "devices_requested": devices,
        "sharded": requested is not None,
        "stack_path_predicted": stack_path,
        "stack_path_actual": spec["stack_path"],
        "stack_path_matched": stack_path == spec["stack_path"],
        "device_map_requested": requested,
        "device_map_applied": getattr(model, "hf_device_map", None),
        "layers_per_device": spec["layers_per_device"],
        "param_bytes_per_device": spec["param_bytes_per_device"],
        "mixer_counts": spec["mixer_counts"],
        "ffn_counts": spec["ffn_counts"],
    }
    if requested is not None:
        report["placement"] = placement.describe(requested, spec["stack_path"])
        # accelerate is free to ignore a map it cannot satisfy. If it did, the
        # AR/DFlash comparison this placement exists for is no longer
        # controlled, so the record says so rather than the run proceeding
        # quietly.
        applied = getattr(model, "hf_device_map", None) or {}
        report["placement_honoured"] = all(
            str(applied.get(key, value)).replace("cuda:", "") ==
            str(value).replace("cuda:", "")
            for key, value in requested.items()
            if key in applied
        )
    return model, {"target": report, "taxonomy": spec}


def load_draft(
    draft_id: str,
    *,
    device: int | None = None,
    dtype=torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> tuple[torch.nn.Module, dict]:
    """Load one drafter replica onto a single device, never sharded."""
    from transformers import AutoConfig

    from ..model import DFlash2DraftModel, DFlashDraftModel

    config = AutoConfig.from_pretrained(draft_id)
    cls = (
        DFlash2DraftModel
        if "DFlash2DraftModel" in (config.architectures or [])
        else DFlashDraftModel
    )
    device = device if device is not None else placement.draft_device()
    model = (
        cls.from_pretrained(
            draft_id, attn_implementation=attn_implementation, dtype=dtype
        )
        .to(f"cuda:{device}")
        .eval()
    )
    spec = taxonomy.describe(model, role="draft")
    return model, {
        "draft": {
            "model_id": draft_id,
            "model_class": cls.__name__,
            "device": f"cuda:{device}",
            "dtype": str(dtype),
            "block_size": getattr(model, "block_size", None),
            "target_layer_ids": list(getattr(model, "target_layer_ids", []) or []),
            "param_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "mixer_counts": spec["mixer_counts"],
        },
        "taxonomy": spec,
    }


def load_pair(
    key: str,
    *,
    devices: list[int] | None = None,
    shard: bool = True,
    with_draft: bool = True,
):
    """Target, drafter and tokenizer for a registry key, with the pairing checked."""
    from transformers import AutoConfig, AutoTokenizer

    pair = models.resolve(key)
    target_config = AutoConfig.from_pretrained(pair.target)
    draft_config = AutoConfig.from_pretrained(pair.draft)
    pairing = models.check_pairing(target_config, draft_config)
    if not pairing["compatible"]:
        raise ValueError(
            f"drafter {pair.draft} is not compatible with target {pair.target}: "
            f"{pairing['problems']}"
        )

    target, target_report = load_target(pair.target, devices=devices, shard=shard)
    draft = draft_report = None
    if with_draft:
        draft, draft_report = load_draft(
            pair.draft, device=(devices or env.usable_devices())[0]
        )
    tokenizer = AutoTokenizer.from_pretrained(pair.target)
    return target, draft, tokenizer, {
        "pair": pair.__dict__.copy(),
        "pairing": pairing,
        **target_report,
        **({"draft_report": draft_report} if draft_report else {}),
    }
