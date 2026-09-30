"""What each layer of a model actually is, read off the modules themselves.

The three targets in this sweep disagree on every axis a layer index could be
used to guess. Qwen3-8B is 36 full-attention layers with a dense MLP.
Qwen3.5-9B interleaves 24 gated-delta-net layers with 8 full-attention ones,
dense MLP throughout. Qwen3.5-35B-A3B is 30 GDN plus 10 full-attention, every
one of them MoE. ``full_attention_interval`` in a config describes the pattern
but not reliably the instantiated stack, and the drafter's own layers follow a
third rule (sliding windows).

So the mixer and the FFN are treated as two independent axes and both are read
from the instantiated module tree, cross-checked against ``layer_types`` when
the config has one. Nothing here keys off a layer number.

Axes
----
mixer : ``full_attention`` | ``sliding_attention`` | ``gdn`` | ``other``
ffn   : ``dense`` | ``moe`` | ``moe+shared`` | ``other``
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from torch import nn

MIXER_FULL = "full_attention"
MIXER_SLIDING = "sliding_attention"
MIXER_GDN = "gdn"
MIXER_OTHER = "other"

FFN_DENSE = "dense"
FFN_MOE = "moe"
FFN_MOE_SHARED = "moe+shared"
FFN_OTHER = "other"


def text_config(config):
    """The text tower's config, past any conditional-generation wrapper."""
    return getattr(config, "text_config", None) or config


def decoder_layers(model: nn.Module, expected: int | None = None) -> nn.ModuleList:
    """The decoder stack, however deeply the wrapper nests it.

    Matching on the expected length is what rules out a vision tower's own
    block list, which on Qwen3.5 sits in the same model object.
    """
    if expected is None:
        expected = text_config(model.config).num_hidden_layers
    candidates = [
        (name, module._modules["layers"])
        for name, module in model.named_modules()
        if isinstance(module._modules.get("layers"), nn.ModuleList)
        and len(module._modules["layers"]) == expected
    ]
    if not candidates:
        raise ValueError(
            f"No decoder stack of {expected} layers on {type(model).__name__}"
        )
    candidates.sort(key=lambda item: item[0].count("."))
    return candidates[0][1]


def _child_names(layer: nn.Module) -> set[str]:
    return set(layer._modules)


def classify_mixer(layer: nn.Module, declared: str | None) -> tuple[str, str | None]:
    """(axis value, module attribute) for the token mixer of one layer.

    ``declared`` is the layer's own ``block_type``/``layer_types`` entry when
    it has one. It decides full vs sliding attention, which are the same
    module class differing only by a window, and it is cross-checked against
    the modules that were actually built.
    """
    children = _child_names(layer)
    if "linear_attn" in children:
        return MIXER_GDN, "linear_attn"
    if "self_attn" in children:
        attention = layer._modules["self_attn"]
        window = getattr(attention, "sliding_window", None)
        if declared in (MIXER_FULL, MIXER_SLIDING):
            return declared, "self_attn"
        return (MIXER_SLIDING if window else MIXER_FULL), "self_attn"
    for name in ("mixer", "attn", "attention", "temporal_mixer"):
        if name in children:
            return MIXER_OTHER, name
    return MIXER_OTHER, None


def classify_ffn(layer: nn.Module) -> tuple[str, str | None, dict]:
    """(axis value, module attribute, MoE shape facts) for one layer's FFN."""
    children = _child_names(layer)
    name = next((n for n in ("mlp", "feed_forward", "ffn") if n in children), None)
    if name is None:
        return FFN_OTHER, None, {}
    module = layer._modules[name]
    sub = _child_names(module)
    if "experts" in sub or "gate" in sub and "experts" in sub:
        experts = module._modules.get("experts")
        shared = "shared_expert" in sub or "shared_experts" in sub
        facts = {
            "num_experts": getattr(experts, "num_experts", None),
            "expert_intermediate_dim": getattr(experts, "intermediate_dim", None),
            "top_k": getattr(module._modules.get("gate"), "top_k", None),
            "has_shared_expert": shared,
            "expert_weights_fused": isinstance(
                getattr(experts, "gate_up_proj", None), nn.Parameter
            ),
        }
        return (FFN_MOE_SHARED if shared else FFN_MOE), name, facts
    return FFN_DENSE, name, {}


def module_device(module: nn.Module) -> str | None:
    for parameter in module.parameters(recurse=True):
        return str(parameter.device)
    for buffer in module.buffers(recurse=True):
        return str(buffer.device)
    return None


@dataclass
class LayerSpec:
    """One decoder layer, on both axes, with where it lives and what it holds."""

    index: int
    module_path: str
    mixer: str
    mixer_module: str | None
    ffn: str
    ffn_module: str | None
    device: str | None
    declared_type: str | None
    param_bytes: int = 0
    moe: dict = field(default_factory=dict)
    shapes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _mixer_shapes(layer: nn.Module, mixer: str, config) -> dict:
    """The shape facts that set this mixer's per-token state and GEMM sizes."""
    if mixer == MIXER_GDN:
        module = layer._modules.get("linear_attn")
        if module is None:
            return {}
        return {
            "conv_dim": getattr(module, "conv_dim", None),
            "conv_kernel_size": getattr(module, "conv_kernel_size", None),
            "num_k_heads": getattr(module, "num_k_heads", None),
            "num_v_heads": getattr(module, "num_v_heads", None),
            "head_k_dim": getattr(module, "head_k_dim", None),
            "head_v_dim": getattr(module, "head_v_dim", None),
        }
    module = layer._modules.get("self_attn")
    if module is None:
        return {}
    head_dim = getattr(module, "head_dim", None) or getattr(config, "head_dim", None)
    return {
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": head_dim,
        "sliding_window": getattr(module, "sliding_window", None),
    }


def describe(model: nn.Module, *, role: str = "target") -> dict:
    """The full per-layer taxonomy of one model.

    ``role`` is ``target`` or ``draft``; it is carried into every diagnostic
    record so a layer index is never ambiguous between the two models.
    """
    config = text_config(model.config)
    layers = decoder_layers(model)
    declared = list(getattr(config, "layer_types", []) or [])
    stack_path = next(
        name
        for name, module in model.named_modules()
        if module is layers
    )

    specs: list[LayerSpec] = []
    for index, layer in enumerate(layers):
        declared_type = declared[index] if index < len(declared) else None
        # The config's name for a GDN layer is "linear_attention"; the axis
        # value is "gdn". Normalise before comparing.
        normalised = (
            MIXER_GDN if declared_type == "linear_attention" else declared_type
        )
        mixer, mixer_module = classify_mixer(layer, normalised)
        ffn, ffn_module, moe = classify_ffn(layer)
        specs.append(
            LayerSpec(
                index=index,
                module_path=f"{stack_path}.{index}",
                mixer=mixer,
                mixer_module=mixer_module,
                ffn=ffn,
                ffn_module=ffn_module,
                device=module_device(layer),
                declared_type=declared_type,
                param_bytes=sum(
                    p.numel() * p.element_size() for p in layer.parameters()
                ),
                moe=moe,
                shapes=_mixer_shapes(layer, mixer, config),
            )
        )

    mismatches = [
        s.index
        for s in specs
        if s.declared_type is not None
        and (MIXER_GDN if s.declared_type == "linear_attention" else s.declared_type)
        != s.mixer
    ]

    return {
        "role": role,
        "model_class": type(model).__name__,
        "stack_path": stack_path,
        "num_layers": len(specs),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "layers": [s.as_dict() for s in specs],
        "mixer_counts": _counts(s.mixer for s in specs),
        "ffn_counts": _counts(s.ffn for s in specs),
        "devices": sorted({s.device for s in specs if s.device}),
        "layers_per_device": _counts(s.device for s in specs),
        "declared_vs_built_mismatch": mismatches,
        "param_bytes_total": sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        "param_bytes_per_device": _param_bytes_per_device(model),
    }


def _counts(values) -> dict:
    counts: dict = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _param_bytes_per_device(model: nn.Module) -> dict:
    """Weight bytes each card holds, counting a shared storage once.

    Tied embeddings and an untied-but-shared lm_head alias the same storage;
    adding both would overstate the weight term that every memory budget in
    the sweep is measured against.
    """
    seen: set[tuple] = set()
    totals: dict = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        device = str(tensor.device)
        totals[device] = totals.get(device, 0) + storage.nbytes()
    return totals


def per_token_state_bytes(spec: dict, dtype_bytes: int = 2) -> dict:
    """Analytic per-token decode state, split by where it comes from.

    This is the closed form the measured component split is checked against.
    An attention layer costs 2 * kv_heads * head_dim * dtype per token; a GDN
    layer costs nothing per token -- its recurrent and conv states are fixed
    size. That difference is the whole of research question 1, so it is worth
    having both the prediction and the measurement.
    """
    attention_per_token = 0
    gdn_recurrent = 0
    gdn_conv = 0
    for layer in spec["layers"]:
        shapes = layer["shapes"]
        if layer["mixer"] in (MIXER_FULL, MIXER_SLIDING):
            kv = shapes.get("num_key_value_heads")
            head = shapes.get("head_dim")
            if kv and head:
                attention_per_token += 2 * kv * head * dtype_bytes
        elif layer["mixer"] == MIXER_GDN:
            v_heads = shapes.get("num_v_heads")
            k_dim = shapes.get("head_k_dim")
            v_dim = shapes.get("head_v_dim")
            conv_dim = shapes.get("conv_dim")
            kernel = shapes.get("conv_kernel_size")
            if v_heads and k_dim and v_dim:
                # The recurrent state is carried in fp32 by the torch kernels.
                gdn_recurrent += v_heads * k_dim * v_dim * 4
            if conv_dim and kernel:
                gdn_conv += conv_dim * kernel * dtype_bytes
    return {
        "attention_kv_bytes_per_token": attention_per_token,
        "gdn_recurrent_state_bytes": gdn_recurrent,
        "gdn_conv_state_bytes": gdn_conv,
        "gdn_state_bytes_total": gdn_recurrent + gdn_conv,
    }
