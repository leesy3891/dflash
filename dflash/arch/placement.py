"""Explicit layer sharding: the same target placement for AR and for DFlash.

``device_map="auto"`` packs layers by free memory at load time, so the same
model lands differently depending on what else was running. Every comparison
in this sweep is between two runs of the same target -- AR against DFlash, one
S against another -- and a placement that drifts between them turns a
placement difference into an apparent architecture difference.

So the map is built here, from the layer count and a fixed device list, and
written into the record. Contiguous blocks of layers, one block per device, so
a forward walks the devices in order and crosses each boundary once. Within a
layer nothing is split: every GEMM, GDN recurrence and MoE expert keeps the
shape it would have on a single card. That is the point -- the research
question is about what a layer costs, and tensor parallelism would change the
shapes being asked about.

One consequence, stated here because it is easy to misread later: while device
*i* runs its block, the other devices have no kernels. That idle is the
structure of pipeline-style layer sharding at batch 1, not a stall inside a
kernel and not DFlash overhead. :mod:`dflash.arch.events` classifies it as
``structural_shard_wait`` and it is reported separately from everything else.
"""

from __future__ import annotations

from .env import usable_devices


def split_layers(num_layers: int, devices: list[int]) -> list[int]:
    """Which device each layer index goes to: contiguous, near-equal blocks.

    The remainder goes to the earliest devices, so device 0 -- which also
    carries the embedding table -- is never the one handed the extra layer
    when it is also the tightest.
    """
    if not devices:
        raise ValueError("no usable devices")
    count = len(devices)
    base, extra = divmod(num_layers, count)
    sizes = [base + (1 if i >= count - extra else 0) for i in range(count)]
    assignment: list[int] = []
    for device, size in zip(devices, sizes):
        assignment.extend([device] * size)
    return assignment


def build_device_map(
    model_class_prefix: str,
    num_layers: int,
    stack_path: str,
    devices: list[int] | None = None,
    *,
    extras: dict | None = None,
) -> dict:
    """An accelerate ``device_map`` pinning each layer to a named device.

    ``stack_path`` is the dotted path to the decoder ``ModuleList`` as
    :func:`taxonomy.describe` found it, so this never guesses whether the
    stack is at ``model.layers`` or ``model.language_model.layers``.
    """
    devices = devices or usable_devices()
    assignment = split_layers(num_layers, devices)
    device_map = {
        f"{stack_path}.{index}": f"cuda:{device}"
        for index, device in enumerate(assignment)
    }
    # Everything outside the stack rides with the layer it is adjacent to:
    # embeddings and the rotary cache with the first layer, the final norm and
    # the output head with the last. Putting the head anywhere else would add
    # a vocab-sized transfer to every verify.
    first = f"cuda:{assignment[0]}"
    last = f"cuda:{assignment[-1]}"
    root = stack_path.rsplit(".layers", 1)[0] if ".layers" in stack_path else ""
    prefix = f"{root}." if root else ""
    device_map.update({
        f"{prefix}embed_tokens": first,
        f"{prefix}rotary_emb": first,
        f"{prefix}norm": last,
        "lm_head": last,
    })
    if root.endswith(".language_model"):
        # A conditional-generation wrapper also carries a vision tower. Text
        # prompts never run it, but accelerate needs every parameter placed;
        # it sits with the embeddings so it adds resident weight to the first
        # shard only, and that weight is counted in the per-device report.
        device_map[root.rsplit(".language_model", 1)[0] + ".visual"] = first
    if extras:
        device_map.update(extras)
    return device_map


def describe(device_map: dict, stack_path: str) -> dict:
    """A record-friendly summary: layers per device and the boundary count."""
    layers: dict[int, str] = {}
    for key, value in device_map.items():
        if key.startswith(f"{stack_path}."):
            suffix = key[len(stack_path) + 1:]
            if suffix.isdigit():
                layers[int(suffix)] = value
    ordered = [layers[i] for i in sorted(layers)]
    boundaries = [
        i for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1]
    ]
    per_device: dict = {}
    for device in ordered:
        per_device[device] = per_device.get(device, 0) + 1
    return {
        "num_layers": len(ordered),
        "layers_per_device": per_device,
        "device_order": list(dict.fromkeys(ordered)),
        "boundary_layer_indices": boundaries,
        "num_crossings_per_forward": len(boundaries),
        "non_layer_placements": {
            key: value
            for key, value in device_map.items()
            if not key.startswith(f"{stack_path}.")
        },
    }


def draft_device(devices: list[int] | None = None) -> int:
    """Where the one drafter replica lives.

    The drafter is never sharded: it is six layers, and splitting them would
    add three device crossings to a call that is meant to be cheap. It sits on
    the first device, which is also where ``dflash_generate`` keeps its
    bookkeeping, so the only tensors that cross a boundary are the target's
    selected hidden states.
    """
    devices = devices or usable_devices()
    return devices[0]
