"""Component-resolved memory: what the decode state is made of, per device.

``_cache_bytes`` in :mod:`dflash.model` walks a cache and sums every CUDA
storage it can reach. That is the right total, but it is not an attention KV
figure: on Qwen3.5 the same number also contains the gated-delta-net recurrent
states, the conv states, and -- while ``record_past`` is on -- a conv recording
buffer that is O(S) rather than O(1). Reporting it as "target KV" is what
makes a hybrid target look like it has a large KV cache when the whole point
of the architecture is that it does not.

This module splits the same walk into named components and keeps each on the
device it lives on.

Components
----------
``attention_kv``
    Keys and values of full- or sliding-attention layers. O(S).
``gdn_recurrent``
    Gated-delta-net recurrent states. Fixed size, independent of S.
``gdn_conv_working``
    The last ``conv_kernel_size`` columns of each conv state -- what the next
    forward actually reads. Fixed size.
``gdn_conv_recording``
    Everything a conv state holds beyond that working set. This exists only
    because ``activate_past_recording`` keeps the full history so ``crop`` can
    roll back, and during prefill it is O(S). It is DFlash's rollback
    machinery, not the target's decode state, and is charged accordingly.

Two sizes are reported for every component:

``logical_bytes``
    ``numel * element_size`` -- what the tensor claims to be.
``storage_bytes``
    The allocation actually kept alive. These differ whenever a tensor is a
    view into a larger buffer, which is exactly what ``crop`` produces: it
    slices rather than copies, so a cropped cache still pins the uncropped
    allocation until the next ``cat`` replaces it. A run's real high-water
    mark follows ``storage_bytes``.
"""

from __future__ import annotations

import torch

from .taxonomy import MIXER_FULL, MIXER_GDN, MIXER_SLIDING

ATTENTION_KV = "attention_kv"
GDN_RECURRENT = "gdn_recurrent"
GDN_CONV_WORKING = "gdn_conv_working"
GDN_CONV_RECORDING = "gdn_conv_recording"
UNCLASSIFIED = "unclassified"

COMPONENTS = (
    ATTENTION_KV,
    GDN_RECURRENT,
    GDN_CONV_WORKING,
    GDN_CONV_RECORDING,
    UNCLASSIFIED,
)


def _empty_totals() -> dict:
    return {
        name: {"logical_bytes": 0, "storage_bytes": 0, "tensors": 0}
        for name in COMPONENTS
    }


def _add(bucket: dict, component: str, logical: int, storage: int) -> None:
    entry = bucket[component]
    entry["logical_bytes"] += logical
    entry["storage_bytes"] += storage
    entry["tensors"] += 1


class _StorageLedger:
    """Charges each distinct allocation to exactly one component.

    A cache can reach the same storage twice -- a cropped view alongside the
    buffer it views into, or an attention layer and a linear layer sharing a
    lazily-initialised empty tensor. Counting it twice would inflate the
    total; counting it nowhere would lose it. It is charged to whichever
    component reaches it first, and the duplicate is recorded as an alias so
    the double-count is visible rather than silent.
    """

    def __init__(self) -> None:
        self.seen: dict[tuple, str] = {}
        self.aliases: list[dict] = []

    def charge(self, tensor: torch.Tensor, component: str) -> int:
        """Storage bytes to charge here: full size once, zero on re-sight."""
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in self.seen:
            if self.seen[key] != component:
                self.aliases.append(
                    {
                        "device": str(tensor.device),
                        "charged_to": self.seen[key],
                        "also_reached_by": component,
                        "bytes": storage.nbytes(),
                    }
                )
            return 0
        self.seen[key] = component
        return storage.nbytes()


def _layer_kinds(cache) -> list[str]:
    kinds = []
    for layer in getattr(cache, "layers", []):
        has_kv = getattr(layer, "keys", None) is not None
        has_linear = bool(getattr(layer, "conv_states", None)) or bool(
            getattr(layer, "recurrent_states", None)
        )
        if has_linear and has_kv:
            kinds.append("hybrid")
        elif has_linear:
            kinds.append("linear")
        else:
            kinds.append("attention")
    return kinds


def cache_components(cache, *, mixers: list[str] | None = None) -> dict:
    """Split one cache into named components, per device.

    ``mixers`` is the per-layer mixer axis from :func:`taxonomy.describe`. It
    is only used to label a layer whose cache object carries both an attention
    and a linear slot -- ``LinearAttentionAndFullAttentionLayer`` is
    instantiated for every layer of a hybrid model, so the cache alone cannot
    say which half a given layer actually uses. When it is omitted the split
    falls back to which slots hold data, which is correct after the first
    forward and merely uninformative before it.
    """
    ledger = _StorageLedger()
    per_device: dict[str, dict] = {}
    per_layer: list[dict] = []

    def bucket(device: str) -> dict:
        return per_device.setdefault(device, _empty_totals())

    def account(tensor, component: str, device_hint: str | None = None) -> tuple[int, int]:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return 0, 0
        if not tensor.is_cuda:
            return 0, 0
        device = str(tensor.device)
        logical = tensor.numel() * tensor.element_size()
        storage = ledger.charge(tensor, component)
        _add(bucket(device), component, logical, storage)
        return logical, storage

    for index, layer in enumerate(getattr(cache, "layers", [])):
        declared = mixers[index] if mixers and index < len(mixers) else None
        entry: dict = {"layer": index, "mixer": declared, "components": {}}

        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor) and keys.numel():
            logical = 0
            storage = 0
            for tensor in (keys, values):
                a, b = account(tensor, ATTENTION_KV)
                logical += a
                storage += b
            entry["components"][ATTENTION_KV] = {
                "logical_bytes": logical,
                "storage_bytes": storage,
                "seq_len": int(keys.shape[-2]),
                "device": str(keys.device),
            }

        recurrent = getattr(layer, "recurrent_states", None) or {}
        for slot, tensor in (recurrent.items() if isinstance(recurrent, dict) else []):
            logical, storage = account(tensor, GDN_RECURRENT)
            if logical:
                entry["components"].setdefault(GDN_RECURRENT, {
                    "logical_bytes": 0, "storage_bytes": 0, "device": str(tensor.device),
                    "shape": tuple(tensor.shape), "dtype": str(tensor.dtype),
                })
                entry["components"][GDN_RECURRENT]["logical_bytes"] += logical
                entry["components"][GDN_RECURRENT]["storage_bytes"] += storage

        conv = getattr(layer, "conv_states", None) or {}
        kernels = getattr(layer, "conv_kernel_size", None) or {}
        for slot, tensor in (conv.items() if isinstance(conv, dict) else []):
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            kernel = kernels.get(slot) if isinstance(kernels, dict) else None
            columns = tensor.shape[-1]
            element = tensor.element_size()
            per_column = (tensor.numel() // max(columns, 1)) * element
            working_columns = min(kernel or columns, columns)
            working_logical = per_column * working_columns
            recording_logical = per_column * max(columns - working_columns, 0)

            # One allocation backs both halves, so the storage is charged to
            # the working set and the recording buffer carries the excess the
            # view keeps alive above its own logical size.
            device = str(tensor.device)
            storage_total = ledger.charge(tensor, GDN_CONV_WORKING)
            _add(bucket(device), GDN_CONV_WORKING, working_logical, min(storage_total, working_logical))
            _add(
                bucket(device),
                GDN_CONV_RECORDING,
                recording_logical,
                max(storage_total - working_logical, 0),
            )
            entry["components"][GDN_CONV_WORKING] = {
                "logical_bytes": working_logical,
                "columns": working_columns,
                "device": device,
            }
            if recording_logical or storage_total > working_logical:
                entry["components"][GDN_CONV_RECORDING] = {
                    "logical_bytes": recording_logical,
                    "storage_bytes": max(storage_total - working_logical, 0),
                    "columns": columns - working_columns,
                    "device": device,
                    "recording_active": bool(getattr(layer, "record_past", False)),
                }
        per_layer.append(entry)

    totals = _empty_totals()
    for device_totals in per_device.values():
        for name, entry in device_totals.items():
            totals[name]["logical_bytes"] += entry["logical_bytes"]
            totals[name]["storage_bytes"] += entry["storage_bytes"]
            totals[name]["tensors"] += entry["tensors"]

    return {
        "total": totals,
        "per_device": per_device,
        "per_layer": per_layer,
        "aliases": ledger.aliases,
        "layer_kinds": _layer_kinds(cache),
        "sum_storage_bytes": sum(
            entry["storage_bytes"] for entry in totals.values()
        ),
        "sum_logical_bytes": sum(
            entry["logical_bytes"] for entry in totals.values()
        ),
    }


def tensor_components(named: dict) -> dict:
    """Per-device sizes of loose tensors: selected hidden, context feature, ...

    Same two-size convention as the cache split, and the same rule that a
    storage shared between two names is charged once. The drafter's context
    feature is a ``cat`` of the target's selected hidden states, so the two are
    distinct allocations; a future implementation that made the feature a view
    would show up here as an alias rather than as a silent halving.
    """
    ledger = _StorageLedger()
    per_device: dict[str, dict] = {}
    detail: dict = {}
    for name, value in named.items():
        tensors = []
        if isinstance(value, torch.Tensor):
            tensors = [value]
        elif isinstance(value, (list, tuple)):
            tensors = [t for t in value if isinstance(t, torch.Tensor)]
        logical = storage = 0
        devices: dict[str, int] = {}
        for tensor in tensors:
            if not tensor.is_cuda or tensor.numel() == 0:
                continue
            device = str(tensor.device)
            size = tensor.numel() * tensor.element_size()
            charged = ledger.charge(tensor, name)
            logical += size
            storage += charged
            devices[device] = devices.get(device, 0) + size
            slot = per_device.setdefault(device, {})
            slot[name] = slot.get(name, 0) + charged
        detail[name] = {
            "logical_bytes": logical,
            "storage_bytes": storage,
            "per_device_logical_bytes": devices,
            "num_tensors": len(tensors),
        }
    return {"detail": detail, "per_device": per_device, "aliases": ledger.aliases}


def device_allocator_state() -> dict:
    """Live and peak allocator bytes for every visible device, right now."""
    state = {}
    for index in range(torch.cuda.device_count()):
        try:
            stats = torch._C._cuda_memoryStats(index)["allocated_bytes"]["all"]
            reserved = torch._C._cuda_memoryStats(index)["reserved_bytes"]["all"]
            state[f"cuda:{index}"] = {
                "allocated_current": stats["current"],
                "allocated_peak": stats["peak"],
                "reserved_current": reserved["current"],
                "reserved_peak": reserved["peak"],
            }
        except (AttributeError, KeyError, RuntimeError):
            state[f"cuda:{index}"] = None
    return state


def summarise(components: dict) -> dict:
    """The one-line version: component totals in bytes, storage-based."""
    return {
        name: entry["storage_bytes"] for name, entry in components["total"].items()
    }
