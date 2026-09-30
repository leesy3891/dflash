"""Which kernel implementation actually runs, recorded rather than assumed.

Qwen3.5's gated-delta-net and causal-conv entry points in Transformers are
wrapped by ``use_kernel_func_from_hub_with_fallback``, which resolves one of
three implementations at import time: a Hub kernel, the original package (FLA
for the delta rule, ``causal_conv1d`` for the convolution), or a pure-torch
reference written inline in the modeling file. The choice is invisible at the
call site and changes the measured cost by a large factor.

So it is read out of the wrapper's closure rather than inferred from whether a
package happens to be installed -- installed and actually selected are
different things, and a record that says "fla 0.3.2 is present" does not say
which function ran.

Numbers taken under different backends are not aggregated. A sweep that
changed backend partway through is a sweep with two populations in it.
"""

from __future__ import annotations

import importlib

TORCH_FALLBACK = "torch_reference"
FLA = "fla"
CAUSAL_CONV1D = "causal_conv1d"
HUB_KERNEL = "hub_kernel"
UNKNOWN = "unknown"

# (module path, attribute) pairs whose implementation decides GDN cost.
GDN_ENTRY_POINTS = (
    ("transformers.models.qwen3_5.modeling_qwen3_5", "torch_chunk_gated_delta_rule"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "torch_recurrent_gated_delta_rule"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "causal_conv1d_fn"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "causal_conv1d_update"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "torch_chunk_gated_delta_rule"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "torch_recurrent_gated_delta_rule"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "causal_conv1d_fn"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "causal_conv1d_update"),
)


def _unwrap_implementation(function):
    """The callable a fallback wrapper will actually dispatch to.

    The wrapper closes over ``implementation``; walking the closure is what
    distinguishes "the torch reference was selected" from "FLA was selected",
    which the function's own name cannot, since both are reached through an
    entry point named ``torch_*``.
    """
    seen = set()
    candidates = []
    stack = [function]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        closure = getattr(current, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) and getattr(value, "__module__", None):
                candidates.append(value)
                stack.append(value)
    return candidates


def classify(function) -> dict:
    """Name the implementation behind one wrapped entry point."""
    candidates = _unwrap_implementation(function)
    modules = [getattr(c, "__module__", "") or "" for c in candidates]
    resolved = UNKNOWN
    if any(m.startswith("fla.") or m == "fla" for m in modules):
        resolved = FLA
    elif any(m.startswith("causal_conv1d") for m in modules):
        resolved = CAUSAL_CONV1D
    elif any("kernels" in m for m in modules):
        resolved = HUB_KERNEL
    elif any(m.startswith("transformers.models.") for m in modules):
        resolved = TORCH_FALLBACK
    return {
        "resolved": resolved,
        "candidate_modules": sorted({m for m in modules if m}),
        "qualname": getattr(function, "__qualname__", None),
    }


def probe() -> dict:
    """Every GDN/conv entry point, with what it resolved to and why.

    Entry points whose module is not imported are reported as not-loaded
    rather than missing: the Qwen3-8B runs never import the hybrid modeling
    files at all, and that is correct, not a gap in the record.
    """
    entries: dict = {}
    for module_path, name in GDN_ENTRY_POINTS:
        key = f"{module_path.rsplit('.', 1)[-1]}.{name}"
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            entries[key] = {
                "resolved": UNKNOWN,
                "na_reason": f"module not importable: {type(exc).__name__}",
            }
            continue
        function = getattr(module, name, None)
        if function is None:
            entries[key] = {"resolved": UNKNOWN, "na_reason": "attribute absent"}
            continue
        entries[key] = classify(function)

    packages = {}
    for name in (FLA, CAUSAL_CONV1D, "flash_attn"):
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            packages[name] = None

    resolved = {entry["resolved"] for entry in entries.values()}
    resolved.discard(UNKNOWN)
    # FLA ships the delta-rule kernels and causal_conv1d the conv kernels, so
    # the fully optimized GDN layer legitimately resolves to two packages.
    # What must not happen is a mix of optimized and torch-reference entry
    # points: that is a configuration nobody deploys, and its acceptance and
    # latency belong to neither arm.
    uses_reference = TORCH_FALLBACK in resolved
    uses_optimized = bool(resolved - {TORCH_FALLBACK})
    homogeneous = not (uses_reference and uses_optimized)
    if not resolved:
        label = "none"
    elif not uses_optimized:
        label = "torch_reference"
    elif homogeneous:
        label = "+".join(sorted(resolved))
    else:
        label = "mixed"
    return {
        "entry_points": entries,
        "packages_installed": packages,
        "distinct_backends": sorted(resolved),
        "homogeneous": homogeneous,
        "config_label": label,
        "summary": (
            f"every GDN/conv entry point is {label}"
            if homogeneous
            else "MIXED optimized and torch-reference kernels -- "
            "do not aggregate these numbers with either arm"
        ),
    }


def attention_backend(model) -> dict:
    """Which attention implementation the loaded target is configured for."""
    config = model.config
    text = getattr(config, "text_config", None) or config
    return {
        "attn_implementation": getattr(
            config, "_attn_implementation", None
        ) or getattr(text, "_attn_implementation", None),
        "requested": getattr(config, "attn_implementation", None),
    }


def describe(model=None) -> dict:
    data = probe()
    if model is not None:
        data["attention"] = attention_backend(model)
    return data
