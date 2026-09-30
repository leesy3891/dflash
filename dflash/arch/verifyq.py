"""How target verify scales with the number of tokens verified at once.

A verify of ``q`` tokens is the same forward an AR step makes, widened. The
question is how much of its cost is fixed -- weight reads, cache reads, kernel
launches, shard crossings -- and how much grows with ``q``, because that ratio
is what decides whether a wider block is free or expensive.

Every measurement starts from the same state. That matters more here than in
most microbenchmarks: a verify mutates the KV cache and the GDN recurrent
state, so a naive loop measures ``q`` tokens against a cache that is
``iteration * q`` tokens longer each time, and reports cache growth as
``q``-scaling. So the cache is snapshotted once and restored before every
repetition.

Restoring is not free -- it copies the whole KV cache -- so its cost is
measured and reported on its own rather than folded into the verify time.

What this is not
----------------
These numbers are a per-call cost at a fixed ``q``. They are not an end-to-end
speedup. Real speedup depends on acceptance, which decides how often a block
of width ``q`` yields ``q`` tokens rather than one, and that is measured by
running the actual decode loop, not here. The two are reported separately and
a ratio of ``q``-costs must not be presented as a speedup.
"""

from __future__ import annotations

import statistics
import time

import torch

from ..model import _crop_to, _make_cache
from . import statemem, taxonomy


def snapshot_cache(cache) -> list[dict]:
    """A full copy of every state tensor, enough to restore exactly."""
    saved = []
    for layer in getattr(cache, "layers", []):
        entry: dict = {}
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.numel():
            entry["keys"] = keys.clone()
            entry["values"] = layer.values.clone()
        for name in ("recurrent_states", "conv_states"):
            slot = getattr(layer, name, None)
            if isinstance(slot, dict):
                entry[name] = {
                    key: (value.clone() if isinstance(value, torch.Tensor) else value)
                    for key, value in slot.items()
                }
        entry["has_previous_state"] = dict(
            getattr(layer, "has_previous_state", {}) or {}
        )
        saved.append(entry)
    return saved


def restore_cache(cache, saved: list[dict]) -> None:
    """Put the cache back exactly as it was at the snapshot.

    The recurrent state is written with ``copy_`` rather than rebound, because
    the layer may hold a static address for it; the conv state is rebound,
    because ``crop`` changes its shape and a copy into the old shape would
    fail.
    """
    for layer, entry in zip(getattr(cache, "layers", []), saved):
        if "keys" in entry:
            layer.keys = entry["keys"].clone()
            layer.values = entry["values"].clone()
        if "recurrent_states" in entry:
            for key, value in entry["recurrent_states"].items():
                if isinstance(value, torch.Tensor):
                    target = layer.recurrent_states.get(key)
                    if isinstance(target, torch.Tensor) and target.shape == value.shape:
                        target.copy_(value)
                    else:
                        layer.recurrent_states[key] = value.clone()
        if "conv_states" in entry:
            for key, value in entry["conv_states"].items():
                layer.conv_states[key] = (
                    value.clone() if isinstance(value, torch.Tensor) else value
                )
        if entry.get("has_previous_state"):
            layer.has_previous_state.update(entry["has_previous_state"])


def _sync_time(device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def measure(
    target,
    input_ids: torch.Tensor,
    *,
    widths: tuple[int, ...] = (1, 4, 8, 16),
    repeats: int = 5,
    warmup: int = 2,
    cache_policy_recording: bool = True,
    block_ids: torch.Tensor | None = None,
) -> dict:
    """Verify cost at each width, from an identical starting state.

    The verified block is the target's own greedy continuation of the prompt
    unless ``block_ids`` is given: the anchor token prefill produces, then
    what the target would emit after it. That is what an accepted draft
    block contains, and on a MoE target it is what decides which experts a
    verify touches -- random token ids route to experts no real block would.
    ``cache_policy_recording=False`` is the AR decode path, so its ``q=1`` row
    is an AR step; with recording on it is not (see :mod:`dflash.arch.ar`).
    """
    device = next(target.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    spec = taxonomy.describe(target)
    mixers = [entry["mixer"] for entry in spec["layers"]]
    position_ids = torch.arange(
        prompt_len + max(widths) + 1, device=device
    ).unsqueeze(0)

    cache = _make_cache(target.config)
    if not cache_policy_recording:
        for layer in cache.layers:
            layer.record_past = False

    prefill = target(
        input_ids=input_ids,
        position_ids=position_ids[:, :prompt_len],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    anchor = prefill.logits[:, -1, :].argmax(dim=-1).to(device)
    prefill = None
    _crop_to(cache, prompt_len)

    snapshot_start = _sync_time(device)
    saved = snapshot_cache(cache)
    snapshot_s = _sync_time(device) - snapshot_start

    components = statemem.cache_components(cache, mixers=mixers)
    snapshot_bytes = sum(
        tensor.numel() * tensor.element_size()
        for entry in saved
        for value in entry.values()
        if isinstance(value, (dict, torch.Tensor))
        for tensor in (
            value.values() if isinstance(value, dict) else [value]
        )
        if isinstance(tensor, torch.Tensor)
    )

    if block_ids is None:
        tokens = [anchor]
        for offset in range(max(widths) - 1):
            step = target(
                input_ids=tokens[-1].view(1, 1),
                position_ids=position_ids[:, prompt_len + offset:prompt_len + offset + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            tokens.append(step.logits[:, -1, :].argmax(dim=-1).to(device))
        block = torch.stack(tokens, dim=1).view(1, -1)
        block_source = "target_greedy_continuation"
        restore_cache(cache, saved)
    else:
        block = block_ids.to(device)
        block_source = "given"

    results = []
    restore_times: list[float] = []
    for width in widths:
        samples: list[float] = []
        for iteration in range(warmup + repeats):
            restore_start = _sync_time(device)
            restore_cache(cache, saved)
            restore_s = _sync_time(device) - restore_start

            start = _sync_time(device)
            target(
                input_ids=block[:, :width],
                position_ids=position_ids[:, prompt_len:prompt_len + width],
                past_key_values=cache,
                use_cache=True,
            )
            elapsed = _sync_time(device) - start
            if iteration >= warmup:
                samples.append(elapsed)
                restore_times.append(restore_s)
        median = statistics.median(samples)
        results.append({
            "q": width,
            "median_s": median,
            "min_s": min(samples),
            "max_s": max(samples),
            "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "samples": samples,
            "s_per_token": median / width,
        })

    base = results[0]["median_s"] if results else None
    for row in results:
        row["cost_over_q1"] = (row["median_s"] / base) if base else None
        # Cost attributable to widening, above what a single token already
        # paid. At q=1 this is zero by construction.
        row["marginal_s_over_q1"] = row["median_s"] - base if base else None
        row["marginal_s_per_extra_token"] = (
            (row["median_s"] - base) / (row["q"] - 1) if base and row["q"] > 1 else None
        )

    return {
        "prompt_tokens": prompt_len,
        "widths": list(widths),
        "repeats": repeats,
        "warmup": warmup,
        "cache_policy": "recording" if cache_policy_recording else "native",
        "block_source": block_source,
        "block_ids": block[0].tolist(),
        "block_ids_tensor": block.detach().cpu(),
        "results": results,
        "snapshot_s": snapshot_s,
        "snapshot_bytes": snapshot_bytes,
        "mean_restore_s": (
            statistics.mean(restore_times) if restore_times else None
        ),
        "cache_components_bytes": statemem.summarise(components),
        "interpretation": {
            "fixed_cost_s": base,
            "note": (
                "Each row is the cost of one verify call at that width from an "
                "identical cache. marginal_s_per_extra_token is the slope "
                "above q=1. Do not read cost_over_q1 as a speedup: the "
                "speedup a width buys depends on acceptance, which is "
                "measured by the decode loop, not by this benchmark."
            ),
            "wasted_work_model": (
                "For a block of width q whose first a tokens are accepted, "
                "the necessary work is approximately the q=a+1 row and the "
                "rest is work the rejection discarded. That is an estimate "
                "from this curve, not a measurement of the rejected suffix: "
                "the tokens are verified in one kernel and their cost is not "
                "separable per position."
            ),
        },
    }
