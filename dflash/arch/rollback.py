"""Does rejecting a draft token actually restore the target's state?

Speculative decoding is only lossless if rolling the cache back to the
accepted prefix leaves the target in exactly the state it would have been in
had it only ever seen that prefix. For a full-attention target that is a
slice: drop the rejected keys and values and the next forward recomputes from
what remains.

A gated-delta-net layer has no per-token state to slice. It carries a
recurrent state that has already absorbed every token of the verify block,
and ``LinearAttentionCacheLayerMixin.crop`` -- the function
``DFlash``'s ``_crop_to`` calls -- only trims ``conv_states``. It never
touches ``recurrent_states``. ``update_recurrent_state`` writes in place with
``copy_``, so the pre-verify value is gone by the time the rejection is known.

This module measures the consequence rather than asserting it. It runs the
same continuation two ways -- once through a rolled-back cache, once through a
cache that only ever saw the accepted prefix -- and compares the next-token
logits and the cache tensors themselves, per layer and per mixer type. A
full-attention target is the control: it should agree bit for bit.

The comparison is structured so a reader can tell *where* a divergence comes
from, because "acceptance is a bit low" and "the state is wrong" look the same
from the outside.
"""

from __future__ import annotations

import torch

from ..model import _crop_to, _make_cache
from . import taxonomy


def _next_logits(target, cache, input_ids, position_ids):
    out = target(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return out.logits[:, -1, :].float()


def _snapshot_states(cache) -> list[dict]:
    """A detached copy of every per-layer state tensor, for comparison."""
    snapshot = []
    for layer in getattr(cache, "layers", []):
        entry: dict = {}
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.numel():
            entry["keys"] = keys.detach().clone()
            entry["values"] = layer.values.detach().clone()
        recurrent = getattr(layer, "recurrent_states", None)
        if isinstance(recurrent, dict):
            for slot, tensor in recurrent.items():
                if isinstance(tensor, torch.Tensor) and tensor.numel():
                    entry[f"recurrent[{slot}]"] = tensor.detach().clone()
        conv = getattr(layer, "conv_states", None)
        if isinstance(conv, dict):
            for slot, tensor in conv.items():
                if isinstance(tensor, torch.Tensor) and tensor.numel():
                    entry[f"conv[{slot}]"] = tensor.detach().clone()
        snapshot.append(entry)
    return snapshot


def _compare_states(rolled: list[dict], clean: list[dict], mixers: list[str]) -> list[dict]:
    rows = []
    for index, (a, b) in enumerate(zip(rolled, clean)):
        mixer = mixers[index] if index < len(mixers) else None
        for name in sorted(set(a) | set(b)):
            left, right = a.get(name), b.get(name)
            if left is None or right is None:
                rows.append({
                    "layer": index, "mixer": mixer, "tensor": name,
                    "status": "present_on_one_side_only",
                    "rolled_shape": None if left is None else tuple(left.shape),
                    "clean_shape": None if right is None else tuple(right.shape),
                })
                continue
            if left.shape != right.shape:
                rows.append({
                    "layer": index, "mixer": mixer, "tensor": name,
                    "status": "shape_mismatch",
                    "rolled_shape": tuple(left.shape),
                    "clean_shape": tuple(right.shape),
                })
                continue
            delta = (left.float() - right.float()).abs()
            scale = right.float().abs().max().item()
            rows.append({
                "layer": index, "mixer": mixer, "tensor": name,
                "status": "equal" if bool(torch.equal(left, right)) else "differs",
                "shape": tuple(left.shape),
                "max_abs_diff": delta.max().item(),
                "mean_abs_diff": delta.mean().item(),
                "reference_max_abs": scale,
                "relative_max_diff": (delta.max().item() / scale) if scale else None,
            })
    return rows


@torch.inference_mode()
def check(
    target,
    input_ids: torch.Tensor,
    *,
    block_size: int = 16,
    accepted: int = 4,
    probe_tokens: int = 8,
    compare_states: bool = True,
) -> dict:
    """One rollback experiment.

    The prompt is prefilled, a ``block_size``-wide verify block is run through
    the cache, ``accepted`` of its tokens are kept and the rest rolled back.
    The continuation is then compared against a cache built only from the
    accepted prefix.

    ``probe_tokens`` continuation tokens are scored rather than one, because a
    GDN recurrence that is wrong by a little shows up more clearly a few steps
    after the rollback than immediately at it.
    """
    device = next(target.parameters()).device
    spec = taxonomy.describe(target)
    mixers = [layer["mixer"] for layer in spec["layers"]]
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    if accepted >= block_size:
        raise ValueError("accepted must be smaller than block_size for a rollback")

    total = prompt_len + block_size + probe_tokens
    position_ids = torch.arange(total, device=device).unsqueeze(0)

    # A deterministic pseudo-block. Its content does not matter -- what is
    # under test is whether the state rolls back, not whether the tokens were
    # good guesses -- but it must be the same on both paths.
    generator = torch.Generator(device="cpu").manual_seed(0)
    vocab = taxonomy.text_config(target.config).vocab_size
    block = torch.randint(
        0, vocab, (1, block_size), generator=generator, dtype=torch.long
    ).to(device)
    probe = torch.randint(
        0, vocab, (1, probe_tokens), generator=generator, dtype=torch.long
    ).to(device)
    accepted_ids = block[:, :accepted]

    # ---- path A: verify the whole block, then roll back ---------------------
    cache_a = _make_cache(target.config)
    target(
        input_ids=input_ids,
        position_ids=position_ids[:, :prompt_len],
        past_key_values=cache_a,
        use_cache=True,
        logits_to_keep=1,
    )
    _crop_to(cache_a, prompt_len)
    target(
        input_ids=block,
        position_ids=position_ids[:, prompt_len:prompt_len + block_size],
        past_key_values=cache_a,
        use_cache=True,
    )
    _crop_to(cache_a, prompt_len + accepted)
    rolled_states = _snapshot_states(cache_a) if compare_states else []
    logits_a = _next_logits(
        target, cache_a, probe,
        position_ids[:, prompt_len + accepted:prompt_len + accepted + probe_tokens],
    )

    # ---- reference paths ----------------------------------------------------
    # Two of them, because they answer different questions and a hybrid target
    # separates them.
    #
    # `segmented` prefills the prompt and then feeds the accepted tokens as
    # their own forward -- the same call segmentation path A used, minus the
    # rejected suffix. Any difference against path A is the rollback, and
    # nothing else.
    #
    # `one_shot` prefills prompt and accepted tokens together. It is what a
    # fresh request would do, and it differs from `segmented` on a GDN target
    # even with no rollback involved, because the chunked delta-rule kernel
    # does not reproduce across a call boundary (the same effect
    # `prefill_chunking_safe` refuses prefill chunking for). Comparing only
    # against this one would charge that to the rollback.
    def _reference(one_shot: bool):
        cache = _make_cache(target.config)
        if one_shot:
            target(
                input_ids=torch.cat([input_ids, accepted_ids], dim=1),
                position_ids=position_ids[:, :prompt_len + accepted],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            target(
                input_ids=input_ids,
                position_ids=position_ids[:, :prompt_len],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            _crop_to(cache, prompt_len)
            if accepted:
                target(
                    input_ids=accepted_ids,
                    position_ids=position_ids[:, prompt_len:prompt_len + accepted],
                    past_key_values=cache,
                    use_cache=True,
                )
        _crop_to(cache, prompt_len + accepted)
        states = _snapshot_states(cache) if compare_states else []
        logits = _next_logits(
            target, cache, probe,
            position_ids[
                :, prompt_len + accepted:prompt_len + accepted + probe_tokens
            ],
        )
        return states, logits

    references = {}
    for name, one_shot in (("segmented", False), ("one_shot", True)):
        clean_states, logits_b = _reference(one_shot)
        delta = (logits_a - logits_b).abs()
        argmax_match = bool(
            (logits_a.argmax(dim=-1) == logits_b.argmax(dim=-1)).all().item()
        )
        state_rows = (
            _compare_states(rolled_states, clean_states, mixers)
            if compare_states else []
        )
        by_mixer: dict = {}
        for row in state_rows:
            bucket = by_mixer.setdefault(
                row["mixer"],
                {"tensors": 0, "equal": 0, "differs": 0, "max_abs_diff": 0.0},
            )
            bucket["tensors"] += 1
            if row["status"] == "equal":
                bucket["equal"] += 1
            elif row["status"] == "differs":
                bucket["differs"] += 1
                bucket["max_abs_diff"] = max(
                    bucket["max_abs_diff"], row.get("max_abs_diff") or 0.0
                )
        # Which state tensors differ, named, so "the recurrent state is the
        # one that does not roll back" is a reading of the data rather than
        # an interpretation laid over it.
        differing = sorted({
            row["tensor"].split("[")[0]
            for row in state_rows if row["status"] == "differs"
        })
        equal_kinds = sorted({
            row["tensor"].split("[")[0]
            for row in state_rows if row["status"] == "equal"
        })
        references[name] = {
            "lossless": argmax_match and all(
                row["status"] == "equal" for row in state_rows
            ),
            "next_token_argmax_match": argmax_match,
            "clean_top1": logits_b.argmax(dim=-1).tolist(),
            "logits_max_abs_diff": delta.max().item(),
            "logits_mean_abs_diff": delta.mean().item(),
            "state_comparison_by_mixer": by_mixer,
            "differing_tensor_kinds": differing,
            "matching_tensor_kinds": equal_kinds,
            "state_rows": state_rows if compare_states else [],
        }

    return {
        "prompt_tokens": prompt_len,
        "block_size": block_size,
        "accepted": accepted,
        "rejected": block_size - accepted,
        "probe_tokens": probe_tokens,
        "mixer_counts": spec["mixer_counts"],
        "rolled_top1": logits_a.argmax(dim=-1).tolist(),
        "references": references,
        # The headline verdict is against the segmented reference, which is
        # the one that isolates the rollback.
        "lossless": references["segmented"]["lossless"],
        "next_token_argmax_match":
            references["segmented"]["next_token_argmax_match"],
        "logits_max_abs_diff": references["segmented"]["logits_max_abs_diff"],
        "state_comparison_by_mixer":
            references["segmented"]["state_comparison_by_mixer"],
        "state_rows": references["segmented"]["state_rows"],
    }


def summarise(result: dict) -> str:
    """A human-readable verdict for the console."""
    lines = [
        f"prompt={result['prompt_tokens']} block={result['block_size']} "
        f"accepted={result['accepted']} rejected={result['rejected']}",
        f"  next-token argmax match : {result['next_token_argmax_match']}",
        f"  logits max |diff|       : {result['logits_max_abs_diff']:.6g}",
    ]
    for name, reference in result["references"].items():
        lines.append(f"  -- vs {name} reference --")
        lines.append(
            f"     argmax match={reference['next_token_argmax_match']} "
            f"logits max|diff|={reference['logits_max_abs_diff']:.6g}"
        )
        for mixer, bucket in sorted(reference["state_comparison_by_mixer"].items()):
            lines.append(
                f"     {mixer:<16} tensors={bucket['tensors']:<4} "
                f"equal={bucket['equal']:<4} differs={bucket['differs']:<4} "
                f"max|diff|={bucket['max_abs_diff']:.6g}"
            )
        lines.append(
            f"     differs: {reference['differing_tensor_kinds'] or 'none'} | "
            f"matches: {reference['matching_tensor_kinds'] or 'none'}"
        )
    lines.append(
        f"  VERDICT (vs segmented): "
        f"{'lossless' if result['lossless'] else 'NOT lossless'}"
    )
    return "\n".join(lines)
