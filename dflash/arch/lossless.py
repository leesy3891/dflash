"""How far a run strays from lossless, measured at every verify step.

``rollback.check`` answers a yes/no question once: after a rejection, is the
target's state the one it would have had without the rejected tokens? On a
GDN target it is not -- ``crop`` rolls back attention KV and the conv window
but leaves the recurrent state holding the whole verify block. This module
puts a number on that during an actual DFlash run, step by step.

At each verify step it reconstructs the state a lossless implementation would
have, by restoring the recurrent state saved just before the verify and
replaying only the committed tokens (anchor + accepted drafts) through the
target. That replay is the reference. Against it, per step:

``state``
    relative Frobenius error of every GDN layer's recurrent state,
    ``|S_stock - S_exact| / |S_exact|``.
``next token``
    the target's distribution for the *next* anchor (the bonus/correction
    token this step emitted) computed from the stock state and from the exact
    one: KL(exact || stock), max |delta logit|, and whether the argmax flips.
    This is where the error becomes visible: a flipped argmax is a token the
    lossless decoder would not have produced.

Two references, because they answer different questions:

``incremental``
    replay from the state *this run* had before the verify. Isolates the
    error this one rejection added.
``accumulated``
    replay from a shadow state that has only ever seen committed tokens,
    carried across steps. The drift of the stock run from a lossless run
    along the same token sequence.

Modes:

``measure``
    everything above, then the stock state -- recurrent, conv window and the
    block's KV slice -- is put back bit for bit, so the run continues on
    exactly the trajectory the ``perf`` pass takes. Checked, not assumed: the
    sweep compares this pass's tokens against ``perf``'s.
``correct``
    the exact state is kept. That makes the run lossless (for greedy
    decoding, output must then match AR up to kernel-width numerics), so its
    acceptance against the stock run's is the acceptance the bug costs. Its
    timing includes the replay and is not a latency result: an efficient fix
    would have the kernel emit per-token states, not re-run the block.

Floors. A replay at a different width is not bitwise the verify: steps where
every draft was accepted fold nothing extra into the state, so their error is
the replay's own numeric floor, reported separately. A full-attention target
has no recurrent state; running the audit on it measures the floor of the
logit comparison alone and is the control.
"""

from __future__ import annotations

import statistics

import torch

MEASURE = "measure"
CORRECT = "correct"


def _linear_layers(cache):
    for index, layer in enumerate(getattr(cache, "layers", [])):
        recurrent = getattr(layer, "recurrent_states", None)
        if isinstance(recurrent, dict) and any(
            isinstance(t, torch.Tensor) for t in recurrent.values()
        ):
            yield index, layer


def _attention_layers(cache):
    for index, layer in enumerate(getattr(cache, "layers", [])):
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.numel():
            yield index, layer


def _save(cache, field: str) -> dict:
    out = {}
    for index, layer in _linear_layers(cache):
        for slot, tensor in getattr(layer, field).items():
            if isinstance(tensor, torch.Tensor):
                out[(index, slot)] = tensor.detach().clone()
    return out


def _restore_recurrent(cache, saved: dict) -> None:
    # In place: the cache writes recurrent states with ``copy_`` and some
    # kernels hold the tensor, so the object must stay the same.
    for (index, slot), tensor in saved.items():
        cache.layers[index].recurrent_states[slot].copy_(tensor)


def _restore_conv(cache, saved: dict) -> None:
    # After a crop the conv state is a kernel-width view; replacing it with a
    # kernel-width tensor of the same values is what the next forward sees.
    for (index, slot), tensor in saved.items():
        cache.layers[index].conv_states[slot] = tensor.clone()


def _forward(target, cache, ids, positions):
    out = target(
        input_ids=ids,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return out.logits[0, -1].float()


def _relative_errors(stock: dict, exact: dict) -> dict:
    errors = {}
    for key, reference in exact.items():
        diff = (stock[key].float() - reference.float()).norm()
        scale = reference.float().norm()
        errors[f"{key[0]}.{key[1]}"] = (
            (diff / scale).item() if scale > 0 else diff.item()
        )
    return errors


def _logit_divergence(exact: torch.Tensor, other: torch.Tensor) -> dict:
    exact = exact.to(other.device)
    log_exact = torch.log_softmax(exact, dim=-1)
    log_other = torch.log_softmax(other, dim=-1)
    top = int(exact.argmax())
    return {
        "kl_exact_to_stock": float(
            (log_exact.exp() * (log_exact - log_other)).sum().clamp_min(0)
        ),
        "max_abs_logit_diff": float((exact - other).abs().max()),
        "argmax_flip": bool(int(other.argmax()) != top),
        "exact_top1_prob": float(log_exact[top].exp()),
        "stock_prob_of_exact_top1": float(log_other[top].exp()),
        "exact_top1_rank_under_stock": int((other > other[top]).sum()),
    }


class VerifyAudit:
    """Handed to ``dflash_generate(verify_audit=...)``; one per run."""

    def __init__(self, mode: str = MEASURE, keep_per_layer: bool = True) -> None:
        if mode not in (MEASURE, CORRECT):
            raise ValueError(f"unknown audit mode {mode!r}")
        self.mode = mode
        self.keep_per_layer = keep_per_layer
        self.steps: list[dict] = []
        self._before_recurrent: dict = {}
        self._before_conv: dict = {}
        self._shadow: dict | None = None

    # -- hooks -------------------------------------------------------------

    def before_verify(self, cache) -> None:
        self._before_recurrent = _save(cache, "recurrent_states")
        self._before_conv = _save(cache, "conv_states")
        if self._shadow is None:
            # The state after prefill has seen only the prompt, so it is
            # exact; the shadow starts there and only ever sees commits.
            self._shadow = {k: v.clone() for k, v in self._before_recurrent.items()}

    @torch.no_grad()
    def after_commit(
        self,
        target,
        cache,
        *,
        block_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_start: int,
        produced: int,
        verify_size: int,
        next_anchor: torch.Tensor | None,
    ) -> None:
        end = block_start + produced
        committed = block_ids[:, :produced]
        committed_positions = position_ids[:, block_start:end]
        anchor_position = position_ids[:, end:end + 1]

        stock_recurrent = _save(cache, "recurrent_states")
        stock_conv = _save(cache, "conv_states")
        stock_kv = {
            index: (
                layer.keys[..., block_start:end, :].clone(),
                layer.values[..., block_start:end, :].clone(),
            )
            for index, layer in _attention_layers(cache)
            if layer.keys.shape[-2] == end
        }

        step: dict = {
            "step": len(self.steps),
            "block_start": block_start,
            "verify_size": verify_size,
            "committed": produced,
            # Tokens the verify folded into the recurrent state that the
            # crop could not take back out.
            "rejected_folded": verify_size - produced,
        }

        stock_logits = None
        if next_anchor is not None:
            stock_logits = _forward(target, cache, next_anchor, anchor_position)
            cache.crop(-1)
            _restore_recurrent(cache, stock_recurrent)
            _restore_conv(cache, stock_conv)

        def replay(start_state: dict):
            # Back to the pre-verify prefix: attention by slicing, the conv
            # window and recurrent state from what was saved before verify
            # (the conv crop cannot reach back past its kernel window).
            cache.crop(-produced)
            _restore_conv(cache, self._before_conv)
            _restore_recurrent(cache, start_state)
            _forward(target, cache, committed, committed_positions)
            cache.crop(0)  # conv back to kernel width; attention untouched
            exact_recurrent = _save(cache, "recurrent_states")
            exact_conv = _save(cache, "conv_states")
            logits = None
            if next_anchor is not None:
                logits = _forward(target, cache, next_anchor, anchor_position)
                cache.crop(-1)
                _restore_recurrent(cache, exact_recurrent)
                _restore_conv(cache, exact_conv)
            return exact_recurrent, logits

        references = {"incremental": self._before_recurrent}
        if self.mode == MEASURE:
            # In correct mode the live state is the shadow, so the two
            # references coincide and one replay is enough.
            references["accumulated"] = self._shadow

        exact_states = {}
        for name, start_state in references.items():
            exact_recurrent, exact_logits = replay(start_state)
            exact_states[name] = exact_recurrent
            errors = _relative_errors(stock_recurrent, exact_recurrent)
            entry: dict = {
                "state_rel_err_mean": (
                    statistics.fmean(errors.values()) if errors else None
                ),
                "state_rel_err_max": max(errors.values()) if errors else None,
            }
            if self.keep_per_layer and errors:
                entry["state_rel_err_per_layer"] = errors
            if stock_logits is not None and exact_logits is not None:
                entry.update(_logit_divergence(exact_logits, stock_logits))
            step[name] = entry

        if self.mode == MEASURE:
            self._shadow = exact_states["accumulated"]
            # Put the stock state back exactly: replayed KV for the block is
            # numerically close to the verify's but not bitwise, and the point
            # of this mode is to follow the perf pass's trajectory.
            for index, (keys, values) in stock_kv.items():
                layer = cache.layers[index]
                layer.keys[..., block_start:end, :].copy_(keys)
                layer.values[..., block_start:end, :].copy_(values)
            _restore_recurrent(cache, stock_recurrent)
            _restore_conv(cache, stock_conv)
        # CORRECT: the cache already holds the incremental replay, which --
        # since every earlier step was also corrected -- is the exact state.
        self.steps.append(step)

    # -- reduction ---------------------------------------------------------

    def summary(self) -> dict:
        return summarise(self.steps, mode=self.mode)


def _stats(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def summarise(steps: list[dict], *, mode: str) -> dict:
    """Per-run reduction. Rejection steps and floor steps are kept apart."""
    out: dict = {"mode": mode, "num_steps": len(steps)}
    rejecting = [s for s in steps if s["rejected_folded"] > 0]
    floor = [s for s in steps if s["rejected_folded"] == 0]
    out["steps_with_rejection"] = len(rejecting)
    out["steps_full_accept"] = len(floor)
    out["rejected_tokens_folded_total"] = sum(s["rejected_folded"] for s in steps)

    for reference in ("incremental", "accumulated"):
        present = [s for s in steps if reference in s]
        if not present:
            continue
        block: dict = {}
        for label, subset in (("rejecting", [s for s in rejecting if reference in s]),
                              ("floor", [s for s in floor if reference in s])):
            block[label] = {
                "state_rel_err_mean": _stats(
                    [s[reference]["state_rel_err_mean"] for s in subset]
                ),
                "state_rel_err_max": _stats(
                    [s[reference]["state_rel_err_max"] for s in subset]
                ),
                "kl_exact_to_stock": _stats(
                    [s[reference].get("kl_exact_to_stock") for s in subset]
                ),
                "max_abs_logit_diff": _stats(
                    [s[reference].get("max_abs_logit_diff") for s in subset]
                ),
            }
        with_logits = [s for s in present if "argmax_flip" in s[reference]]
        flips = [s for s in with_logits if s[reference]["argmax_flip"]]
        block["next_token_steps"] = len(with_logits)
        block["argmax_flips"] = len(flips)
        block["argmax_flip_rate"] = (
            len(flips) / len(with_logits) if with_logits else None
        )
        block["first_flip_step"] = flips[0]["step"] if flips else None
        # Where along the run the drift is: the error at the last step is the
        # accumulated drift after the whole output, not an average.
        block["final_state_rel_err_mean"] = present[-1][reference]["state_rel_err_mean"]
        out[reference] = block
    return out


def token_agreement(reference: list[int], other: list[int]) -> dict:
    """Where two greedy outputs part ways, for the output-level comparison."""
    length = min(len(reference), len(other))
    first = next(
        (i for i in range(length) if reference[i] != other[i]), None
    )
    matched = sum(1 for i in range(length) if reference[i] == other[i])
    return {
        "compared_tokens": length,
        "identical": first is None and len(reference) == len(other),
        "first_divergence": first,
        "common_prefix": length if first is None else first,
        "positional_match_rate": matched / length if length else None,
    }
