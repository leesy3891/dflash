"""The autoregressive baseline, measured before the drafter is loaded.

``cmd_sweep`` runs every AR condition with only the target resident and loads
the drafter afterwards, so the AR peaks contain the target's weights, cache and
activations and nothing of the drafter's. Records say so in
``drafter_resident``.

Two things this does not do, both deliberate.

It does not use ``block_size=1`` through ``dflash_generate``. That run still
loads the drafter onto the GPU, so its memory baseline includes the drafter's
weights and its cache object -- which is precisely the overhead the comparison
is trying to isolate. A memory baseline has to come from a process where the
drafter was never constructed.

It does not silently pick a cache policy. ``dflash_generate`` calls
``activate_past_recording()`` so a rejected block can be cropped, and on a
hybrid target that is not a neutral bookkeeping flag: it changes which kernel
runs. ``Qwen3_5GatedDeltaNet.forward`` takes the fused single-token path
``causal_conv1d_update`` only when ``seq_len == 1 and not record_past``.
With recording on, every decode step instead goes through the general
``causal_conv1d_fn`` path and the conv state keeps growing until a crop.

So a single AR number would have to choose between two different questions:

``native``
    No recording. What autoregressive decoding of this target actually costs,
    which is the speedup denominator a reader cares about.
``recording``
    Recording on **and cropped every step**, matching what
    ``dflash_generate`` actually does. Isolates the cost of being able to
    roll back from the cost of speculation.

The crop is not optional. With ``record_past`` on, ``update_conv_state``
keeps the concatenated history instead of the last ``conv_kernel_size``
columns, so a loop that never calls ``crop`` grows its conv state by one
column per token and its per-step cost grows with it. ``dflash_generate``
crops after every verify, which bounds the buffer. A "recording" baseline
without the crop therefore measures unbounded buffer growth rather than
DFlash's policy, and reports a penalty several times the real one.

Both policies are measured. The difference between them is DFlash's rollback
tax on the target, and on a full-attention target it should be nil.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
from transformers import DynamicCache

from ..model import _crop_to
from . import statemem, taxonomy

NATIVE = "native"
RECORDING = "recording"


def make_cache(config, policy: str) -> DynamicCache:
    cache = DynamicCache(config=config)
    if policy == RECORDING:
        cache.activate_past_recording()
    elif policy != NATIVE:
        raise ValueError(f"unknown cache policy {policy!r}")
    return cache


def _cuda_time() -> float:
    """A timestamp taken after the device has caught up.

    Guarded rather than assumed: the loop's correctness -- cache policy, crop,
    EOS trimming -- is device-independent and is exercised on CPU, where a
    synchronize would raise.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def generate(
    target,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_token_ids: list[int] | None = None,
    cache_policy: str = NATIVE,
    temperature: float = 0.0,
    ignore_eos: bool = False,
    component_probe=None,
    mixers: list[str] | None = None,
    eos_check_interval: int = 8,
    crop_each_step: bool | None = None,
    phase_callback=None,
) -> SimpleNamespace:
    """Greedy AR decode of one request, timed and component-resolved.

    ``ignore_eos`` is what separates a shape-controlled run -- exactly
    ``max_new_tokens`` tokens, so the token count is a controlled variable --
    from a natural one that stops at EOS. The two are never averaged together;
    the returned ``stopped_naturally`` flag is what keeps them apart.

    **No host synchronisation inside the decode loop.** The obvious way to
    write this loop -- ``produced.append(int(next_token.item()))`` and an EOS
    test per step -- forces a device-to-host copy after every token, and the
    CPU cannot launch step *n+1* until step *n*'s result has come back. That
    turns the baseline into a latency-bound loop that a real implementation
    would not be, and on a layer-sharded target it drains the pipeline at
    every token.

    It also makes the comparison unfair in a specific direction. DFlash
    synchronises roughly once per *verify step*; at block 16 and ~0.49
    acceptance that is about one sync per six committed tokens. An AR loop
    that syncs once per token therefore pays ~6x more sync overhead per token
    than the thing it is the denominator for, and every bit of that difference
    lands in the reported speedup.

    So tokens are written into a device-side buffer and never read back
    mid-loop. With ``ignore_eos`` there is nothing to read at all. Without it,
    the stop condition is checked every ``eos_check_interval`` steps, which
    costs one sync per interval instead of one per token and may run a few
    tokens past EOS; those are trimmed from the output and reported as
    ``eos_overrun_tokens``, and the latency metric is per *executed* token so
    that the overrun is charged rather than hidden.
    """
    device = next(target.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]
    config = taxonomy.text_config(target.config)
    cache = make_cache(target.config, cache_policy)
    # Recording without cropping is not a policy anything runs; see the module
    # docstring. Callers can still force either way to measure the difference.
    if crop_each_step is None:
        crop_each_step = cache_policy == RECORDING

    stop = (
        torch.tensor(stop_token_ids, dtype=torch.long, device=device)
        if stop_token_ids and not ignore_eos
        else None
    )
    position_ids = torch.arange(
        prompt_len + max_new_tokens + 1, device=device
    ).unsqueeze(0)

    components: list[dict] = []

    def snapshot(label: str, token_index: int | None = None) -> None:
        if component_probe is None:
            return
        split = component_probe(
            target_cache=cache,
            draft_cache=_EMPTY,
            live={"selected_hidden": None, "context_feature": None},
            draft_weight_bytes=0,
        )
        split["label"] = label
        split["decode_token"] = token_index
        components.append(split)

    def phase(label):
        if phase_callback is not None:
            phase_callback(label, {})

    prefill_start = _cuda_time()
    phase("prefill: target forward")
    output = target(
        input_ids=input_ids,
        position_ids=position_ids[:, :prompt_len],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    next_token = (
        output.logits[:, -1, :].argmax(dim=-1)
        if temperature == 0
        else torch.multinomial(
            torch.softmax(output.logits[:, -1, :].float() / temperature, dim=-1), 1
        ).squeeze(-1)
    )
    ttft = _cuda_time() - prefill_start
    snapshot("prefill_end")

    # Tokens live on the device for the whole loop. `tokens[0]` is the first
    # token, which prefill produced.
    tokens = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    tokens[0] = next_token[0]
    # A device-side sticky flag: once any produced token is a stop token it
    # stays set, so an interval check never misses an EOS that happened
    # between two checks.
    hit_stop = torch.zeros((), dtype=torch.bool, device=device)
    if stop is not None:
        hit_stop |= torch.isin(next_token, stop).any()

    executed = 1
    stopped_naturally = False
    decode_start = _cuda_time()
    step_times: list[float] = []
    while executed < max_new_tokens:
        index = prompt_len + executed - 1
        step_start = time.perf_counter()
        phase("decode: ar step")
        output = target(
            input_ids=next_token.view(1, 1),
            position_ids=position_ids[:, index:index + 1],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = (
            output.logits[:, -1, :].argmax(dim=-1)
            if temperature == 0
            else torch.multinomial(
                torch.softmax(output.logits[:, -1, :].float() / temperature, dim=-1), 1
            ).squeeze(-1)
        )
        tokens[executed] = next_token[0]
        if crop_each_step:
            # What dflash_generate does after every verify. Under the
            # recording policy this is what stops the conv history growing
            # without bound; it is a no-op for the attention layers, whose
            # crop(0) returns immediately.
            _crop_to(cache, prompt_len + executed)
        if stop is not None:
            hit_stop |= torch.isin(next_token, stop).any()
        # These are launch-side timings, not completion times: without a
        # synchronize the host returns as soon as the kernels are queued.
        # They are kept for shape only; the reported latency comes from the
        # synchronized span around the whole loop.
        step_times.append(time.perf_counter() - step_start)
        executed += 1
        # The only read-back in the loop, and only when EOS can stop it.
        if stop is not None and executed % eos_check_interval == 0:
            if bool(hit_stop.item()):
                stopped_naturally = True
                break
        if component_probe is not None and executed in (2, 16, 64, 256):
            snapshot("decode", executed)

    phase(None)
    decode_time = _cuda_time() - decode_start
    snapshot("run_end")

    # One read-back at the end, outside the timed region.
    produced = tokens[:executed].tolist()
    eos_overrun = 0
    if stop is not None:
        if not stopped_naturally:
            stopped_naturally = bool(hit_stop.item())
        if stopped_naturally:
            stop_set = set(stop_token_ids or ())
            for position, token in enumerate(produced):
                if token in stop_set:
                    # Tokens after the stop were executed but are not output.
                    eos_overrun = len(produced) - (position + 1)
                    produced = produced[: position + 1]
                    break

    return SimpleNamespace(
        role="ar",
        cache_policy=cache_policy,
        num_input_tokens=prompt_len,
        num_output_tokens=len(produced),
        num_executed_tokens=executed,
        eos_overrun_tokens=eos_overrun,
        output_ids=produced,
        time_to_first_token_s=ttft,
        decode_latency_s=decode_time,
        total_latency_s=ttft + decode_time,
        # TPOT over the decode phase only, excluding the first token, which is
        # prefill's. Divided by tokens actually *executed*, so an EOS overrun
        # is charged to the per-token cost instead of making it look cheaper.
        # With ignore_eos, executed and delivered are the same number.
        time_per_output_token_s=(
            decode_time / (executed - 1) if executed > 1 else None
        ),
        time_per_delivered_token_s=(
            decode_time / (len(produced) - 1) if len(produced) > 1 else None
        ),
        step_times_s=step_times,
        stopped_naturally=stopped_naturally,
        ignore_eos=ignore_eos,
        crop_each_step=crop_each_step,
        eos_check_interval=(None if stop is None else eos_check_interval),
        syncs_in_decode_loop=(
            0 if stop is None else executed // eos_check_interval
        ),
        hit_length_cap=executed >= max_new_tokens,
        components=components,
        num_layers=config.num_hidden_layers,
    )


class _Empty:
    """Stands in for the absent draft cache so the probe signature is shared."""

    layers: list = []


_EMPTY = _Empty()


def weight_bytes_per_device(model) -> dict:
    return taxonomy._param_bytes_per_device(model)


def baseline_memory(model, cache, mixers=None) -> dict:
    """Weights plus resolved cache components, per device, drafter-free."""
    return {
        "weights_per_device": weight_bytes_per_device(model),
        "cache": statemem.summarise(
            statemem.cache_components(cache, mixers=mixers)
        ),
        "allocator": statemem.device_allocator_state(),
    }
