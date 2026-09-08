import time
from types import SimpleNamespace
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        round(start + (i * span) / (num_draft_layers - 1))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int] | None,
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def _decoder_layers(target: nn.Module, expected: int) -> nn.ModuleList:
    """The target's decoder layer list, however deeply the wrapper nests it.

    Qwen3 keeps it at ``target.model.layers``; Qwen3.5 wraps a text model in a
    conditional-generation head, so it sits one level further down. Matching on
    the expected length rules out a vision tower's own stack.
    """
    candidates = [
        (name, module._modules["layers"])
        for name, module in target.named_modules()
        if isinstance(module._modules.get("layers"), nn.ModuleList)
        and len(module._modules["layers"]) == expected
    ]
    if not candidates:
        raise ValueError(
            f"Could not find a decoder stack of {expected} layers on "
            f"{type(target).__name__}"
        )
    candidates.sort(key=lambda item: item[0].count("."))
    return candidates[0][1]


class HiddenStateTap:
    """Capture only the target layers whose output DFlash actually injects.

    ``output_hidden_states=True`` materialises every layer's residual stream —
    37 tensors of (1, N, 4096) for Qwen3-8B — when the drafter reads five of
    them. During prefill that is the single largest tensor in the run: ~19 GB at
    64k input, enough on its own to push an otherwise comfortable run off a 48 GB
    card. Forward hooks on the wanted layers keep exactly the same tensors and
    let the rest be freed as the forward walks the stack.

    The tensors captured here are the layer outputs, which is what
    ``hidden_states[layer_id + 1]`` is. The final entry of ``hidden_states`` is
    the only one HuggingFace normalises, and ``build_target_layer_ids`` never
    selects the last layer, so the two paths agree exactly.
    """

    def __init__(
        self, target: nn.Module, layer_ids: list[int], num_target_layers: int,
        device: "torch.device | None" = None,
    ):
        self.layers = _decoder_layers(target, num_target_layers)
        self.layer_ids = list(layer_ids)
        # With the target sharded across GPUs the wanted layers land on
        # different devices; gather them where the drafter runs.
        self.device = device
        if max(self.layer_ids) >= num_target_layers - 1:
            raise ValueError(
                "HiddenStateTap cannot serve the final layer, whose hidden state "
                "HuggingFace reports after the final norm"
            )
        self.captured: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _hook(self, layer_id: int):
        def hook(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            if self.device is not None and state.device != self.device:
                state = state.to(self.device)
            self.captured[layer_id] = state
        return hook

    def __enter__(self) -> "HiddenStateTap":
        for layer_id in self.layer_ids:
            self._handles.append(
                self.layers[layer_id].register_forward_hook(self._hook(layer_id))
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captured.clear()

    def states(self) -> list[torch.Tensor]:
        return [self.captured[layer_id] for layer_id in self.layer_ids]

    def feature(self) -> torch.Tensor:
        return torch.cat(self.states(), dim=-1)

    def release(self) -> None:
        """Drop references so the captured activations can be freed."""
        self.captured.clear()


def prefill_chunking_safe(target: nn.Module) -> bool:
    """Whether slicing the prefill reproduces a single full forward.

    For a full-attention target it does: measured on Qwen3-8B over a 13217-token
    prompt, chunked and one-shot prefill agree on the next-token argmax at
    100.000% of positions. A hybrid target does not -- Qwen3.5-9B keeps 24 of
    its 32 layers on a gated-delta-rule recurrence, and splitting the prefill
    changes its predictions at ~7% of positions (93.2% agreement at chunk 4096,
    93.7% at 8192). That is a different model output, not rounding, so chunking
    is refused there rather than silently distorting acceptance.
    """
    config = getattr(target.config, "text_config", None) or target.config
    layer_types = getattr(config, "layer_types", None)
    return not layer_types or all(kind == "full_attention" for kind in layer_types)


def _target_step(target, tap, layer_ids, want_hidden: bool, **kwargs):
    """Run the target and hand back the residual streams DFlash injects.

    Returns ``(output, selected, accounted)``: the model output, the wanted
    layers' hidden states in ``layer_ids`` order, and the tensors to charge to
    the drafter's hidden-state overhead. Under the tap those two coincide;
    under ``output_hidden_states`` the target materialised every layer, so the
    whole tuple is what the run actually paid for.
    """
    if not want_hidden:
        return target(**kwargs, output_hidden_states=False), None, ()
    if tap is not None:
        with tap:
            output = target(**kwargs, output_hidden_states=False)
            selected = tap.states()
        return output, selected, selected
    output = target(**kwargs, output_hidden_states=True)
    offset = 1
    return output, [output.hidden_states[i + offset] for i in layer_ids], output.hidden_states


def _sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None

    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def _sample_probs(probs: torch.Tensor) -> torch.Tensor:
    shape = probs.shape[:-1]
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(shape)


def _validate_sampling(temperature: float, top_p: float, top_k: int) -> None:
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("temperature and top_k must be non-negative, and top_p in (0, 1].")


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    _validate_sampling(temperature, top_p, top_k)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    return _sample_probs(_sampling_probs(logits, temperature, top_p, top_k))


def _rejection_sample(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_indices: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor]:
    gamma = draft_tokens.shape[1]
    p = target_probs[:, :gamma].gather(-1, draft_tokens[..., None])[..., 0]
    if draft_indices is None:
        q = draft_probs.gather(-1, draft_tokens[..., None])[..., 0]
    else:
        q = (draft_probs * (draft_indices == draft_tokens[..., None])).sum(-1)
    accepted = (
        (torch.rand_like(q) * q < p).to(torch.int32).cumprod(-1).sum(-1)[0].item()
    )
    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0]

    residual = target_probs[0, accepted].clone()
    if draft_indices is None:
        residual.sub_(draft_probs[0, accepted])
    else:
        residual.scatter_add_(0, draft_indices[0, accepted], -draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[0, accepted],
    )
    return accepted, _sample_probs(residual[None])[0]


def _draft_config(config):
    return getattr(config, "dflash_config", {})


def _draft_value(config, name, default=None):
    return _draft_config(config).get(name, getattr(config, name, default))


def _raw_input_embeddings(
    target: nn.Module, input_ids: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """The drafter's noise embedding, taken from the target's embedding table.

    Reads the weight directly rather than calling the module, which on a sharded
    target means accelerate's device hooks never run: if the embedding table
    landed on a different card from the drafter, both the lookup and the result
    have to be bridged by hand.
    """
    weight = target.get_input_embeddings().weight
    embedding = F.embedding(input_ids.to(weight.device), weight) * scale
    return embedding.to(input_ids.device)


def _output_head(target: nn.Module) -> nn.Module:
    head = getattr(target, "lm_head", None)
    return head if head is not None else target.get_output_embeddings()


def _make_cache(config):
    cache = DynamicCache(config=config)
    cache.activate_past_recording()
    return cache


def _crop_to(cache, length):
    remove = cache.get_seq_length() - length
    cache.crop(-remove)


def _cache_bytes(cache) -> int:
    """CUDA bytes held by a cache, counting each storage once.

    Walks the per-layer tensors rather than assuming a key/value pair, so it
    also covers the conv and recurrent states of hybrid targets.
    """
    seen: set[tuple] = set()
    total = 0
    pending = list(getattr(cache, "layers", []))
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                storage = item.untyped_storage()
                key = (item.device, storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif hasattr(item, "__dict__"):
            pending.extend(item.__dict__.values())
    return total


def _all_device_peak(reserved: bool = False) -> int:
    """Peak CUDA bytes summed over every visible device.

    Identical to the single-device figure when the model sits on one card, and
    the only meaningful total when the target is sharded across several.
    """
    read = (
        torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated
    )
    return sum(read(index) for index in range(torch.cuda.device_count()))


def _reset_device_peaks() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _device_bytes(index: int, field: str) -> int:
    """One allocator counter, without the public wrapper's overhead.

    ``torch.cuda.memory_allocated`` and friends rebuild and flatten the entire
    stats dictionary on every call: 81us measured, against 10us for the raw
    binding. At one read per decoder layer that difference lands directly in
    the reported per-token latency, so the fast path is the one that keeps the
    profiler from changing what it measures.
    """
    try:
        return torch._C._cuda_memoryStats(index)["allocated_bytes"]["all"][field]
    except (AttributeError, KeyError):  # pragma: no cover - old torch
        return (
            torch.cuda.max_memory_allocated(index)
            if field == "peak"
            else torch.cuda.memory_allocated(index)
        )


def _all_device_live() -> int:
    """Bytes live on every visible device right now."""
    return sum(
        _device_bytes(index, "current")
        for index in range(torch.cuda.device_count())
    )


class PeakTracker:
    """Where the run's peak allocation lands, not just how large it is.

    ``torch.cuda.max_memory_allocated`` is a per-device maximum over the whole
    run, so summing it across a sharded target adds maxima that never coexisted.
    Measured on a 32k Qwen3.5-9B prefill split over two cards: device 0 peaks at
    20.69 GB while running its own layers and has fallen back to 14.29 GB long
    before device 1 peaks at 24.51, so the naive sum reports 45.20 GB against a
    true simultaneous 32.91 -- a 12.29 GB (27%) overcount.

    Cutting the run into short intervals fixes it. Within one interval only one
    device is doing work, so the sum of per-interval device maxima is tight, and
    the maximum over intervals is what gets reported. On a single device the two
    agree exactly: a maximum over time is the maximum over any partition of it,
    so single-GPU figures are unchanged by this.

    Every interval carries the label of the operation running in it plus the
    decode token and drafter step in flight, so the record says which operation
    set the peak.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.num_devices = torch.cuda.device_count() if self.enabled else 0
        self.peak_bytes = 0
        self.peak_site: dict | None = None
        self.peak_per_device: list[int] = []
        self._device_maxima = [0] * self.num_devices
        self._label = "startup"
        self._site: dict = {}
        self._decode_token = None
        self._draft_step = None
        self._open_decode_token = None
        self._open_draft_step = None
        if self.enabled:
            _reset_device_peaks()

    def context(self, *, decode_token=None, draft_step=None) -> None:
        """Name the decode position the *following* intervals belong to.

        This does not touch the interval already in flight. An interval that
        opened during prefill and is still open when the decode loop starts is
        prefill's, and stamping it with decode token 0 would report a prefill
        peak as a decode-time one.
        """
        self._decode_token = decode_token
        self._draft_step = draft_step

    def boundary(self, label: str, **site) -> int:
        """Close the interval that just ran and open one called ``label``.

        Returns the closed interval's peak, which is what the draft-activation
        probe needs and saves it a second round of resets.
        """
        if not self.enabled:
            return 0
        per_device = [
            _device_bytes(index, "peak") for index in range(self.num_devices)
        ]
        total = sum(per_device)
        self._device_maxima = [
            max(seen, now) for seen, now in zip(self._device_maxima, per_device)
        ]
        if total > self.peak_bytes:
            self.peak_bytes = total
            self.peak_per_device = per_device
            self.peak_site = {
                "operation": self._label,
                # Captured when this interval opened -- see context().
                "decode_token": self._open_decode_token,
                "draft_step": self._open_draft_step,
                **self._site,
            }
        _reset_device_peaks()
        self._label, self._site = label, site
        self._open_decode_token = self._decode_token
        self._open_draft_step = self._draft_step
        return total

    def watch(self, layers) -> list:
        """Open an interval before every target layer.

        A sharded prefill only attributes its peak correctly if the intervals
        are short enough that one device is working in each, and one decoder
        layer is that unit. Pre-hooks rather than post-hooks so the interval
        contains the layer it is named after.
        """
        if not self.enabled:
            return []
        def make_hook(index):
            # A pre-hook that returns anything replaces the layer's arguments,
            # and boundary() returns the closed interval's size, so swallow it.
            def hook(module, args):
                self.boundary(f"target layer {index}", layer=index)

            return hook

        return [
            layer.register_forward_pre_hook(make_hook(index))
            for index, layer in enumerate(layers)
        ]

    def finish(self) -> None:
        self.boundary("finished")

    @property
    def sum_of_device_maxima_bytes(self) -> int:
        """The old, over-counting figure, kept so records stay comparable."""
        return sum(self._device_maxima)


def _tensor_bytes(tensors) -> int:
    """CUDA bytes held by an iterable of tensors, counting each storage once."""
    seen: set[tuple] = set()
    total = 0
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


class _GpuTimer:
    """Accumulates GPU time for a region using CUDA events.

    The draft/verify loop already synchronises once per step (the acceptance
    count is read back with ``.item()``), but work *within* a step is still
    queued asynchronously, so wall-clock timers cannot separate the drafter
    from the target. CUDA events can, without adding a sync per region; pending
    pairs are drained periodically so the event list stays bounded.
    """

    _DRAIN_EVERY = 512

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.elapsed_ms = 0.0
        self._pending: list[tuple] = []
        self._start = None

    def __enter__(self):
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, *exc_info) -> bool:
        if self.enabled:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((self._start, end))
            if len(self._pending) >= self._DRAIN_EVERY:
                self._drain()
        return False

    def _drain(self) -> None:
        if not self._pending:
            return
        self._pending[-1][1].synchronize()
        self.elapsed_ms += sum(start.elapsed_time(end) for start, end in self._pending)
        self._pending.clear()

    @property
    def seconds(self) -> float | None:
        if not self.enabled:
            return None
        self._drain()
        return self.elapsed_ms / 1000.0


def module_bytes(module: nn.Module) -> int:
    """CUDA bytes held by a module's parameters and buffers."""
    seen: set[tuple] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def _attention_mask(query, key, *, is_causal, sliding_window):
    query_position = key.shape[-2] - query.shape[-2] + torch.arange(
        query.shape[-2], device=query.device
    )[:, None]
    key_position = torch.arange(key.shape[-2], device=query.device)[None, :]
    visible = torch.ones(
        (query.shape[-2], key.shape[-2]), dtype=torch.bool, device=query.device
    )
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    return visible[None, None]


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    block_size: int | None = None,
    return_stats: bool = False,
    profile_draft_memory: bool = False,
    profile_draft_latency: bool = True,
    hidden_states: str = "full",
    prefill_chunk: int | None = None,
):
    _validate_sampling(temperature, top_p, top_k)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size

    # Everything DFlash keeps between steps lives with the drafter. On one GPU
    # that is the target's device too; on a sharded target it is one shard, and
    # the boundaries that cross a device are moved explicitly.
    device = next(model.parameters()).device
    output_ids = torch.full(
        (1, max_length + 1), model.mask_token_id, dtype=torch.long, device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = _make_cache(target.config)
    past_key_values_draft = _make_cache(model.config)

    timing = return_stats and profile_draft_latency
    draft_forward_timer = _GpuTimer(timing)
    context_feature_timer = _GpuTimer(timing)
    target_forward_timer = _GpuTimer(timing)
    # The target's context feature is what DFlash injects into every draft
    # layer, and materialising it forces output_hidden_states on the target.
    # Both are drafter overhead and neither is charged to the baseline.
    hidden_states_bytes = 0
    context_feature_bytes = 0
    # Two ways to get the target residual streams the drafter is conditioned
    # on: ask for all of them, or hook the handful DFlash reads. They agree
    # exactly, but at 64k input the full tuple is ~19 GB on its own, so the tap
    # is what keeps a long-context run on one card. See HiddenStateTap.
    if hidden_states not in ("selective", "full"):
        raise ValueError(
            f"Unknown hidden_states mode '{hidden_states}'; use 'selective' or 'full'"
        )
    tracker = PeakTracker(bool(return_stats))
    tap = (
        HiddenStateTap(
            target,
            model.target_layer_ids,
            int(_draft_value(model.config, "num_target_layers")),
            device=device,
        )
        if block_size > 1 and hidden_states == "selective"
        else None
    )

    target_config = getattr(target.config, "text_config", None) or target.config
    watch_handles = tracker.watch(
        _decoder_layers(target, target_config.num_hidden_layers)
    )

    prefill_start = _cuda_time() if return_stats else None
    # Prefill in slices when asked. The KV cache carries the sequence forward —
    # including the recurrent state of a hybrid target — so the result is the
    # same, but the per-call activation of an attention kernel is bounded by the
    # chunk rather than the context. Qwen3.5's linear-attention prefill needs
    # ~14 GB at 32k and twice that at 64k, which is what puts a 64k run off a
    # 48 GB card long before any of DFlash's own tensors do.
    if prefill_chunk and not prefill_chunking_safe(target):
        raise ValueError(
            "--prefill-chunk is only valid for a full-attention target; this "
            "one has recurrent layers whose prefill does not reproduce across a "
            "chunk boundary. See prefill_chunking_safe()."
        )
    chunk = prefill_chunk if prefill_chunk else num_input_tokens
    feature_chunks = []
    for begin in range(0, num_input_tokens, chunk):
        end = min(begin + chunk, num_input_tokens)
        tracker.boundary("prefill: target forward", chunk=begin // chunk)
        output, selected, accounted = _target_step(
            target,
            tap,
            model.target_layer_ids,
            block_size > 1,
            input_ids=input_ids[:, begin:end],
            position_ids=position_ids[:, begin:end],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
        )
        if block_size > 1:
            hidden_states_bytes = max(hidden_states_bytes, _tensor_bytes(accounted))
            with context_feature_timer:
                feature_chunks.append(torch.cat(selected, dim=-1))
            # The prefill streams are the largest tensors in the run; drop them
            # the moment this chunk's context feature has been built.
            selected = accounted = None

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(
        output.logits, temperature, top_p, top_k
    )
    # Closed for every configuration, not just DFlash: without it the last
    # prefill layer's interval stays open into the first verify, and the peak
    # of a baseline run gets reported against a prefill label.
    tracker.boundary("prefill: first token")
    if block_size > 1:
        tracker.boundary("prefill: context-feature concat")
        with context_feature_timer:
            target_hidden = (
                feature_chunks[0]
                if len(feature_chunks) == 1
                else torch.cat(feature_chunks, dim=1)
            )
        feature_chunks = None
        context_feature_bytes = max(
            context_feature_bytes, _tensor_bytes([target_hidden])
        )
    _crop_to(past_key_values_target, num_input_tokens)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None
    if len({p.device for p in target.parameters()}) < 2:
        # One device: per-step boundaries already resolve the peak exactly, and
        # a read per layer per decode step would show up in the reported tpot.
        for handle in watch_handles:
            handle.remove()
        watch_handles = []

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    accepted_lengths = []
    proposed_lengths = []
    draft_activation_bytes = 0
    start = num_input_tokens
    stop_tokens = (
        torch.tensor(stop_token_ids, dtype=output_ids.dtype, device=output_ids.device)
        if stop_token_ids is not None
        else None
    )

    stopped = stop_tokens is not None and torch.isin(output_ids[:, start], stop_tokens).any()
    while start + 1 < max_length and not stopped:
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start : start + verify_size].clone()
        block_position_ids = position_ids[:, start : start + verify_size]
        tracker.context(
            decode_token=start - num_input_tokens, draft_step=len(acceptance_lengths)
        )
        if verify_size > 1:
            before_draft_bytes = _all_device_live() if profile_draft_memory else 0
            tracker.boundary("decode: draft forward")
            noise_embedding = _raw_input_embeddings(
                target,
                block_output_ids,
                float(_draft_value(model.config, "input_embedding_scale", 1.0)),
            )
            with draft_forward_timer:
                draft_hidden = model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, start - target_hidden.shape[1] : start + verify_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                )[:, 1 - verify_size :, :]
            _crop_to(past_key_values_draft, start)
            draft_peak_bytes = tracker.boundary("decode: draft logits")
            if profile_draft_memory:
                # Subtract whatever the call left behind — the first draft call
                # populates the whole draft KV cache, which is persistent, not
                # activation. Netting it out keeps this term transient-only so
                # it does not double-count draft_cache_bytes.
                resident = max(before_draft_bytes, _all_device_live())
                draft_activation_bytes = max(
                    draft_activation_bytes, draft_peak_bytes - resident
                )
            if isinstance(model, DFlash2DraftModel):
                draft_tokens, draft_indices, draft_probs = model.propose(
                    draft_hidden,
                    block_output_ids[:, 0],
                    _output_head(target),
                    temperature,
                )
                block_output_ids[:, 1:] = draft_tokens
            else:
                draft_logits = model.compute_logits(
                    draft_hidden, _output_head(target)
                ).to(draft_hidden.device)
                if temperature > 0:
                    draft_probs = _sampling_probs(
                        draft_logits, temperature, top_p, top_k
                    )
                    block_output_ids[:, 1:] = _sample_probs(draft_probs)
                    draft_indices = None
                else:
                    block_output_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)
        tracker.boundary("decode: target verify")
        with target_forward_timer:
            output, selected, accounted = _target_step(
                target,
                tap,
                model.target_layer_ids,
                verify_size > 1,
                input_ids=block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
            )

        if temperature > 0:
            target_probs = _sampling_probs(output.logits, temperature, top_p, top_k)
            if verify_size > 1:
                acceptance_length, bonus = _rejection_sample(
                    block_output_ids[:, 1:],
                    target_probs,
                    draft_probs,
                    draft_indices,
                )
            else:
                acceptance_length = 0
                bonus = _sample_probs(target_probs[:, -1])[0]
        else:
            posterior = torch.argmax(output.logits, dim=-1)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            bonus = posterior[:, acceptance_length][0]
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = bonus
        produced = min(acceptance_length + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_indices = torch.isin(
                output_ids[0, start + 1 : start + produced + 1], stop_tokens
            ).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                produced = stop_indices[0].item() + 1
                stopped = True
        start += produced
        _crop_to(past_key_values_target, start)
        acceptance_lengths.append(produced)
        accepted_lengths.append(min(acceptance_length, produced))
        proposed_lengths.append(verify_size - 1)

        if verify_size > 1:
            hidden_states_bytes = max(hidden_states_bytes, _tensor_bytes(accounted))
            tracker.boundary("decode: context-feature concat")
            with context_feature_timer:
                target_hidden = torch.cat(selected, dim=-1)[:, :produced, :]
            context_feature_bytes = max(
                context_feature_bytes, _tensor_bytes([target_hidden])
            )

    tracker.finish()
    for handle in watch_handles:
        handle.remove()

    output_ids = output_ids[:, :min(start + 1, max_length)]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    end_time = _cuda_time()
    total_decode_time = end_time - decode_start
    num_proposed = sum(proposed_lengths)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        total_latency=end_time - prefill_start,
        decode_latency=total_decode_time,
        acceptance_lengths=acceptance_lengths,
        accepted_lengths=accepted_lengths,
        proposed_lengths=proposed_lengths,
        num_accepted_tokens=sum(accepted_lengths),
        num_proposed_tokens=num_proposed,
        num_verify_steps=len(acceptance_lengths),
        num_draft_calls=sum(1 for n in proposed_lengths if n > 0),
        num_full_gamma_proposals=sum(
            1 for n in proposed_lengths if n == block_size - 1
        ),
        gamma=block_size - 1,
        # Interval-resolved: the largest simultaneous total, not the sum of
        # per-device maxima that never coexisted. See PeakTracker.
        peak_memory_bytes=tracker.peak_bytes,
        peak_memory_sum_device_maxima_bytes=tracker.sum_of_device_maxima_bytes,
        peak_memory_per_device_bytes=tracker.peak_per_device,
        peak_site=tracker.peak_site,
        peak_memory_reserved_bytes=_all_device_peak(reserved=True),
        draft_activation_bytes=draft_activation_bytes if profile_draft_memory else None,
        draft_cache_bytes=_cache_bytes(past_key_values_draft),
        target_cache_bytes=_cache_bytes(past_key_values_target),
        target_hidden_states_bytes=hidden_states_bytes,
        context_feature_bytes=context_feature_bytes,
        draft_forward_s=draft_forward_timer.seconds,
        context_feature_s=context_feature_timer.seconds,
        target_forward_s=target_forward_timer.seconds,
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        is_causal = getattr(config, "is_causal", None)
        self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if (
            attention_mask is None
            and (self.is_causal or self.sliding_window is not None)
        ):
            attention_mask = _attention_mask(
                q,
                k,
                is_causal=self.is_causal,
                sliding_window=self.sliding_window,
            )
        attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_conv = None
        self.mlp_conv = None

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        if attention_kernel is not None:
            hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        hidden_states = residual + hidden_states
        return hidden_states


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


class CandidateSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        rank = int(_draft_value(config, "selector_rank"))
        self.top_k = int(_draft_value(config, "selector_top_k"))
        self.predecessor_codebook = nn.Embedding(config.vocab_size, rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, rank)
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)

    def select(self, hidden, logits, anchor_ids, temperature):
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path, q_rows = [], []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            if temperature > 0:
                q = _sampling_probs(scores[:, None], temperature)[:, 0]
                index = _sample_probs(q)
                q_rows.append(q)
            else:
                index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return (
            torch.stack(path, dim=1),
            candidates,
            torch.stack(q_rows, dim=1) if q_rows else None,
        )


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules: ClassVar[list[str]] = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = _draft_value(
            config,
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = int(_draft_value(config, "block_size", 16))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden, output_head):
        logits = output_head(hidden)
        logits = logits * float(_draft_value(self.config, "output_multiplier", 1.0))
        softcap = _draft_value(self.config, "final_logit_softcapping")
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits

    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


class DFlash2DraftModel(DFlashDraftModel):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("key_mapping", {
            f"candidate_selector.{name}": f"candidate_selector.{name}.weight"
            for name in ("predecessor_codebook", "successor_codebook")
        })
        return super().from_pretrained(*args, **kwargs)

    def __init__(self, config) -> None:
        super().__init__(config)
        kernel_size = int(_draft_value(config, "conv_kernel_size"))
        group_size = int(_draft_value(config, "conv_group_size"))
        for layer in self.layers:
            layer.attention_conv = GroupedDynamicCausalConv(
                config.hidden_size, kernel_size, group_size
            )
            layer.mlp_conv = GroupedDynamicCausalConv(
                config.hidden_size, kernel_size, group_size
            )
        self.candidate_selector = CandidateSelector(config)
        self.post_init()

    def propose(self, hidden, anchor_ids, output_head, temperature):
        return self.candidate_selector.select(
            hidden,
            self.compute_logits(hidden, output_head),
            anchor_ids,
            temperature,
        )
