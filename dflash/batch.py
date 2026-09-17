"""Batched DFlash and baseline decoding, for the batch-size sweep.

``dflash_generate`` runs one request. Batching it is not a matter of widening
the batch dimension: every row accepts a different number of draft tokens per
step, so after one verify the rows sit at different cache lengths, different
positions, and feed the drafter different numbers of new context rows. This
module keeps those per row.

* Caches are preallocated ``(B, capacity, H_kv, d)`` storages with one valid
  length per row. A forward writes each row at its own offset; attention is
  FlashAttention-2 varlen with ``seqused_k``, which reads each row's keys up to
  its own length and aligns the causal mask to that end. A rejected block needs
  no crop -- the next step overwrites it. Nothing is re-concatenated per step,
  which ``DynamicCache`` does to the whole KV cache (BATCH_PROFILING_PLAN.md
  §2.4), and no mask is passed to SDPA, whose masked kernel measured 4x slower
  than flash on a 32k single-token decode.
* Prefill runs one request at a time straight into its row, followed by the
  drafter's own prefill (its first forward over the whole prompt). Only the
  decode loop is batched. See BATCH_PROFILING_PLAN.md §4.1.
* Qwen3.5's gated-delta-rule layers are rolled back exactly. transformers'
  cache ``crop`` restores the conv window but never the recurrent state, so
  every rejected token used to stay folded into it (§2.2). Here the verify's
  per-layer inputs are kept and the accepted prefix is replayed from the
  pre-verify state, with ``g = beta = 0`` on the rejected steps -- an identity
  update -- so rows with different acceptance share one call.
* The GDN conv buffer is not recorded during prefill, which needs no rollback;
  recording it held 384 KiB per prompt token until the first verify (§2.3).
* One host sync per decode step (the acceptance read-back). Lengths the
  kernels need on the host are tracked there rather than read back.
"""

from __future__ import annotations

import time
import weakref

import torch
import torch.nn.functional as F

from . import model as model_module

ATTN_NAME = "dflash_batch"
_ALIGN = 16

# The target's attention layers are reached through HF's attention registry,
# which passes no cache handle. Like model._DRAFT_STAGES this is set by the
# engine for the duration of a batch and cleared afterwards.
_ACTIVE: "TargetCache | None" = None


def _round_up(value: int, multiple: int = _ALIGN) -> int:
    return -(-value // multiple) * multiple


def _storage_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


# ---------------------------------------------------------------------------
# Attention over per-row storages
# ---------------------------------------------------------------------------


def attend(query, keys, values, used, max_used: int, *, causal: bool, window, scale,
           meta: dict | None = None, uniform: bool = False):
    """Attention of a query block against each row's first ``used[b]`` keys.

    ``query`` is ``(R, H, q, d)`` as HF hands it over, ``keys``/``values`` the
    rows' storages ``(R, capacity, H_kv, d)``. The query block is the last ``q``
    of a row's visible keys, so a causal mask is aligned to ``used[b]``, and a
    window counts back from each query's own position. ``uniform`` says every
    row has the same ``used``. Returns ``(R, q, H, d)``.

    Three kernels compute the same thing and none wins everywhere -- measured on
    the A6000, dense flash (split-KV) is 3x faster than varlen flash for a single
    decode row, the masked memory-efficient kernel beats varlen for sixteen
    16-token rows, and varlen is the only fast one for ragged rows at a long
    context. The first call of each shape times the applicable ones and keeps
    the fastest (see ``tuned_kernels``), so each decoding mode is measured on the
    best kernel available to it rather than on one that favours the other.
    """
    rows, heads, q_len, dim = query.shape
    causal = causal and q_len > 1
    meta = {} if meta is None else meta
    args = (query, keys, values, used, max_used, causal, window, scale, meta)
    if not query.is_cuda:
        return _masked(*args)
    options = ["varlen"]
    if uniform and window is None and (q_len == 1 or q_len == max_used or not causal):
        options.insert(0, "dense")
    if q_len <= 64:
        options.append("masked")
    key = (rows, q_len, heads, keys.shape[2], dim, causal, window,
           _round_up(max_used, 1024), tuple(options))
    choice = _TUNED.get(key)
    if choice is None:
        choice = _TUNED[key] = _tune(options, args)
    return _KERNELS[choice](*args)


_TUNED: dict = {}


def tuned_kernels() -> dict:
    """Which kernel each attention shape runs on, for the record."""
    return {
        f"rows={k[0]} q={k[1]} heads={k[2]}/{k[3]} d={k[4]} causal={k[5]} "
        f"window={k[6]} keys<={k[7]}": v
        for k, v in _TUNED.items()
    }


def _tune(options, args) -> str:
    if len(options) == 1:
        return options[0]
    best, best_time = options[0], float("inf")
    for name in options:
        kernel = _KERNELS[name]
        kernel(*args)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            kernel(*args)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if elapsed < best_time:
            best, best_time = name, elapsed
    return best


def _dense(query, keys, values, used, max_used, causal, window, scale, meta):
    # Only offered when every row sees exactly max_used keys and the causal
    # mask, if any, is square -- where SDPA's top-left alignment is correct.
    heads, kv_heads = query.shape[1], keys.shape[2]
    out = F.scaled_dot_product_attention(
        query,
        keys[:, :max_used].transpose(1, 2),
        values[:, :max_used].transpose(1, 2),
        is_causal=causal, scale=scale, enable_gqa=heads != kv_heads,
    )
    return out.transpose(1, 2)


def _masked(query, keys, values, used, max_used, causal, window, scale, meta):
    rows, heads, q_len, dim = query.shape
    kv_heads = keys.shape[2]
    group = heads // kv_heads
    # One step's meta is shared by every layer, and the drafter mixes full and
    # sliding layers of the same shape, so the mask kind is part of the key.
    key = ("mask", rows, q_len, group, max_used, causal, window)
    if key not in meta:
        slots = torch.arange(max_used, device=query.device)
        positions = used[:, None] - q_len + torch.arange(q_len, device=query.device)
        visible = slots[None, None, :] < used[:, None, None]
        if causal:
            visible = visible & (slots[None, None, :] <= positions[:, :, None])
        if window is not None:
            visible = visible & (positions[:, :, None] - slots[None, None, :] < window)
            if not causal:
                visible = visible & (slots[None, None, :] - positions[:, :, None] < window)
        mask = visible.expand(rows, q_len, max_used)[:, None].repeat(1, 1, group, 1)
        meta[key] = torch.zeros(mask.shape, dtype=query.dtype, device=query.device).masked_fill(
            ~mask, float("-inf")
        )
    folded = query.reshape(rows, kv_heads, group * q_len, dim)
    out = F.scaled_dot_product_attention(
        folded,
        keys[:, :max_used].transpose(1, 2),
        values[:, :max_used].transpose(1, 2),
        attn_mask=meta[key], scale=scale,
    )
    return out.reshape(rows, heads, q_len, dim).transpose(1, 2)


def _varlen(query, keys, values, used, max_used, causal, window, scale, meta):
    """FlashAttention-2 varlen over the rows' storages, keys cut at seqused_k.

    The query heads are not folded into the query length here: with only the
    KV heads left to parallelise over, the kernel ran 4-7x slower.
    """
    rows, heads, q_len, dim = query.shape
    capacity, kv_heads = keys.shape[1], keys.shape[2]
    key = ("varlen", rows, q_len, capacity)
    if key not in meta:
        steps = torch.arange(rows + 1, device=query.device, dtype=torch.int32)
        meta[key] = (steps * q_len, steps * capacity, used.to(torch.int32))
    cu_q, cu_k, used32 = meta[key]
    left = right = None
    if window is not None:
        left = window - 1
        right = 0 if causal else window - 1
    out = torch.ops.aten._flash_attention_forward(
        query.transpose(1, 2).reshape(rows * q_len, heads, dim),
        keys.reshape(rows * capacity, kv_heads, dim),
        values.reshape(rows * capacity, kv_heads, dim),
        cu_q, cu_k, q_len, max_used, 0.0, causal, False,
        scale=scale, window_size_left=left, window_size_right=right,
        seqused_k=used32,
    )[0]
    return out.view(rows, q_len, heads, dim)


_KERNELS = {"dense": _dense, "masked": _masked, "varlen": _varlen}


def _target_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    cache = _ACTIVE
    if cache is None:
        # No batched cache (e.g. a plain cacheless forward): ordinary causal SDPA.
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=query.shape[2] > 1, scale=scaling,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return out.transpose(1, 2), None
    return cache.attend(module.layer_idx, query, scaling), None


def use_batch_attention(target, enabled: bool = True) -> None:
    """Route the target's attention through the batched path, or back to SDPA."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_NAME, _target_attention)
    name = ATTN_NAME if enabled else "sdpa"
    for module in target.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = name


# ---------------------------------------------------------------------------
# Gated-delta-rule capture. The GDN layer calls this module-level function by
# name, so wrapping the attribute is enough to see its inputs.
# ---------------------------------------------------------------------------

_CHUNK = None


def _install_gdn_capture() -> None:
    global _CHUNK
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5
    except ImportError:  # pragma: no cover - older transformers
        return
    if _CHUNK is not None:
        return
    _CHUNK = qwen3_5.torch_chunk_gated_delta_rule

    def capturing(query, key, value, g=None, beta=None, **kwargs):
        cache = _ACTIVE
        if cache is not None and cache.decode:
            # The torch fallback pads a sequence to a whole chunk and then runs a
            # Python loop over the chunk's rows: a 16-token verify at the default
            # chunk of 64 spends three quarters of that loop on padding. The
            # chunked recurrence is exact for any chunk size.
            kwargs["chunk_size"] = cache.gdn_chunk
            if cache.capture:
                cache.captured[cache.gdn_layer] = (query, key, value, g, beta)
        return _CHUNK(query, key, value, g=g, beta=beta, **kwargs)

    qwen3_5.torch_chunk_gated_delta_rule = capturing


class _GdnLayerView:
    """What a Qwen3.5 GDN layer reads off ``cache.layers[i]``.

    Holds the cache weakly: ``cache.layers`` holds the view, and a strong
    reference back would make a cycle that keeps every finished batch's KV and
    GDN state on the card until the cyclic GC happens to run.
    """

    def __init__(self, cache: "TargetCache", index: int):
        self._cache, self._index = weakref.proxy(cache), index

    @property
    def record_past(self) -> bool:
        # In decode always take the record path, so the conv input comes back
        # through update_conv_state and can be rolled back per row.
        return self._cache.decode

    @property
    def conv_states(self):
        return [self._cache.conv[self._index][self._cache.rows]]

    @property
    def recurrent_states(self):
        return [self._cache.rec[self._index][self._cache.rows]]


class TargetCache:
    """Per-row target state: static KV for attention layers, GDN conv/recurrent state.

    Duck-types the parts of transformers' ``Cache`` the Qwen3 and Qwen3.5
    decoder layers call. Mask creation is skipped for an attention
    implementation with no registered mask function, so ``update`` and the GDN
    methods are the whole interface.
    """

    def __init__(self, config, batch: int, capacity: int, dtype, device):
        text = getattr(config, "text_config", None) or config
        self.types = list(
            getattr(text, "layer_types", None) or ["full_attention"] * text.num_hidden_layers
        )
        heads = text.num_attention_heads
        kv_heads = text.num_key_value_heads
        head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
        self.batch, self.capacity = batch, capacity
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.conv: dict[int, torch.Tensor] = {}
        self.rec: dict[int, torch.Tensor] = {}
        self.layers: list = []
        element = torch.tensor([], dtype=dtype).element_size()
        for index, kind in enumerate(self.types):
            if kind == "linear_attention":
                key_dim = text.linear_num_key_heads * text.linear_key_head_dim
                value_dim = text.linear_num_value_heads * text.linear_value_head_dim
                self.conv[index] = torch.zeros(
                    batch, 2 * key_dim + value_dim, text.linear_conv_kernel_dim,
                    dtype=dtype, device=device,
                )
                self.rec[index] = torch.zeros(
                    batch, text.linear_num_value_heads, text.linear_key_head_dim,
                    text.linear_value_head_dim, dtype=torch.float32, device=device,
                )
                self.layers.append(_GdnLayerView(self, index))
            else:
                shape = (batch, capacity, kv_heads, head_dim)
                self.keys[index] = torch.zeros(shape, dtype=dtype, device=device)
                self.values[index] = torch.zeros(shape, dtype=dtype, device=device)
                self.layers.append(None)
        self.kv_bytes_per_token = len(self.keys) * 2 * kv_heads * head_dim * element
        self.lengths = torch.zeros(batch, dtype=torch.long, device=device)
        # Per-forward state, set by prefill_row / decode_step.
        self.rows = slice(None)
        self.prefill = True
        self.decode = False
        self.capture = False
        self._row_offset = torch.arange(batch, device=device) * capacity
        self._start = 0
        self._flat = None
        self._used = None
        self._max_used = 0
        self._uniform = False
        self._meta: dict = {}
        # Chunk size of the GDN chunked recurrence in decode (verify and replay).
        self.gdn_chunk = 16
        # Rollback scratch for one verify.
        self.gdn_layer = None
        self.conv_full: dict[int, torch.Tensor] = {}
        self.rec_new: dict[int, torch.Tensor] = {}
        self.captured: dict[int, tuple] = {}

    # -- per-forward setup --------------------------------------------------

    def prefill_row(self, row: int, length: int) -> None:
        self.rows = slice(row, row + 1)
        self.prefill, self.decode, self.capture = True, False, False
        self._used = torch.full((1,), length, dtype=torch.long, device=self.lengths.device)
        self._max_used = length
        self._uniform = True
        self._meta = {}

    def decode_step(self, q_len: int, capture: bool, max_length: int, min_length: int) -> None:
        """Every row writes ``q_len`` tokens at its own length.

        ``max_length``/``min_length`` bound the rows' lengths, tracked on the
        host by the caller so that setting up a step never waits for the GPU.
        """
        self.rows = slice(None)
        self.prefill, self.decode, self.capture = False, True, capture
        self._used = self.lengths + q_len
        self._max_used = max_length + q_len
        self._uniform = max_length == min_length
        self._start = min_length
        if not self._uniform:
            write = self.lengths[:, None] + torch.arange(q_len, device=self.lengths.device)
            self._flat = (self._row_offset[:, None] + write).reshape(-1)
        self._meta = {}

    # -- attention layers ---------------------------------------------------

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Write this forward's K/V into the rows' storage.

        Kept to one cheap op per tensor: at batch 1 the decode step is bound by
        how fast the host issues kernels, so per-layer indexing overhead is
        wall-clock time. The return value is HF's contract only -- attend()
        reads the storage, so the inputs are handed back untouched.
        """
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        new_keys, new_values = key_states.transpose(1, 2), value_states.transpose(1, 2)
        if self.prefill:
            length = key_states.shape[2]
            keys[self.rows, :length].copy_(new_keys)
            values[self.rows, :length].copy_(new_values)
        elif self._uniform:
            end = self._start + key_states.shape[2]
            keys[:, self._start : end].copy_(new_keys)
            values[:, self._start : end].copy_(new_values)
        else:
            tail = keys.shape[2:]
            keys.view(-1, *tail).index_copy_(0, self._flat, new_keys.reshape(-1, *tail))
            values.view(-1, *tail).index_copy_(0, self._flat, new_values.reshape(-1, *tail))
        return key_states, value_states

    def attend(self, layer_idx, query, scale):
        return attend(
            query, self.keys[layer_idx][self.rows], self.values[layer_idx][self.rows],
            self._used, self._max_used, causal=True, window=None, scale=scale,
            meta=self._meta, uniform=self._uniform,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._max_used

    # -- gated-delta-rule layers -------------------------------------------

    def has_previous_state(self, layer_idx=None, state_idx=None) -> bool:
        return self.decode

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
        self.gdn_layer = layer_idx
        window = self.conv[layer_idx].shape[-1]
        if not self.decode:
            # Prefill: keep the last `window` inputs and nothing else. A copy,
            # not a view, so the prompt-length buffer is freed with the forward.
            tail = conv_states[..., -window:]
            if tail.shape[-1] < window:
                tail = F.pad(tail, (window - tail.shape[-1], 0))
            self.conv[layer_idx][self.rows] = tail
            return conv_states
        full = torch.cat([self.conv[layer_idx], conv_states], dim=-1)
        self.conv_full[layer_idx] = full
        return full

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx=0, **kwargs):
        if not self.decode:
            self.rec[layer_idx][self.rows] = recurrent_states
        else:
            # Kept apart: the pre-verify state is what a rollback replays from.
            self.rec_new[layer_idx] = recurrent_states
        return recurrent_states

    def commit(self, kept: torch.Tensor, max_kept: int | None = None) -> None:
        """Advance every row by ``kept[b]`` tokens of the verify just run.

        Attention layers need only the length: whatever lies past it is never
        read and is overwritten next step. GDN layers get the conv window that
        ends at the last kept token, and a recurrent state replayed over exactly
        the kept prefix. ``max_kept`` (the largest ``kept``, known on the host)
        lets the replay skip the steps no row kept; the torch fallback's cost is
        a Python loop over them.
        """
        for index, full in self.conv_full.items():
            window = self.conv[index].shape[-1]
            gather = (kept[:, None, None] + torch.arange(window, device=kept.device)).expand(
                -1, full.shape[1], -1
            )
            self.conv[index].copy_(torch.gather(full, 2, gather))
        for index, new_state in self.rec_new.items():
            captured = self.captured.get(index)
            if captured is None:
                # Single-token verify (the baseline): a row either keeps its one
                # token or, once finished, nothing.
                keep = (kept > 0)[:, None, None, None]
                self.rec[index].copy_(torch.where(keep, new_state, self.rec[index]))
                continue
            query, key, value, g, beta = captured
            span = query.shape[1] if max_kept is None else max_kept
            if span == 0:
                continue
            query, key, value, g, beta = (x[:, :span] for x in (query, key, value, g, beta))
            steps = torch.arange(span, device=kept.device)
            valid = (steps[None, :] < kept[:, None])[..., None]          # (B, span, 1)
            _, state = _CHUNK(
                query, key, value,
                g=g * valid, beta=beta * valid,
                chunk_size=span,
                initial_state=self.rec[index],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            self.rec[index].copy_(state)
        self.lengths += kept
        self.conv_full.clear()
        self.rec_new.clear()
        self.captured.clear()

    # -- accounting ---------------------------------------------------------

    def kv_bytes(self) -> int:
        return _storage_bytes(list(self.keys.values()) + list(self.values.values()))

    def gdn_bytes(self) -> int:
        scratch = list(self.conv_full.values()) + list(self.rec_new.values())
        for item in self.captured.values():
            scratch.extend(item)
        return _storage_bytes(
            list(self.conv.values()) + list(self.rec.values()) + scratch
        )


class DraftCache:
    """Per-row draft KV with one trash slot for rows' padding.

    A draft call writes the context K/V of the tokens the last verify kept --
    a different count per row, padded to the largest -- then the noise block.
    Padded context entries are pointed at the trash slot (the last one, never
    inside any row's visible range), so a row's real entries are never
    clobbered.
    """

    def __init__(self, config, batch: int, capacity: int, dtype, device):
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        shape = (batch, capacity + 1, kv_heads, head_dim)
        layers = config.num_hidden_layers
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.trash = capacity
        self.lengths = torch.zeros(batch, dtype=torch.long, device=device)
        self._row_offset = torch.arange(batch, device=device) * (capacity + 1)
        self.rows = slice(None)
        self._contiguous = None
        self._flat = None
        self._used = None
        self._max_used = 0
        self._uniform = False
        self._meta: dict = {}

    def prefill_row(self, row: int, length: int) -> None:
        """One row's first call: ``length`` tokens written from position 0."""
        self.rows = slice(row, row + 1)
        self._contiguous = length
        self._flat = None
        self._used = torch.full((1,), length, dtype=torch.long, device=self.lengths.device)
        self._max_used = length
        self._uniform = True
        self._meta = {}

    def decode_step(self, write: torch.Tensor, used: torch.Tensor, max_used: int,
                    uniform: bool) -> None:
        """``write``: (B, n) slots per row; ``used``: (B,) keys a row may see,
        the noise block being the last of them; ``uniform``: all rows equal."""
        self.rows = slice(None)
        self._contiguous = None
        self._flat = (self._row_offset[:, None] + write).reshape(-1)
        self._used = used
        self._max_used = max_used
        self._uniform = uniform
        self._meta = {}

    def attend(self, layer_idx, query, key, value, *, is_causal, sliding_window, scaling):
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        with model_module._stage("cache_update"):
            if self._contiguous is not None:
                keys[self.rows, : self._contiguous].copy_(key.transpose(1, 2))
                values[self.rows, : self._contiguous].copy_(value.transpose(1, 2))
            else:
                # Padded context entries all point at the trash slot; which of
                # them lands there last does not matter.
                tail = keys.shape[2:]
                keys.view(-1, *tail).index_copy_(0, self._flat, key.transpose(1, 2).reshape(-1, *tail))
                values.view(-1, *tail).index_copy_(0, self._flat, value.transpose(1, 2).reshape(-1, *tail))
        with model_module._stage("attention"):
            return attend(
                query, keys[self.rows], values[self.rows], self._used, self._max_used,
                causal=is_causal, window=sliding_window, scale=scaling, meta=self._meta,
                uniform=self._uniform,
            )

    def kv_bytes(self) -> int:
        return _storage_bytes(self.keys + self.values)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _sync(cuda: bool) -> None:
    if cuda:
        torch.cuda.synchronize()


class BatchEngine:
    """Runs one batch of requests, DFlash (block > 1) or baseline (block 1).

    Both modes go through the same caches, attention kernels and bookkeeping,
    so a same-batch speedup compares decoding strategies and nothing else.
    """

    def __init__(
        self, target, draft, *, eos_ids, eos_mode: str = "suppress",
        draft_stages: bool = True,
    ):
        if isinstance(draft, model_module.DFlash2DraftModel):
            raise NotImplementedError("the batched engine supports DFlashDraftModel only")
        if eos_mode not in ("suppress", "ignore"):
            raise ValueError("eos_mode is 'suppress' or 'ignore'")
        self.target, self.draft = target, draft
        self.device = next(draft.parameters()).device
        self.dtype = next(draft.parameters()).dtype
        self.cuda = self.device.type == "cuda"
        self.eos = torch.tensor(sorted(set(eos_ids)), device=self.device)
        self.eos_mode = eos_mode
        self.head = model_module._output_head(target)
        self.embed_scale = float(
            model_module._draft_value(draft.config, "input_embedding_scale", 1.0)
        )
        self.mask_token = int(draft.mask_token_id)
        self.layer_ids = list(draft.target_layer_ids)
        self.num_target_layers = int(
            model_module._draft_value(draft.config, "num_target_layers")
        )
        self.draft_weight_bytes = model_module.module_bytes(draft)
        self.target_weight_bytes = model_module.module_bytes(target)
        self.draft_stages = draft_stages
        use_batch_attention(target, True)
        _install_gdn_capture()

    def _pick(self, logits):
        """Greedy token with EOS suppressed, and whether EOS would have won."""
        raw = logits.argmax(-1)
        hit = torch.isin(raw, self.eos)
        if self.eos_mode == "ignore":
            return raw, hit
        return logits.index_fill(-1, self.eos, float("-inf")).argmax(-1), hit

    def _mark(self):
        if not self.cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _elapsed(start, end) -> float | None:
        if start is None or end is None:
            return None
        return start.elapsed_time(end) / 1000.0

    @torch.inference_mode()
    def run_batch(self, prompts: list[torch.Tensor], num_tokens: int, block_size: int,
                  on_step=None) -> dict:
        """Generate exactly ``num_tokens`` per prompt (the prefill token included).

        ``on_step(step)`` is called before every decode step and with ``None``
        once the loop ends -- the kernel profiler's hook.
        """
        global _ACTIVE
        batch, q_len = len(prompts), block_size
        dflash = block_size > 1
        device = self.device
        lengths = [int(p.numel()) for p in prompts]
        capacity = _round_up(max(lengths) + num_tokens + 2 * q_len)
        live: dict = {"selected": None, "feature": None}
        caches: dict = {"target": None, "draft": None}

        def probe() -> dict:
            target, draft = caches["target"], caches["draft"]
            return {
                "target_kv_bytes": target.kv_bytes() if target else 0,
                "gdn_state_bytes": target.gdn_bytes() if target else 0,
                "draft_kv_bytes": draft.kv_bytes() if draft else 0,
                "selected_hidden_bytes": model_module._tensor_bytes(live["selected"] or ()),
                "context_feature_bytes": model_module._tensor_bytes(
                    () if live["feature"] is None else (live["feature"],)
                ),
                "draft_weight_bytes": self.draft_weight_bytes,
                "allocated_bytes": model_module._all_device_live(),
            }

        phases = model_module.PhaseMemory(self.cuda, probe)
        tracker = model_module.PeakTracker(self.cuda, on_interval=phases.interval)

        def phase(label: str) -> None:
            tracker.boundary(label)
            phases.begin(label)

        stages = model_module._DraftStages(self.cuda and self.draft_stages and dflash)
        model_module._DRAFT_STAGES = stages if stages.enabled else None
        try:
            phase("setup: allocate caches")
            tcache = TargetCache(self.target.config, batch, capacity, self.dtype, device)
            caches["target"] = tcache
            dcache = (
                DraftCache(self.draft.config, batch, capacity + q_len, self.dtype, device)
                if dflash else None
            )
            caches["draft"] = dcache
            _ACTIVE = tcache
            tap = (
                model_module.HiddenStateTap(
                    self.target, self.layer_ids, self.num_target_layers, device=device
                )
                if dflash else None
            )

            # ---------------- prefill, one request at a time ----------------
            first = torch.empty(batch, dtype=torch.long, device=device)
            first_eos = torch.zeros(batch, dtype=torch.bool, device=device)
            proposals = (
                torch.empty(batch, q_len - 1, dtype=torch.long, device=device)
                if dflash else None
            )
            prefill_s, drafter_prefill_s = [], []
            for row, ids in enumerate(prompts):
                _sync(self.cuda)
                started = time.perf_counter()
                length = lengths[row]
                tcache.prefill_row(row, length)
                phase("prefill: target forward")
                kwargs = dict(
                    input_ids=ids[None],
                    position_ids=torch.arange(length, device=device)[None],
                    past_key_values=tcache, use_cache=True, logits_to_keep=1,
                )
                selected = None
                if tap is not None:
                    with tap:
                        output = self.target(**kwargs)
                        selected = tap.states()
                    live["selected"] = selected
                else:
                    output = self.target(**kwargs)
                phase("prefill: first token")
                token, hit = self._pick(output.logits[:, -1])
                first[row] = token[0]
                first_eos[row] = hit[0]
                del output
                tcache.lengths[row] = length
                if dflash:
                    phase("prefill: context-feature build")
                    feature = torch.cat(selected, dim=-1)
                    selected = None
                    live["selected"], live["feature"] = None, feature
                    _sync(self.cuda)
                    drafter_started = time.perf_counter()
                    phase("prefill: drafter prefill")
                    if stages.enabled:
                        stages.bucket = stages.first
                    block = torch.full((1, q_len), self.mask_token, dtype=torch.long, device=device)
                    block[0, 0] = first[row]
                    dcache.prefill_row(row, length + q_len)
                    hidden = self.draft(
                        target_hidden=feature,
                        noise_embedding=model_module._raw_input_embeddings(
                            self.target, block, self.embed_scale
                        ),
                        position_ids=torch.arange(length + q_len, device=device)[None],
                        past_key_values=dcache,
                        use_cache=True,
                    )[:, 1:]
                    proposals[row] = self._pick(self.draft.compute_logits(hidden, self.head))[0][0]
                    dcache.lengths[row] = length
                    feature = hidden = None
                    live["feature"] = None
                    _sync(self.cuda)
                    drafter_prefill_s.append(time.perf_counter() - drafter_started)
                _sync(self.cuda)
                prefill_s.append(time.perf_counter() - started)

            # ---------------- batched decode ----------------
            generated = [1] * batch
            active = [num_tokens > 1] * batch
            row_lengths = list(lengths)           # host copy of tcache.lengths
            gen = torch.ones(batch, dtype=torch.long, device=device)
            tokens = [[value] for value in first.tolist()]
            natural_eos = [0 if value else None for value in first_eos.tolist()]
            kept_log: list[list[int]] = [[] for _ in range(batch)]
            finished_at: list[int | None] = [None] * batch
            offsets = torch.arange(q_len, device=device)
            last = first
            steps, events = [], []
            steady_calls = 0
            _sync(self.cuda)
            decode_started = previous = time.perf_counter()
            step = 0
            while any(active):
                if on_step is not None:
                    on_step(step)
                active_at_start = sum(active)
                block = torch.cat([last[:, None], proposals], dim=1) if dflash else last[:, None]
                mark = {"verify": self._mark()}
                phase("decode: target verify")
                tcache.decode_step(
                    q_len, capture=dflash,
                    max_length=max(row_lengths), min_length=min(row_lengths),
                )
                kwargs = dict(
                    input_ids=block,
                    position_ids=tcache.lengths[:, None] + offsets,
                    past_key_values=tcache, use_cache=True, logits_to_keep=0,
                )
                if tap is not None:
                    with tap:
                        output = self.target(**kwargs)
                        selected = tap.states()
                    live["selected"] = selected
                else:
                    output = self.target(**kwargs)
                mark["accept"] = self._mark()
                phase("decode: accept/rollback")
                posterior, hit = self._pick(output.logits)
                del output
                if dflash:
                    accepted = (block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
                else:
                    accepted = torch.zeros(batch, dtype=torch.long, device=device)
                kept = (accepted + 1) * (gen < num_tokens)
                gen += torch.minimum(kept, num_tokens - gen)
                bonus = posterior.gather(1, accepted[:, None])[:, 0]
                previous_lengths = tcache.lengths.clone()
                host = torch.cat(
                    [kept[:, None], block, bonus[:, None], hit.long()], dim=1
                ).tolist()
                now = time.perf_counter()
                # After the read-back, so the GDN replay can stop at the longest
                # kept prefix instead of running the whole block.
                tcache.commit(kept, max(record[0] for record in host))
                mark["accept_end"] = self._mark()
                produced = 0
                for row in range(batch):
                    record = host[row]
                    row_lengths[row] += record[0]
                    if not active[row]:
                        continue
                    kept_row = record[0]
                    new = record[2 : 1 + kept_row] + [record[1 + q_len]]
                    hits = record[2 + q_len :]
                    room = num_tokens - generated[row]
                    take = min(kept_row, room)
                    if natural_eos[row] is None:
                        for k in range(take):
                            if hits[k]:
                                natural_eos[row] = generated[row] + k
                                break
                    tokens[row].extend(new[:take])
                    kept_log[row].append(kept_row)
                    generated[row] += take
                    produced += take
                    if generated[row] >= num_tokens:
                        active[row] = False
                        finished_at[row] = step
                steps.append({"wall_s": now - previous, "active": active_at_start, "tokens": produced})
                previous = now

                if dflash and any(active):
                    mark["ctx"] = self._mark()
                    phase("decode: context-feature build")
                    widest = max(record[0] for record in host)
                    feature = torch.cat(selected, dim=-1)[:, :widest]
                    selected = None
                    live["selected"], live["feature"] = None, feature
                    mark["draft"] = self._mark()
                    phase("decode: draft forward")
                    if stages.enabled:
                        stages.bucket = stages.steady
                    columns = torch.arange(widest, device=device)
                    context_slots = previous_lengths[:, None] + columns
                    writes = torch.where(
                        columns[None, :] < kept[:, None], context_slots, dcache.trash
                    )
                    noise_slots = tcache.lengths[:, None] + offsets
                    dcache.decode_step(
                        torch.cat([writes, noise_slots], dim=1),
                        tcache.lengths + q_len,
                        max(row_lengths) + q_len,
                        max(row_lengths) == min(row_lengths),
                    )
                    noise = torch.full((batch, q_len), self.mask_token, dtype=torch.long, device=device)
                    noise[:, 0] = bonus
                    hidden = self.draft(
                        target_hidden=feature,
                        noise_embedding=model_module._raw_input_embeddings(
                            self.target, noise, self.embed_scale
                        ),
                        position_ids=torch.cat([context_slots, noise_slots], dim=1),
                        past_key_values=dcache,
                        use_cache=True,
                    )[:, 1:]
                    phase("decode: draft logits")
                    proposals = self._pick(self.draft.compute_logits(hidden, self.head))[0]
                    feature = hidden = None
                    live["feature"] = None
                    dcache.lengths.copy_(tcache.lengths)
                    mark["draft_end"] = self._mark()
                    steady_calls += 1
                selected = None
                live["selected"] = None
                last = bonus
                events.append(mark)
                step += 1
            if on_step is not None:
                on_step(None)
            _sync(self.cuda)
            decode_s = previous - decode_started
            tracker.finish()
            phases.finish()
            for record, mark in zip(steps, events):
                record["verify_s"] = self._elapsed(mark["verify"], mark["accept"])
                record["accept_s"] = self._elapsed(mark["accept"], mark["accept_end"])
                record["ctx_s"] = self._elapsed(mark.get("ctx"), mark.get("draft"))
                record["draft_s"] = self._elapsed(mark.get("draft"), mark.get("draft_end"))
            return {
                "prompt_lengths": lengths,
                "capacity": capacity,
                "tokens": tokens,
                "kept": kept_log,
                "natural_eos_at": natural_eos,
                "finished_at_step": finished_at,
                "prefill_s": prefill_s,
                "drafter_prefill_s": drafter_prefill_s,
                "decode_s": decode_s,
                "steps": steps,
                "steady_draft_calls": steady_calls,
                "stage_s": stages.seconds(),
                "peak_bytes": tracker.peak_bytes,
                "peak_site": tracker.peak_site,
                "reserved_bytes": torch.cuda.memory_reserved() if self.cuda else 0,
                "phase_memory": phases.summary(),
                "target_kv_alloc_bytes": tcache.kv_bytes(),
                "draft_kv_alloc_bytes": dcache.kv_bytes() if dcache else 0,
                "gdn_state_bytes": tcache.gdn_bytes(),
                "target_weight_bytes": self.target_weight_bytes,
                "draft_weight_bytes": self.draft_weight_bytes,
                "attention_kernels": tuned_kernels(),
            }
        except torch.OutOfMemoryError as exc:
            exc.dflash_phase = tracker._label
            raise
        finally:
            _ACTIVE = None
            model_module._DRAFT_STAGES = None
