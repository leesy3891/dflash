"""Static multi-request DFlash. Each target/draft decode forward has global B rows.

Sequential unchunked prefill is common to AR and DFlash. Attention caches scatter
into preallocated storage; GDN verify is transactional, with mixer-only accepted
prefix replay. Finished rows remain in the kernels, but never commit output.
"""
from __future__ import annotations

import copy
import time
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from transformers import DynamicCache

from .model import HiddenStateTap, _decoder_layers, _raw_input_embeddings, _output_head


def _mask(query_pos, key_pos, valid, causal, window):
    distance = query_pos[:, :, None] - key_pos[:, None, :]
    mask = valid[:, None, :].expand(-1, query_pos.shape[1], -1)
    if causal:
        mask = mask & (distance >= 0)
    if window is not None:
        mask = mask & (distance < window)
        if not causal:
            mask = mask & (distance > -window)
    return mask[:, None]


def attend(query, keys, values, used, maximum, *, causal, window, scale, meta):
    """Reference ragged SDPA; mask cache includes every semantic parameter."""
    q = query.shape[-2]
    used = used.to(query.device)
    key = (causal, window, q, maximum, str(query.device), tuple(used.tolist()))
    if key not in meta:
        kpos = torch.arange(maximum, device=query.device)[None].expand(query.shape[0], -1)
        qpos = used[:, None] - q + torch.arange(q, device=query.device)
        meta[key] = _mask(qpos, kpos, kpos < used[:, None], causal, window)
    out = F.scaled_dot_product_attention(query, keys[:, :maximum].transpose(1, 2),
        values[:, :maximum].transpose(1, 2), attn_mask=meta[key], scale=scale, enable_gqa=True)
    return out.transpose(1, 2)


class TargetCache(DynamicCache):
    """HF GDN state plus preallocated attention KV, indexed by per-row length."""
    def __init__(self, config, lengths, capacity):
        super().__init__(config=config)
        self.lengths = list(lengths)
        self.capacity = capacity
        self.query_width = 0
        self.write_lengths = list(lengths)
        self.buffers = {}

    def prepare(self, lengths, width):
        self.lengths = list(lengths)
        self.write_lengths = list(lengths)
        self.query_width = width

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        b, h, q, d = key_states.shape
        if layer_idx not in self.buffers:
            self.buffers[layer_idx] = [torch.zeros((b, h, self.capacity, d),
                device=key_states.device, dtype=key_states.dtype) for _ in range(2)]
        positions = torch.tensor(self.write_lengths, device=key_states.device)[:, None] + torch.arange(q, device=key_states.device)
        indices = positions[:, None, :, None].expand(b, h, q, d)
        for dest, src in zip(self.buffers[layer_idx], (key_states, value_states)):
            dest.scatter_(2, indices, src)
        end = max(self.write_lengths) + q
        return tuple(t[:, :, :end] for t in self.buffers[layer_idx])

    def get_seq_length(self, layer_idx=0):
        return max(self.lengths, default=0)

    def get_mask_sizes(self, query_length, layer_idx=0):
        return max(self.lengths, default=0) + query_length, 0

    def import_row(self, cache, row):
        """Copy a completed single-row prefill; never share recurrent storage."""
        for i, source in enumerate(cache.layers):
            if hasattr(source, 'keys') and source.keys is not None:
                if i not in self.buffers:
                    _, h, _, d = source.keys.shape
                    self.buffers[i] = [torch.zeros((len(self.lengths), h, self.capacity, d),
                        device=source.keys.device, dtype=source.keys.dtype) for _ in range(2)]
                for dest, src in zip(self.buffers[i], (source.keys, source.values)):
                    dest[row:row+1, :, :src.shape[-2]].copy_(src)
            if hasattr(source, 'recurrent_states'):
                dest = self.layers[i]
                for name in ('conv_states', 'recurrent_states'):
                    for k, tensor in getattr(source, name).items():
                        if tensor is None:
                            continue
                        if getattr(dest, name)[k] is None:
                            getattr(dest, name)[k] = tensor.new_zeros((len(self.lengths), *tensor.shape[1:]))
                        getattr(dest, name)[k][row:row+1].copy_(tensor)
                for name in ('is_conv_states_initialized', 'is_recurrent_states_initialized', 'has_previous_state', 'conv_kernel_size'):
                    setattr(dest, name, copy.copy(getattr(source, name)))
                dest.device, dest.dtype = source.device, source.dtype

    def masks(self, device, layer_types, window=None):
        q = self.query_width
        positions = torch.tensor(self.lengths, device=device)[:, None] + torch.arange(q, device=device)
        kpos = torch.arange(max(self.lengths)+q, device=device)[None].expand(len(self.lengths), -1)
        valid = kpos < torch.tensor(self.lengths, device=device)[:, None] + q
        result = {'linear_attention': None}
        for kind in set(layer_types):
            if kind != 'linear_attention':
                result[kind] = _mask(positions, kpos, valid, True, window if kind == 'sliding_attention' else None)
        return result


class DraftCache(TargetCache):
    """Conditioning KV with physical sliding-window ring storage.

    Noise entries are temporary and overwritten next call; only context enters
    the persistent history. A scratch concatenation is used for attention, not
    a full-history cat/copy on every update.
    """
    def __init__(self, config, lengths, capacity):
        super().__init__(config, [0] * len(lengths), capacity)
        self.windows = [config.sliding_window if k == 'sliding_attention' else None
                        for k in config.layer_types]
        self.views = {}

    def prepare_context(self, starts, context_lengths, context_width, width):
        self.lengths = list(starts)
        self.context_lengths = list(context_lengths)
        self.context_width = context_width
        self.query_width = width
        self.views.clear()

    def update(self, keys, values, layer_idx, *args, **kwargs):
        b, h, _, d = keys.shape
        device = keys.device
        window = self.windows[layer_idx]
        capacity = min(self.capacity, window) if window else self.capacity
        if layer_idx not in self.buffers:
            self.buffers[layer_idx] = [keys.new_zeros((b, h, capacity, d)) for _ in range(2)]
        ctx = self.context_width
        # Prefill can be much longer than the sliding ring; retain only the
        # final window per row. No duplicate scatter indices are allowed.
        for row, length in enumerate(self.context_lengths):
            take = min(length, capacity)
            start = self.lengths[row]
            pos = torch.arange(start + length - take, start + length, device=device) % capacity
            for buf, src in zip(self.buffers[layer_idx], (keys, values)):
                buf[row, :, pos] = src[row, :, length-take:length]
        ends = [s+c for s,c in zip(self.lengths, self.context_lengths)]
        history = min(max(ends), capacity)
        offsets = torch.arange(history, device=device)[None].expand(b, -1)
        end_tensor = torch.tensor(ends, device=device)[:, None]
        starts = (end_tensor - capacity).clamp_min(0) if window else torch.zeros_like(end_tensor)
        hpos = starts + offsets
        valid = hpos < end_tensor
        gather = (hpos % capacity)[:, None, :, None].expand(b, h, history, d)
        # Full layers return a view; sliding layers gather only their window.
        history_kv = [buf.gather(2, gather) if window else buf[:, :, :history] for buf in self.buffers[layer_idx]]
        noise_pos = end_tensor + torch.arange(self.query_width, device=device)
        self.views[layer_idx] = (torch.cat((hpos, noise_pos), -1),
            torch.cat((valid, torch.ones_like(noise_pos, dtype=torch.bool)), -1), noise_pos)
        return tuple(torch.cat((buf, src[:, :, ctx:]), dim=2) for buf, src in zip(history_kv, (keys, values)))

    def attention_mask(self, query, key, *, causal, window, layer_idx):
        kpos, valid, qpos = self.views[layer_idx]
        return _mask(qpos, kpos, valid, causal, window)


class GDNTransaction:
    """Save only fixed-size states and mixer inputs, never attention KV."""
    def __init__(self, target, cache):
        self.cache = cache
        self.entries = []
        self.handles = []
        config = getattr(target.config, 'text_config', target.config)
        for i, layer in enumerate(_decoder_layers(target, config.num_hidden_layers)):
            mixer = getattr(layer, 'linear_attn', None)
            if mixer is not None:
                self.handles.append(mixer.register_forward_pre_hook(self._capture(i), with_kwargs=True))

    def _capture(self, index):
        def hook(module, args, kwargs):
            state = self.cache.layers[index]
            saved = {name: {k: v.clone() for k,v in getattr(state, name).items() if v is not None}
                     for name in ('conv_states', 'recurrent_states')}
            hidden = args[0] if args else kwargs['hidden_states']
            self.entries.append((module, state, hidden, saved))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def restore_replay(self, kept):
        self.close()
        for module, state, hidden, saved in self.entries:
            if all(n in (0, hidden.shape[1]) for n in kept):
                for name, tensors in saved.items():
                    for k, old in tensors.items():
                        now = getattr(state, name)[k]
                        active = torch.tensor([n > 0 for n in kept], device=now.device)
                        now.copy_(torch.where(active.view(-1, *([1]*(now.ndim-1))), now, old))
                continue
            for name, tensors in saved.items():
                for k, v in tensors.items():
                    getattr(state, name)[k].copy_(v)
            # Replay all B rows in each mixer call. Freeze rows once their
            # accepted prefix ends; padding computation remains measurable.
            for offset in range(max(kept, default=0)):
                before = {name: {k: v.clone() for k,v in getattr(state, name).items() if v is not None}
                          for name in ('conv_states', 'recurrent_states')}
                module(hidden_states=hidden[:, offset:offset+1], cache_params=self.cache)
                for name, tensors in before.items():
                    for k, old in tensors.items():
                        now = getattr(state, name)[k]
                        active = torch.tensor([n > offset for n in kept], device=now.device)
                        now.copy_(torch.where(active.view(-1, *([1]*(now.ndim-1))), now, old))
        self.entries.clear()


def sync():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


class BatchEngine:
    def __init__(self, target, draft=None, *, eos_ids=(), draft_stages=False, observer=None):
        self.target, self.draft = target, draft
        self.eos_ids = list(eos_ids or [])
        self.observer = observer
        self.config = getattr(target.config, 'text_config', target.config)
        self.device = target.get_input_embeddings().weight.device
        self.draft_device = next(draft.parameters()).device if draft is not None else self.device

    def phase(self, name, **metadata):
        return self.observer.phase(name, **metadata) if self.observer else nullcontext()

    def choose(self, logits, natural):
        if not natural and self.eos_ids:
            logits = logits.clone()
            logits[..., self.eos_ids] = -float('inf')
        return logits.argmax(-1)

    @torch.inference_mode()
    def run_batch(self, prompts, max_new_tokens, block_size=16, *, natural_eos=False):
        if not prompts or max_new_tokens < 1 or block_size < 1:
            raise ValueError('nonempty prompts, positive output cap and block required')
        if block_size > 1 and self.draft is None:
            raise ValueError('DFlash requires a matching drafter')
        b = len(prompts)
        lengths = [len(p) for p in prompts]
        if min(lengths) < 1:
            raise ValueError('empty prompt')
        capacity = max(lengths) + max_new_tokens + block_size
        cache = TargetCache(self.config, lengths, capacity)
        dc = DraftCache(self.draft.config, lengths, capacity) if block_size > 1 else None
        tap = HiddenStateTap(self.target, self.draft.target_layer_ids,
                            self.config.num_hidden_layers, self.draft_device) if dc else None
        tokens, stamps, kept_log = [[] for _ in prompts], [[] for _ in prompts], [[] for _ in prompts]
        active = [True] * b
        features = []
        sync()
        started = time.perf_counter()
        for row, prompt in enumerate(prompts):
            pc = DynamicCache(config=self.config)
            with self.phase('target_prefill', request=row):
                with tap if tap else nullcontext():
                    out = self.target(input_ids=prompt.to(self.device)[None], past_key_values=pc,
                                      use_cache=True, logits_to_keep=1)
                    selected = tap.states() if tap else None
                first = int(self.choose(out.logits[:, -1], natural_eos).item())
            now = time.perf_counter() - started
            tokens[row].append(first)
            stamps[row].append(now)
            active[row] = max_new_tokens > 1 and not (natural_eos and first in self.eos_ids)
            with self.phase('cache_import', request=row):
                cache.import_row(pc, row)
            if selected:
                with self.phase('hidden_capture_projection', request=row):
                    features.append(torch.cat(selected, dim=-1))
            if self.observer:
                self.observer.memory(cache, dc, selected, features[-1] if features else None, None,
                                     phase='prefill_resident')
            del pc, out, selected
        sync()
        prefill_done = time.perf_counter() - started
        feature_lengths = lengths.copy()
        draft_starts = [0] * b
        if dc:
            hidden = torch.cat([F.pad(f, (0,0,0,max(lengths)-f.shape[1])) for f in features], 0)
            features.clear()
        steps = []
        step = 0
        first_commit = None
        while any(active):
            if self.observer:
                self.observer.step = step
                self.observer.active_rows = active.copy()
            width = block_size
            anchor = [r[-1] for r in tokens]
            block = torch.full((b, width), self.draft.mask_token_id if dc else 0,
                               device=self.draft_device, dtype=torch.long)
            block[:, 0] = torch.tensor(anchor, device=self.draft_device)
            positions = torch.tensor(lengths, device=self.draft_device)[:, None] + torch.arange(width, device=self.draft_device)
            if dc:
                with self.phase('first_draft' if step == 0 else 'draft'):
                    dc.prepare_context(draft_starts, feature_lengths, hidden.shape[1], width)
                    ctxpos = torch.tensor(draft_starts, device=self.draft_device)[:,None] + torch.arange(hidden.shape[1], device=self.draft_device)
                    dh = self.draft(target_hidden=hidden,
                        noise_embedding=_raw_input_embeddings(self.target, block),
                        position_ids=torch.cat((ctxpos, positions), -1), past_key_values=dc, use_cache=True)
                    logits = self.draft.compute_logits(dh[:,1:].to(_output_head(self.target).weight.device), _output_head(self.target))
                    block[:,1:] = self.choose(logits, natural_eos).to(self.draft_device)
                    del dh, logits
                draft_starts = lengths.copy()
            cache.prepare(lengths, width)
            transaction = GDNTransaction(self.target, cache) if dc or not all(active) else None
            try:
                with self.phase('target_verify' if dc else 'ar_decode'):
                    with tap if tap else nullcontext():
                        out = self.target(input_ids=block.to(self.device), position_ids=positions.to(self.device),
                            attention_mask=cache.masks(self.device, self.config.layer_types,
                                getattr(self.config, 'sliding_window', None)),
                            past_key_values=cache, use_cache=True)
                        selected = tap.states() if tap else None
                with self.phase('sampling'):
                    posterior = self.choose(out.logits, natural_eos).to(self.draft_device)
                    matches = (block[:,1:] == posterior[:,:-1]).int().cumprod(-1).sum(-1).tolist()
                    block_cpu, post_cpu = block.tolist(), posterior.tolist()
            finally:
                if transaction:
                    transaction.close()
            # Anchor was committed by prefill/previous step. Commit only new
            # accepted drafts plus correction/bonus, clipped by EOS and cap.
            kept, accepted, committed, proposed, bonuses, active_before = [], [], [], [], [], active.copy()
            commit_time = time.perf_counter() - started
            for row in range(b):
                if not active[row]:
                    kept.append(0); accepted.append(0); committed.append(0); proposed.append(0)
                    bonuses.append(0)
                    continue
                count = 0
                accepted_here = 0
                candidates = block_cpu[row][1:matches[row]+1] + [post_cpu[row][matches[row]]]
                for j, token in enumerate(candidates):
                    if len(tokens[row]) == max_new_tokens:
                        break
                    tokens[row].append(token); stamps[row].append(commit_time)
                    count += 1
                    accepted_here += int(j < matches[row])
                    if natural_eos and token in self.eos_ids:
                        break
                # Cache prefix includes old anchor and newly accepted draft
                # inputs; leave the final committed output as the next anchor.
                retained = count
                kept.append(retained)
                kept_log[row].append(retained)
                accepted.append(accepted_here); committed.append(count); proposed.append(width-1)
                bonuses.append(count-accepted_here)
                active[row] = len(tokens[row]) < max_new_tokens and not (natural_eos and tokens[row][-1] in self.eos_ids)
            if first_commit is None:
                first_commit = commit_time
            if self.observer:
                self.observer.committed(step, kept, width, active_before)
                self.observer.memory(cache, dc, selected, hidden if dc else None, transaction,
                                     phase='verify_before_rollback')
            with self.phase('rollback_replay'):
                if transaction:
                    transaction.restore_replay(kept)
            with self.phase('cache_update'):
                if dc:
                    hidden = torch.cat(selected, -1)
                    feature_lengths = kept.copy()
                lengths = [n+k for n,k in zip(lengths, kept)]
                cache.lengths = lengths.copy()
            if self.observer:
                self.observer.memory(cache, dc, selected, hidden if dc else None, transaction)
            steps.append(dict(step=step, active_rows=active_before, proposed=proposed, accepted=accepted,
                committed=committed, correction_or_bonus=bonuses, kept=kept,
                active_after=active.copy(), output_lengths=list(map(len,tokens)),
                target_logical_lengths=lengths.copy(),
                draft_logical_lengths=draft_starts.copy() if dc else None,
                nominal_verify_width=width, actual_verify_tokens=b*width,
                active_verify_tokens=sum(active_before)*width,
                padding_tokens=(b-sum(active_before))*width,
                rejected_or_uncommitted_input_tokens=sum(width-k for k,a in zip(kept,active_before) if a),
                ghost_committed_tokens=0))
            del out, selected, posterior, transaction
            step += 1
        sync()
        end = time.perf_counter()-started
        decode = end-prefill_done
        total = sum(map(len,tokens))
        return dict(tokens=tokens, kept=kept_log, steps=steps,
            e2e_s=end, decode_s=decode, prefill_s=prefill_done,
            ttft_s=[s[0] for s in stamps], first_speculative_commit_s=first_commit if dc else None,
            request_latency_s=[s[-1] for s in stamps],
            tpot_s=[(s[-1]-s[0])/(len(s)-1) if len(s)>1 else None for s in stamps],
            committed_output_tokens=total, decode_committed_output_tokens=total-b,
            output_tok_s=total/end, decode_tok_s=(total-b)/decode if decode>0 else None,
            batch_decode_throughput_all_output_tok_s=total/decode if decode>0 else None,
            input_lengths=[len(p) for p in prompts],
            output_timestamps_s=stamps, prefill_policy='sequential_unchunked',
            cache_policy='preallocated_scatter; draft_sliding_ring; GDN_snapshot_mixer_replay',
            static_finished_rows='computed_and_frozen', attention_backend='sdpa')
