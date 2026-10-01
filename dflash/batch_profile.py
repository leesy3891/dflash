"""Diagnostic passes only. Perf never installs these hooks or synchronizes layers."""
from __future__ import annotations
import contextlib
import json
import time
import statistics
from pathlib import Path
import torch


def storages(groups):
    seen, result = set(), {}
    for label, tensors in groups.items():
        for tensor in tensors:
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity in seen:
                continue
            seen.add(identity)
            key = f'{label}/{tensor.device}'
            result[key] = result.get(key, 0) + storage.nbytes()
    return result


def layer_metadata(target, role):
    from .model import _decoder_layers
    config = getattr(target.config, 'text_config', target.config)
    layers = _decoder_layers(target, config.num_hidden_layers) if role == 'target' else target.layers
    output = []
    for i, layer in enumerate(layers):
        mixer = getattr(layer, 'layer_type', None) or config.layer_types[i]
        ffn = 'MoE(+shared)' if hasattr(layer.mlp, 'experts') else 'dense'
        output.append((i, layer, mixer, ffn))
    return output


class Observer:
    def __init__(self, run_id, mode, destination):
        self.run_id, self.mode, self.destination = run_id, mode, Path(destination)
        self.events, self.memory_records, self.routing = [], [], []
        self.stack, self.handles = [], []
        self.step, self.active_rows = -1, []
        self.current_phase = 'setup'
        self.pending_routes = []
        self.commits = {}
        self.expert_samples = []
        self.microbenchmarks = []

    @contextlib.contextmanager
    def phase(self, name, **metadata):
        previous = self.current_phase
        self.current_phase = name
        with self.span(kind='phase', phase=name, **metadata):
            yield
        self.current_phase = previous

    @contextlib.contextmanager
    def span(self, **metadata):
        event_id = len(self.events)
        event = dict(run=self.run_id, event=event_id, parent=self.stack[-1] if self.stack else None,
            request=None, step=self.step, phase=self.current_phase, role=None, layer=None,
            module=None, device=None, stream=None, shape=None, backend=None,
            active_rows=self.active_rows.copy(),request_ids=list(range(len(self.active_rows))))
        event.update(metadata)
        event['host_start_ns'] = time.perf_counter_ns()
        self.events.append(event)
        self.stack.append(event_id)
        label = f'batch::{self.run_id}::{event_id}::{event["phase"]}::{event.get("module") or "phase"}'
        with torch.profiler.record_function(label):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(label)
            try:
                yield event
            finally:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
                event['host_end_ns'] = time.perf_counter_ns()
                event['host_inclusive_ns'] = event['host_end_ns']-event['host_start_ns']
                self.stack.pop()

    def install(self, target, draft=None):
        self.weight_bytes = storages({'target_weights': target.parameters(),
            'draft_weights': draft.parameters() if draft else []})
        for role, model in [('target', target), ('drafter', draft)]:
            if model is None:
                continue
            outer = [('embedding', model.get_input_embeddings()), ('output_head', model.get_output_embeddings())] if role=='target' else [
                ('context_projection',model.fc),('context_norm',model.hidden_norm),('final_norm',model.norm)]
            for name, module in outer:
                if module is not None:
                    self._watch(module, dict(role=role,layer=None,module=name,mixer='other',ffn='other',kind='operator'))
            for layer_id, layer, mixer, ffn in layer_metadata(model, role):
                for name, module in layer.named_modules():
                    # layer -> operators. Norms/projections/experts retain their
                    # actual module names; kernel attribution uses deepest span.
                    self._watch(module, dict(role=role, layer=layer_id, module=name or 'layer',
                                mixer=mixer, ffn=ffn, kind='layer' if not name else 'operator'))
                    if self.mode == 'moe' and 'TopKRouter' in type(module).__name__:
                        self.handles.append(module.register_forward_hook(self._route(layer_id, name)))
                    if self.mode == 'moe' and name == 'mlp.experts':
                        def sample(mod, args, layer_id=layer_id):
                            if self.current_phase == 'target_verify' and self.step == 0:
                                self.expert_samples.append((layer_id, mod, [x.detach().clone() for x in args]))
                        self.handles.append(module.register_forward_pre_hook(sample))

    def committed(self, step, kept, width, active):
        self.commits[step] = dict(kept=list(kept), width=width, active=list(active))

    def _watch(self, module, info):
        calls = []
        def pre(mod, args, kwargs):
            tensor = next((x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)), None)
            device = tensor.device if tensor is not None else next(mod.parameters(), torch.empty(0)).device
            span = self.span(**info, device=str(device), shape=list(tensor.shape) if tensor is not None else None,
                stream=int(torch.cuda.current_stream(device).cuda_stream) if device.type == 'cuda' else None,
                backend=type(mod).__module__+'.'+type(mod).__name__)
            event = span.__enter__()
            calls.append(span)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                event['_cuda'] = (start, end, device)
        def post(mod, args, output):
            span = calls.pop()
            event = self.events[self.stack[-1]]
            if '_cuda' in event:
                _, end, device = event['_cuda']
                with torch.cuda.device(device):
                    end.record()
            span.__exit__(None, None, None)
        self.handles.extend([module.register_forward_pre_hook(pre, with_kwargs=True),
                             module.register_forward_hook(post, always_call=True)])

    def _route(self, layer, module):
        def hook(mod, args, output):
            # No host read inside the forward. This is a separate routing pass.
            if self.current_phase != 'target_verify':
                return
            self.pending_routes.append((dict(run=self.run_id, step=self.step, phase=self.current_phase,
                layer=layer, module=module, active_rows=self.active_rows.copy(), num_experts=mod.num_experts,
                hidden_size=mod.hidden_dim),
                output[2].detach().clone()))
        return hook

    def memory(self, target_cache, draft_cache, selected, feature, transaction, phase=None):
        if self.mode != 'memory':
            return
        groups = {'target_attention_kv': [t for pair in target_cache.buffers.values() for t in pair],
            'gdn_conv': [], 'gdn_recurrent': [], 'rollback_recording': [], 'gdn_verify_inputs': [],
            'draft_conditioning_kv': [t for pair in draft_cache.buffers.values() for t in pair] if draft_cache else [],
            'selected_hidden': selected or [], 'context_feature': [feature]}
        for layer in target_cache.layers:
            for attribute, label in [('conv_states','gdn_conv'),('recurrent_states','gdn_recurrent')]:
                groups[label].extend(getattr(layer, attribute, {}).values())
        if transaction:
            for _, _, hidden, saved in transaction.entries:
                groups['gdn_verify_inputs'].append(hidden)
                groups['rollback_recording'].extend(t for tensors in saved.values() for t in tensors.values())
        self.memory_records.append(dict(run=self.run_id, step=self.step, phase=phase or self.current_phase,
            components={**storages(groups), **self.weight_bytes}, target_logical_lengths=target_cache.lengths.copy(),
            allocated_per_device=[torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())],
            measurement='simultaneous component snapshot at named boundary; not component maxima'))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        for event in self.events:
            if '_cuda' in event:
                start, end, _ = event.pop('_cuda')
                event['same_device_cuda_inclusive_ms'] = start.elapsed_time(end)
        child_times = {}
        for e in self.events:
            child_times[e['parent']] = child_times.get(e['parent'], 0)+e['host_inclusive_ns']
        for e in self.events:
            e['host_exclusive_ns'] = e['host_inclusive_ns']-child_times.get(e['event'],0)
        for info, indices in self.pending_routes:
            ids = indices.cpu()
            counts = torch.bincount(ids.flatten(), minlength=info['num_experts'])
            step=self.commits[info['step']]
            q=step['width']; active=torch.tensor(step['active'])[:,None]
            kept=torch.tensor(step['kept'])[:,None]
            rejected=(torch.arange(q)[None]>=kept)&active
            ids3=ids.view(len(step['kept']),q,-1)
            rejected_counts=torch.bincount(ids3[rejected].flatten(),minlength=info['num_experts'])
            active_ids=ids3[active.expand(-1,q)]
            self.routing.append(dict(**info, expert_ids=ids.tolist(), unique_experts=int((counts>0).sum()),
                active_unique_experts=int(active_ids.unique().numel()),
                rejected_input_tokens=int(rejected.sum()),rejected_expert_assignments=rejected_counts.tolist(),
                rejected_router_flops_estimate=int(rejected.sum())*2*info['hidden_size']*info['num_experts'],
                tokens_per_expert=counts.tolist(), max_mean_load=float(counts.max()/counts.float().mean()),
                weight_dram_bytes=None, weight_dram_reason='expert union is not a DRAM measurement'))
        self.pending_routes.clear()
        # A separate post-generation isolated expert microbenchmark. Its smaller
        # gathered shape is not the in-situ time "wasted" by rejected tokens.
        with torch.inference_mode():
            for layer, module, args in self.expert_samples:
                step=self.commits[0];q=step['width']
                reject=torch.tensor([active and j>=kept for active,kept in zip(step['active'],step['kept']) for j in range(q)],device=args[0].device)
                hidden, ids, weights=args
                n=int(reject.sum())
                hidden_dim=hidden.shape[-1]
                intermediate=module.gate_up_proj.shape[-2]//2
                entry=dict(layer=layer,step=0,rejected_tokens=n,
                    rejected_expert_gemm_flops_estimate=6*n*ids.shape[-1]*hidden_dim*intermediate,
                    formula='6*N_rejected*top_k*hidden*expert_intermediate; activation/shared expert excluded',
                    note='isolated gathered suffix; not measured in-situ waste; warm-cache CUDA-event microbenchmark')
                if n and hidden.is_cuda:
                    small=[x[reject] for x in args]
                    module(*small)
                    torch.cuda.synchronize(hidden.device)
                    timings=[]
                    for _ in range(3):
                        with torch.cuda.device(hidden.device):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record();module(*small);end.record();end.synchronize()
                            timings.append(start.elapsed_time(end))
                    entry['isolated_suffix_ms_median']=statistics.median(timings)
                    entry['isolated_suffix_ms_samples']=timings
                else:
                    entry['isolated_suffix_ms_median']=None
                    entry['reason']='no rejected inputs or CPU correctness pass'
                self.microbenchmarks.append(entry)
        self.expert_samples.clear()
        self.destination.mkdir(parents=True, exist_ok=True)
        for name, data in [('events',self.events),('memory',self.memory_records),('routing',self.routing),
                           ('rejected_expert_microbench',self.microbenchmarks)]:
            with (self.destination/f'{name}.jsonl').open('w') as f:
                for row in data:
                    f.write(json.dumps(row)+'\n')


def reconstruct_memory_peak(initial, traces, final, limit):
    """Timestamp-ordered aggregate allocator occupancy, with tie uncertainty.

    Allocator timestamps share the CPU clock across devices. Equal timestamps
    do not establish cross-device ordering: provide a lower/upper bound instead
    of calling a sum of unrelated per-device maxima a simultaneous peak.
    """
    if any(len(t)>=limit for t in traces):
        return dict(status='NA',reason='allocator history ring may have wrapped')
    timeline={}
    for device,events in enumerate(traces):
        for event in events:
            if event['action'] not in ('alloc','free_requested'): continue
            stamp=event.get('time_us',event.get('time_ns'))
            if stamp is None:
                return dict(status='NA',reason='installed allocator does not expose timestamps')
            delta=event['size']*(1 if event['action']=='alloc' else -1)
            timeline.setdefault(stamp,[]).append((device,delta))
    current=list(initial);lower=upper=sum(current)
    for stamp,changes in sorted(timeline.items()):
        before=sum(current)
        # All allocation events before frees give the conservative tie bound.
        upper=max(upper,before+sum(max(0,d) for _,d in changes))
        for device,delta in changes: current[device]+=delta
        lower=max(lower,sum(current))
        if min(current)<0:
            return dict(status='NA',reason='incomplete or inconsistent allocator event history')
    if current!=list(final):
        return dict(status='NA',reason='reconstructed final allocation differs from allocator counters',
                    reconstructed=current,observed=list(final))
    return dict(status='measured' if lower==upper else 'timestamp_tie_bounds',
        simultaneous_aggregate_peak_bytes=lower if lower==upper else None,
        simultaneous_peak_lower_bound_bytes=lower,simultaneous_peak_upper_bound_bytes=upper,
        metric='PyTorch allocated bytes; CUDA context/external allocations excluded')


class MemoryHistory:
    def __init__(self, destination):
        self.destination=Path(destination)
        self.limit=2_000_000

    def start(self):
        self.initial=[torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
        torch.cuda.memory._record_memory_history(enabled='all',context=None,
            max_entries=self.limit,clear_history=True)

    def finish(self):
        snapshot=torch.cuda.memory._snapshot()
        final=[torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
        torch.cuda.memory._record_memory_history(enabled=None)
        traces=snapshot['device_traces']
        # Allocator records are saved even when this runtime cannot provide a
        # reliable timestamp ordering. No reconstruction is fabricated.
        self.destination.mkdir(parents=True,exist_ok=True)
        (self.destination/'allocator_history.json').write_text(json.dumps(dict(initial=self.initial,final=final,traces=traces)))
        result=reconstruct_memory_peak(self.initial,traces,final,self.limit)
        (self.destination/'aggregate_peak.json').write_text(json.dumps(result,indent=2))
        return result
