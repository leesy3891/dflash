"""Reproducible three-shard batch profiling CLI. No data parallel replicas."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys
import time
import traceback
import faulthandler

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
 'qwen3-8b': ('Qwen/Qwen3-8B','z-lab/Qwen3-8B-DFlash-b16'),
 'qwen3.5-9b': ('Qwen/Qwen3.5-9B','z-lab/Qwen3.5-9B-DFlash'),
 'qwen3.5-35b-a3b': (str(Path.home()/'models/Qwen3.5-35B-A3B'),'z-lab/Qwen3.5-35B-A3B-DFlash'),
}
SCHEMA = 'dflash.batch.v3'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    def encode(value):
        if isinstance(value, set): return sorted(value)
        if isinstance(value, Path): return str(value)
        raise TypeError(f'Unsupported record value: {type(value).__name__}')
    tmp.write_text(json.dumps(obj, indent=2, default=encode)+'\n'); tmp.replace(path)


def command_output(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=45)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def snapshot(repo):
    from huggingface_hub import hf_hub_download
    if Path(repo).is_dir():
        return str(Path(repo).resolve())
    # Some valid local checkpoints intentionally omit model-card images.
    # Resolve the pinned config, then validate all weight shards ourselves.
    return str(Path(hf_hub_download(repo, 'config.json', local_files_only=True)).parent)


def checkpoint_inventory(path):
    """Validate shard index against safetensors headers without loading weights."""
    path = Path(path)
    index_path = path/'model.safetensors.index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    names = sorted(set(index['weight_map'].values())) if index else ['model.safetensors']
    headers, sizes, header_hashes = {}, {}, {}
    for name in names:
        file = path/name
        with file.open('rb') as f:
            raw = f.read(8)
            length = struct.unpack('<Q',raw)[0]
            header_bytes = f.read(length)
            header = json.loads(header_bytes)
        header_hashes[name] = hashlib.sha256(header_bytes).hexdigest()
        sizes[name] = file.stat().st_size
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        for key, value in tensors.items():
            lo, hi = value['data_offsets']
            if not 0 <= lo <= hi <= sizes[name]-8-length:
                raise ValueError(f'Invalid safetensors offsets: {file}:{key}')
            if key in headers:
                raise ValueError(f'Duplicate tensor: {key}')
            headers[key] = name
    if index and headers != index['weight_map']:
        raise ValueError(f'Shard index/header mismatch: {path}')
    return dict(path=str(path), revision=path.name, shard_bytes=sizes,
        header_sha256=header_hashes, tensor_count=len(headers),
        integrity='index/header/offset/file-size; full weight payload hash not computed')


def select_backend(backend):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as dense
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
    records = {}
    for module in (dense, moe):
        for name in ('causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
            original = inspect.unwrap(getattr(module,name))
            if backend == 'torch':
                selected = original
            else:
                if name.startswith('causal'):
                    import causal_conv1d
                    selected = getattr(causal_conv1d, name)
                else:
                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
                    selected = chunk_gated_delta_rule if 'chunk' in name else fused_recurrent_gated_delta_rule
            # Adapter freezes the actual callable; HF wrappers can otherwise
            # silently resolve a different installed package.
            params = inspect.signature(selected).parameters
            var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            def call(*a, _fn=selected, _keys=params, _var=var_kw, **kw):
                return _fn(*a, **(kw if _var else {k:v for k,v in kw.items() if k in _keys}))
            setattr(module, name, call)
            records[module.__name__+'.'+name] = selected.__module__+'.'+selected.__name__
    return records


def load_models(model_name, mode, backend):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from .model import DFlashDraftModel, _decoder_layers
    print('stage: checkpoint/config validation', flush=True)
    target_id, draft_id = PRESETS[model_name]
    target_path, draft_path = snapshot(target_id), snapshot(draft_id)
    inventories = {'target':checkpoint_inventory(target_path), 'draft':checkpoint_inventory(draft_path)}
    config = AutoConfig.from_pretrained(target_path, local_files_only=True)
    draft_config = AutoConfig.from_pretrained(draft_path, local_files_only=True)
    text = getattr(config,'text_config',config)
    if draft_config.num_target_layers != text.num_hidden_layers or draft_config.hidden_size != text.hidden_size:
        raise ValueError('drafter/target hidden width or depth mapping mismatch')
    ids = draft_config.dflash_config['target_layer_ids']
    if min(ids)<0 or max(ids)>=text.num_hidden_layers-1:
        raise ValueError('Invalid selective hidden mapping')
    klass = AutoModelForImageTextToText if hasattr(config,'text_config') else AutoModelForCausalLM
    with init_empty_weights():
        skeleton = klass.from_config(config)
    layers = _decoder_layers(skeleton, text.num_hidden_layers)
    names = {id(m):n for n,m in skeleton.named_modules()}
    # Never use a root '' fallback alongside child overrides: Accelerate's
    # root AlignDevicesHook would temporarily move every shard to GPU 0.
    placement = {}
    for i, layer in enumerate(layers):
        placement[names[id(layer)]] = min(i*3//len(layers),2)
    layer_roots = tuple(placement)
    for name, mod in skeleton.named_modules():
        if any(name == root or name.startswith(root+'.') for root in layer_roots):
            continue
        if any(True for _ in mod.parameters(recurse=False)) or any(True for _ in mod.buffers(recurse=False)):
            if not name:
                raise ValueError('Root-owned tensors require an explicit architecture placement adapter')
            placement[name] = 0
        if name.endswith('.norm') and name.rsplit('.',1)[0] == names[id(layers)].rsplit('.',1)[0]:
            placement[name] = 2
    placement['lm_head'] = 2
    del skeleton, layers
    print('stage: target weight load and three-shard dispatch', flush=True)
    target, loading = klass.from_pretrained(target_path, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map=placement, local_files_only=True, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'Target load errors: {loading}')
    target.eval()
    actual = [{str(p.device) for p in layer.parameters()} for layer in _decoder_layers(target, text.num_hidden_layers)]
    expected = [{f'cuda:{min(i*3//text.num_hidden_layers,2)}'} for i in range(text.num_hidden_layers)]
    if actual != expected:
        raise ValueError(f'Actual target placement differs from planned contiguous shards: {actual}')
    print('stage: target loaded', flush=True)
    draft = None
    if mode == 'dflash':
        draft, draft_loading = DFlashDraftModel.from_pretrained(draft_path, config=draft_config,
            dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
        if any(draft_loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError(f'Drafter load errors: {draft_loading}')
        draft = draft.to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    print('stage: resolving fixed backend', flush=True)
    return target, draft, tokenizer, dict(checkpoints=inventories, target_placement=placement, draft_placement=0 if draft else None,
        target_loading=loading, rope={'target':getattr(text,'rope_parameters',None),'draft':getattr(draft_config,'rope_parameters',None)},
        rope_policy='checkpoint_native; extrapolated lengths labelled; no per-condition scaling',
        attention='sdpa', gdn=select_backend(backend), target_config=text.to_dict(), draft_config=draft_config.to_dict())


def prepare_prompts(destination, model, length, rows=32, seed=42):
    """Freeze distinct LongBench documents and every composition contributor.

    Token-level head/tail truncation gives exactly S tokens without decode /
    re-encode drift. Long inputs concatenate distinct source documents, recorded
    as a shape workload, not a LongBench accuracy evaluation.
    """
    import zipfile
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    target_path = snapshot(PRESETS[model][0])
    tok = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    archive_path = Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',local_files_only=True))
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    tasks = ['narrativeqa','qasper','multifieldqa_en','hotpotqa']
    pools = {}
    with zipfile.ZipFile(archive_path) as z:
        for task in tasks:
            records = [json.loads(line) for line in z.read(f'data/{task}.jsonl').splitlines()]
            order = list(range(len(records))); random.Random(seed).shuffle(order)
            pools[task] = [(i, records[i]) for i in order]
    output = []
    for row in range(rows):
        task = tasks[row % len(tasks)]
        pool = pools[task]
        sources, context = [], []
        offset = row // len(tasks)
        first = pool[offset % len(pool)][1]
        suffix = tok.encode('\nQuestion: '+first.get('input','Summarize the document.')+'\nAnswer:', add_special_tokens=False)
        for j in range(len(pool)):
            index, document = pool[(offset+j) % len(pool)]
            piece = tok.encode(document['context']+'\n\n', add_special_tokens=False)
            context.extend(piece)
            sources.append(dict(task=task, split='full', index=index, id=document.get('_id',document.get('id')),
                source_sha256=digest(document), contributed_tokens=len(piece)))
            if len(context) + len(suffix) >= length:
                break
        if len(context)+len(suffix)<length:
            raise ValueError('Insufficient distinct LongBench source tokens')
        ids = context+suffix
        half = length//2
        ids = ids[:half] + ids[-(length-half):]
        output.append(dict(request=row, task=task, sources=sources, composed=len(sources)>1,
            input_ids=ids, token_sha256=digest(ids), actual_input_tokens=len(ids)))
    if len({x['token_sha256'] for x in output}) != rows:
        raise ValueError('Duplicate batch prompts')
    payload = dict(schema=SCHEMA, model=model, length=length, seed=seed, archive_sha256=archive_hash,
        archive=str(archive_path), tokenizer_revision=Path(target_path).name,
        template='raw context + Question/Answer; token head/tail truncation; no chat template', rows=output)
    write_json(destination,payload)
    return payload


def environment():
    import torch
    versions = {}
    for name in ('torch','transformers','flash-attn','flash-linear-attention','causal-conv1d','accelerate'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(git=command_output(['git','rev-parse','HEAD']), branch=command_output(['git','branch','--show-current']),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT/'dflash').glob('batch*.py'))+[ROOT/'dflash/model.py']},
        versions=versions, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpus=[dict(logical=i,name=torch.cuda.get_device_name(i),memory_bytes=torch.cuda.get_device_properties(i).total_memory,
                   uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())],
        topology=command_output(['nvidia-smi','topo','-m']),
        p2p=[[torch.cuda.can_device_access_peer(i,j) if i!=j else True for j in range(3)] for i in range(3)])


def run(args):
    import torch
    from .batch import BatchEngine, sync
    from .batch_profile import Observer, MemoryHistory
    dest = Path(args.output)
    dest.mkdir(parents=True,exist_ok=True)
    status_path = dest/'status.json'
    if status_path.exists() and not args.force:
        old = json.loads(status_path.read_text())
        if old.get('status') == 'complete':
            print('Already complete:',dest); return
    # A crashed attempt must not contribute partial repeats to its replacement.
    # Preserve its records under a separate directory before restarting.
    if (dest/'raw.jsonl').exists():
        archive=dest/f'incomplete-{time.time_ns()}'
        archive.mkdir()
        for name in ('raw.jsonl','manifest.json','status.json','summary.json','preflight.json'):
            if (dest/name).exists(): (dest/name).rename(archive/name)
    write_json(status_path,dict(status='running',started=time.time(),command=sys.argv))
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        if torch.cuda.device_count()!=3:
            raise ValueError('Exactly three visible GPUs are required; no automatic placement changes')
        if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='batch':
            raise ValueError('Only batch branch execution is allowed')
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        target,draft,tok,manifest = load_models(args.model,args.mode,args.backend)
        # Watchdog tracebacks are a loader diagnostic, never a perf-pass hook.
        faulthandler.cancel_dump_traceback_later()
        print('stage: model ready, collecting environment', flush=True)
        frozen = json.loads(Path(args.prompts).read_text())
        if frozen['model'] != args.model or frozen['length'] != args.length or len(frozen['rows'])<args.batch:
            raise ValueError('Frozen prompt manifest does not match condition')
        rows = frozen['rows'][:args.batch]
        for row in rows:
            if digest(row['input_ids'])!=row['token_sha256'] or len(row['input_ids'])!=args.length:
                raise ValueError('Prompt token hash/length mismatch')
        prompts = [torch.tensor(r['input_ids'],dtype=torch.long) for r in rows]
        eos = tok.eos_token_id
        if isinstance(eos,int): eos=[eos]
        engine = BatchEngine(target,draft,eos_ids=eos)
        manifest.update(schema=SCHEMA,model=args.model,mode=args.mode,backend_family='transformers_layer_sharding',
            pass_type=args.pass_type,condition=vars(args),environment=environment(),
            prompt_manifest=str(Path(args.prompts).resolve()),prompt_hashes=[r['token_sha256'] for r in rows],
            workload='natural_eos' if args.natural else 'shape_controlled_eos_suppressed',
            seed=args.seed,dtype='bfloat16',greedy=True,global_batch=args.batch,
            beyond_draft_training_length=args.model=='qwen3.5-35b-a3b' and args.length>40000,
            native_rope_extrapolation=args.length+args.tokens>min(manifest['target_config']['max_position_embeddings'],manifest['draft_config']['max_position_embeddings']),
            missing_metrics={'ncu':'NA: run counter pass separately; no bandwidth/compute classification from scaling',
                'simultaneous_aggregate_peak':'memory pass reconstructs allocator history; other passes NA',
                'bridge_main':'not run; B=1 study is on another branch; no merged baseline'})
        write_json(dest/'manifest.json',manifest)
        print('stage: full-condition warmup', flush=True)
        # A minimal allocator budget guard. It is explicitly a lower bound,
        # never a prediction that activations will fit.
        cfg = manifest['target_config']; dcfg=manifest['draft_config']
        capacity=args.length+args.tokens+args.block
        estimates=[0,0,0]
        for i,kind in enumerate(cfg['layer_types']):
            if kind != 'linear_attention':
                estimates[min(i*3//cfg['num_hidden_layers'],2)] += args.batch*capacity*cfg['num_key_value_heads']*cfg['head_dim']*4
        if draft:
            for kind in dcfg['layer_types']:
                n=min(capacity,dcfg['sliding_window']) if kind=='sliding_attention' else capacity
                estimates[0]+=args.batch*n*dcfg['num_key_value_heads']*dcfg['head_dim']*4
            estimates[0]+=args.batch*args.length*dcfg['hidden_size']*len(draft.target_layer_ids)*2
        free=[torch.cuda.mem_get_info(i)[0] for i in range(3)]
        write_json(dest/'preflight.json',dict(cache_and_context_lower_bound_bytes=estimates,free_bytes=free,
            omitted='GDN states, transient activations, prefill duplication and scratch; actual OOM still possible'))
        if any(need>available for need,available in zip(estimates,free)):
            write_json(status_path,dict(status='capacity_preflight_rejected',estimate=estimates,free=free)); return
        block=args.block if args.mode=='dflash' else 1
        # Full-condition warmup, discarded. A separate smoke gate must pass
        # before the queue schedules this point.
        warm=engine.run_batch(prompts,args.tokens,block,natural_eos=args.natural)
        expected_tokens, expected_kept = warm['tokens'], warm['kept']
        del warm
        repeats=args.repeats if args.pass_type=='perf' else 1
        results=[]
        for repeat in range(repeats):
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            sync()
            for i in range(3): torch.cuda.reset_peak_memory_stats(i)
            observer=None; profiler=None; history=None; aggregate=None
            if args.pass_type!='perf':
                observer=Observer(f'{dest.name}-{repeat}',args.pass_type,dest/f'diagnostic-{repeat}')
                observer.install(target,draft); engine.observer=observer
            if args.pass_type=='trace':
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,profile_memory=False,with_stack=False)
                profiler.__enter__()
            if args.pass_type=='nsys':
                torch.cuda.profiler.start()
            if args.pass_type=='memory':
                history=MemoryHistory(dest/f'diagnostic-{repeat}')
                history.start()
            try:
                result=engine.run_batch(prompts,args.tokens,block,natural_eos=args.natural)
            finally:
                if history: aggregate=history.finish()
                if args.pass_type=='nsys': torch.cuda.profiler.stop()
                if profiler:
                    profiler.__exit__(None,None,None)
                    profiler.export_chrome_trace(str(dest/'trace.json'))
                if observer: observer.close()
                engine.observer=None
            if result['tokens']!=expected_tokens or result['kept']!=expected_kept:
                raise AssertionError('Profiler/repeat changed output or acceptance')
            result.update(schema=SCHEMA,repeat=repeat,pass_type=args.pass_type,
                peak_allocated_per_device_bytes=[torch.cuda.max_memory_allocated(i) for i in range(3)],
                peak_reserved_per_device_bytes=[torch.cuda.max_memory_reserved(i) for i in range(3)],
                simultaneous_aggregate_peak_bytes=aggregate.get('simultaneous_aggregate_peak_bytes') if aggregate else None,
                simultaneous_memory_measurement=aggregate,
                profiler_output_and_acceptance_equal=True)
            result['sum_per_device_maxima_bytes']=sum(result['peak_allocated_per_device_bytes'])
            results.append(result)
            with (dest/'raw.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
        summary={key:dict(median=statistics.median(r[key] for r in results),
            stdev=statistics.stdev(r[key] for r in results) if len(results)>1 else None,
            min=min(r[key] for r in results),max=max(r[key] for r in results))
            for key in ('e2e_s','decode_s','output_tok_s','decode_tok_s')}
        write_json(dest/'summary.json',dict(repeats=len(results),metrics=summary))
        write_json(status_path,dict(status='complete',finished=time.time(),repeats=len(results)))
    except torch.cuda.OutOfMemoryError as exc:
        write_json(status_path,dict(status='oom',reason=str(exc),traceback=traceback.format_exc()))
    except Exception as exc:
        write_json(status_path,dict(status='unsupported' if isinstance(exc,(ImportError,NotImplementedError)) else 'failed',
            reason=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


def smoke(args):
    """Real-checkpoint BF16 tests including B1/B2 and diagnostic perturbation.

    AR is loaded in a different process by the queue. This worker only writes
    outputs; comparison in the queue gates all scientific profiling.
    """
    run(args)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=PRESETS,required=True)
    p.add_argument('--mode',choices=['ar','dflash'],default='dflash')
    p.add_argument('--backend',choices=['fla','torch'],default='fla')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--length',type=int,default=4096)
    p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--block',type=int,choices=[4,8,16],default=16)
    p.add_argument('--pass-type',choices=['perf','memory','trace','moe','nsys','ncu'],default='perf')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--natural',action='store_true')
    p.add_argument('--prompts',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--force',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.prepare:
        prepare_prompts(args.prompts,args.model,args.length)
    else:
        if not 1<=args.batch<=32 or args.repeats<3:
            raise ValueError('B must be 1..32 and perf repeats >=3')
        run(args)

if __name__=='__main__': main()
