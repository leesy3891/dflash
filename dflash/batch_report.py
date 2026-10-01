"""Compare matched perf runs; report diagnostic attribution without double counting."""
from __future__ import annotations
import bisect
import csv
import heapq
import json
from pathlib import Path
import statistics
import sys
from .batch_bench import write_json


def csv_write(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not fields: fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not result or start>result[-1][1]: result.append([start,end])
        else: result[-1][1]=max(result[-1][1],end)
    return result


def duration(intervals): return sum(e-s for s,e in union(intervals))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def trace_report(folder):
    """Kernel attribution follows external/launch correlation into deepest scope.

    CPU exclusive time and GPU interval union have separate columns. Never add
    parent+child or all-device CUDA duration to request wall time.
    """
    folder=Path(folder)
    trace_path=folder/'trace.json'
    if not trace_path.exists(): return
    destination=folder/'analysis';destination.mkdir(exist_ok=True)
    metadata=read_jsonl(folder/'diagnostic-0/events.jsonl')
    events=json.loads(trace_path.read_text())['traceEvents']
    spans={}; operations={}; kernels=[];copies=[];runtime={}
    for e in events:
        if e.get('ph')!='X' or 'dur' not in e: continue
        name=e.get('name','');cat=e.get('cat','');a=e.get('args',{})
        thread=(e.get('pid'),e.get('tid'))
        if name.startswith('batch::'):
            try: event_id=int(name.split('::')[2])
            except (ValueError,IndexError): continue
            spans.setdefault(thread,[]).append((e['ts'],e['ts']+e['dur'],event_id))
        if cat=='cpu_op' and 'External id' in a:
            operations.setdefault(thread,[]).append(e)
        if 'runtime' in cat or 'driver' in cat:
            runtime[a.get('correlation')]=a.get('External id')
        if cat=='kernel': kernels.append(e)
        if 'memcpy' in cat or 'memset' in cat: copies.append(e)
    owners={}
    for thread,ops in operations.items():
        ranges=sorted(spans.get(thread,[]));j=0;heap=[]
        for e in sorted(ops,key=lambda x:x['ts']):
            ts=e['ts']
            while j<len(ranges) and ranges[j][0]<=ts:
                start,end,index=ranges[j];heapq.heappush(heap,(-start,end,index));j+=1
            while heap and heap[0][1]<ts: heapq.heappop(heap)
            if heap: owners[e['args']['External id']]=heap[0][2]
    attributed=[];by_gpu={};by_operator={};copy_gpu={}
    for e in kernels:
        args=e.get('args',{});device=args.get('device',e.get('pid'))
        index=owners.get(args.get('External id'),owners.get(runtime.get(args.get('correlation'))))
        info=metadata[index] if index is not None else {}
        interval=[e['ts'],e['ts']+e['dur']]
        row=dict(run=folder.name,kernel=e['name'],device=device,stream=args.get('stream'),
            start_us=e['ts'],duration_us=e['dur'],event=index,phase=info.get('phase','unattributed'),
            role=info.get('role'),layer=info.get('layer'),module=info.get('module'),
            mixer=info.get('mixer'),ffn=info.get('ffn'),step=info.get('step'),shape=info.get('shape'),
            correlation=args.get('correlation'),external_id=args.get('External id'),
            bottleneck='undetermined',confidence='NA',reason='timeline alone does not establish bandwidth/compute')
        attributed.append(row)
        by_gpu.setdefault(device,[]).append(interval)
        key=(row['phase'],row['role'],row['layer'],row['module'],row['mixer'],row['ffn'],device)
        by_operator.setdefault(key,[]).append(interval)
    for e in copies:
        device=e.get('args',{}).get('device',e.get('pid'))
        copy_gpu.setdefault(device,[]).append([e['ts'],e['ts']+e['dur']])
    csv_write(destination/'operator_kernels.csv',attributed)
    operator_rows=[]
    for key,intervals in by_operator.items():
        row=dict(zip(['phase','role','layer','module','mixer','ffn','device'],key))
        row.update(exclusive_kernel_interval_union_us=duration(intervals),kernel_sum_us=sum(e-s for s,e in intervals),
            kernels=len(intervals),bottleneck='undetermined',evidence='NCU counters unavailable in this trace',confidence='NA')
        operator_rows.append(row)
    csv_write(destination/'operators.csv',operator_rows)
    all_work=union([interval for group in list(by_gpu.values())+list(copy_gpu.values()) for interval in group])
    cpu_ranges=[(s,e) for group in spans.values() for s,e,_ in group]
    origin=min((s for s,e in cpu_ranges),default=0);end=max((e for s,e in cpu_ranges),default=0)
    wall=max(0,end-origin)
    gpu_rows=[]
    for device in sorted(set(by_gpu)|set(copy_gpu)):
        busy=duration(by_gpu.get(device,[]));copy=duration(copy_gpu.get(device,[]))
        work=duration(by_gpu.get(device,[])+copy_gpu.get(device,[]))
        gpu_rows.append(dict(device=device,kernel_busy_union_us=busy,copy_union_us=copy,
            kernel_copy_overlap_us=busy+copy-work,device_no_activity_us=max(0,wall-work),
            interpretation='No-activity includes structural shard/dependency wait, not kernel DRAM stall'))
    csv_write(destination/'gpu_idle_transfer.csv',gpu_rows)
    write_json(destination/'timeline_summary.json',dict(traced_host_window_us=wall,
        global_activity_union_us=duration(all_work),global_no_work_or_unattributed_wall_us=max(0,wall-duration(all_work)),
        simultaneous_gpu_kernel_sum_is_not_wall=True,unattributed_kernels=sum(r['event'] is None for r in attributed)))
    gaps=[];cursor=origin
    for start,finish in all_work:
        if start>cursor:gaps.append(dict(start_us=cursor,end_us=start,duration_us=start-cursor,
            classification='unknown',reason='Requires correlated dependency/launch evidence; no automatic DRAM-stall attribution'))
        cursor=max(cursor,finish)
    if end>cursor:gaps.append(dict(start_us=cursor,end_us=end,duration_us=end-cursor,classification='unknown',reason='wall residual'))
    csv_write(destination/'gaps.csv',gaps,['start_us','end_us','duration_us','classification','reason'])
    csv_write(destination/'transfer.csv',[dict(name=e['name'],start_us=e['ts'],duration_us=e['dur'],**e.get('args',{})) for e in copies])
    csv_write(destination/'roofline.csv',[dict(operator=r['module'],layer=r['layer'],phase=r['phase'],device=r['device'],
        dram_bytes=None,l2_bytes=None,flops=None,bandwidth=None,tensor_pipe=None,occupancy=None,
        status='NA',reason='NCU not collected; traffic cannot be inferred from expert union') for r in operator_rows])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(13,3))
        for device,intervals in by_gpu.items():
            ax.broken_barh([((s-origin)/1000,(e-s)/1000) for s,e in union(intervals)],(int(device)-.3,.6))
        ax.set_yticks([0,1,2]);ax.set_yticklabels(['GPU 0','GPU 1','GPU 2'])
        ax.set_xlabel('Trace time (ms)');ax.set_title('3 GPU kernel activity; blanks are not evidence of kernel stalls')
        fig.tight_layout();fig.savefig(destination/'gpu_timeline.png',dpi=160);plt.close(fig)
        phases=sorted({r['phase'] for r in operator_rows});layers=sorted({r['layer'] for r in operator_rows if r['layer'] is not None})
        if phases and layers:
            matrix=[[sum(r['exclusive_kernel_interval_union_us'] for r in operator_rows if r['phase']==phase and r['layer']==layer)/1000 for layer in layers] for phase in phases]
            fig,ax=plt.subplots(figsize=(13,max(3,len(phases)*.4)))
            chart=ax.imshow(matrix,aspect='auto');ax.set_yticks(range(len(phases)),phases);ax.set_xlabel('Layer');fig.colorbar(chart,ax=ax,label='Attributed kernel ms (not wall)')
            fig.tight_layout();fig.savefig(destination/'phase_layer_heatmap.png',dpi=160);plt.close(fig)
    except ImportError:
        write_json(destination/'plot_status.json',dict(status='NA',reason='matplotlib unavailable'))


def report(root):
    root=Path(root);dest=root/'report';dest.mkdir(parents=True,exist_ok=True)
    statuses=[];runs=[];events=[];routing=[];memory=[];microbench=[]
    for status_path in sorted((root/'runs').glob('*/status.json')):
        folder=status_path.parent;status=json.loads(status_path.read_text())
        statuses.append(dict(run=folder.name,**status))
        if status.get('status')!='complete':continue
        manifest=json.loads((folder/'manifest.json').read_text());raw=read_jsonl(folder/'raw.jsonl')
        runs.append((folder,manifest,raw))
        for diagnostic in folder.glob('diagnostic-*'):
            for name,bucket in [('events',events),('routing',routing),('memory',memory),('rejected_expert_microbench',microbench)]:
                path=diagnostic/f'{name}.jsonl'
                if path.exists():bucket.extend(read_jsonl(path))
        if (folder/'trace.json').exists() and not (folder/'analysis/timeline_summary.json').exists():trace_report(folder)
    plan_path=root/'plan.json'
    if plan_path.exists():
        seen={r['run'] for r in statuses}
        for job in json.loads(plan_path.read_text())['jobs']:
            if job['id'] not in seen:
                statuses.append(dict(run=job['id'],status='queued_not_run'))
    csv_write(dest/'status.csv',statuses)
    csv_write(dest/'phase.csv',[e for e in events if e['kind']=='phase'])
    csv_write(dest/'layer.csv',[e for e in events if e['kind']=='layer'])
    csv_write(dest/'operator_host.csv',[e for e in events if e['kind']=='operator'])
    csv_write(dest/'routing.csv',routing);csv_write(dest/'memory.csv',memory)
    csv_write(dest/'rejected_expert_microbench.csv',microbench)
    lookup={};comparisons=[];perturb=[]
    def condition(m):
        c=m['condition']
        return (m['model'],c['batch'],c['length'],c['tokens'],c['natural'],c['backend'],tuple(m['prompt_hashes']),
                json.dumps(m['target_placement'],sort_keys=True))
    for folder,m,raw in runs:
        if m['pass_type']=='perf':lookup[(condition(m),m['mode'],m['condition']['block'] if m['mode']=='dflash' else 0)]=(folder,m,raw)
    for folder,m,raw in runs:
        block=m['condition']['block'] if m['mode']=='dflash' else 0
        key=condition(m)
        if m['pass_type']!='perf':
            base=lookup.get((key,m['mode'],block))
            if base:
                baseline=statistics.median(r['e2e_s'] for r in base[2])
                perturb.append(dict(run=folder.name,baseline=base[0].name,pass_type=m['pass_type'],
                    e2e_overhead_fraction=statistics.median(r['e2e_s'] for r in raw)/baseline-1,
                    output_equal=raw[0]['tokens']==base[2][0]['tokens'],acceptance_equal=raw[0]['kept']==base[2][0]['kept']))
            continue
        if m['mode']!='dflash':continue
        baseline=lookup.get((key,'ar',0))
        if not baseline:continue
        ar=baseline[2]
        matched_revision=m['checkpoints']['target']==baseline[1]['checkpoints']['target']
        if not matched_revision or m['gdn']!=baseline[1]['gdn']:continue
        accepted=sum(sum(s['accepted']) for s in raw[0]['steps']);proposed=sum(sum(s['proposed']) for s in raw[0]['steps'])
        c=m['condition']
        valid=raw[0]['tokens']==ar[0]['tokens'] and len(ar)>=3 and len(raw)>=3
        tpots=sorted(t for t in raw[0]['tpot_s'] if t is not None)
        comparisons.append(dict(model=m['model'],batch=c['batch'],S=c['length'],output=c['tokens'],block=c['block'],natural=c['natural'],
            comparison_valid=valid,invalid_reason=None if valid else 'output mismatch or fewer than three repeats',
            e2e_speedup=statistics.median(r['e2e_s'] for r in ar)/statistics.median(r['e2e_s'] for r in raw) if valid else None,
            decode_speedup=statistics.median(r['decode_s'] for r in ar)/statistics.median(r['decode_s'] for r in raw) if valid else None,
            committed_decode_tok_s=statistics.median(r['decode_tok_s'] for r in raw),
            all_output_over_decode_wall_tok_s=statistics.median(r['batch_decode_throughput_all_output_tok_s'] for r in raw),
            extra_per_device_peak_bytes=[statistics.median(r['peak_allocated_per_device_bytes'][i] for r in raw)-statistics.median(r['peak_allocated_per_device_bytes'][i] for r in ar) for i in range(3)],
            acceptance=accepted/proposed if proposed else None,
            tpot_p50_s=statistics.median(tpots) if tpots else None,
            tpot_p95_s=tpots[max(0,int(.95*len(tpots)+.999)-1)] if tpots else None,
            padding_tokens=sum(s['padding_tokens'] for s in raw[0]['steps']),
            ar_repeats=len(ar),dflash_repeats=len(raw),output_equal=raw[0]['tokens']==ar[0]['tokens'],
            bottleneck='undetermined',evidence='NCU counters required; scaling alone is not proof'))
    csv_write(dest/'ar_comparison.csv',comparisons);csv_write(dest/'profiler_perturbation.csv',perturb)
    write_json(dest/'coverage.json',dict(recorded=len(statuses),complete=len(runs),paired_perf=len(comparisons),
        status_counts={s:sum(r['status']==s for r in statuses) for s in sorted({r['status'] for r in statuses})}))

if __name__=='__main__': report(sys.argv[1])
