"""Persistent, fail-closed three-GPU queue with correctness gates and resume."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from .batch_bench import ROOT, PRESETS, write_json, prepare_prompts

GPUS = ['GPU-ef8b9800-1a20-174f-32b5-b3e9935e1db2',
        'GPU-8911bb88-5fdb-401b-808e-4be44cc6acd5',
        'GPU-8c8a823b-24e8-e395-532d-b619ae2522d5']


def fingerprint():
    files=sorted((ROOT/'dflash').glob('batch*.py'))+[ROOT/'dflash/model.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def make_plan(path, extended=False):
    jobs=[]
    def add(model,stage,b,s,o=256,block=16,pass_type='perf',natural=False):
        for mode in ('ar','dflash'):
            # AR is invariant to draft width. Reuse its baseline, never load a drafter.
            if mode=='ar' and block!=16: continue
            jobs.append(dict(model=model,stage=stage,batch=b,length=s,tokens=o,block=block,
                pass_type=pass_type,mode=mode,natural=natural))
    for model in PRESETS:
        for b in (1,2): add(model,'smoke',b,128,16)
        add(model,'smoke_natural',2,128,32,natural=True)
        add(model,'smoke_profile',2,128,16,pass_type='memory')
    # Initial comparisons isolate memory footprint and B*verify width.
    for model in PRESETS:
        for s,b in [(4096,2),(4096,8),(16384,2),(16384,8),(4096,32)]:
            add(model,'representative',b,s)
        for s,b in [(4096,2),(4096,8),(16384,8)]:
            for pass_type in ('memory','trace'):
                add(model,'diagnostic',b,s,pass_type=pass_type)
            if model=='qwen3.5-35b-a3b':
                add(model,'diagnostic',b,s,pass_type='moe')
        for pass_type in ('nsys','ncu'):
            add(model,'counter_timeline',8,4096,pass_type=pass_type)
        add(model,'natural',8,4096,natural=True)
        # Width/output sweeps are separate, sparse, not a Cartesian product.
        for block in (4,8): add(model,'width',8,4096,block=block)
        for output in (1024,4096): add(model,'output',2,4096,o=output)
    if extended:
        for model in PRESETS:
            for s in (4096,8192,16384,32768,65536):
                for b in (2,8):
                    if s not in (4096,16384): add(model,'context_expansion',b,s)
            for b in (4,16,32):
                for s in (4096,16384):
                    if (b,s)!=(32,4096): add(model,'batch_expansion',b,s)
    priority={'smoke':0,'smoke_natural':0,'smoke_profile':0,'representative':1,
        'context_expansion':2,'batch_expansion':3,'diagnostic':4,'counter_timeline':5,
        'natural':6,'width':7,'output':8}
    jobs.sort(key=lambda j:(priority[j['stage']],list(PRESETS).index(j['model'])))
    for i,job in enumerate(jobs):
        job['id']=f'{i:04d}-{job["model"]}-{job["stage"]}-{job["mode"]}-b{job["batch"]}-s{job["length"]}-o{job["tokens"]}-k{job["block"]}-{job["pass_type"]}'
    plan=dict(schema='dflash.batch.queue.v3',created=time.time(),gpus=GPUS,
        backend='fla',source_fingerprint=fingerprint(),jobs=jobs,
        scope='smoke B1/B2 only; research B2..32; 3 fixed layer shards; target-only AR subprocesses',
        candidate_grid=dict(S=[4096,8192,16384,32768,65536],B=[2,4,8,16,32],block=[4,8,16],output=[256,1024,4096]),
        extended=extended)
    write_json(path,plan)
    return plan


def gpu_idle(gpus):
    try:
        # NVML UUID lookup scans the unhealthy/unselected physical GPU 0 on
        # this host and can hang. Query the recorded NVML indices directly,
        # then verify UUID identity; CUDA jobs still select by UUID.
        selection=','.join(str(GPUS.index(uuid)+1) for uuid in gpus)
        result=subprocess.run(['nvidia-smi','-i',selection,
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=30,check=True)
        readings={}
        for line in result.stdout.strip().splitlines():
            uuid,mem,util=(s.strip() for s in line.split(','))
            readings[uuid]=dict(memory_mib=int(mem),utilization=int(util))
        apps=subprocess.run(['nvidia-smi','-i',selection,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=30,check=True)
        occupied={line.split(',')[0].strip() for line in apps.stdout.splitlines() if ',' in line}
        okay=set(readings)==set(gpus) and all(v['memory_mib']<1024 and v['utilization']<=5 for v in readings.values())
        return okay and not occupied.intersection(gpus),dict(readings=readings,compute_apps=apps.stdout)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        return False,dict(reason='GPU query failed; fail closed',error=str(exc))


def raw_first(folder):
    return json.loads((folder/'raw.jsonl').read_text().splitlines()[0])


def gate(plan, model, root):
    jobs=[j for j in plan['jobs'] if j['model']==model and j['stage'].startswith('smoke')]
    results={}
    for j in jobs:
        folder=root/'runs'/j['id']
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if status.get('status')!='complete': return False,dict(reason='smoke incomplete or failed',job=j['id'],status=status)
        results[(j['stage'],j['batch'],j['mode'])]=raw_first(folder)
    for stage,b in [('smoke',1),('smoke',2),('smoke_natural',2),('smoke_profile',2)]:
        ar,df=(results[(stage,b,m)] for m in ('ar','dflash'))
        if ar['tokens']!=df['tokens']:
            return False,dict(reason='AR/DFlash token mismatch',stage=stage,batch=b,
                ar_tokens=ar['tokens'],dflash_tokens=df['tokens'],tolerance='exact greedy tokens; no hidden tolerance')
    for mode in ('ar','dflash'):
        one,two=(results[('smoke',b,mode)] for b in (1,2))
        if one['tokens'][0]!=two['tokens'][0] or one['kept'][0]!=two['kept'][0]:
            return False,dict(reason='B1/B2 row0 output or acceptance mismatch',mode=mode)
        diag=results[('smoke_profile',2,mode)]
        if diag['tokens']!=two['tokens'] or diag['kept']!=two['kept']:
            return False,dict(reason='profiler perturbation changed output/acceptance',mode=mode)
    return True,dict(status='passed',tolerance='exact output tokens and acceptance',
        comparisons=['B1/B2','AR/DFlash','natural EOS','profiler on/off'])


def worker(plan_path,root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path(plan_path).read_text())
    lock=(root/'worker.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Queue worker already active')
    state=root/'queue_status.json'
    write_json(root/'worker.json',dict(pid=os.getpid(),plan=str(Path(plan_path).resolve()),started=time.time()))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(plan['gpus']),'OMP_NUM_THREADS':'4',
         'HF_HUB_OFFLINE':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True','PYTHONUNBUFFERED':'1'}
    locks=[]
    for uuid in sorted(plan['gpus']):
        handle=open('/tmp/dflash-batch-'+uuid+'.lock','a+')
        locks.append(handle)
    # Cooperative locks are not a cluster scheduler allocation. Actual device
    # occupancy is checked again before *every* subprocess.
    for handle in locks: fcntl.flock(handle,fcntl.LOCK_EX)
    for job in plan['jobs']:
        folder=root/'runs'/job['id']
        status_path=folder/'status.json'
        if status_path.exists():
            old=json.loads(status_path.read_text())
            if old.get('status') in ('complete','oom','capacity_preflight_rejected','skipped_correctness_gate','failed','unsupported'):
                continue
        if fingerprint()!=plan['source_fingerprint']:
            write_json(state,dict(status='blocked_source_changed',job=job['id'],reason='Create a new plan after review; existing results preserved'))
            return
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='batch':
            write_json(state,dict(status='blocked_branch_changed',job=job['id'])); return
        if not job['stage'].startswith('smoke'):
            passed,evidence=gate(plan,job['model'],root)
            write_json(root/'gates'/f'{job["model"]}.json',evidence)
            if not passed:
                write_json(status_path,dict(status='skipped_correctness_gate',evidence=evidence));continue
        prompt=root/'prompts'/f'{job["model"]}-s{job["length"]}.json'
        if not prompt.exists():
            write_json(state,dict(status='preparing_prompts',job=job['id']))
            try: prepare_prompts(prompt,job['model'],job['length'])
            except Exception as exc:
                write_json(status_path,dict(status='failed',reason='prompt preparation: '+str(exc)));continue
        while True:
            available,readings=gpu_idle(plan['gpus'])
            if available: break
            write_json(state,dict(status='waiting_for_gpus',pid=os.getpid(),job=job['id'],updated=time.time(),gpu=readings))
            time.sleep(30)
        # The source/branch may have changed while the worker waited for GPUs.
        if fingerprint()!=plan['source_fingerprint'] or subprocess.check_output(
                ['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='batch':
            write_json(state,dict(status='blocked_source_or_branch_changed',job=job['id']))
            return
        args=[sys.executable,'-u','-m','dflash.batch_bench','--model',job['model'],'--mode',job['mode'],
            '--backend',plan['backend'],'--batch',str(job['batch']),'--length',str(job['length']),
            '--tokens',str(job['tokens']),'--block',str(job['block']),'--pass-type',job['pass_type'],
            '--repeats','3','--prompts',str(prompt),'--output',str(folder)]
        if job['natural']: args.append('--natural')
        folder.mkdir(parents=True,exist_ok=True)
        if job['pass_type']=='nsys':
            executable=shutil.which('nsys')
            if not executable:
                write_json(status_path,dict(status='unsupported',reason='Nsight Systems missing'));continue
            args=[executable,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                  '--capture-range=cudaProfilerApi','--capture-range-end=stop',
                  '--output='+str(folder/'timeline')]+args
        elif job['pass_type']=='ncu':
            executable=shutil.which('ncu') or '/opt/nvidia/nsight-compute/2024.1.1/ncu'
            if not Path(executable).exists():
                write_json(status_path,dict(status='unsupported',reason='Nsight Compute missing'));continue
            # Separate pass: representative first 24 GEMM/attention/GDN kernels
            # in verify, not every kernel and not an E2E performance sample.
            sections=['SpeedOfLight','MemoryWorkloadAnalysis','ComputeWorkloadAnalysis',
                      'Occupancy','WarpStateStats','SpeedOfLight_HierarchicalTensorRooflineChart']
            args=[executable,'--target-processes','all','--replay-mode','kernel','--cache-control','none',
                '--clock-control','none','--nvtx','--nvtx-include','regex:batch::.*::(target_verify|ar_decode)::phase/',
                '--kernel-name','regex:.*(gemm|Gemm|matmul|attention|gated_delta|fwd_kernel).*',
                '--filter-mode','per-gpu','--launch-count','24','--export',str(folder/'counters')]+[
                    item for section in sections for item in ('--section',section)]+args
        write_json(folder/'command.json',dict(argv=args,env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','HF_HUB_OFFLINE','PYTORCH_ALLOC_CONF')}))
        with (folder/'process.log').open('a') as log:
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(state,dict(status='running',pid=os.getpid(),child_pid=child.pid,job=job['id'],started=time.time()))
            code=child.wait()
        if not status_path.exists() or json.loads(status_path.read_text()).get('status')=='running':
            write_json(status_path,dict(status='failed',reason='worker exited without final record',returncode=code))
        if job['pass_type']=='ncu' and (folder/'counters.ncu-rep').exists():
            with (folder/'counters.csv').open('w') as csv_file:
                subprocess.run([executable,'--import',str(folder/'counters.ncu-rep'),'--csv','--page','raw'],stdout=csv_file,check=False)
        if job['pass_type']=='nsys' and (folder/'timeline.nsys-rep').exists():
            subprocess.run([executable,'export','--type','sqlite','--output',str(folder/'timeline.sqlite'),str(folder/'timeline.nsys-rep')],check=False)
        # Persist a reviewable report after each condition, including failures.
        subprocess.run([sys.executable,'-m','dflash.batch_report',str(root)],cwd=ROOT,check=False)
    write_json(state,dict(status='finished',pid=os.getpid(),finished=time.time(),
        note='Inspect run statuses: finished queue does not imply all scientific conditions succeeded'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','start','worker','status'])
    parser.add_argument('--root',default='record_arch_batch/campaign')
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    path=root/'plan.json'
    if args.action=='plan':
        if path.exists(): raise SystemExit('Plan exists; use a new --root to preserve provenance')
        print(json.dumps(make_plan(path,args.extended),indent=2))
    elif args.action=='status':
        print((root/'queue_status.json').read_text() if (root/'queue_status.json').exists() else 'not started')
    elif args.action=='worker': worker(path,root)
    else:
        if not path.exists(): make_plan(path,args.extended)
        # Verify an existing worker PID before detaching another.
        if (root/'worker.json').exists():
            pid=json.loads((root/'worker.json').read_text())['pid']
            try:
                os.kill(pid,0)
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'dflash.batch_queue' in cmd:
                    raise SystemExit(f'Worker already active: {pid}')
            except (ProcessLookupError,FileNotFoundError): pass
        with (root/'worker.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','dflash.batch_queue','worker','--root',str(root)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,root=str(root),status='worker_started')))

if __name__=='__main__': main()
