"""Submit the whole bounded revision suite once, with at most eight simultaneous GPUs."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from common import save, sha, verify

STAGES=('cpu_preflight','gpu_preflight','core','lomo_ablation','aggregate')


def commands(root,partition,cpu_partition,environment,max_gpus,jobs):
    result={}
    for kind in STAGES:
        cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--job-name=gaze_rebuttal_'+kind,
             '--chdir='+str(root),'--output='+str(root/'logs'/(kind+'_%A_%a.out')),
             '--error='+str(root/'logs'/(kind+'_%A_%a.err')),'--export=ALL,GAZE_CONDA_ENV='+environment]
        gpu=kind in ('gpu_preflight','core','lomo_ablation')
        cmd+=['--partition='+(partition if gpu else cpu_partition),'--cpus-per-task=4','--mem=32G',
              '--time='+('06:00:00' if kind in ('core','lomo_ablation') else '01:00:00')]
        if gpu: cmd+=['--gres=gpu:1']
        if kind=='gpu_preflight': cmd+=['--dependency=afterok:'+jobs['cpu_preflight']]
        if kind=='core': cmd+=['--array=0-56%'+str(max_gpus),'--dependency=afterok:'+jobs['gpu_preflight']]
        if kind=='lomo_ablation':
            # Core must finish (success or failure); preflight must have succeeded.
            cmd+=['--array=0-99%'+str(max_gpus),'--dependency=afterok:'+jobs['gpu_preflight']+',afterany:'+jobs['core']]
        if kind=='aggregate': cmd+=['--dependency=afterany:'+jobs['core']+':'+jobs['lomo_ablation']]
        result[kind]=cmd+[str(root/'worker.sh'),kind]
    return result


def submit(root,partition='gpu-l4-24g',cpu_partition='cpu-zen4,cpu-skylake',environment='MetaboliteRNA',max_gpus=8,dry_run=False):
    root=Path(root).resolve(); verify(root)
    if not 1<=max_gpus<=8: raise ValueError('Maximum GPU concurrency must be between1 and8.')
    for value in [partition,cpu_partition,environment]:
        if not re.fullmatch(r'[A-Za-z0-9_,.-]+',value): raise ValueError('Invalid partition/environment name.')
    if ',' in partition: raise ValueError('Use one GPU partition for consistent hardware.')
    tasks=json.loads((root/'tasks.json').read_text()); protocol=json.loads((root/'research_protocol.json').read_text())
    if len(tasks)!=157 or [t['stage'] for t in tasks]!=['core']*57+['lomo_ablation']*100:
        raise ValueError('Unexpected tasks/stage ordering.')
    if protocol['training_gpu_hour_ceiling']!=942: raise ValueError('Unexpected training ceiling.')
    settings=dict(partition=partition,cpu_partition=cpu_partition,environment=environment,max_gpus=max_gpus,
                  manifest_sha256=sha(root/'manifest.json'))
    if dry_run:
        preview=commands(root,partition,cpu_partition,environment,max_gpus,{s:'<'+s.upper()+'_ID>' for s in STAGES})
        for cmd in preview.values(): print(shlex.join(cmd))
        return dict(preview=True,tasks=157,max_gpus=max_gpus,training_gpu_hour_ceiling=942,total_gpu_hour_ceiling=943)
    available=set(subprocess.check_output(['sinfo','-h','-o','%P'],text=True).replace('*','').split())
    missing=(set(partition.split(','))|set(cpu_partition.split(',')))-available
    if missing: raise ValueError('Partitions unavailable: '+', '.join(sorted(missing)))
    (root/'logs').mkdir(exist_ok=True)
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'submission.json'
        record=json.loads(path.read_text()) if path.exists() else dict(settings=settings,training_gpu_hour_ceiling=942,total_gpu_hour_ceiling=943,jobs={})
        if record['settings']!=settings: raise ValueError('Existing submission settings differ; refusing to alter an active run.')
        for kind in STAGES:
            existing=record['jobs'].get(kind)
            if existing:
                if re.fullmatch(r'\d+',existing.get('job_id','')):
                    print(f"Reusing {kind} job {existing['job_id']}",flush=True); continue
                raise RuntimeError(kind+' has an uncertain submission receipt. Inspect Slurm accounting; no duplicate submitted.')
            ids={s:record['jobs'].get(s,{}).get('job_id','<PENDING>') for s in STAGES}
            cmd=commands(root,partition,cpu_partition,environment,max_gpus,ids)[kind]
            entry=dict(state='attempting',command=cmd); record['jobs'][kind]=entry; save(path,record)
            print(shlex.join(cmd),flush=True)
            result=subprocess.run(cmd,text=True,capture_output=True)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                entry.update(state='uncertain',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                save(path,record); raise RuntimeError('sbatch did not return a valid job ID:\n'+result.stdout+result.stderr)
            entry.update(state='submitted',job_id=job); save(path,record)
        return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition',default='gpu-l4-24g'); p.add_argument('--cpu-partition',default='cpu-zen4,cpu-skylake')
    p.add_argument('--conda-env',default='MetaboliteRNA'); p.add_argument('--max-gpus',type=int,default=8)
    p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    try: result=submit(Path.cwd(),a.partition,a.cpu_partition,a.conda_env,a.max_gpus,a.dry_run)
    except (ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(str(exc),file=sys.stderr); raise SystemExit(2)
    print(json.dumps(result,indent=2))
