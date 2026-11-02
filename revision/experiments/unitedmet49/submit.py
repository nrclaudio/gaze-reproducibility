"""Submit 49 CPU fits, a preflight, and verification, with durable receipts."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
from common import save,sha,verify

STAGES=['preflight','fits','review']


def command(root,stage,jobs,partition,environment,concurrency):
    cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes',
         '--job-name=um_lomo_'+stage,'--chdir='+str(root),
         '--output='+str(root/'logs'/(stage+'_%A_%a.out')),
         '--error='+str(root/'logs'/(stage+'_%A_%a.err')),
         '--partition='+partition,'--cpus-per-task=2','--mem=16G',
         '--time='+('04:00:00' if stage=='fits' else '01:00:00'),
         '--export=ALL,GAZE_CONDA_ENV='+environment]
    if stage=='fits':cmd+=['--array=0-48%'+str(concurrency),'--dependency=afterok:'+jobs['preflight']]
    if stage=='review':cmd+=['--dependency=afterany:'+jobs['fits']]
    return cmd+[str(root/'worker.sh'),stage]


def submit(root,partition,environment,concurrency,dry_run=False):
    root=Path(root).resolve();verify(root)
    assert len(json.loads((root/'tasks.json').read_text()))==49
    if not 1<=concurrency<=8:raise ValueError('Concurrency must be 1–8 CPU tasks.')
    for value in [partition,environment]:
        if not re.fullmatch(r'[A-Za-z0-9_,.-]+',value):raise ValueError('Invalid partition or environment.')
    if dry_run:
        for stage in STAGES:
            print(shlex.join(command(root,stage,{s:'<'+s+'_ID>' for s in STAGES},partition,environment,concurrency)))
        return {'dry_run':True,'cpu_fits':49,'cpu_concurrency':concurrency,'gpus':0}
    available=set(subprocess.check_output(['sinfo','-h','-o','%P'],text=True).replace('*','').split())
    if set(partition.split(','))-available:raise ValueError('Requested CPU partitions unavailable.')
    (root/'logs').mkdir(exist_ok=True)
    settings=dict(partition=partition,environment=environment,concurrency=concurrency,manifest_sha256=sha(root/'manifest.json'))
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'submission.json'
        record=json.loads(path.read_text()) if path.exists() else dict(settings=settings,jobs={})
        if record['settings']!=settings:raise ValueError('Existing submission settings differ.')
        for stage in STAGES:
            old=record['jobs'].get(stage)
            if old:
                if re.fullmatch(r'\d+',old.get('job_id','')):
                    print(f"Reusing {stage}: {old['job_id']}",flush=True);continue
                raise RuntimeError('Uncertain previous submission. Inspect accounting; no duplicate submitted.')
            ids={s:record['jobs'].get(s,{}).get('job_id','<PENDING>') for s in STAGES}
            cmd=command(root,stage,ids,partition,environment,concurrency)
            entry=dict(state='attempting',command=cmd);record['jobs'][stage]=entry;save(path,record)
            print(shlex.join(cmd),flush=True)
            result=subprocess.run(cmd,text=True,capture_output=True)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                entry.update(state='uncertain',stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
                save(path,record);raise RuntimeError('Submission failed: '+result.stderr)
            entry.update(state='submitted',job_id=job);save(path,record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition',default='cpu-zen4,cpu-skylake');p.add_argument('--conda-env',default='MetaboliteRNA')
    p.add_argument('--max-jobs',type=int,default=8);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    print(json.dumps(submit(Path.cwd(),a.partition,a.conda_env,a.max_jobs,a.dry_run),indent=2))
