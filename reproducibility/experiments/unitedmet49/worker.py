"""Run one fixed target or a small, unscored preflight."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import pyro
from common import save,sha,verify
from model import training_matrix,fit,predict


def run(root,index=0,smoke=False):
    root=Path(root).resolve(); verify(root)
    task=json.loads((root/'tasks.json').read_text())[index]
    settings=json.loads((root/'protocol.json').read_text())['settings']
    assert task['index']==index
    out=root/('preflight' if smoke else 'results')/task['id']
    out.mkdir(parents=True,exist_ok=True)
    with (out/'claim.json').open('x') as f: json.dump(dict(task=task,smoke=smoke),f)
    started=time.monotonic()
    status=dict(state='running',smoke=smoke,task=task,manifest_sha256=sha(root/'manifest.json'),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),python=sys.version,
                torch=str(torch.__version__),pyro=str(pyro.__version__))
    save(out/'status.json',status)
    try:
        torch.set_num_threads(settings['threads'])
        with np.load(root/'inputs/fit.npz') as z: data=dict(z)
        column=task['column']
        met,rna=data['met'],data['rna']
        if smoke:
            # No assessment outcomes are loaded or scored during preflight.
            met=met[:24]; rna=rna[:24,:16]
            kept=[column]+[j for j in range(met.shape[1]) if j!=column and np.isfinite(met[:,j]).sum()>2][:7]
            met=met[:,kept];column=0
        matrix=training_matrix(met,rna,column)
        fitted,diagnostic=fit(root,matrix,column,dimensions=3 if smoke else settings['dimensions'],
            steps=3 if smoke else settings['steps'],lr=settings['lr'],seed=settings['seed'])
        prediction=predict(fitted,column,draws=8 if smoke else settings['posterior_draws'],seed=settings['seed'])
        if not np.isfinite(prediction).all(): raise ValueError('Invalid predictions.')
        np.savez_compressed(out/'posterior.npz',**fitted)
        if not smoke:
            # Assessment truth never enters the fit process or saved posterior.
            np.savez_compressed(out/'predictions.npz',predictions=prediction,
                sample_ids=data['sample_ids'],feature=np.array(task['feature']),node=np.array(task['node']))
        save(out/'config.json',dict(task=task,settings=settings,diagnostic=diagnostic,
            matrix_sha256=__import__('hashlib').sha256(matrix.tobytes()).hexdigest(),
            training_input_sha256=sha(root/'inputs/fit.npz'),assessment_loaded=False,
            observed_training_metabolites=int(np.isfinite(matrix[:,:met.shape[1]]).any(0).sum()),
            steps_run=len(fitted['losses']),smoke=smoke))
        verify(root)
        status.update(state='complete',elapsed_seconds=time.monotonic()-started,
            artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'})
        save(out/'status.json',status)
        print(task['id']+': '+('preflight' if smoke else 'fit')+' complete.',flush=True)
    except BaseException as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started)
        save(out/'status.json',status); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--task',type=int,default=0)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.root,a.task,a.smoke)
