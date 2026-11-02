"""Fail before the array on data/mask/configuration or real training-path errors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from common import save,sha,verify


def cpu(root):
    sys.path.insert(0,str(root/'code'))
    import torch
    import mudata as mu
    from src.data.sample_splits import manifest_splits
    from src.data.dataset_smiles import FiLMDatasetWithSMILES
    from protocol import make_tasks
    torch.set_num_threads(4)
    subprocess.run([sys.executable,'-m','unittest','-v','test_protocol'],cwd=root,check=True)
    m=mu.read_h5mu(root/'inputs/camp_reanalysis.h5mu')
    saved=json.loads((root/'inputs/splits.json').read_text())
    for kind in ['lomo','sample']:
        for phase in [1,2]:
            actual=manifest_splits(root/f'inputs/{kind}_phase{phase}.csv',m['rna'].obs,m['metabolomics'].obs)
            if actual!=saved[kind][str(phase)]: raise ValueError('Frozen sample split mismatch.')
    panel=json.loads((root/'inputs/lomo_panel.json').read_text())['retained']
    if make_tasks(panel)!=json.loads((root/'tasks.json').read_text()): raise ValueError('Task inventory mismatch.')
    dataset=FiLMDatasetWithSMILES(str(root/'inputs/camp_reanalysis.h5mu'),str(root/'inputs/graph.pt'),str(root/'inputs/gene_mapping.json'),device=torch.device('cpu'))
    a=np.load(root/'inputs/assessment.npz',allow_pickle=False)
    checked=[]
    for target in panel:
        col=list(a['features']).index(target['feature'])
        row=int(np.flatnonzero(np.isfinite(a['targets'][:,col]))[0])
        values=dataset.metabolomics_dense[row]
        dataset.excluded_target_nodes=set()
        original,mask=dataset._targets_for_values(values)
        if not mask[target['node']]: raise ValueError('Historical target not measurable.')
        dataset.excluded_target_nodes={target['node']}
        y,mask=dataset._targets_for_values(values)
        if mask[target['node']] or not torch.isnan(y[target['node']]): raise ValueError('LOMO target not masked.')
        perturbed=values.copy(); perturbed[col]=123456.
        y2,mask2=dataset._targets_for_values(perturbed)
        if not torch.equal(mask,mask2) or not torch.allclose(y,y2,equal_nan=True):
            raise ValueError('Held target perturbation changes supervised training labels.')
        checked.append(target['node'])
    record=dict(state='passed',manifest_sha256=sha(root/'manifest.json'),masked_nodes=checked,
        tasks=157,split_counts={k:{p:{s:len(v) for s,v in data.items()} for p,data in phases.items()} for k,phases in saved.items()})
    save(root/'preflight/cpu.json',record); return record


def smoke(root,gpu=True):
    import torch
    if gpu and not torch.cuda.is_available(): raise RuntimeError('GPU preflight has no CUDA device.')
    from protocol import VARIANTS
    from worker import run
    tasks=json.loads((root/'tasks.json').read_text()); first=tasks[0]['node']
    indices=[next(t['index'] for t in tasks if t['kind']=='lomo' and t['node']==first and t['variant']==v and t['seed']==42) for v in VARIANTS]
    indices.append(next(t['index'] for t in tasks if t['kind']=='sample' and t['variant']=='full' and t['seed']==42))
    records=[run(root,index,smoke=True) for index in indices]
    record=dict(state='passed',manifest_sha256=sha(root/'manifest.json'),smoke_tasks=[r['task']['id'] for r in records],
                device='cuda' if gpu else 'cpu',assessment_outcomes_scored=False)
    save(root/'preflight'/('gpu.json' if gpu else 'local_smoke.json'),record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--gpu',action='store_true'); p.add_argument('--local-smoke',action='store_true'); a=p.parse_args()
    root=a.root.resolve(); verify(root)
    result=smoke(root,gpu=a.gpu) if a.gpu or a.local_smoke else cpu(root)
    print(json.dumps(result,indent=2))
