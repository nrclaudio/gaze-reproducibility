"""Verify released files and recompute manuscript summaries; optional CPU checkpoint replay."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def correlation(y,p,minimum=3):
 keep=np.isfinite(y);y=y[keep];p=p[keep]
 assert np.isfinite(p).all()
 if len(y)<minimum or np.ptp(y)==0:return float('nan')
 return 0.0 if np.ptp(p)==0 else float(spearmanr(y,p).statistic)

def verify_files():
 manifest=json.loads((ROOT/'release_manifest.json').read_text())
 for name,expected in manifest['files'].items():
  p=ROOT/name
  if not p.is_file() or sha(p)!=expected:raise ValueError('Release file differs: '+name)
 for folder in (ROOT/'experiments').iterdir():
  for name,expected in json.loads((folder/'manifest.json').read_text())['files'].items():
   assert sha(folder/name)==expected,(folder,name)
 return len(manifest['files'])

def metrics():
 table=pd.read_csv(ROOT/'manuscript_data/lomo49.csv');rows=[];fits=0;ablation=[]
 datasets={}
 for folder in ['lomo36','lomo10','lomo3']:
  root=ROOT/'experiments'/folder
  for task in json.loads((root/'tasks.json').read_text()):
   out=root/'results'/task['id'];status=json.loads((out/'status.json').read_text())
   assert status['state']=='complete',task['id']
   for name,expected in status['artifacts'].items():
    if name.endswith('.pt') and not (out/name).exists():continue
    assert sha(out/name)==expected,(task['id'],name)
   z=dict(np.load(out/'predictions.npz'));fits+=1
   assert str(z['checkpoint_sha256'])==status['artifacts']['phase2/best_model.pt']
   if task['kind']=='lomo' and task['variant']=='full' and task['seed']==42:datasets[task['id']]=z
   corrs=[correlation(z['targets'][:,j],z['predictions'][:,j],minimum=10) for j in range(z['targets'].shape[1])]
   if folder=='lomo36':ablation.append(dict(**task,correlations=corrs))
 for row in table.itertuples():
  z=datasets[row.task_id];y=z['targets'][:,0];p=z['predictions'][:,0];keep=np.isfinite(y)
  rho=correlation(y,p);r2=float(r2_score(y[keep],p[keep]));assert np.isclose(rho,row.pooled_spearman,atol=1e-10);assert np.isclose(r2,row.pooled_r2,atol=1e-5)
  rows.append((rho,r2))
 assert fits==170 and len(rows)==49
 source=pd.read_csv(ROOT/'manuscript_data/ablation_seed_metrics.csv')
 panel=set(json.loads((ROOT/'experiments/lomo36/research_protocol.json').read_text())['lomo_ablation_targets'][i]['node'] for i in range(5))
 for row in source.itertuples():
  selected=[t for t in ablation if t['kind']==row.kind and t['variant']==row.variant and t['seed']==row.seed and (row.kind=='sample' or t['node'] in panel)]
  values=[v for t in selected for v in t['correlations'] if np.isfinite(v)]
  assert len(values)==row.targets and np.isclose(np.median(values),row.median_pooled_spearman,atol=1e-10),(row,values)
 um=np.load(ROOT/'experiments/unitedmet49/review/predictions.npz');um_values=[];higher=0
 for row in table.itertuples():
  z=datasets[row.task_id];j=np.flatnonzero(um['nodes']==row.node).item()
  assert np.array_equal(um['sample_ids'],z['sample_ids'])
  rho=correlation(z['targets'][:,0],um['predictions'][:,j]);um_values.append(rho);higher+=row.pooled_spearman>rho
 external={}
 for mode in ['cohort','cohort_plus_targets']:
  run=ROOT/'external/runs'/mode;z=np.load(run/'evaluation_primary/predictions.npz');meta=pd.read_csv(run/'evaluation_primary/samples.csv')
  nodes=json.loads((ROOT/'external/artifacts/heldout_target_nodes.json').read_text()) if mode=='cohort_plus_targets' else z['node_indices']
  cols=np.flatnonzero(np.isin(z['node_indices'],nodes))
  for cohort in ['RC18','RC20']:
   ix=np.flatnonzero(meta.holdout_cohort.eq(cohort));v=[correlation(z['targets'][ix,j],z['predictions'][ix,j]) for j in cols]
   external[mode+'/'+cohort]=float(np.nanmedian(v))
 return dict(verified_gaze_fits=fits,lomo_targets=len(rows),lomo_median_spearman=float(np.median(np.array(rows)[:,0])),lomo_median_r2=float(np.median(np.array(rows)[:,1])),lomo_positive_r2=int((np.array(rows)[:,1]>0).sum()),ablation_seed_summaries=len(source),unitedmet_median_spearman=float(np.median(um_values)),gaze_higher_than_unitedmet=int(higher),external_median_spearman=external)

def replay():
 import torch
 sys.path.insert(0,str(ROOT/'external/code'))
 from src.config import ExperimentConfig
 from src.data.dataset_smiles import FiLMDatasetWithSMILES
 from src.models.factory import create_model
 torch.set_num_threads(2);art=ROOT/'external/artifacts'
 data=FiLMDatasetWithSMILES(str(art/'camp_reanalysis.h5mu'),str(art/'graph.pt'),str(art/'gene_mapping.json'),device=torch.device('cpu'))
 ids=pd.Index(data.mudata['rna'].obs.original_index.astype(str));checks=[]
 for mode in ['cohort','cohort_plus_targets']:
  run=ROOT/'external/runs'/mode;cfg=ExperimentConfig.from_dict(json.loads((run/'phase2/experiment_config.json').read_text()))
  model=create_model(cfg.model,embedding_dim=data.embedding_dim,gene_dim=data.gene_dim,smiles_dim=data.smiles_dim)
  model.load_state_dict(torch.load(run/'phase2/best_model.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
  saved=np.load(run/'evaluation_primary/predictions.npz');meta=pd.read_csv(run/'evaluation_primary/samples.csv',dtype=str,keep_default_na=False)
  for cohort in ['RC18','RC20']:
   ix=np.flatnonzero(meta.holdout_cohort.eq(cohort))
   for pos in ix[[0,len(ix)//2,-1]]:
    sample=data[ids.get_loc(meta.rna_original_id.iloc[pos])];sample.y[:]=float('nan');sample.mask[:]=False
    with torch.no_grad():actual=model(sample,compute_physics=False)[0].reshape(-1)[saved['node_indices']].numpy()
    expected=saved['predictions'][pos];assert np.allclose(actual,expected,rtol=1e-4,atol=1e-3),(mode,cohort,pos)
    checks.append(float(np.max(np.abs(actual-expected))))
 return dict(checkpoint_files=2,samples_replayed=len(checks),max_abs_difference=max(checks),rtol=1e-4,atol=1e-3,targets_hidden_during_inference=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 report=dict(files_verified=verify_files(),**metrics())
 if a.replay:report['checkpoint_replay']=replay()
 print(json.dumps(report,indent=2))
 if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
