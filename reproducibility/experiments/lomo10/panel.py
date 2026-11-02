"""Run and verify the separately frozen ten-target addition to the LOMO panel."""
import argparse
import fcntl
import getpass
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import warnings
import numpy as np
from common import save,sha,verify


def read(path):return json.loads(Path(path).read_text())


def validate(root):
    root=Path(root).resolve(); base=root.parent; verify(root); verify(base)
    plan=read(root/'panel_plan.json')
    if sha(base/'manifest.json')!=plan['base_manifest_sha256'] or sha(root/'base_manifest.json')!=plan['base_manifest_sha256']:
        raise ValueError('Original bundle identity changed.')
    for name,digest in plan['reused_files_sha256'].items():
        if sha(root/name)!=digest or sha(base/name)!=digest:
            raise ValueError('Original model code or inputs changed: '+name)
    tasks=read(root/'tasks.json')
    if tasks!=plan['additional_tasks'] or len(tasks)!=10:
        raise ValueError('Expected the ten predeclared additions.')
    existing=plan['original_core_tasks']; combined=plan['combined_panel']['retained']
    if len(existing)!=36 or len(combined)!=46 or len({t['node'] for t in existing+tasks})!=46:
        raise ValueError('Duplicate or missing target in combined panel.')
    if {t['node'] for t in existing+tasks}!={p['node'] for p in combined}:
        raise ValueError('Combined target identities differ.')
    for i,task in enumerate(tasks):
        if task['index']!=i or task['kind']!='lomo' or task['variant']!='full' or task['seed']!=42:
            raise ValueError('Addition is not a matched full-model seed42 LOMO task.')
    for task in existing:
        if sha(base/'results'/task['id']/'status.json')!=plan['existing_status_sha256'][task['id']]:
            raise ValueError('An existing core result changed.')
    return plan


def preflight(root):
    plan=validate(root);sys.path.insert(0,str(root/'code'))
    import torch
    import mudata as mu
    from src.data.metabolite_mapping import metabolite_feature_nodes
    from src.data.sample_splits import manifest_splits
    from src.data.dataset_smiles import FiLMDatasetWithSMILES
    from worker import configuration,prediction_layout
    torch.set_num_threads(4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        data=mu.read_h5mu(root/'inputs/camp_reanalysis.h5mu')
    graph=torch.load(root/'inputs/graph.pt',map_location='cpu',weights_only=False)
    nodes=metabolite_feature_nodes(data['metabolomics'].var,
        {v:i for i,v in enumerate(graph.hmdb_ids) if v},{v:i for i,v in enumerate(graph.kegg_ids) if v})
    values=data['metabolomics'].X
    values=values.toarray() if hasattr(values,'toarray') else np.asarray(values)
    spec=importlib.util.spec_from_file_location('frozen_panel_selector',root/'panel_selector.py')
    selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
    actual=selector.select_historical_lomo_panel(read(root/'historical_lomo_panel.json'),data['metabolomics'].var,
        nodes,graph,np.isfinite(values).sum(0))
    if actual!=plan['combined_panel']:raise ValueError('Panel selection is not reproduced by identifiers/coverage.')
    splits=read(root/'inputs/splits.json')['lomo']
    for phase in [1,2]:
        if manifest_splits(root/f'inputs/lomo_phase{phase}.csv',data['rna'].obs,data['metabolomics'].obs)!=splits[str(phase)]:
            raise ValueError('Sample splits changed.')
    dataset=FiLMDatasetWithSMILES(str(root/'inputs/camp_reanalysis.h5mu'),str(root/'inputs/graph.pt'),
        str(root/'inputs/gene_mapping.json'),device=torch.device('cpu'))
    checked=[]
    for task in plan['additional_tasks']:
        layout=prediction_layout(root,task); column=int(layout['columns'][0])
        if int(np.isfinite(values[:,column]).sum())!=task['n_observed']:raise ValueError('Target observation coverage changed.')
        row=int(np.flatnonzero(np.isfinite(values[:,column]))[0]); observations=dataset.metabolomics_dense[row]
        dataset.excluded_target_nodes=set()
        _,mask=dataset._targets_for_values(observations)
        if not mask[task['node']]:raise ValueError('Target not represented in dataset mapping.')
        for phase in [1,2]:
            config=configuration(root,task,phase,root/'preflight'/task['id']/f'phase{phase}')
            if config.data.excluded_target_nodes!=[task['node']] or config.training.validation_objective!='prediction_mse':
                raise ValueError('Target mask or MSE selection changed.')
            dataset.excluded_target_nodes=set(config.data.excluded_target_nodes)
            y,mask=dataset._targets_for_values(observations)
            perturbed=observations.copy();perturbed[column]=123456.
            y2,mask2=dataset._targets_for_values(perturbed)
            if mask[task['node']] or not torch.isnan(y[task['node']]) or not torch.equal(mask,mask2) or not torch.allclose(y,y2,equal_nan=True):
                raise ValueError('Held target values influence supervised labels.')
        checked.append(task['node'])
    result=dict(state='passed',manifest_sha256=sha(root/'manifest.json'),masked_nodes=checked,
        combined_targets=46,additional_targets=10,both_phases_masked=True,model_and_inputs_identical=True)
    save(root/'preflight/cpu.json',result);return result


def train(root,index):
    plan=validate(root);check=read(root/'preflight/cpu.json')
    if check['state']!='passed' or check['manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Missing matching input preflight.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('A Slurm GPU allocation is required.')
    if str(torch.__version__)!=plan['expected_torch_version']:raise RuntimeError('Original PyTorch environment required.')
    if not 0<=index<10:raise ValueError('Invalid task index.')
    from worker import run
    return run(root,index)


def combined_summary(root,require_checkpoints=False):
    plan=validate(root);base=root.parent;sys.path.insert(0,str(root/'code'))
    from aggregate import verify_task,write_csv
    from reporting_stats import metrics_rows,score
    output=root/'combined_review';output.mkdir(exist_ok=True)
    inventory=[];metrics=[];details={};fingerprints=set()
    for source,tasks in [(base,plan['original_core_tasks']),(root,plan['additional_tasks'])]:
        for task in tasks:
            row=dict(task_id=task['id'],historical_fold=task['historical_fold'],node=task['node'],feature=task['feature'],
                source='existing36' if source==base else 'addition10')
            try:
                data,detail=verify_task(source,task,require_checkpoints)
                fingerprint=detail['initialization']['full_reference_parameter_sha256'];fingerprints.add(fingerprint)
                rows=metrics_rows(data,data['predictions'])
                pooled=next(r for r in rows if r['group']=='__pooled_secondary__')
                primary,_=score(data['targets'],data['predictions'],data['groups'],'lomo')
                row.update(state='verified',within_study_spearman=primary if np.isfinite(primary) else None,
                    pooled_spearman=pooled['spearman'],pooled_r2=pooled['r2'],n=pooled['n'],
                    selected_epoch=detail['selected_epoch'])
                metrics.extend(dict(task_id=task['id'],**r) for r in rows);details[task['id']]=detail
            except (OSError,ValueError,KeyError,TypeError) as exc:
                row.update(state='missing_or_unverified',reason=str(exc))
            inventory.append(row)
    if len(fingerprints)>1:raise ValueError('Original and added fits have different shared initial weights.')
    inventory.sort(key=lambda row:row['historical_fold'])
    complete=all(row['state']=='verified' for row in inventory)
    def median(key):
        values=[r[key] for r in inventory if r['state']=='verified' and r[key] is not None]
        return float(np.median(values)) if values else None
    summary=dict(state='complete' if complete else 'incomplete',expected=46,
        verified=sum(r['state']=='verified' for r in inventory),
        original_verified=sum(r['state']=='verified' and r['source']=='existing36' for r in inventory),
        additional_verified=sum(r['state']=='verified' and r['source']=='addition10' for r in inventory),
        final_panel_metrics=None,excluded=plan['combined_panel']['excluded'],
        manifest_sha256=sha(root/'manifest.json'),base_manifest_sha256=sha(base/'manifest.json'),
        require_checkpoints=require_checkpoints,details=details,
        qualifications=plan['qualifications'],single_seed=True,independent_cohort_validation=False)
    if complete:
        summary['final_panel_metrics']=dict(median_pooled_spearman=median('pooled_spearman'),
            median_pooled_r2=median('pooled_r2'),positive_pooled_r2=sum(r['pooled_r2'] is not None and r['pooled_r2']>0 for r in inventory),
            median_within_study_spearman=median('within_study_spearman'),
            pooled_spearman_scorable_targets=sum(r['pooled_spearman'] is not None for r in inventory),
            pooled_r2_scorable_targets=sum(r['pooled_r2'] is not None for r in inventory),
            within_study_scorable_targets=sum(r['within_study_spearman'] is not None for r in inventory))
    write_csv(output/'inventory.csv',inventory);write_csv(output/'per_target_metrics.csv',metrics)
    save(output/'summary.json',summary)
    lines=[f"Combined LOMO panel: {summary['verified']}/46 verified ({summary['original_verified']}/36 reused, {summary['additional_verified']}/10 new). State: {summary['state']}.",
        'One seed; fixed-graph metabolite-label holdout. This is not independent cohort validation.',
        'Four historical targets remain excluded: three unresolved feature collisions and one distinct-compound mismatch.']
    if complete:lines.append(json.dumps(summary['final_panel_metrics']))
    else:lines.append('No final panel median is reported until all46 target fits pass verification.')
    lines.append(plan['qualifications'])
    (output/'findings.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return 0 if complete else 1


def submission_commands(root,settings,cap,wait_ids,ids):
    commands={}
    for kind in ['preflight','train','aggregate']:
        cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--job-name=gaze_panel_'+kind,
            '--chdir='+str(root),'--output='+str(root/'logs'/('panel_'+kind+'_%A_%a.out')),
            '--error='+str(root/'logs'/('panel_'+kind+'_%A_%a.err')),'--export=ALL,GAZE_CONDA_ENV='+settings['environment'],
            '--cpus-per-task=4','--mem=32G']
        if kind=='train':
            dependency='afterok:'+ids.get('preflight','<PREFLIGHT_ID>')
            if wait_ids:dependency+=',afterany:'+':'.join(wait_ids)
            cmd+=['--partition='+settings['partition'],'--gres=gpu:1','--time=08:00:00',
                  '--array=0-9%'+str(cap),'--dependency='+dependency]
        else:
            cmd+=['--partition='+settings['cpu_partition'],'--time=01:00:00']
            if kind=='aggregate':cmd+=['--dependency=afterany:'+ids.get('train','<TRAIN_ID>')]
        commands[kind]=cmd+[str(root/'panel.sh'),kind]
    return commands


def submit(root,max_gpus=8,dry_run=False):
    if not 1<=max_gpus<=8:raise ValueError('GPU concurrency must be between1 and8.')
    validate(root);base=root.parent
    original=read(base/('submission.json' if (base/'submission.json').exists() else 'remote_submission.json'))
    settings=original['settings'];recovery=base/'recovery_20261030/submission.json'
    if dry_run:
        for cmd in submission_commands(root,settings,max_gpus,['<ACTIVE_RECOVERY_IDS>'],{}).values():print(shlex.join(cmd))
        return dict(dry_run=True,training_jobs=10,training_gpu_hour_ceiling=80,maximum_concurrency=max_gpus,
            existing_fits_reused=36,combined_targets=46,
            concurrency_policy='Wait for active recorded original/recovery GPU jobs before starting up to8. If recovery is not fully submitted, cap additions at5 to reserve3 recovery GPUs.')
    active_tokens=subprocess.check_output(['squeue','-h','-u',getpass.getuser(),'-o','%F %A'],text=True).split()
    active={s.split('_')[0] for s in active_tokens if re.fullmatch(r'\d+(?:_.*)?',s)}
    available=set(subprocess.check_output(['sinfo','-h','-o','%P'],text=True).replace('*','').split())
    if (set(settings['partition'].split(','))|set(settings['cpu_partition'].split(',')))-available:
        raise ValueError('Original partitions unavailable.')
    prior_gpu=[v.get('job_id') for k,v in original['jobs'].items() if k in ['gpu_preflight','core','lomo_ablation']]
    recovery_ready=False
    if recovery.exists():
        rec=read(recovery)
        if rec['identity']['base_manifest_sha256']!=sha(base/'manifest.json'):raise ValueError('Recovery bundle mismatch.')
        recovery_ids=[rec['jobs'].get(k,{}).get('job_id') for k in ['refit','predict']]
        recovery_ready=all(isinstance(j,str) and re.fullmatch(r'\d+',j) for j in recovery_ids)
        prior_gpu+=recovery_ids
    wait_ids=sorted(set(j for j in prior_gpu if j in active))
    cap=max_gpus if recovery_ready else min(max_gpus,5)
    (root/'logs').mkdir(exist_ok=True);journal=root/'submission.json'
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        identity=dict(manifest_sha256=sha(root/'manifest.json'),requested_max_gpus=max_gpus,settings=settings)
        record=read(journal) if journal.exists() else dict(identity=identity,cap=cap,wait_ids=wait_ids,
            training_gpu_hour_ceiling=80,jobs={})
        if record['identity']!=identity:raise ValueError('Submission settings changed; refusing duplicate fits.')
        for kind in ['preflight','train','aggregate']:
            previous=record['jobs'].get(kind)
            if previous:
                if re.fullmatch(r'\d+',previous.get('job_id','')):
                    print('Reusing '+kind+' job '+previous['job_id'],flush=True);continue
                raise RuntimeError('Uncertain submission receipt; inspect Slurm accounting before any retry.')
            ids={k:v['job_id'] for k,v in record['jobs'].items()}
            cmd=submission_commands(root,settings,record['cap'],record['wait_ids'],ids)[kind]
            entry=dict(state='attempting',command=cmd);record['jobs'][kind]=entry;save(journal,record)
            result=subprocess.run(cmd,text=True,capture_output=True);job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                entry.update(state='uncertain',stdout=result.stdout,stderr=result.stderr);save(journal,record)
                raise RuntimeError('Panel submission failed: '+result.stdout+result.stderr)
            entry.update(state='submitted',job_id=job);save(journal,record);print(kind+': '+job,flush=True)
        return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','train','aggregate','submit'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--task',type=int);p.add_argument('--max-gpus',type=int,default=8)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--require-checkpoints',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.action=='preflight':print(json.dumps(preflight(root),indent=2))
    elif a.action=='train':train(root,a.task)
    elif a.action=='aggregate':raise SystemExit(combined_summary(root,a.require_checkpoints))
    else:print(json.dumps(submit(root,a.max_gpus,a.dry_run),indent=2))
