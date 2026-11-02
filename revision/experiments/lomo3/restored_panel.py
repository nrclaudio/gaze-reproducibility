"""Run the three source-resolved targets and verify the complete 49-target panel."""
import argparse
import fcntl
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
    root=Path(root).resolve();base=root.parent;verify(root);verify(base)
    plan=read(root/'restored_plan.json');previous=base/plan['previous_bundle'];verify(previous)
    if sha(base/'manifest.json')!=plan['base_manifest_sha256'] or sha(previous/'manifest.json')!=plan['previous_manifest_sha256']:
        raise ValueError('Existing bundle identity changed.')
    for name,digest in plan['reused_files_sha256'].items():
        if sha(root/name)!=digest or sha(base/name)!=digest:
            raise ValueError('Training code or input changed: '+name)
    tasks=read(root/'tasks.json');prior_plan=read(previous/'panel_plan.json')
    if tasks!=plan['additional_tasks'] or len(tasks)!=3 or {t['node'] for t in tasks}!={845,1887,1894}:
        raise ValueError('Expected exactly the three source-resolved targets.')
    if plan['original_core_tasks']!=prior_plan['original_core_tasks'] or plan['previous_additional_tasks']!=prior_plan['additional_tasks']:
        raise ValueError('Existing target declarations changed.')
    all_tasks=plan['original_core_tasks']+plan['previous_additional_tasks']+tasks
    if len(all_tasks)!=49 or len({t['node'] for t in all_tasks})!=49:
        raise ValueError('Duplicate or missing historical target.')
    if {t['node'] for t in all_tasks}!={t['node'] for t in plan['combined_panel']['retained']}:
        raise ValueError('Combined panel does not match reviewed target identities.')
    if [t['node'] for t in plan['combined_panel']['excluded']]!=[1120]:
        raise ValueError('Expected only the androsterone exclusion.')
    for i,task in enumerate(tasks):
        if task['index']!=i or task['kind']!='lomo' or task['variant']!='full' or task['seed']!=42:
            raise ValueError('Task configuration differs from the fixed protocol.')
    for task in plan['original_core_tasks']:
        if sha(base/'results'/task['id']/'status.json')!=plan['existing_status_sha256'][task['id']]:
            raise ValueError('An existing core result changed.')
    return plan


def preflight(root):
    plan=validate(root);sys.path.insert(0,str(root/'code'))
    import torch
    import mudata as mu
    from src.data.sample_splits import manifest_splits
    from src.data.dataset_smiles import FiLMDatasetWithSMILES
    from worker import configuration,prediction_layout
    torch.set_num_threads(4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        dataset=FiLMDatasetWithSMILES(str(root/'inputs/camp_reanalysis.h5mu'),str(root/'inputs/graph.pt'),
            str(root/'inputs/gene_mapping.json'),device=torch.device('cpu'))
    data=dataset.mudata;values=np.asarray(dataset.metabolomics_dense)
    old=dict(np.load(root.parent/'inputs/assessment.npz',allow_pickle=False))
    new=dict(np.load(root/'inputs/assessment.npz',allow_pickle=False))
    if set(old)!=set(new):raise ValueError('Assessment array keys changed.')
    for key in old:
        if key!='nodes':np.testing.assert_array_equal(old[key],new[key])
    changes=np.flatnonzero(old['nodes']!=new['nodes'])
    expected={t['feature']:t['node'] for t in plan['additional_tasks']}
    if {str(old['features'][i]):int(new['nodes'][i]) for i in changes}!=expected or not np.all(old['nodes'][changes]==-1):
        raise ValueError('Assessment changes exceed the three reviewed assignments.')
    splits=read(root/'inputs/splits.json')['lomo']
    for phase in [1,2]:
        if manifest_splits(root/f'inputs/lomo_phase{phase}.csv',data['rna'].obs,data['metabolomics'].obs)!=splits[str(phase)]:
            raise ValueError('Sample partitions changed.')
    checked=[]
    for task in plan['additional_tasks']:
        layout=prediction_layout(root,task);column=int(layout['columns'][0]);node=task['node']
        if int(np.isfinite(values[:,column]).sum())!=task['n_observed']:
            raise ValueError('Target observation count changed.')
        np.testing.assert_array_equal(layout['targets'][:,0],values[:,column])
        if np.any(dataset.feature_nodes==node):raise ValueError('Restored target has unexpected supervised training labels.')
        for phase in [1,2]:
            config=configuration(root,task,phase,root/'preflight'/task['id']/f'phase{phase}')
            if config.data.excluded_target_nodes!=[node] or config.training.validation_objective!='prediction_mse':
                raise ValueError('Target-label exclusion or epoch selection changed.')
            dataset.excluded_target_nodes=set(config.data.excluded_target_nodes)
            for observations in values:
                y,mask=dataset._targets_for_values(observations)
                perturbed=observations.copy();perturbed[column]=123456.
                y2,mask2=dataset._targets_for_values(perturbed)
                if mask[node] or not torch.isnan(y[node]) or not torch.equal(mask,mask2) or not torch.allclose(y,y2,equal_nan=True):
                    raise ValueError('Target measurements affect supervised labels.')
        checked.append(node)
    result=dict(state='passed',manifest_sha256=sha(root/'manifest.json'),masked_nodes=checked,
        specimens_checked=len(values),both_phases_masked=True,training_inputs_unchanged=True,
        only_assessment_mapping_changed=True,combined_targets=49)
    save(root/'preflight/cpu.json',result);return result


def train(root,index):
    plan=validate(root);check=read(root/'preflight/cpu.json')
    if check['state']!='passed' or check['manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Missing matching input preflight.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('A Slurm GPU allocation is required.')
    if str(torch.__version__)!=plan['expected_torch_version']:raise RuntimeError('Original PyTorch environment required.')
    if not isinstance(index,int) or not 0<=index<3:raise ValueError('Invalid task index.')
    from worker import run
    return run(root,index)


def combined_summary(root,require_checkpoints=False):
    plan=validate(root);base=root.parent;sys.path.insert(0,str(root/'code'))
    from aggregate import verify_task,write_csv
    from reporting_stats import metrics_rows,score
    output=root/'combined_review';output.mkdir(exist_ok=True)
    inventory=[];metrics=[];details={};fingerprints=set()
    sources=[(base,plan['original_core_tasks'],'existing36'),
        (base/plan['previous_bundle'],plan['previous_additional_tasks'],'addition10'),
        (root,plan['additional_tasks'],'restored3')]
    for source,tasks,label in sources:
        for task in tasks:
            row=dict(task_id=task['id'],historical_fold=task['historical_fold'],node=task['node'],feature=task['feature'],source=label)
            try:
                data,detail=verify_task(source,task,require_checkpoints)
                fingerprints.add(detail['initialization']['full_reference_parameter_sha256'])
                rows=metrics_rows(data,data['predictions']);pooled=next(r for r in rows if r['group']=='__pooled_secondary__')
                primary,_=score(data['targets'],data['predictions'],data['groups'],'lomo')
                row.update(state='verified',within_study_spearman=primary if np.isfinite(primary) else None,
                    pooled_spearman=pooled['spearman'],pooled_r2=pooled['r2'],n=pooled['n'],selected_epoch=detail['selected_epoch'])
                metrics.extend(dict(task_id=task['id'],**r) for r in rows);details[task['id']]=detail
            except (OSError,ValueError,KeyError,TypeError) as exc:
                row.update(state='missing_or_unverified',reason=str(exc))
            inventory.append(row)
    if len(fingerprints)>1:raise ValueError('Original and added models have different shared initial weights.')
    inventory.sort(key=lambda row:row['historical_fold']);complete=all(r['state']=='verified' for r in inventory)
    def median(key):
        values=[r[key] for r in inventory if r['state']=='verified' and r[key] is not None]
        return float(np.median(values)) if values else None
    counts={label:sum(r['state']=='verified' and r['source']==label for r in inventory) for _,_,label in sources}
    summary=dict(state='complete' if complete else 'incomplete',expected=49,
        verified=sum(counts.values()),verified_by_source=counts,final_panel_metrics=None,
        excluded=plan['combined_panel']['excluded'],manifest_sha256=sha(root/'manifest.json'),
        base_manifest_sha256=plan['base_manifest_sha256'],previous_manifest_sha256=plan['previous_manifest_sha256'],
        require_checkpoints=require_checkpoints,details=details,qualifications=plan['qualifications'],single_seed=True)
    if complete:
        summary['final_panel_metrics']=dict(median_pooled_spearman=median('pooled_spearman'),
            median_pooled_r2=median('pooled_r2'),positive_pooled_r2=sum(r['pooled_r2'] is not None and r['pooled_r2']>0 for r in inventory),
            median_within_study_spearman=median('within_study_spearman'),
            pooled_spearman_scorable_targets=sum(r['pooled_spearman'] is not None for r in inventory),
            pooled_r2_scorable_targets=sum(r['pooled_r2'] is not None for r in inventory),
            within_study_scorable_targets=sum(r['within_study_spearman'] is not None for r in inventory))
    write_csv(output/'inventory.csv',inventory);write_csv(output/'per_target_metrics.csv',metrics);save(output/'summary.json',summary)
    lines=[f"Complete eligible LOMO panel: {summary['verified']}/49 verified. State: {summary['state']}.",
        f"Verified sources: {counts['existing36']}/36 existing, {counts['addition10']}/10 lookup additions, {counts['restored3']}/3 annotation restorations.",
        plan['qualifications']]
    lines.append(json.dumps(summary['final_panel_metrics']) if complete else 'No complete-panel median is reported until all 49 target fits pass verification.')
    (output/'findings.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return 0 if complete else 1


def submission_commands(root,settings,cap,previous_train,ids):
    commands={}
    for kind in ['preflight','train','aggregate']:
        cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--job-name=gaze_restored_'+kind,
            '--chdir='+str(root),'--output='+str(root/'logs'/('restored_'+kind+'_%A_%a.out')),
            '--error='+str(root/'logs'/('restored_'+kind+'_%A_%a.err')),
            '--export=ALL,GAZE_CONDA_ENV='+settings['environment'],'--cpus-per-task=4','--mem=32G']
        if kind=='train':
            cmd+=['--partition='+settings['partition'],'--gres=gpu:1','--time=08:00:00',
                '--array=0-2%'+str(cap),'--dependency=afterok:'+ids.get('preflight','<PREFLIGHT_ID>')+',afterany:'+previous_train]
        else:
            cmd+=['--partition='+settings['cpu_partition'],'--time=01:00:00']
            if kind=='aggregate':cmd+=['--dependency=afterany:'+ids.get('train','<TRAIN_ID>')]
        commands[kind]=cmd+[str(root/'restored_panel.sh'),kind]
    return commands


def submit(root,max_gpus=8,dry_run=False):
    if not 1<=max_gpus<=8:raise ValueError('GPU concurrency must be between 1 and 8.')
    plan=validate(root);base=root.parent
    original=read(base/('submission.json' if (base/'submission.json').exists() else 'remote_submission.json'))
    settings=original['settings'];cap=min(3,max_gpus)
    if dry_run:
        for cmd in submission_commands(root,settings,cap,'<TEN_TARGET_ARRAY_ID>',{}).values():print(shlex.join(cmd))
        return dict(dry_run=True,training_jobs=3,training_gpu_hour_ceiling=24,
            total_addition_gpu_hour_ceiling=104,maximum_concurrency=max_gpus,combined_targets=49,
            dependency='Three-target array starts only after the ten-target array terminates.')
    previous=read(base/plan['previous_bundle']/'submission.json')
    if previous['identity']['manifest_sha256']!=plan['previous_manifest_sha256']:
        raise ValueError('Previous submission does not match the ten-target bundle.')
    if previous['identity']['requested_max_gpus']!=max_gpus:
        raise ValueError('Use the same concurrency limit as the ten-target submission.')
    previous_train=previous['jobs'].get('train',{}).get('job_id','')
    if not re.fullmatch(r'\d+',previous_train):raise ValueError('The ten-target training array has no confirmed job ID.')
    available=set(subprocess.check_output(['sinfo','-h','-o','%P'],text=True).replace('*','').split())
    if (set(settings['partition'].split(','))|set(settings['cpu_partition'].split(',')))-available:
        raise ValueError('Original partitions unavailable.')
    (root/'logs').mkdir(exist_ok=True);journal=root/'submission.json'
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        identity=dict(manifest_sha256=sha(root/'manifest.json'),requested_max_gpus=max_gpus,
            previous_train_job_id=previous_train,settings=settings)
        record=read(journal) if journal.exists() else dict(identity=identity,cap=cap,training_gpu_hour_ceiling=24,jobs={})
        if record['identity']!=identity:raise ValueError('Submission identity changed; refusing duplicate fits.')
        for kind in ['preflight','train','aggregate']:
            prior=record['jobs'].get(kind)
            if prior:
                if re.fullmatch(r'\d+',prior.get('job_id','')):
                    print('Reusing '+kind+' job '+prior['job_id'],flush=True);continue
                raise RuntimeError('Uncertain submission receipt; inspect Slurm accounting before any retry.')
            ids={k:v['job_id'] for k,v in record['jobs'].items()}
            cmd=submission_commands(root,settings,record['cap'],previous_train,ids)[kind]
            entry=dict(state='attempting',command=cmd);record['jobs'][kind]=entry;save(journal,record)
            result=subprocess.run(cmd,text=True,capture_output=True);job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                entry.update(state='uncertain',stdout=result.stdout,stderr=result.stderr);save(journal,record)
                raise RuntimeError('Restored-target submission failed: '+result.stdout+result.stderr)
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
