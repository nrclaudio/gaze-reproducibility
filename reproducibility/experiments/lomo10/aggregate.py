"""Verify the fixed experiment inventory and report every completed or missing task."""
import argparse
import csv
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from common import save, sha, verify
from protocol import VARIANTS, SEEDS, architecture
from reporting_stats import metrics_rows, score, paired_comparison, holm
from worker import prediction_layout, configuration


def read(path): return json.loads(Path(path).read_text())


def verify_task(root,task,require_checkpoints=False):
    output=root/'results'/task['id']; status=read(output/'status.json')
    if status['state']!='complete' or status['smoke'] or status['task']!=task:
        raise ValueError('Not a completed production task.')
    if status['manifest_sha256']!=sha(root/'manifest.json'): raise ValueError('Wrong frozen bundle.')
    artifacts=status['artifacts']; absent=[]
    required=['predictions.npz','selection.json','claim.json']
    for phase in [1,2]:
        required += [f'phase{phase}/'+name for name in ['best_model.pt','experiment_config.json','sample_indices.json',
            'initialization.json','training_history.json','epoch_diagnostics.csv','test_results.json']]
    required+=['phase1/best_validation_model.pt']
    if not set(required).issubset(artifacts): raise ValueError('Missing required recorded artifacts.')
    for name,digest in artifacts.items():
        path=output/name
        if not path.resolve().is_relative_to(output.resolve()): raise ValueError('Unsafe artifact path.')
        if path.suffix=='.pt' and not path.exists() and not require_checkpoints:
            absent.append(name); continue
        if sha(path)!=digest: raise ValueError('Changed or missing result artifact: '+name)
    selection=read(output/'selection.json'); epoch=selection['selected_epoch']
    if not isinstance(epoch,int) or not 1<=epoch<=300: raise ValueError('Invalid selected epoch.')
    histories=[read(output/f'phase{p}/training_history.json') for p in [1,2]]
    mse=np.asarray(histories[0]['val_prediction_mse'])
    if len(mse)>300 or not np.isfinite(mse).all() or int(np.argmin(mse))+1!=epoch:
        raise ValueError('Wrong MSE-selected epoch.')
    if not np.array_equal(mse,np.asarray(histories[0]['val_loss'])) or selection['selection_mse']!=float(mse[epoch-1]):
        raise ValueError('Selection objective/value mismatch.')
    if len(histories[1]['train_loss'])!=epoch: raise ValueError('Refit epoch count mismatch.')
    split_record=read(root/'inputs/splits.json')
    for phase,h in zip([1,2],histories):
        if not np.isfinite(h['train_loss']).all(): raise ValueError('Non-finite training loss.')
        saved=read(output/f'phase{phase}/experiment_config.json')
        expected=configuration(root,task,phase,output/f'phase{phase}',epochs=300 if phase==1 else epoch).to_dict()
        # Local and remote absolute paths differ; their contents are hash-bound.
        saved['training']['save_dir']=expected['training']['save_dir']
        for key in ['mudata_path','graph_path','gene_mapping_path','split_manifest_path']:
            saved['data'][key]=expected['data'][key]
        if saved!=expected: raise ValueError('Training configuration mismatch in phase'+str(phase))
        if read(output/f'phase{phase}/sample_indices.json')!=split_record[task['kind']][str(phase)]:
            raise ValueError('Actual fitted sample split differs from the frozen manifest.')
        if read(output/f'phase{phase}/initialization.json')!=selection['initialization']:
            raise ValueError('Initialization mismatch.')
        if read(output/f'phase{phase}/test_results.json')['total_epochs_run']!=len(h['train_loss']):
            raise ValueError('Incomplete training history.')
    if selection['objective']!='prediction_mse' or selection['target_exclusions']!=([task['node']] if task['kind']=='lomo' else []):
        raise ValueError('Held-out target masking or objective mismatch.')
    layout=prediction_layout(root,task)
    data=dict(np.load(output/'predictions.npz',allow_pickle=False))
    for key,value in layout.items():
        same=np.array_equal(value,data[key],equal_nan=True) if np.issubdtype(value.dtype,np.floating) else np.array_equal(value,data[key])
        if not same: raise ValueError('Assessment identity/truth mismatch: '+key)
    pred=data['predictions']
    if pred.shape!=layout['targets'].shape or not np.isfinite(pred).all(): raise ValueError('Invalid predictions.')
    if str(data['checkpoint_sha256'])!=artifacts['phase2/best_model.pt']: raise ValueError('Predictions/checkpoint identity mismatch.')
    return data,dict(selected_epoch=epoch,elapsed_seconds=status['elapsed_seconds'],
         parameters=selection['initialization']['parameters'],initialization=selection['initialization'],
         checkpoints_not_downloaded=absent)


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def aggregate(root,output=None,repetitions=1000,require_checkpoints=False):
    root=Path(root).resolve(); output=Path(output or root/'summary'); output.mkdir(parents=True,exist_ok=True)
    verify(root); sys.path.insert(0,str(root/'code'))
    tasks=read(root/'tasks.json'); results={}; inventory=[]; metrics=[]; details={}
    for task in tasks:
        row=dict(task_id=task['id'],kind=task['kind'],variant=task['variant'],seed=task['seed'],stage=task['stage'])
        if not (root/'results'/task['id']/'status.json').exists():
            row.update(state='missing'); inventory.append(row); continue
        try:
            data,detail=verify_task(root,task,require_checkpoints)
            results[task['id']]=data; details[task['id']]=detail
            primary,_=score(data['targets'],data['predictions'],data['groups'],task['kind'])
            row.update(state='verified',primary_spearman=primary if np.isfinite(primary) else None,
                       selected_epoch=detail['selected_epoch'],parameters=detail['parameters'],elapsed_seconds=detail['elapsed_seconds'])
            for metric in metrics_rows(data,data['predictions']): metrics.append(dict(task_id=task['id'],**metric))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            row.update(state='unverified',reason=str(exc))
        inventory.append(row)
    # Common full-model initialization must match across every fit with the same seed.
    for seed in SEEDS:
        fingerprints={details[t['id']]['initialization']['full_reference_parameter_sha256']
                      for t in tasks if t['seed']==seed and t['id'] in details}
        if len(fingerprints)>1: raise ValueError('Reference initialization differs across matched seed fits.')
    panel=read(root/'inputs/lomo_panel.json')['retained']; comparisons=[]; sections=[]
    for kind in ['sample','lomo']:
        if kind=='sample':
            identities=lambda variant,seed:[f'sample_{variant}_s{seed}']
        else:
            identities=lambda variant,seed:[f"lomo_n{p['node']}_{variant}_s{seed}" for p in panel[:5]]
        for variant in VARIANTS[1:]:
            needed=[i for method in ['full',variant] for seed in SEEDS for i in identities(method,seed)]
            missing=[i for i in needed if i not in results]
            entry=dict(kind=kind,variant=variant,reference='full',state='incomplete',missing_tasks=missing)
            if not missing:
                anchor=results[identities('full',SEEDS[0])[0]]
                truth=np.concatenate([results[i]['targets'] for i in identities('full',SEEDS[0])],axis=1)
                arrays=[]
                for method in ['full',variant]:
                    values=[]
                    for seed in SEEDS:
                        selected=[results[i] for i in identities(method,seed)]
                        for d in selected:
                            if not np.array_equal(d['sample_ids'],anchor['sample_ids']) or not np.array_equal(d['groups'],anchor['groups']):
                                raise ValueError('Ablation sample pairing differs.')
                        if not np.array_equal(np.concatenate([d['targets'] for d in selected],axis=1),truth,equal_nan=True):
                            raise ValueError('Ablation outcome pairing differs.')
                        values.append(np.concatenate([d['predictions'] for d in selected],axis=1))
                    arrays.append(np.asarray(values))
                entry.update(paired_comparison(truth,arrays[0],arrays[1],anchor['groups'],kind,repetitions))
                entry['state']='complete'
            comparisons.append(entry)
        family=[r for r in comparisons if r['kind']==kind]
        adjusted=holm([r.get('exact_seed_sign_p',1.) for r in family])
        for row,p in zip(family,adjusted):
            if row['state']=='complete': row['holm_seed_sign_p']=p
    core_ids=[f"lomo_n{p['node']}_full_s42" for p in panel]
    core=[r for r in inventory if r['task_id'] in core_ids and r['state']=='verified']
    verified=len(results); complete=verified==len(tasks)
    lines=[f'Verified {verified}/{len(tasks)} training tasks. State: '+('complete.' if complete else 'incomplete.'),
        'Retrospective reviewer analyses. No new independent external-validation claim.',
        'Full LOMO panel:36 targets,seed42. LOMO ablations:5 predefined targets,3 seeds. Sample ablations:one held source study,3 seeds.',
        f"LOMO panel: {len(core)}/36 verified. Sample fits: {sum(r['state']=='verified' and r['kind']=='sample' for r in inventory)}/21 verified.",
        'All variant effects below are ablation minus full; negative favors the full model.',
        'Bootstrap intervals are descriptive,conditional on these targets/studies. Exact seed sign tests have only3 seed replicates.',
        'MSE and R2 use the existing relative-abundance scale. Loss/flux diagnostics do not validate biological fluxes.']
    if len(core)==36:
        values=[r['primary_spearman'] for r in core if r['primary_spearman'] is not None]
        lines.append(f'Full LOMO panel median of target within-study medians: {float(np.median(values)):.4f} ({len(values)}/36 scorable targets).')
    for row in comparisons:
        if row['state']=='complete':
            lo,hi=row['descriptive_bootstrap_ci95']
            lines.append(f"{row['kind']} {row['variant']}: effect {row['mean_effect']:+.4f}; descriptive95% interval [{lo:+.4f},{hi:+.4f}]; Holm seed sign p={row['holm_seed_sign_p']:.3f}.")
        else: lines.append(f"{row['kind']} {row['variant']}: incomplete ({len(row['missing_tasks'])} required fits unavailable).")
    if not complete: lines.append('Missing or unverified fits are not successful experiments; inspect inventory.csv and Slurm logs/accounting.')
    lines.append('Checkpoint files may remain on ALICE; download with --checkpoints for independent local weight verification.')
    write_csv(output/'inventory.csv',inventory); write_csv(output/'per_target_metrics.csv',metrics)
    save(output/'comparisons.json',comparisons)
    save(output/'verification.json',dict(state='complete' if complete else 'incomplete',verified=verified,expected=len(tasks),
        manifest_sha256=sha(root/'manifest.json'),details=details,inventory=inventory,
        checkpoint_files_required=require_checkpoints,bootstrap_repetitions=repetitions))
    (output/'findings.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path); p.add_argument('--require-checkpoints',action='store_true')
    a=p.parse_args(); raise SystemExit(aggregate(a.root,a.output,require_checkpoints=a.require_checkpoints))
