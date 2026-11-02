"""One fixed two-phase fit; no holdout-driven selection or automatic retries."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from common import save, sha, verify
from protocol import architecture, matched_model


def configuration(root, task, phase, destination, epochs=300, smoke=False):
    from src.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, PhysicsConfig
    arch, physics = architecture(task['variant'])
    manifest = root / f"inputs/{task['kind']}_phase{phase}.csv"
    if smoke:
        import pandas as pd
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        keep = list(frame.index[frame.partition.eq('train')][:4])
        if phase == 1: keep += list(frame.index[frame.partition.eq('validation')][:4])
        frame.loc[~frame.index.isin(keep), 'partition'] = 'excluded'
        manifest = destination.parent / f'smoke_phase{phase}.csv'
        frame.to_csv(manifest, index=False)
    return ExperimentConfig(seed=task['seed'], config_name=task['id'] + f'_phase{phase}',
        model=ModelConfig(variant='enhanced', hidden_dim=128, num_layers=3, dropout=.2, enhanced_config=arch.to_dict()),
        data=DataConfig(mudata_path=str(root/'inputs/camp_reanalysis.h5mu'), graph_path=str(root/'inputs/graph.pt'),
            gene_mapping_path=str(root/'inputs/gene_mapping.json'), split_manifest_path=str(manifest),
            mudata_sha256=sha(root/'inputs/camp_reanalysis.h5mu'), split_manifest_sha256=sha(manifest),
            excluded_target_nodes=[task['node']] if task['kind']=='lomo' else [],
            batch_size=4, gradient_accumulation_steps=4, num_workers=0, pin_memory=False),
        training=TrainingConfig(learning_rate=.001, max_epochs=epochs, patience=20 if phase==1 else epochs+1,
            save_dir=str(destination), validation_objective='prediction_mse', mask_target_smiles=.5),
        physics=PhysicsConfig(**physics))


def fit(config):
    import torch
    import src.training.trainer as module
    from src.training.callbacks import DiagnosticsCallback
    original_factory, original_diagnostics = module.create_model, module.DiagnosticsCallback
    initialized = []
    def factory(model_config, **dimensions):
        model, record = matched_model(model_config, config.seed, **dimensions)
        initialized.append(record)
        return model
    class Diagnostics(DiagnosticsCallback):
        def on_batch_end(self, *args, **kwargs): pass  # Epoch aggregates are recorded below.
        def on_epoch_end(self, trainer, epoch, logs):
            row = dict(epoch=epoch+1)
            for key in ['train_loss','val_loss','val_configured_loss','val_prediction_mse']:
                row[key] = logs.get(key, float('nan'))
            for category in ['loss_components','flux_stats']:
                for key, value in logs.get(category, {}).items(): row[category+'_'+key] = value
            path = Path(self.model_dir) / 'epoch_diagnostics.csv'
            with path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if epoch == 0: writer.writeheader()
                writer.writerow(row)
    module.create_model, module.DiagnosticsCallback = factory, Diagnostics
    try:
        result = module.train_experiment(config,
            scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300))
    finally:
        module.create_model, module.DiagnosticsCallback = original_factory, original_diagnostics
    if len(initialized)!=1: raise ValueError('Expected one fresh initialization per phase.')
    save(Path(config.training.save_dir)/'initialization.json', initialized[0])
    return result, initialized[0]


def prediction_layout(root, task):
    a = dict(np.load(root/'inputs/assessment.npz', allow_pickle=False))
    splits = json.loads((root/'inputs/splits.json').read_text())
    if task['kind']=='lomo':
        rows = np.arange(len(a['sample_ids']))
        columns = np.flatnonzero(a['features']==task['feature'])
        if len(columns)!=1 or int(a['nodes'][columns[0]])!=task['node']: raise ValueError('Target identity mismatch.')
        groups = a['groups'][rows]
    else:
        rows = np.asarray(splits['sample']['2']['holdout'], dtype=int)
        columns = a['sample_columns']; groups = a['strata'][rows]
    return dict(indices=rows, columns=columns, groups=groups, sample_ids=a['sample_ids'][rows],
                features=a['features'][columns], nodes=a['nodes'][columns], targets=a['targets'][np.ix_(rows,columns)])


def predict(root, config, task, output, smoke=False):
    import torch
    from src.data.dataset_smiles import FiLMDatasetWithSMILES
    from src.models.factory import create_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FiLMDatasetWithSMILES(config.data.mudata_path, config.data.graph_path, config.data.gene_mapping_path, device=device)
    model = create_model(config.model, embedding_dim=dataset.embedding_dim, gene_dim=dataset.gene_dim, smiles_dim=dataset.smiles_dim).to(device)
    checkpoint = output/'phase2/best_model.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    layout = prediction_layout(root, task)
    if smoke:
        # Predict source fitting rows only; no held sample/metabolite outcome is scored.
        layout['indices'] = np.array(json.loads((output/'phase2/sample_indices.json').read_text())['train'][:2])
    predictions = []
    with torch.no_grad():
        for index in layout['indices']:
            data = dataset[int(index)].to(device)
            data.y = torch.full_like(data.y, float('nan')); data.mask = torch.zeros_like(data.mask)
            pred = model(data, compute_physics=False)[0].reshape(-1)
            predictions.append(pred[torch.as_tensor(layout['nodes'], device=device)].cpu().numpy())
    predictions = np.asarray(predictions)
    if not np.isfinite(predictions).all(): raise ValueError('Non-finite frozen-model prediction.')
    if smoke:
        save(output/'prediction_smoke.json', dict(rows=len(predictions), columns=predictions.shape[1], finite=True,
            assessment_outcomes_scored=False, checkpoint_sha256=sha(checkpoint)))
    else:
        np.savez_compressed(output/'predictions.npz', **layout, predictions=predictions,
                            checkpoint_sha256=np.array(sha(checkpoint)))


def run(root, task_index, smoke=False):
    root = Path(root).resolve(); verify(root)
    sys.path.insert(0, str(root/'code'))
    import torch
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK','4'))))
    tasks = json.loads((root/'tasks.json').read_text()); task = tasks[task_index]
    if task['index'] != task_index: raise ValueError('Task index mismatch.')
    output = root/('smoke' if smoke else 'results')/task['id']
    output.mkdir(parents=True, exist_ok=True)
    # Atomic claim: never silently overwrite a partially completed fit.
    with (output/'claim.json').open('x') as f: json.dump(dict(task=task, smoke=smoke), f)
    start = time.time()
    record = dict(task=task, smoke=smoke, state='running', manifest_sha256=sha(root/'manifest.json'),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'), device='cuda' if torch.cuda.is_available() else 'cpu',
                  torch_version=str(torch.__version__))
    save(output/'status.json', record)
    try:
        config1 = configuration(root, task, 1, output/'phase1', epochs=1 if smoke else 300, smoke=smoke)
        result1, init1 = fit(config1)
        history = json.loads((output/'phase1/training_history.json').read_text())
        values = np.asarray(history['val_prediction_mse'])
        if not np.isfinite(values).all() or not np.isfinite(history['train_loss']).all():
            raise ValueError('Non-finite selection training/validation history.')
        selected = int(np.argmin(values))+1
        if result1['best_epoch']!=selected or not np.array_equal(values, np.asarray(history['val_loss'])):
            raise ValueError('Checkpoint selection did not use ordinary validation MSE.')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        config2 = configuration(root, task, 2, output/'phase2', epochs=selected, smoke=smoke)
        result2, init2 = fit(config2)
        history2 = json.loads((output/'phase2/training_history.json').read_text())
        if result2['total_epochs_run']!=selected or not np.isfinite(history2['train_loss']).all():
            raise ValueError('Refit did not finish the selected number of epochs.')
        if init1!=init2: raise ValueError('Two phases did not start from identical weights.')
        save(output/'selection.json', dict(selected_epoch=selected, selection_mse=float(values[selected-1]),
             objective='prediction_mse', target_exclusions=config1.data.excluded_target_nodes, initialization=init1))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        predict(root, config2, task, output, smoke=smoke)
        verify(root)
        record.update(state='complete', elapsed_seconds=time.time()-start,
            artifacts={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                       if p.is_file() and p.name!='status.json'})
        save(output/'status.json', record)
        return record
    except Exception as exc:
        record.update(state='failed', error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.time()-start)
        save(output/'status.json', record)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--task',type=int,required=True); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); run(a.root,a.task,a.smoke)
