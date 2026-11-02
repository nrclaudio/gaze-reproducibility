"""Fresh, manifest-verified retrospective cohort holdout; no holdout scoring here."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, PhysicsConfig
from src.models.components import EnhancedModelConfig
from src.data.sample_splits import verify_sha256
from src.training.trainer import train_experiment


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_artifacts(artifacts):
    summary = json.loads((artifacts / 'preparation_summary.json').read_text())
    for name, expected in summary['artifact_hashes'].items():
        verify_sha256(artifacts / name, expected)
    return summary


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for folder in ['src', 'scripts']
            for p in sorted((ROOT / folder).rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['cohort', 'cohort_plus_targets'], default='cohort')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true', help='Two small development batches only; not a scientific result.')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.accumulation, args.threads) < 1:
        parser.error('Epoch, patience, batch, accumulation and thread counts must be positive.')
    artifacts, out = args.artifacts.resolve(), args.output.resolve()
    summary = verify_artifacts(artifacts)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    hashes = code_hashes()
    status = {'state': 'running', 'smoke_only': args.smoke, 'mode': args.mode, 'seed': args.seed,
              'started_unix': time.time(), 'artifact_hashes': summary['artifact_hashes'], 'code_hashes': hashes,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'cuda': torch.version.cuda, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                              'packages': {name: importlib.metadata.version(name) for name in
                                  ['torch-geometric', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'mudata', 'anndata', 'h5py']}},
              'protocol': 'Retrospective cohort holdout. Fresh initialization in both phases; no historical weights or encoders loaded. Fixed historical architecture/weights are not independent of prior development. Source-study preprocessing retained; no fitted model-level transform.',
              'schedule': 'CosineAnnealingLR with the same fixed horizon in both phases; final duration chosen only using development validation loss.',
              'primary_analysis': 'Within-cohort, per-target Pearson correlation, minimum 3 observed outcomes; report all target counts. Tumour and normal strata are descriptive. Pooled correlation is secondary.',
              'target_panel': '20 nodes chosen by development coverage only; their labels are masked in fitting and epoch selection in cohort_plus_targets mode.'}
    def save_status():
        (out / 'run_status.json').write_text(json.dumps(status, indent=2))
    save_status()
    try:
        nodes = json.loads((artifacts / 'heldout_target_nodes.json').read_text()) if args.mode == 'cohort_plus_targets' else []
        manifests = []
        for phase in [1, 2]:
            path = artifacts / f'phase{phase}_manifest.csv'
            if args.smoke:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                for role in ['train', 'validation']:
                    rows = frame.index[frame.partition.eq(role)]
                    frame.loc[rows[2 * args.batch_size:], 'partition'] = 'excluded'
                path = out / f'smoke_phase{phase}_manifest.csv'
                frame.to_csv(path, index=False)
            manifests.append(path)
        arch = EnhancedModelConfig.with_features(['residual_connections', 'cross_modal_attention', 'mcr',
                                                 'flux_messages', 'mass_balance_projection'])
        arch.layer_type = 'Relational'
        arch.structural_loss_weight = 1.0
        def config(phase, epochs):
            return ExperimentConfig(seed=args.seed, config_name=f'{args.mode}_phase{phase}',
                model=ModelConfig(variant='enhanced', hidden_dim=128, num_layers=3, dropout=0.2,
                                  use_directionality=True, enhanced_config=arch.to_dict()),
                data=DataConfig(mudata_path=str(artifacts / 'camp_reanalysis.h5mu'),
                    graph_path=str(artifacts / 'graph.pt'), gene_mapping_path=str(artifacts / 'gene_mapping.json'),
                    split_manifest_path=str(manifests[phase - 1]),
                    mudata_sha256=summary['artifact_hashes']['camp_reanalysis.h5mu'],
                    split_manifest_sha256=digest(manifests[phase - 1]), excluded_target_nodes=nodes,
                    batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
                    num_workers=0, pin_memory=False),
                training=TrainingConfig(learning_rate=0.001, max_epochs=epochs,
                    patience=args.patience if phase == 1 else epochs + 1,
                    save_dir=str(out / f'phase{phase}'), mask_target_smiles=0.5),
                physics=PhysicsConfig(lambda_balance=0.5, lambda_correlation=0.1, lambda_variance=1.0))
        horizon = 1 if args.smoke else args.epochs
        def schedule(optimizer):
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon)
        first = train_experiment(config(1, horizon), scheduler_factory=schedule)
        chosen = first['best_epoch']
        history = json.loads((out / 'phase1/training_history.json').read_text())
        if not history['val_loss'] or not np.isfinite(history['val_loss']).all() or chosen < 1:
            raise RuntimeError('No finite validation-based epoch selection; final training refused.')
        status.update(state='phase2', selected_epoch=chosen)
        save_status()
        second = train_experiment(config(2, chosen), scheduler_factory=schedule)
        if second['total_epochs_run'] != chosen:
            raise RuntimeError('Final training stopped before the selected epoch.')
        final_history = json.loads((out / 'phase2/training_history.json').read_text())
        if not np.isfinite(final_history['train_loss']).all():
            raise RuntimeError('Non-finite final training loss; checkpoint cannot be evaluated.')
        if code_hashes() != hashes:
            raise RuntimeError('Code changed during training; refusing to mark this run complete.')
        status.update(state='smoke_complete' if args.smoke else 'training_complete',
            final_checkpoint_sha256=digest(out / 'phase2/best_model.pt'),
            final_config_sha256=digest(out / 'phase2/experiment_config.json'),
            phase1_config_sha256=digest(out / 'phase1/experiment_config.json'), finished_unix=time.time())
        save_status()
        print(json.dumps({k: status[k] for k in ['state', 'selected_epoch', 'final_checkpoint_sha256']}, indent=2))
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        save_status()
        raise


if __name__ == '__main__':
    main()
