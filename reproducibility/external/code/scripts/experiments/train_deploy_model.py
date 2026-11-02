#!/usr/bin/env python3
"""Train a deployment model using two-phase training.

Phase 1: Train with a 90/10 train/val split to find the optimal epoch via
          early stopping (ReduceLROnPlateau + patience).
Phase 2: Retrain from scratch on ALL data for exactly ``best_epoch`` epochs
          using CosineAnnealingLR.

This ensures the final deployed model is trained on 100% of data without
overfitting, because we know exactly when to stop.

Usage (local smoke test):
    python scripts/experiments/train_deploy_model.py \
        --phase1-epochs 10 --batch-size 1 --output-dir /tmp/deploy_test

Usage (cluster):
    sbatch scripts/slurm/submit_deploy_model.sh
"""

import argparse
import os
import sys
import torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, PhysicsConfig
from src.models.components import EnhancedModelConfig
from src.training.trainer import train_experiment
from src.inference.normalization import compute_gene_statistics
from src.data.datamodule import FiLMDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description='Two-phase deployment model training'
    )
    parser.add_argument('--phase1-epochs', type=int, default=300,
                        help='Max epochs for Phase 1 (default: 300)')
    parser.add_argument('--phase1-patience', type=int, default=20,
                        help='Early-stopping patience for Phase 1 (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--output-dir', type=str,
                        default='deployed_models/relational_full',
                        help='Output directory for model artifacts')
    parser.add_argument('--mask-target-smiles', type=float, default=0.5,
                        help='Probability of masking target SMILES during training (default: 0.5)')
    parser.add_argument('--lambda-bal', type=float, default=2.0,
                        help='Balance loss weight (default: 2.0)')
    parser.add_argument('--lambda-corr', type=float, default=0.5,
                        help='Correlation loss weight (default: 0.5)')
    parser.add_argument('--lambda-var', type=float, default=0.0,
                        help='Variance loss weight (default: 0.0)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    return parser.parse_args()


def _build_arch_config(args):
    """Build the architecture config shared by both phases."""
    arch_config = EnhancedModelConfig.with_features([
        'residual_connections',
        'cross_modal_attention',
        'mcr',
        'flux_messages',
        'mass_balance_projection',
    ])
    arch_config.layer_type = 'Relational'
    arch_config.structural_loss_weight = 1.0
    return arch_config


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    arch_config = _build_arch_config(args)

    # ------------------------------------------------------------------
    # Phase 1: find optimal epoch with a validation split
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: Finding optimal epoch (90/10 train/val split)")
    print("=" * 70)
    print(f"  Max epochs:    {args.phase1_epochs}")
    print(f"  Patience:      {args.phase1_patience}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size:    {args.batch_size} (accum {args.gradient_accumulation_steps})")
    print(f"  Mask SMILES:   {args.mask_target_smiles}")
    print(f"  Physics:       bal={args.lambda_bal}, corr={args.lambda_corr}, var={args.lambda_var}")
    print(f"  LR schedule:   ReduceLROnPlateau (default)")
    print()

    phase1_dir = os.path.join(output_dir, '_phase1')

    phase1_config = ExperimentConfig(
        seed=args.seed,
        config_name='deploy_phase1_find_epoch',
        data=DataConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_fraction=0.9,
            val_fraction=0.1,
            num_workers=0,
            pin_memory=False,
        ),
        model=ModelConfig(
            variant='enhanced',
            num_layers=args.num_layers,
            hidden_dim=128,
            dropout=0.2,
            use_directionality=True,
            enhanced_config=arch_config,
        ),
        training=TrainingConfig(
            learning_rate=args.learning_rate,
            max_epochs=args.phase1_epochs,
            patience=args.phase1_patience,
            save_dir=phase1_dir,
            mask_target_smiles=args.mask_target_smiles,
        ),
        physics=PhysicsConfig(
            lambda_balance=args.lambda_bal,
            lambda_correlation=args.lambda_corr,
            lambda_variance=args.lambda_var,
        ),
    )

    phase1_results = train_experiment(phase1_config)
    best_epoch = phase1_results['best_epoch']
    total_run = phase1_results['total_epochs_run']

    print()
    print("-" * 70)
    print(f"Phase 1 complete: ran {total_run} epochs, best epoch = {best_epoch}")
    print(f"  Val-based R2:  {phase1_results['test_r2']:.4f}")
    print("-" * 70)
    print()

    # ------------------------------------------------------------------
    # Phase 2: retrain on ALL data for exactly best_epoch epochs
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"PHASE 2: Retraining on ALL data for {best_epoch} epochs")
    print("=" * 70)
    print(f"  Epochs:        {best_epoch}")
    print(f"  LR schedule:   CosineAnnealingLR (T_max={best_epoch})")
    print()

    phase2_config = ExperimentConfig(
        seed=args.seed,
        config_name='deploy_relational_full_detgate',
        data=DataConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_fraction=1.0,
            val_fraction=0.0,
            num_workers=0,
            pin_memory=False,
        ),
        model=ModelConfig(
            variant='enhanced',
            num_layers=args.num_layers,
            hidden_dim=128,
            dropout=0.2,
            use_directionality=True,
            enhanced_config=arch_config,
        ),
        training=TrainingConfig(
            learning_rate=args.learning_rate,
            max_epochs=best_epoch,
            patience=best_epoch + 1,  # disable early stopping
            save_dir=output_dir,
            mask_target_smiles=args.mask_target_smiles,
        ),
        physics=PhysicsConfig(
            lambda_balance=args.lambda_bal,
            lambda_correlation=args.lambda_corr,
            lambda_variance=args.lambda_var,
        ),
    )

    def cosine_scheduler(optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_epoch)

    phase2_results = train_experiment(phase2_config, scheduler_factory=cosine_scheduler)

    print()
    print("-" * 70)
    print(f"Phase 2 complete: trained {phase2_results['total_epochs_run']} epochs on all data")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Compute gene statistics for SpatialRefiner
    # ------------------------------------------------------------------
    print("\nComputing gene statistics from full dataset...")
    data_module = FiLMDataModule(config=phase2_config.data, seed=args.seed)
    data_module.setup()
    gene_stats_path = os.path.join(output_dir, 'gene_stats.pt')
    compute_gene_statistics(data_module.dataset, gene_stats_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DEPLOYMENT TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Phase 1: best epoch = {best_epoch} (out of {total_run})")
    print(f"  Phase 2: trained {phase2_results['total_epochs_run']} epochs on all data")
    print()
    print(f"Artifacts saved to: {output_dir}")
    print(f"  best_model.pt          - model weights (Phase 2)")
    print(f"  experiment_config.json  - full config for reproducibility")
    print(f"  gene_stats.pt          - bulk gene statistics for SpatialRefiner")
    print(f"  test_results.json      - metrics (nan expected, no test set)")
    print(f"  _phase1/               - Phase 1 artifacts (can be deleted)")


if __name__ == '__main__':
    main()
