#!/usr/bin/env python3
"""
Run LOMO cross-validation for architectural ablation studies.
This script handles different architectural configurations for the Enhanced model.
"""

import argparse
import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
from src.models.components import EnhancedModelConfig
from src.evaluation.lomo_cross_validation import run_lomo_cross_validation
from src.utils.data_paths import get_standard_data_paths


def get_arch_configurations() -> dict:
    """Get architectural configurations for systematic ablation testing.
    
    Returns:
        dict: Configuration specifications for each ablation condition
    """
    return {
        # Level 0: Legacy Baseline
        'baseline_legacy': {
            'description': 'Legacy Homogeneous GNN (No Relational, No Structural Loss)',
            'config_method': 'original_model_config',
            'expected_improvement': 'N/A (Old Reference)',
        },
        
        # Level 1: Relational Core (Minimum Viable Graph Model)
        'relational_core': {
            'description': 'Relational HeteroConv + Structural Loss + 100% Blindfold',
            'config_method': 'feature_combo',
            'config_arg': [], # No extras
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
                'use_residual_connections': False, # No residuals
            },
            'training_settings': {
                'mask_target_smiles': 1.0 # 100% Blindfold
            },
            'expected_improvement': 'Topological dependency via manifold alignment (Pure Graph)',
        },

        # Level 1b: Relational Core + GPE (Graph Positional Encodings)
        'relational_core_gpe': {
            'description': 'Level 1 + Graph Positional Encodings (100% Blindfold)',
            'config_method': 'feature_combo',
            'config_arg': ['graph_positional_encodings'],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
                'use_residual_connections': False,
            },
            'training_settings': {
                'mask_target_smiles': 1.0
            },
            'expected_improvement': 'Explicit topology awareness in pure-graph mode (most informative GPE test)',
        },
        
        # Level 2: Relational + Residuals (Deep Model)
        'relational_residuals': {
            'description': 'Relational + Residuals + LayerNorm + 50% Blindfold',
            'config_method': 'feature_combo',
            'config_arg': ['residual_connections'],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5 # Partial Blindfold for robustness
            },
            'expected_improvement': 'Deeper propagation and stability',
        },
        
        # Level 3: Relational + Attention (Manifold Aligned)
        'relational_attention': {
            'description': 'Relational + Residuals + Recursive Cross-Modal Attention',
            'config_method': 'feature_combo',
            'config_arg': ['residual_connections', 'cross_modal_attention'],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5
            },
            'expected_improvement': 'Layer-wise manifold alignment',
        },
        
        # Level 4: Relational + MCR (Readout)
        'relational_mcr': {
            'description': 'Relational + Attention + Metabolite-Conditioned Reader',
            'config_method': 'feature_combo',
            'config_arg': ['residual_connections', 'cross_modal_attention', 'mcr'],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5
            },
            'expected_improvement': 'Refined readout using neighbor attention',
        },

        # Level 4b: Relational + Multi-Hop MCR
        'relational_mcr_multihop': {
            'description': 'Level 4 with 2-hop MCR (extends receptive field)',
            'config_method': 'feature_combo',
            'config_arg': ['residual_connections', 'cross_modal_attention', 'mcr_multihop'],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5
            },
            'expected_improvement': 'Extended receptive field for isolated metabolites via multi-hop attention',
        },

        # Level 4c: Relational + Multi-Hop MCR + GPE
        'relational_mcr_multihop_gpe': {
            'description': 'Level 4 with 2-hop MCR + Graph Positional Encodings',
            'config_method': 'feature_combo',
            'config_arg': ['residual_connections', 'cross_modal_attention', 'mcr_multihop', 'graph_positional_encodings'],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5
            },
            'expected_improvement': 'Combined multi-hop + explicit topology features',
        },

        # Level 5: Relational + Physics (Full Stack)
        'relational_full': {
            'description': 'Relational + Attention + MCR + Physics Constraints',
            'config_method': 'feature_combo',
            'config_arg': [
                'residual_connections',
                'cross_modal_attention',
                'mcr',
                'flux_messages',
                'mass_balance_projection'
            ],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5
            },
            'expected_improvement': 'Maximum biophysical realism',
        },

        # Level 6: Full Stack + GPE + Multi-Hop MCR
        'relational_full_plus': {
            'description': 'Level 5 + GPE + Multi-Hop MCR (everything enabled)',
            'config_method': 'feature_combo',
            'config_arg': [
                'residual_connections',
                'cross_modal_attention',
                'mcr_multihop',
                'graph_positional_encodings',
                'flux_messages',
                'mass_balance_projection'
            ],
            'custom_settings': {
                'layer_type': 'Relational',
                'structural_loss_weight': 1.0,
            },
            'training_settings': {
                'mask_target_smiles': 0.5
            },
            'expected_improvement': 'Full stack with extended receptive field and explicit topology',
        }
    }


def create_model_config(arch_config: str) -> EnhancedModelConfig:
    """Create architectural configuration for the specified ablation condition.
    
    Args:
        arch_config: Architecture configuration name
        
    Returns:
        EnhancedModelConfig: Configured architecture instance
        
    Raises:
        ValueError: If architecture configuration is unknown
    """
    CONFIG_FACTORIES = {
        'original_model_config': lambda: EnhancedModelConfig.original_model_config(),
        'default_config': lambda: EnhancedModelConfig(),
        'ablation_config': lambda arg: EnhancedModelConfig.ablation_config(arg),
        'feature_combo': lambda features: EnhancedModelConfig.with_features(features),
    }
    
    configs = get_arch_configurations()
    
    if arch_config not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown architecture config '{arch_config}'. Available: {available}")
    
    config_spec = configs[arch_config]
    factory_name = config_spec['config_method']
    
    if factory_name not in CONFIG_FACTORIES:
        raise ValueError(f"Unknown config factory method: {factory_name}")
    
    factory = CONFIG_FACTORIES[factory_name]
    
    # Handle configurations that require arguments
    if factory_name == 'ablation_config':
        if 'config_arg' not in config_spec:
            raise ValueError(f"Ablation config '{arch_config}' missing required 'config_arg'")
        return factory(config_spec['config_arg'])
    elif factory_name == 'feature_combo':
        features = config_spec.get('config_arg')
        if features is None:
            raise ValueError(f"Feature combo config '{arch_config}' missing required 'config_arg'")
        if not isinstance(features, (list, tuple)):
            raise ValueError(f"Feature combo config '{arch_config}' expects 'config_arg' as list or tuple of feature names")
        return factory(features)
    else:
        return factory()


def main():
    """Run LOMO ablation for a single architectural configuration."""
    parser = argparse.ArgumentParser(description='Run LOMO CV for architectural ablation')
    parser.add_argument('--arch-config', type=str, required=True,
                        choices=list(get_arch_configurations().keys()),
                        help='Architectural configuration to test')
    parser.add_argument('--lambda-bal', type=float, default=0.0,
                        help='Balance loss weight')
    parser.add_argument('--lambda-corr', type=float, default=0.0,
                        help='Correlation loss weight')
    parser.add_argument('--lambda-var', type=float, default=0.0,
                        help='Variance loss weight')
    parser.add_argument('--lambda-kl', type=float, default=0.0,
                        help='KL divergence loss weight')
    parser.add_argument('--output-dir', type=str, default='lomo_results_ablation',
                        help='Output directory for results')
    parser.add_argument('--n-metabolites', type=int, default=10,
                        help='Number of metabolites to test')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs per LOMO fold')
    parser.add_argument('--learning-rate', type=float, default=0.00075,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size (default: 128 to match normal training)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--pretrained-encoders', type=str, default=None,
                        help='Path to pretrained encoder weights (best_pretrained_encoders.pth)')
    parser.add_argument('--compute-neighbor-correlation', action='store_true',
                        help='Compute neighbour correlation diagnostics during LOMO runs')
    parser.add_argument('--fold-only', type=int, default=None,
                        help='Run only a specific fold index (for SLURM arrays)')
    # Strategy parameters
    parser.add_argument('--graph-identifier', type=str, default='', help='Identifier for graph file (e.g. masked)')
    parser.add_argument('--enable-metabolite-dropout', action='store_true', help='Apply metabolite dropout during training')
    parser.add_argument('--force-residuals', action='store_true',
                        help='Force enable residual connections and layer norm (useful for deep models)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before optimizer step')
    
    # New Relational Arguments
    parser.add_argument('--structural-loss-weight', type=float, default=0.0,
                        help='Weight for structural reconstruction loss')
    parser.add_argument('--mask-target-smiles', type=float, default=0.0,
                        help='Probability of masking target SMILES during training (Blindfold)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    _print_experiment_header(args)
    
    # Get configuration information
    configs = get_arch_configurations()
    config_info = configs[args.arch_config]
    
    _print_experiment_details(args, config_info, device)
    
    # Create output directory
    if args.fold_only is not None:
        # Use stable directory for array jobs (provided by SLURM script)
        output_dir = Path(args.output_dir)
    else:
        # Use timestamped directory for standalone runs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f"architecture_{args.arch_config}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset...")
    mudata_path, graph_path, gene_mapping_path = get_standard_data_paths(identifier=args.graph_identifier)
    print(f"Using data paths:")
    print(f"  MuData: {mudata_path}")
    print(f"  Graph: {graph_path}")
    print(f"  Gene mapping: {gene_mapping_path}")
    
    dataset = FiLMDatasetWithSMILES(
        mudata_path=mudata_path,
        embeddings_graph_path=graph_path,
        gene_mapping_path=gene_mapping_path
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create model configuration
    arch_config = create_model_config(args.arch_config)
    
    # Apply custom settings from config spec (e.g. layer_type='Relational')
    if 'custom_settings' in config_info:
        print(f"Applying custom settings: {config_info['custom_settings']}")
        for key, value in config_info['custom_settings'].items():
            if hasattr(arch_config, key):
                setattr(arch_config, key, value)
            else:
                print(f"Warning: arch_config has no attribute '{key}'")
    
    # Override with CLI args if provided (allows manual tuning)
    if args.structural_loss_weight > 0:
        arch_config.structural_loss_weight = args.structural_loss_weight
        
    # Force residuals if requested (e.g. for deep models)
    if args.force_residuals:
        print("⚡ Forcing Residual Connections + LayerNorm (due to --force-residuals)")
        arch_config.use_residual_connections = True
        arch_config.use_layer_normalization = True
    
    # Model parameters
    model_kwargs = {
        'embedding_dim': dataset.embedding_dim,
        'gene_dim': dataset.gene_dim,
        'smiles_dim': dataset.smiles_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'use_directional': True,
        'config': arch_config
    }
    
    print(f"Model parameters: {model_kwargs}")
    print(f"Architecture config: {arch_config}")
    
    # LOMO parameters
    lomo_params = {
        'training_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'lambda_bal': args.lambda_bal,
        'lambda_corr': args.lambda_corr,
        'lambda_var': args.lambda_var,
        'lambda_kl': args.lambda_kl,
        'seed': args.seed,
        'pretrained_encoders': args.pretrained_encoders,
        'mask_target_smiles': args.mask_target_smiles # Pass this down
    }
    
    # Resolve mask_target_smiles from config if not in CLI
    # If training_settings has it, use it (unless CLI override)
    mask_target_smiles = args.mask_target_smiles
    if 'training_settings' in config_info and 'mask_target_smiles' in config_info['training_settings']:
        # If CLI is default 0.0, use config value
        if args.mask_target_smiles == 0.0:
            mask_target_smiles = config_info['training_settings']['mask_target_smiles']
            # Coerce boolean True to 1.0
            if mask_target_smiles is True:
                mask_target_smiles = 1.0
    
    # Save configuration (only for full runs or the first fold to avoid race conditions)
    if args.fold_only is None or args.fold_only == 0 or args.fold_only == 999:
        config_data = {
            'arch_config': args.arch_config,
            'description': config_info['description'],
            'architecture_features': f"arch_config = EnhancedModelConfig.{config_info['config_method']}({config_info.get('config_arg', '')})",
            'physics_parameters': f'lambda_bal={args.lambda_bal}, lambda_corr={args.lambda_corr}, lambda_var={args.lambda_var}, lambda_kl={args.lambda_kl}',
            'lomo_parameters': lomo_params,
            'model_kwargs': model_kwargs,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'compute_neighbor_correlation': args.compute_neighbor_correlation,
            'mask_target_smiles': mask_target_smiles
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    # Run LOMO cross-validation
    print(f"\nStarting LOMO cross-validation...")
    try:
        results = run_lomo_cross_validation(
            model_class=EnhancedConcatGNNWithSMILES,
            model_kwargs=model_kwargs,
            dataset=dataset,
            device=device,
            output_dir=str(output_dir),
            n_metabolites_to_test=args.n_metabolites,
            training_epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            lambda_bal=args.lambda_bal,
            lambda_corr=args.lambda_corr,
            lambda_var=args.lambda_var,
            lambda_kl=args.lambda_kl,
            pretrained_encoders=args.pretrained_encoders,
            compute_neighbor_correlation=args.compute_neighbor_correlation,
            specific_fold_idx=args.fold_only,
            enable_metabolite_dropout=args.enable_metabolite_dropout,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mask_target_smiles=mask_target_smiles # Pass to runner
        )
        
        # Update config with results (only if full run or final fold)
        if args.fold_only is None:
            analysis = results['analysis']
            config_data['results_summary'] = {
                'mean_r2': float(analysis['mean_r2']),
                'std_r2': float(analysis['std_r2']),
                'mean_rmse': float(analysis['mean_rmse']),
                'std_rmse': float(analysis.get('std_rmse', 0)),  # Add default if missing
                'n_metabolites_tested': len(results['results'])
            }
            
            # Save final results
            with open(output_dir / 'final_results.json', 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            print(f"\n{'='*60}")
            print(f"LOMO ABLATION COMPLETED: {args.arch_config}")
            print(f"{'='*60}")
            print(f"Mean R²: {analysis['mean_r2']:.4f} ± {analysis['std_r2']:.4f}")
            print(f"Mean RMSE: {analysis['mean_rmse']:.4f} ± {analysis.get('std_rmse', 0):.4f}")
            print(f"Metabolites tested: {len(results['results'])}")
            print(f"Results saved to: {output_dir}")
        else:
            print(f"\nFold {args.fold_only} completed successfully.")
        
    except Exception as e:
        error_info = {
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"❌ Error during LOMO cross-validation: {e}")
        raise


def _print_experiment_header(args):
    """Print formatted experiment header."""
    print(f"{'='*80}")
    print(f"LOMO ARCHITECTURAL ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Configuration: {args.arch_config}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _print_experiment_details(args, config_info: dict, device: torch.device):
    """Print detailed experiment configuration."""
    print(f"\n📋 EXPERIMENT CONFIGURATION")
    print(f"{'─'*50}")
    print(f"Architecture: {config_info['description']}")
    print(f"Expected benefit: {config_info.get('expected_improvement', 'Not specified')}")
    print(f"Device: {device}")
    
    print(f"\n🏗️  MODEL ARCHITECTURE")
    print(f"{'─'*50}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"GNN layers: {args.num_layers}")
    print(f"Dropout rate: {args.dropout}")
    
    print(f"\n⚖️  PHYSICS-INFORMED LOSS WEIGHTS")
    print(f"{'─'*50}")
    print(f"Mass balance (λ_bal): {args.lambda_bal}")
    print(f"Flux correlation (λ_corr): {args.lambda_corr}")
    print(f"Flux variance (λ_var): {args.lambda_var}")
    print(f"KL divergence (λ_kl): {args.lambda_kl}")
    
    print(f"\n🧪 EVALUATION SETUP")
    print(f"{'─'*50}")
    print(f"Metabolites to test: {args.n_metabolites}")
    print(f"Training epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.seed}")
    print(f"Compute neighbor correlation: {args.compute_neighbor_correlation}")


if __name__ == "__main__":
    main()
