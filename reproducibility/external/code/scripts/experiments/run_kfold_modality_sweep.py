
import argparse
import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.data.batching import collate_metabolic_data
from src.models.factory import create_model
from src.models.components import EnhancedModelConfig
from src.models.modules.pna_layer import PNALayer
from src.training.trainer_engine import Trainer
from src.training.loss_factory import create_physics_loss
from src.training.callbacks import LoggingCallback
from src.config import ExperimentConfig, PhysicsConfig, DataConfig, ModelConfig, TrainingConfig
from src.utils.data_paths import get_standard_data_paths

def run_kfold_cv(args, config_name, output_dir):
    """Run 5-fold cross-validation for a specific configuration."""
    
    print(f"\n{'='*80}")
    print(f"STARTING K-FOLD CV: {config_name}")
    print(f"{'='*80}")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Dataset
    print(f"Loading dataset with randomization settings:")
    print(f"  - Randomize SMILES: {args.randomize_smiles}")
    print(f"  - Randomize Genes: {args.randomize_genes}")
    print(f"  - Randomize Enzymes: {args.randomize_embeddings}")
    print(f"  - Randomize GPE: {args.randomize_gpe}")
    print(f"  - Random Graph: {args.random_graph}")
    print(f"  - Graph Identifier: {args.graph_identifier}")
    
    mudata_path, graph_path, gene_mapping_path = get_standard_data_paths(identifier=args.graph_identifier)
    
    dataset = FiLMDatasetWithSMILES(
        mudata_path=mudata_path,
        embeddings_graph_path=graph_path,
        gene_mapping_path=gene_mapping_path,
        device=device,
        randomize_smiles=args.randomize_smiles,
        randomize_genes=args.randomize_genes,
        randomize_embeddings=args.randomize_embeddings,
        random_graph=args.random_graph,
        randomize_gpe=args.randomize_gpe,
    )
    
    # Ensure num_nodes is set
    dataset.num_nodes = len(dataset.node_types)
    
    # K-Fold Splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n--- FOLD {fold_idx + 1}/5 ---")
        
        # Create DataLoaders
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_metabolic_data,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_metabolic_data,
            num_workers=4,
            pin_memory=True,
        )
        
        # Configure Enhanced Architecture (Relational + Structural Loss)
        enhanced_config = EnhancedModelConfig(
            layer_type="Relational",
            # Structural Reconstruction Loss (Manifold Alignment)
            structural_loss_weight=1.0,
            # Recursive Cross-Modal Alignment
            use_cross_modal_attention=args.use_attention,
            # Enable Residuals for performance/depth
            use_residual_connections=args.use_residuals,
            use_layer_normalization=True,
            # Graph Positional Encodings
            use_graph_positional_encodings=args.use_gpe,
            # Standard settings
            use_similarity_regularization=False,
            use_flux_messages=False,
            use_mass_balance_projection=False,
            use_mcr=False
        )
        
        # Configure Model
        model_config = ModelConfig(
            variant="enhanced",
            hidden_dim=128,
            num_layers=3,
            dropout=0.2,
            use_directionality=False, # Relational handles direction
            use_flux_messages=False,
            use_mass_balance_projection=False,
            subtract_sample_mean=args.subtract_sample_mean,
            enhanced_config=enhanced_config
        )

        # Initialize Model
        model = create_model(
            model_cfg=model_config,
            embedding_dim=dataset.embedding_dim,
            gene_dim=dataset.gene_dim,
            smiles_dim=dataset.smiles_dim,
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Criterion (MSE Only)
        # We use create_physics_loss but with all weights 0 except MSE implied
        physics_cfg = PhysicsConfig(
            lambda_balance=0.0,
            lambda_correlation=0.0,
            lambda_variance=0.0,
            lambda_kl=0.0
        )
        criterion, _ = create_physics_loss(
            physics_cfg,
            enzyme_confidence=3.0,
            flux_flexibility=0.8,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            contrastive_weight=0.0,
            enable_contrastive=False,
            mask_target_smiles=args.blindfold_prob, # New Argument
            callbacks=[LoggingCallback()]
        )
        
        # Train
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=args.epochs,
            patience=20 # Early stopping
        )
        
        # Record Results
        best_val_loss = min(history['val_loss'])
        best_val_r2 = max(history['val_r2'])
        best_val_rmse = min(history['val_rmse'])
        
        print(f"Fold {fold_idx + 1} Best Validation: Loss={best_val_loss:.4f}, R2={best_val_r2:.4f}, RMSE={best_val_rmse:.4f}")
        
        fold_results.append({
            'fold': fold_idx,
            'val_loss': best_val_loss,
            'val_r2': best_val_r2,
            'val_rmse': best_val_rmse
        })
        
    # Aggregate Results
    r2_scores = [r['val_r2'] for r in fold_results]
    rmse_scores = [r['val_rmse'] for r in fold_results]
    
    summary = {
        'config_name': config_name,
        'mean_r2': float(np.mean(r2_scores)),
        'std_r2': float(np.std(r2_scores)),
        'mean_rmse': float(np.mean(rmse_scores)),
        'std_rmse': float(np.std(rmse_scores)),
        'fold_results': fold_results,
        'args': vars(args)
    }
    
    # Save Results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(output_dir, f"{config_name}_results.json")
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n{'='*80}")
    print(f"COMPLETED: {config_name}")
    print(f"Mean R2: {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
    print(f"Saved to: {save_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run K-Fold Modality Ablation')
    
    # Randomization Flags
    parser.add_argument('--randomize-smiles', action='store_true', help='Randomize SMILES features')
    parser.add_argument('--randomize-genes', action='store_true', help='Randomize Gene features')
    parser.add_argument('--randomize-embeddings', action='store_true', help='Randomize Enzyme embeddings')
    parser.add_argument('--randomize-gpe', action='store_true', help='Randomize Graph Positional Encoding features')
    parser.add_argument('--random-graph', action='store_true', help='Use random graph topology')
    parser.add_argument('--graph-identifier', type=str, default='', help='Identifier for graph file (e.g. masked)')
    
    # Training Params
    parser.add_argument('--config-name', type=str, required=True, help='Name of the configuration')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=0.00075, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--blindfold-prob', type=float, default=0.5, help='Probability of masking target SMILES (Blindfold Test)')
    # New Architectural Toggles
    parser.add_argument('--use-residuals', action='store_true', help='Enable residual connections')
    parser.add_argument('--use-attention', action='store_true', help='Enable recursive cross-modal attention')
    parser.add_argument('--use-gpe', action='store_true', help='Enable Graph Positional Encodings')
    parser.add_argument('--subtract-sample-mean', action='store_true', help='Subtract mean gene expression per sample')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_kfold_cv(args, args.config_name, args.output_dir)

if __name__ == '__main__':
    main()
