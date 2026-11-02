
import argparse
import os
import sys
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
from src.models.components import EnhancedModelConfig
from src.evaluation.lomo_cross_validation import (
    evaluate_model_on_metabolites, 
    analyze_lomo_results, 
    save_lomo_results,
    _compute_metabolite_statistics,
    _select_metabolites_by_coverage
)
from src.utils.data_paths import get_standard_data_paths

def main():
    parser = argparse.ArgumentParser(description='Evaluate Synthetic LOMO Model on All Metabolites')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--graph-identifier', type=str, default='', help='Graph identifier (e.g. masked)')
    parser.add_argument('--n-metabolites', type=int, default=50, help='Number of metabolites to evaluate (default: 50)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset...")
    mudata_path, graph_path, gene_mapping_path = get_standard_data_paths(identifier=args.graph_identifier)
    dataset = FiLMDatasetWithSMILES(
        mudata_path=mudata_path,
        embeddings_graph_path=graph_path,
        gene_mapping_path=gene_mapping_path
    )
    
    # Recreate model structure
    # Note: We need to match the config used during training. 
    # Ideally this would be saved with the checkpoint, but for now we assume the default "Enhanced" config
    # with the parameters inferred from the checkpoint or defaults.
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Infer dimensions from checkpoint if possible
    embedding_dim = dataset.embedding_dim
    gene_dim = dataset.gene_dim
    smiles_dim = dataset.smiles_dim
    
    # Try to load experiment config from the model directory
    model_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(model_dir, 'experiment_config.json')
    
    # Default values
    hidden_dim = 128
    num_layers = 3
    dropout = 0.2
    use_flux_messages = True
    config = EnhancedModelConfig(
        use_cross_modal_attention=True,
        use_residual_connections=True,
        use_similarity_regularization=True,
        use_mcr=True
    )
    
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            exp_config = json.load(f)
            
        # Extract model config
        model_cfg = exp_config.get('model', {})
        hidden_dim = model_cfg.get('hidden_dim', hidden_dim)
        num_layers = model_cfg.get('num_layers', num_layers)
        dropout = model_cfg.get('dropout', dropout)
        use_flux_messages = model_cfg.get('use_flux_messages', True)
        
        # Extract enhanced config
        enhanced_cfg_data = model_cfg.get('enhanced_config')
        if enhanced_cfg_data:
            print("Found enhanced configuration in experiment config.")
            config = EnhancedModelConfig(**enhanced_cfg_data)
        else:
            print("No enhanced_config found in experiment config, using defaults.")
            
    else:
        print(f"Warning: Configuration file not found at {config_path}. Using default/inferred settings.")
        # Heuristics to guess hidden dim from checkpoint
        if 'reaction_encoder.weight' in checkpoint:
            hidden_dim = checkpoint['reaction_encoder.weight'].shape[0]
            print(f"Inferred hidden_dim={hidden_dim} from checkpoint.")
        
    print(f"Initializing model with hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}...")
    print(f"Enhanced Config: {config}")
    
    model = EnhancedConcatGNNWithSMILES(
        embedding_dim=embedding_dim,
        gene_dim=gene_dim,
        smiles_dim=smiles_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        config=config,
        use_flux_messages=use_flux_messages
    )
    
    # Load weights
    print(f"Loading weights from {args.model_path}...")
    try:
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()
    
    # Identify eligible metabolites using the same logic as Real LOMO
    print("Identifying eligible metabolites for evaluation...")
    min_samples_per_metabolite = 20
    (
        metabolite_sample_counts,
        coverage_scores,
        direct_scores,
        far_scores,
    ) = _compute_metabolite_statistics(dataset)

    eligible_metabolites = [
        met_idx
        for met_idx, count in metabolite_sample_counts.items()
        if count >= min_samples_per_metabolite
    ]

    print(f"Found {len(eligible_metabolites)} metabolites with >= {min_samples_per_metabolite} samples")

    # Select subset for testing (stratified by coverage)
    # This ensures we test on the SAME type of distribution as the Real LOMO experiments
    if len(eligible_metabolites) > args.n_metabolites:
        selected_metabolites = _select_metabolites_by_coverage(
            eligible_metabolites, coverage_scores, args.n_metabolites
        )
    else:
        selected_metabolites = eligible_metabolites
        
    # Validate selected metabolites
    first_data = dataset[0]
    if hasattr(first_data, 'metabolite_indices'):
        valid_metabolite_indices = set(first_data.metabolite_indices.cpu().numpy().tolist())
    else:
        valid_metabolite_indices = set([i for i, t in enumerate(first_data.node_types) if t == 'metabolite'])
    
    selected_metabolites = [m for m in selected_metabolites if m in valid_metabolite_indices]
    
    print(f"Selected {len(selected_metabolites)} diverse metabolites for evaluation.")
    
    # Run evaluation on selected metabolites
    print("Starting evaluation...")
    results = evaluate_model_on_metabolites(
        model, 
        dataset, 
        device, 
        metabolite_indices=selected_metabolites,
        batch_size=args.batch_size
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze results
    print("Analyzing results...")
    analysis = analyze_lomo_results(results, args.output_dir)
    save_lomo_results(results, analysis, args.output_dir)
    
    print(f"\nEvaluation complete!")
    print(f"Tested {len(results)} metabolites.")
    print(f"Mean R²: {analysis['mean_r2']:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
