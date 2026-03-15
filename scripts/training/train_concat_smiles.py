"""
Train Concat GNN with SMILES embeddings for metabolite prediction.

This script trains GNN models with configurable components:
- Physics loss terms (mass balance, correlation, variance, flux alignment)
- Enhanced model features (residual connections, cross-modal attention, similarity regularization)
- Flux-based message passing and mass balance projection
"""

import argparse
from copy import deepcopy

# Ignore UserWarnings and FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import ExperimentConfig, load_experiment_config
from src.training.trainer import train_experiment
from src.models.components import EnhancedModelConfig


def build_config_from_args(args, arch_config) -> ExperimentConfig:
    """Construct ExperimentConfig from CLI arguments."""

    enhanced_payload = (
        arch_config.to_dict() if hasattr(arch_config, 'to_dict') else arch_config
    ) if arch_config is not None else None

    config_dict = {
        "seed": args.seed,
        "config_name": args.config_name,
        "model": {
            "variant": "enhanced" if arch_config is not None else "concat",
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_directionality": args.directionality,
            "use_flux_messages": args.use_flux_messages,
            "flux_message_depth": args.flux_message_depth,
            "use_mass_balance_projection": args.use_mass_balance_projection,
            "enhanced_config": enhanced_payload,
        },
        "data": {
            "normalized": args.normalised,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "random_graph": args.random_graph,
        },
        "training": {
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "mask_target_smiles": args.mask_target_smiles,
        },
        "physics": {
            "lambda_balance": args.lambda_bal,
            "lambda_correlation": args.lambda_corr,
            "lambda_variance": args.lambda_var,
            "lambda_kl": args.lambda_kl,
            "use_listwise_ranking": args.use_listwise_ranking,
            "enzyme_confidence": args.enzyme_confidence,
            "flux_flexibility": args.flux_flexibility,
        },
        "contrastive": {
            "enabled": args.enable_contrastive,
            "weight": args.contrastive_weight,
            "temperature": args.contrastive_temperature,
            "pretrained_encoders_path": args.pretrained_encoders,
        },
    }

    return ExperimentConfig.from_dict(config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline Concat GNN with SMILES embeddings')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to experiment configuration (JSON or YAML). CLI flags override when provided.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--random_graph', action='store_true', 
                        help='Train on randomized graph structure')
    parser.add_argument('--compare', action='store_true', 
                        help='Train on both original and random graphs and compare')
    parser.add_argument('--directionality', action='store_true',
                        help='Use directional convolutions in the GNN model')
    parser.add_argument('--normalised', action='store_true',
                        help='Use normalised edge features')
    parser.add_argument('--pretrained_encoders', type=str, default=None,
                        help='Path to contrastive pretrained encoder weights (from contrastive pretraining)')

    parser.add_argument('--lambda_bal', type=float, default=0.0, 
                        help='Weight for mass balance loss (set to 0.01 when using mass balance projection)')
    parser.add_argument('--lambda_corr', type=float, default=0.0, 
                        help='Weight for gene-flux correlation loss (higher = stronger gene-flux coupling)')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='Weight for flux variance loss (prevents flux collapse)')
    parser.add_argument('--lambda_kl', type=float, default=0.0,
                        help='Weight for listwise flux alignment loss (KL divergence)')

    # === Listwise ranking arguments ===
    parser.add_argument('--use_listwise_ranking', action='store_true', default=False,
                        help='Enable listwise ranking loss for flux-gene alignment')
    parser.add_argument('--enzyme_confidence', type=float, default=3.0,
                        help='Confidence in enzyme scores - higher values trust biology more (default: 3.0)')
    parser.add_argument('--flux_flexibility', type=float, default=0.8,
                        help='Flexibility for flux predictions - higher values are more forgiving of model errors (default: 0.8)')

    
    # === Batching arguments ===
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for training (default: 1). Recommended values: 8, 16, 32')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps (default: 1). Use with large batch sizes to avoid OOM.')
    
    # === Model architecture arguments ===
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--learning_rate', type=float, default=0.00075,
                        help='Learning rate for optimizer (default: 0.00075)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of epochs to train (default: 300)')
    
    # === Contrastive learning arguments ===
    parser.add_argument('--enable_contrastive', action='store_true',
                        help='Enable standard contrastive learning (works best with batch_size > 1)')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss (default: 0.1)')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning (default: 0.07)')
    
    # === Architectural Features (Neural Network Modifications) ===
    parser.add_argument('--layer_type', type=str, default="GAT",
                        choices=["GAT", "Directional", "PNA", "Relational"],
                        help='ARCHITECTURE: Backbone GNN layer type')
    parser.add_argument('--structural_loss_weight', type=float, default=0.0,
                        help='ARCHITECTURE: Weight for structural reconstruction loss (manifold alignment)')
    parser.add_argument('--mask_target_smiles', type=float, default=0.0,
                        help='TRAINING: Probability of masking metabolite SMILES (Blindfold Test)')
    
    parser.add_argument('--residual_connections', action='store_true',
                        help='ARCHITECTURE: Add residual connections in GNN layers to prevent over-smoothing')
    parser.add_argument('--cross_modal_attention', action='store_true',
                        help='ARCHITECTURE: Enable cross-modal attention between reactions and metabolites')
    parser.add_argument('--similarity_loss_weight', type=float, default=0.0,
                        help='ARCHITECTURE: Weight for chemical similarity regularization (0.0=disabled, try 0.01)')
    parser.add_argument('--use_flux_messages', action='store_true', default=False,
                        help='ARCHITECTURE: Enable flux-based message passing to metabolite embeddings')
    parser.add_argument('--flux_message_depth', type=int, default=1,
                        help='ARCHITECTURE: GNN layer depth at which to add flux messages (1-indexed)')
    parser.add_argument('--use_mass_balance_projection', action='store_true', default=False,
                        help='ARCHITECTURE: Enable Cholesky-based mass balance projection for exact constraint satisfaction')
    parser.add_argument('--gene-skip', action='store_true', default=False,
                        help='ARCHITECTURE: Skip-connect raw gene expression to MCR prediction head')
    parser.add_argument('--gene-film', action='store_true', default=False,
                        help='ARCHITECTURE: FiLM conditioning on gene expression per GNN layer')
    parser.add_argument('--detection-gate', action='store_true', default=False,
                        help='ARCHITECTURE: Enable detection gate for sparse spatial inference')
    parser.add_argument('--detection-gate-weight', type=float, default=1.0,
                        help='ARCHITECTURE: BCE loss weight for detection gate (default: 1.0)')

    # === Hyperopt support ===
    parser.add_argument('--config_name', type=str, default=None,
                        help='Configuration name for hyperopt trials (used to create unique model directories)')
    
    args = parser.parse_args()
    
    # Create architectural feature configuration if any features are requested
    arch_config = None
    architectural_features = (args.residual_connections or args.cross_modal_attention or
                            args.similarity_loss_weight > 0 or args.use_flux_messages or
                            args.use_mass_balance_projection or args.layer_type != "GAT" or
                            args.structural_loss_weight > 0 or args.gene_skip or args.gene_film or
                            args.detection_gate)
    
    if architectural_features:
        arch_config = EnhancedModelConfig(
            use_cross_modal_attention=args.cross_modal_attention,
            use_residual_connections=args.residual_connections,
            use_layer_normalization=args.residual_connections, # Enable with residuals
            use_similarity_regularization=(args.similarity_loss_weight > 0),
            similarity_loss_weight=args.similarity_loss_weight,
            use_flux_messages=args.use_flux_messages,
            flux_message_depth=args.flux_message_depth,
            use_mass_balance_projection=args.use_mass_balance_projection,
            layer_type=args.layer_type,
            structural_loss_weight=args.structural_loss_weight,
            use_gene_skip=args.gene_skip,
            use_gene_film=args.gene_film,
            use_detection_gate=args.detection_gate,
            detection_gate_weight=args.detection_gate_weight,
        )
        print("🏗️  Using architectural modifications:")
        if args.layer_type != "GAT":
            print(f"  ✓ Layer type: {args.layer_type}")
        if args.structural_loss_weight > 0:
            print(f"  ✓ Structural consistency loss (weight: {args.structural_loss_weight})")
        if args.mask_target_smiles > 0:
            print(f"  ✓ SMILES masking (probability: {args.mask_target_smiles})")
        if args.residual_connections:
            print("  ✓ Residual connections (skip connections in GNN layers)")
        if args.cross_modal_attention:
            print("  ✓ Cross-modal attention (reaction-metabolite interaction)")
        if args.similarity_loss_weight > 0:
            print(f"  ✓ Similarity regularization (weight: {args.similarity_loss_weight})")
        if args.use_flux_messages:
            print(f"  ✓ Flux message passing (at depth {args.flux_message_depth})")
        if args.use_mass_balance_projection:
            print("  ✓ Mass balance projection (Cholesky-based exact constraints)")
        if args.gene_skip:
            print("  ✓ Gene expression skip connection (raw gene → MCR head)")
        if args.gene_film:
            print("  ✓ Gene expression FiLM conditioning (per GNN layer)")
        if args.detection_gate:
            print(f"  ✓ Detection gate (BCE weight: {args.detection_gate_weight})")
    else:
        print("📌 Using baseline ConcatGNN model architecture")
        
    # Print physics configuration
    physics_components = []
    if args.lambda_bal > 0: physics_components.append(f"Balance({args.lambda_bal})")
    if args.lambda_corr > 0: physics_components.append(f"Corr({args.lambda_corr})")
    if args.lambda_var > 0: physics_components.append(f"Var({args.lambda_var})")
    if args.lambda_kl > 0: physics_components.append(f"KL({args.lambda_kl})")
    
    if physics_components:
        print(f"⚡ Physics loss components: {', '.join(physics_components)}")
    else:
        print("📊 Using MSE-only loss")
    
    if args.config:
        experiment_config = load_experiment_config(args.config)
        if arch_config is not None:
            experiment_config.model.variant = 'enhanced'
            experiment_config.model.enhanced_config = arch_config.to_dict()
    else:
        experiment_config = build_config_from_args(args, arch_config)

    if args.random_graph and not args.compare:
        experiment_config.data.random_graph = True

    if args.compare:
        original_config = deepcopy(experiment_config)
        random_config = deepcopy(experiment_config)
        original_config.data.random_graph = False
        random_config.data.random_graph = True

        print('\n=== Training on original graph ===')
        results_orig = train_experiment(original_config)
        print('\n=== Training on randomized graph ===')
        results_rand = train_experiment(random_config)

        print('\nComparison Metrics:')
        print(f"Original graph: {results_orig}")
        print(f"Random graph:   {results_rand}")
    else:
        results = train_experiment(experiment_config)
        print('\nFinal results:')
        print(results)
