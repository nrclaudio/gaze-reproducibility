"""Main training logic for concatenated GNN with SMILES embeddings."""

import torch
import numpy as np
import os
import json

from src.config import ExperimentConfig
from src.data.datamodule import FiLMDataModule
from src.models.components import EnhancedModelConfig
from src.models.factory import create_model
from src.training.loss_factory import create_physics_loss
from src.training.callbacks import DiagnosticsCallback, LoggingCallback
from src.training.trainer_engine import Trainer
from ..utils.data_paths import get_project_root


def train_experiment(experiment_config: ExperimentConfig, *, scheduler_factory=None) -> dict:
    """Train a model using the structured experiment configuration.

    Args:
        experiment_config: Full experiment configuration.
        scheduler_factory: Optional callable ``(optimizer) -> scheduler`` to
            override the default ``ReduceLROnPlateau``.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = experiment_config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_cfg = experiment_config.data
    model_cfg = experiment_config.model
    physics_cfg = experiment_config.physics
    training_cfg = experiment_config.training
    contrastive_cfg = experiment_config.contrastive

    data_module = FiLMDataModule(
        config=data_cfg,
        seed=seed,
        device=device,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    dataset = data_module.dataset
    splits = data_module.splits
    test_indices = splits.get('test', [])

    use_normalised = data_cfg.normalized
    gradient_accumulation_steps = data_cfg.gradient_accumulation_steps
    random_graph = data_cfg.random_graph

    learning_rate = training_cfg.learning_rate
    max_epochs = training_cfg.max_epochs
    patience_setting = training_cfg.patience

    pretrained_encoders = contrastive_cfg.pretrained_encoders_path
    enable_contrastive = contrastive_cfg.enabled
    contrastive_weight = contrastive_cfg.weight
    contrastive_temperature = contrastive_cfg.temperature

    use_directionality = model_cfg.use_directionality
    flux_message_depth = model_cfg.flux_message_depth
    use_flux_messages = model_cfg.use_flux_messages
    use_mass_balance_projection = model_cfg.use_mass_balance_projection
    hidden_dim = model_cfg.hidden_dim
    num_layers = model_cfg.num_layers
    dropout = model_cfg.dropout

    enhanced_config_data = model_cfg.enhanced_config
    enhanced_config = None
    if enhanced_config_data is not None:
        if isinstance(enhanced_config_data, EnhancedModelConfig):
            enhanced_config = enhanced_config_data
        else:
            enhanced_config = EnhancedModelConfig.from_dict(enhanced_config_data)

    λ_bal = physics_cfg.lambda_balance
    λ_corr = physics_cfg.lambda_correlation
    λ_var = physics_cfg.lambda_variance
    λ_kl = physics_cfg.lambda_kl
    use_listwise_ranking = physics_cfg.use_listwise_ranking
    enzyme_confidence = physics_cfg.enzyme_confidence
    flux_flexibility = physics_cfg.flux_flexibility

    epochs = max_epochs if max_epochs is not None else 300
    patience = patience_setting if patience_setting is not None else 10

    base_dir = os.path.join(get_project_root(), 'trained_models')
    model_type = 'concat_smiles'

    if use_normalised:
        model_type = f'{model_type}_normalised'
    if random_graph:
        model_type = f'{model_type}_random'
    if use_directionality:
        model_type = f'{model_type}_directional'
    if λ_bal > 0 or λ_corr > 0 or λ_var > 0 or λ_kl > 0:
        model_type = f'{model_type}_physics'

    if training_cfg.save_dir:
        model_dir = training_cfg.save_dir
    elif experiment_config.config_name:
        model_dir = os.path.join(
            base_dir,
            model_type,
            f'seed_{experiment_config.seed}',
            experiment_config.config_name,
        )
    else:
        model_dir = os.path.join(base_dir, model_type, f'seed_{experiment_config.seed}')
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'experiment_config.json'), 'w') as cfg_handle:
        json.dump(experiment_config.to_dict(), cfg_handle, indent=2)

    embedding_dim = dataset.embedding_dim
    gene_dim = dataset.gene_dim
    smiles_dim = dataset.smiles_dim

    if hidden_dim is None:
        hidden_dim = 128
    if num_layers is None:
        num_layers = 3
    if dropout is None:
        dropout = 0.2

    model_cfg.hidden_dim = hidden_dim
    model_cfg.num_layers = num_layers
    model_cfg.dropout = dropout
    model_cfg.use_flux_messages = use_flux_messages
    model_cfg.flux_message_depth = flux_message_depth
    model_cfg.use_mass_balance_projection = use_mass_balance_projection

    if enhanced_config is not None:
        model_cfg.variant = 'enhanced'
        model_cfg.enhanced_config = enhanced_config

    model = create_model(
        model_cfg,
        embedding_dim=embedding_dim,
        gene_dim=gene_dim,
        smiles_dim=smiles_dim,
    )

    if pretrained_encoders is not None:
        print(f"Loading contrastive pretrained encoders from: {pretrained_encoders}")
        try:
            encoder_checkpoint = torch.load(pretrained_encoders, map_location=device, weights_only=True)
        except TypeError:
            encoder_checkpoint = torch.load(pretrained_encoders, map_location=device)
        except Exception as exc:
            print(f"Warning: Safe weights-only load failed: {exc}")
            print("Falling back to weights_only=False. Proceed only if checkpoint is trusted.")
            encoder_checkpoint = torch.load(pretrained_encoders, map_location=device, weights_only=False)
        encoder_components = ['reaction_encoder', 'metabolite_encoder', 'convs']
        loaded_components = []

        for component in encoder_components:
            if component in encoder_checkpoint and hasattr(model, component):
                try:
                    getattr(model, component).load_state_dict(encoder_checkpoint[component])
                    loaded_components.append(component)
                    print(f"Loaded {component} from contrastive pretraining")
                except Exception as exc:
                    print(f"Warning: Could not load {component}: {exc}")
            else:
                print(f"Warning: Encoder checkpoint missing component: {component}")

        if not loaded_components:
            print("Warning: No encoder components were successfully loaded from contrastive pretraining")

    model.to(device)

    if enable_contrastive:
        model.enable_contrastive_learning(temperature=contrastive_temperature)

    criterion, _ = create_physics_loss(
        physics_cfg,
        enzyme_confidence=enzyme_confidence,
        flux_flexibility=flux_flexibility,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    if scheduler_factory is not None:
        scheduler = scheduler_factory(optimizer)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    diagnostics_callback = DiagnosticsCallback(model_dir)
    logging_callback = LoggingCallback()
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        gradient_accumulation_steps=gradient_accumulation_steps,
        contrastive_weight=contrastive_weight,
        enable_contrastive=enable_contrastive,
        enable_metabolite_dropout=training_cfg.enable_metabolite_dropout,
        mask_target_smiles=training_cfg.mask_target_smiles,
        validation_objective=training_cfg.validation_objective,
        callbacks=[diagnostics_callback, logging_callback],
    )

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        checkpoint_path=os.path.join(model_dir, 'best_validation_model.pt'),
    )

    if history.get('train_loss'):
        print(
            f"Final Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Val Loss: {history['val_loss'][-1]:.4f}"
        )

    has_physics = (λ_bal > 0 or λ_corr > 0 or λ_var > 0 or λ_kl > 0)

    if test_indices:
        test_logs = trainer.test(
            test_loader,
            dataset=dataset,
            test_indices=test_indices,
            has_physics=has_physics,
        )
    else:
        # Historical diagnostics infer "training" as the complement of test.
        # With a separate held-out cohort that would inspect excluded samples.
        test_logs = {key: float('nan') for key in ('test_loss', 'test_r2', 'test_rmse')}

    total_epochs_run = len(history['train_loss'])
    best_epoch = total_epochs_run - trainer.state.patience_counter

    results = {
        'test_loss': float(test_logs['test_loss']),
        'test_r2': float(test_logs['test_r2']),
        'test_rmse': float(test_logs['test_rmse']),
        'best_epoch': best_epoch,
        'total_epochs_run': total_epochs_run,
    }
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(model_dir, 'sample_indices.json'), 'w') as f:
        json.dump(splits, f, indent=2)

    torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

    with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(
        f"Test Loss: {results['test_loss']:.4f}, "
        f"Test R2: {results['test_r2']:.4f}, "
        f"Test RMSE: {results['test_rmse']:.4f}"
    )
    print(f"Model saved to: {model_dir}/best_model.pt")

    return results
