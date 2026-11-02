"""Trainer class coordinating training, validation, and testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

from .callbacks.base import CallbackList, TrainerCallback, BatchOutput
from ..utils.batch_utils import get_batch_size


@dataclass
class TrainerState:
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    epoch: int = 0


class Trainer:
    """Stateful trainer that orchestrates optimisation and evaluation."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion,
        gradient_accumulation_steps: int,
        contrastive_weight: float,
        enable_contrastive: bool,
        enable_metabolite_dropout: bool = False,
        mask_target_smiles: bool = False,
        validation_objective: str = 'configured_loss',
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.contrastive_weight = contrastive_weight
        self.enable_contrastive = enable_contrastive
        self.enable_metabolite_dropout = enable_metabolite_dropout
        self.mask_target_smiles = mask_target_smiles
        if validation_objective not in {'configured_loss', 'prediction_mse'}:
            raise ValueError(f'Unknown validation objective: {validation_objective}')
        self.validation_objective = validation_objective

        self.callbacks = CallbackList(callbacks)
        self.state = TrainerState()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        epochs: int,
        patience: int,
        checkpoint_path=None,
    ) -> Dict[str, List[float]]:
        """Run the training/validation loop."""
        history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_r2': [],
            'val_rmse': [],
            'val_configured_loss': [],
            'val_prediction_mse': [],
            'learning_rate': [],
        }

        self.callbacks.on_train_begin(self)
        best_model_state = None

        for epoch in range(epochs):
            self.state.epoch = epoch
            self.callbacks.on_epoch_begin(self, epoch)

            train_metrics = self._run_training_epoch(train_loader)
            val_metrics = self._run_validation_epoch(val_loader, epoch)

            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_r2'].append(val_metrics['val_r2'])
            history['val_rmse'].append(val_metrics['val_rmse'])
            history['val_configured_loss'].append(val_metrics.get('val_configured_loss', val_metrics['val_loss']))
            history['val_prediction_mse'].append(val_metrics.get('val_prediction_mse', val_metrics['val_rmse']**2))
            history['learning_rate'].append(train_metrics['learning_rate'])

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if np.isfinite(val_metrics['val_loss']):
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            logs = {
                **train_metrics,
                **val_metrics,
                'epoch': epoch,
            }
            self.callbacks.on_validation_end(self, epoch, val_metrics)
            self.callbacks.on_epoch_end(self, epoch, logs)

            improved = val_metrics['val_loss'] < self.state.best_val_loss
            if improved:
                self.state.best_val_loss = val_metrics['val_loss']
                self.state.patience_counter = 0
                # state_dict() contains references to live parameters. Snapshot
                # tensors so later optimizer steps cannot overwrite the best epoch.
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                if checkpoint_path is not None:
                    torch.save(best_model_state, checkpoint_path)
            else:
                self.state.patience_counter += 1

            if self.state.patience_counter >= patience:
                break

        self.callbacks.on_train_end(self)

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history

    def _run_training_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_balance_loss = 0.0
        batch_count = 0
        epoch_loss_components: Dict[str, float] = {}
        epoch_flux_stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'nonzero_fraction': 0.0,
        }

        accumulation_count = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {self.state.epoch+1} Training", unit="batch", leave=False)

        for batch_idx, data in enumerate(train_pbar):
            batch_count += 1
            accumulation_count += 1
            data = data.to(self.device, non_blocking=True)

            if not hasattr(data, 'num_nodes'):
                if hasattr(data, 'x_embedding') and data.x_embedding is not None:
                    if get_batch_size(data) > 1 and hasattr(data, 'batch'):
                        data.num_nodes = data.x_embedding.size(1)
                    else:
                        data.num_nodes = data.x_embedding.size(0)
                else:
                    raise ValueError("Cannot determine num_nodes: x_embedding not available")

            if accumulation_count == 1:
                self.optimizer.zero_grad()

            batch_output, loss_components, flux_stats = self._forward_pass(data)

            scaled_loss = batch_output / self.gradient_accumulation_steps
            scaled_loss.backward()

            if accumulation_count == self.gradient_accumulation_steps:
                self.optimizer.step()
                accumulation_count = 0

            total_loss += batch_output.item()
            total_balance_loss += loss_components.get('Balance', 0.0)

            for key, value in loss_components.items():
                epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value

            for key in epoch_flux_stats.keys():
                if flux_stats and key in flux_stats:
                    epoch_flux_stats[key] += flux_stats[key]

            self.callbacks.on_batch_end(
                self,
                self.state.epoch,
                batch_idx,
                BatchOutput(
                    loss=batch_output.item(),
                    loss_components=loss_components,
                    flux_stats=flux_stats,
                ),
            )

        # Apply any remaining gradients
        if accumulation_count > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        num_batches = max(batch_count, 1)
        avg_loss_components = {k: v / num_batches for k, v in epoch_loss_components.items()}
        avg_flux_stats = {k: v / num_batches for k, v in epoch_flux_stats.items()}

        current_lr = self.optimizer.param_groups[0]['lr']
        return {
            'train_loss': total_loss / num_batches,
            'avg_balance_loss': total_balance_loss / num_batches,
            'loss_components': avg_loss_components,
            'flux_stats': avg_flux_stats,
            'learning_rate': current_lr,
        }

    def _forward_pass(self, data):
        need_embeddings = self.enable_contrastive and getattr(self.model, 'contrastive_module', None) is not None
        
        # Prepare kwargs for model forward pass
        from ..models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
        model_kwargs = {}
        if isinstance(self.model, EnhancedConcatGNNWithSMILES):
            if self.enable_metabolite_dropout:
                model_kwargs['dropout_metabolites'] = True
            if self.mask_target_smiles:
                model_kwargs['mask_target_smiles'] = self.mask_target_smiles

        model_out = self.model(
            data,
            return_embeddings=need_embeddings,
            compute_physics=True,
            **model_kwargs
        )
        preds, deltas, fluxes, embeddings = model_out[:4]
        criterion_loss = self.criterion(preds, data.y, data.mask, deltas, fluxes, self.state.epoch, data)
        loss = criterion_loss
        loss_components = getattr(self.criterion, 'last_loss_components', {})

        if fluxes is not None:
            flux_det = fluxes.detach()
            flux_stats = {
                'mean': float(flux_det.mean().item()),
                'std': float(flux_det.std().item()),
                'min': float(flux_det.min().item()),
                'max': float(flux_det.max().item()),
                'nonzero_fraction': float((flux_det.abs() > 1e-6).float().mean().item()),
            }
        else:
            flux_stats = None

        if self.enable_contrastive and getattr(self.model, 'contrastive_module', None) is not None:
            contrastive_loss = self.model.compute_contrastive_loss(
                data,
                embeddings,
                weight=self.contrastive_weight,
            )
            loss += contrastive_loss
            loss_components['Contrastive'] = contrastive_loss.item()

        # === Handle Structural Reconstruction Loss ===
        if hasattr(self.model, 'last_structural_loss') and self.model.last_structural_loss is not None:
            struct_loss = self.model.last_structural_loss
            loss += struct_loss
            loss_components['Structural'] = struct_loss.item()

        # === Handle Detection Gate Loss ===
        if hasattr(self.model, 'last_detection_loss') and self.model.last_detection_loss is not None:
            det_weight = getattr(self.model, 'detection_gate_weight', 1.0)
            det_loss = self.model.last_detection_loss * det_weight
            loss += det_loss
            loss_components['DetectionGate'] = det_loss.item()

        return loss, loss_components, flux_stats

    def _run_validation_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        val_preds: List[float] = []
        val_targets: List[float] = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                model_out = self.model(
                    data,
                    return_embeddings=False,
                    compute_physics=True,
                )
                preds, deltas, flux_vec, embeddings = model_out[:4]
                loss = self.criterion(preds, data.y, data.mask, deltas, flux_vec, epoch, data)

                mask = data.mask.cpu().numpy()
                preds_np = preds.cpu().numpy()[mask]
                targets_np = data.y.cpu().numpy()[mask]
                valid = ~np.isnan(preds_np) & ~np.isnan(targets_np)
                if np.sum(valid) > 0:
                    val_preds.extend(preds_np[valid])
                    val_targets.extend(targets_np[valid])

                val_loss += loss.item()

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            val_loss = float('nan')

        if val_preds:
            val_r2 = float(r2_score(val_targets, val_preds))
            rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))
        else:
            val_r2 = float('nan')
            rmse = float('nan')

        return {
            'val_loss': rmse**2 if self.validation_objective == 'prediction_mse' else val_loss,
            'val_configured_loss': val_loss,
            'val_prediction_mse': rmse**2,
            'val_r2': val_r2,
            'val_rmse': rmse,
        }

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test(self, test_loader, *, dataset=None, test_indices=None, has_physics=False):
        self.model.eval()
        test_loss = 0.0
        test_preds: List[float] = []
        test_targets: List[float] = []
        test_fluxes: List[np.ndarray] = []
        test_deltas: List[np.ndarray] = []
        test_embeddings: List[np.ndarray] = []
        test_metabolite_preds: List[np.ndarray] = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                model_out = self.model(
                    data,
                    return_embeddings=True,
                    compute_physics=True,
                )
                preds, deltas, flux_vec, embeddings = model_out[:4]
                loss = self.criterion(preds, data.y, data.mask, deltas, flux_vec, self.state.epoch, data)
                test_fluxes.append(flux_vec.cpu().numpy())
                test_deltas.append(deltas.cpu().numpy())
                test_metabolite_preds.append(preds.cpu().numpy())
                if embeddings is not None:
                    test_embeddings.append(embeddings.cpu().numpy())

                mask = data.mask.cpu().numpy()
                preds_np = preds.cpu().numpy()[mask]
                targets_np = data.y.cpu().numpy()[mask]
                valid = ~np.isnan(preds_np) & ~np.isnan(targets_np)
                if np.sum(valid) > 0:
                    test_preds.extend(preds_np[valid])
                    test_targets.extend(targets_np[valid])

                test_loss += loss.item()

        if len(test_loader) > 0:
            test_loss /= len(test_loader)

        if test_preds:
            test_r2 = float(r2_score(test_targets, test_preds))
            test_rmse = float(np.sqrt(mean_squared_error(test_targets, test_preds)))
        else:
            test_r2 = float('nan')
            test_rmse = float('nan')

        logs = {
            'test_loss': test_loss,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_fluxes': test_fluxes,
            'test_deltas': test_deltas,
            'test_embeddings': test_embeddings,
            'test_metabolite_preds': test_metabolite_preds,
            'test_preds': test_preds,
            'test_targets': test_targets,
            'dataset': dataset,
            'test_indices': test_indices,
            'model': self.model,
            'device': self.device,
            'has_physics': has_physics,
        }

        self.callbacks.on_test_end(self, logs)
        return logs
