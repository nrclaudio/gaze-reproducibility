"""Console logging callback for trainer progress."""

from __future__ import annotations

from typing import Any, Dict

from .base import TrainerCallback


class LoggingCallback(TrainerCallback):
    """Prints concise epoch summaries and test metrics."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        if not self.verbose:
            return
        train_loss = logs.get('train_loss')
        val_loss = logs.get('val_loss')
        val_r2 = logs.get('val_r2')
        val_rmse = logs.get('val_rmse')
        lr = logs.get('learning_rate')
        
        # Format main metrics
        message = (
            f"Epoch {epoch+1} | LR {lr:.6f} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Val R² {val_r2:.4f} | Val RMSE {val_rmse:.4f}"
        )

        # Add breakdown of loss components if available
        loss_components = logs.get('loss_components', {})
        if loss_components:
            components_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
            message += f" | {components_str}"

        print(message)

    def on_test_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        if not self.verbose:
            return
        print(
            f"Test Loss: {logs['test_loss']:.4f}, "
            f"Test R2: {logs['test_r2']:.4f}, "
            f"Test RMSE: {logs['test_rmse']:.4f}"
        )
