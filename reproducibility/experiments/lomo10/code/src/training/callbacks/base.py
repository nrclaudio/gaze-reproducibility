"""Callback interfaces for training events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class BatchOutput:
    loss: float
    loss_components: Dict[str, float]
    flux_stats: Optional[Dict[str, float]] = None


class TrainerCallback:
    """Base class for hooks that respond to trainer lifecycle events."""

    def on_train_begin(self, trainer: 'Trainer') -> None:
        pass

    def on_train_end(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_validation_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_test_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        pass

    def on_batch_end(self, trainer: 'Trainer', epoch: int, batch_idx: int, output: BatchOutput) -> None:
        pass


class CallbackList(TrainerCallback):
    """Composite wrapper that forwards events to multiple callbacks."""

    def __init__(self, callbacks: Optional[list[TrainerCallback]] = None) -> None:
        self.callbacks = callbacks or []

    def append(self, callback: TrainerCallback) -> None:
        self.callbacks.append(callback)

    # Delegate event handlers to all callbacks
    def on_train_begin(self, trainer: 'Trainer') -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: 'Trainer') -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_validation_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_validation_end(trainer, epoch, logs)

    def on_test_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_test_end(trainer, logs)

    def on_batch_end(self, trainer: 'Trainer', epoch: int, batch_idx: int, output: BatchOutput) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, epoch, batch_idx, output)
