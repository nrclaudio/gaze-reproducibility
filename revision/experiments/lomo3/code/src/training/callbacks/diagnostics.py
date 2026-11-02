"""Callbacks wrapping existing diagnostic utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import TrainerCallback, BatchOutput
from ..diagnostics import (
    _save_bulk_diagnostics,
    _save_basic_diagnostics,
    _save_test_diagnostics,
    save_per_metabolite_metrics,
)


class DiagnosticsCallback(TrainerCallback):
    """Handles diagnostic saving at key training checkpoints."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.train_history: Dict[str, List[float]] = {}
        self.test_buffers: Dict[str, list] = {}

    def on_batch_end(self, trainer: 'Trainer', epoch: int, batch_idx: int, output: BatchOutput) -> None:
        # Accumulate loss components for logging if desired
        for key, value in output.loss_components.items():
            history = self.train_history.setdefault(key, [])
            history.append(value)

    def on_test_end(self, trainer: 'Trainer', logs: Dict[str, Any]) -> None:
        has_physics = logs.get('has_physics', False)

        _save_test_diagnostics(
            logs.get('test_fluxes', []),
            logs.get('test_deltas', []),
            logs.get('test_embeddings', []),
            logs.get('test_metabolite_preds', []),
            logs.get('test_preds', []),
            logs.get('test_targets', []),
            self.model_dir,
            use_enhanced_physics=has_physics,
        )

        dataset = logs.get('dataset')
        model = logs.get('model')
        device = logs.get('device')
        test_indices = logs.get('test_indices', [])

        if dataset is not None and model is not None and device is not None:
            try:
                save_per_metabolite_metrics(self.model_dir, dataset, model, device, test_indices)
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"Error generating per-metabolite metrics: {exc}")
