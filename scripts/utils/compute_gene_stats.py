#!/usr/bin/env python
"""Compute per-gene statistics from a bulk training dataset.

This is a post-hoc utility for models that were trained before
gene_stats.pt was automatically saved.  It reconstructs the dataset
using the experiment config stored alongside the checkpoint and saves
``gene_stats.pt`` into the same directory.

Usage:
    python scripts/utils/compute_gene_stats.py trained_models/concat_smiles/seed_42/best_model.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config import ExperimentConfig
from src.data.datamodule import FiLMDataModule
from src.inference.normalization import compute_gene_statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gene_stats.pt for a trained model"
    )
    parser.add_argument("model_path", help="Path to best_model.pt checkpoint")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing gene_stats.pt",
    )
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model_path)
    save_path = os.path.join(model_dir, "gene_stats.pt")

    if os.path.exists(save_path) and not args.force:
        print(f"gene_stats.pt already exists at {save_path}. Use --force to overwrite.")
        return

    # Load experiment config
    config_path = os.path.join(model_dir, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            experiment_config = ExperimentConfig.from_dict(json.load(f))
    else:
        print("Warning: experiment_config.json not found; using defaults")
        experiment_config = ExperimentConfig()

    device = torch.device("cpu")

    data_module = FiLMDataModule(
        config=experiment_config.data,
        seed=experiment_config.seed,
        device=device,
    )
    data_module.setup(allow_single_slide_with_teacher=False)
    dataset = data_module.dataset

    compute_gene_statistics(dataset, save_path=save_path)


if __name__ == "__main__":
    main()
