#!/usr/bin/env python3
"""
XGBoost baseline under the same 5-fold CV as the GNN modality ablation.

Trains one XGBoost per metabolite per fold, aggregates predictions, and
saves a results JSON compatible with analyze_modality_results.py.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.utils.data_paths import get_standard_data_paths

import torch


def run_kfold_xgboost(args):
    """Train per-metabolite XGBoost under the same 5-fold CV as the GNN ablation."""

    print("Loading dataset...")
    device = torch.device("cpu")
    mudata_path, graph_path, gene_mapping_path = get_standard_data_paths(identifier="")

    dataset = FiLMDatasetWithSMILES(
        mudata_path=mudata_path,
        embeddings_graph_path=graph_path,
        gene_mapping_path=gene_mapping_path,
        device=device,
    )

    mudata = dataset.mudata

    # Extract RNA features
    rna_data = mudata["rna"].X
    if scipy.sparse.issparse(rna_data):
        rna_data = rna_data.toarray()
    # Drop genes with NaN
    genes_ok = ~np.isnan(rna_data).any(axis=0)
    rna_data = rna_data[:, genes_ok]
    print(f"RNA features: {rna_data.shape}")

    # Extract metabolite targets
    met_data = mudata["metabolomics"].X
    if scipy.sparse.issparse(met_data):
        met_data = met_data.toarray()
    measured_mask = ~np.isnan(met_data)
    n_metabolites = met_data.shape[1]
    print(f"Metabolites: {n_metabolites}, Samples: {met_data.shape[0]}")

    # Same KFold as run_kfold_modality_sweep.py
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n--- FOLD {fold_idx + 1}/5 ---")

        # Standardize RNA features on training fold
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(rna_data[train_indices])
        X_val_all = scaler.transform(rna_data[val_indices])

        fold_preds = []
        fold_targets = []

        for met_idx in range(n_metabolites):
            # Get samples with measured values
            train_mask = measured_mask[train_indices, met_idx]
            val_mask = measured_mask[val_indices, met_idx]

            X_train = X_train_all[train_mask]
            y_train = met_data[train_indices[train_mask], met_idx]
            X_val = X_val_all[val_mask]
            y_val = met_data[val_indices[val_mask], met_idx]

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=args.seed,
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            fold_preds.extend(y_pred.tolist())
            fold_targets.extend(y_val.tolist())

        # Fold-level metrics (pooled across all metabolites, same as GNN eval)
        fold_preds = np.array(fold_preds)
        fold_targets = np.array(fold_targets)
        fold_r2 = float(r2_score(fold_targets, fold_preds))
        fold_rmse = float(np.sqrt(mean_squared_error(fold_targets, fold_preds)))

        print(f"  Fold {fold_idx + 1}: R2={fold_r2:.4f}, RMSE={fold_rmse:.4f}")
        fold_results.append({"fold": fold_idx, "val_r2": fold_r2, "val_rmse": fold_rmse})

    # Aggregate
    r2_scores = [r["val_r2"] for r in fold_results]
    rmse_scores = [r["val_rmse"] for r in fold_results]

    summary = {
        "config_name": "XGBoost_Baseline",
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "mean_rmse": float(np.mean(rmse_scores)),
        "std_rmse": float(np.std(rmse_scores)),
        "fold_results": fold_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "XGBoost_Baseline_results.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"XGBoost 5-Fold CV:  R2 = {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
    print(f"                   RMSE = {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost baseline under 5-fold CV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match GNN ablation)")
    args = parser.parse_args()
    run_kfold_xgboost(args)
