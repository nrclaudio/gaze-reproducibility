"""
Evaluate a pre-trained GAZE model on external UnitedMet ccRCC datasets.

Loads a trained model checkpoint and runs inference on the UnitedMet MuData,
computing per-metabolite and per-dataset metrics (Spearman, Pearson, R², RMSE).

Usage:
    conda run -n MetaboliteRNA python scripts/experiments/evaluate_external.py \
        --model-path <checkpoint.pt> \
        --output-dir results/external_unitedmet/
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.data.batching import collate_metabolic_data
from src.models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
from src.models.components import EnhancedModelConfig
from src.utils.data_paths import get_graph_path, get_gene_mapping_path


def load_model(model_path, dataset, device):
    """Load a pre-trained model from checkpoint, reusing the pattern from evaluate_synthetic_lomo."""
    checkpoint = torch.load(model_path, map_location=device)

    embedding_dim = dataset.embedding_dim
    gene_dim = dataset.gene_dim
    smiles_dim = dataset.smiles_dim

    # Try to load experiment config from the model directory
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "experiment_config.json")

    hidden_dim = 128
    num_layers = 3
    dropout = 0.2
    use_flux_messages = True
    config = EnhancedModelConfig(
        use_cross_modal_attention=True,
        use_residual_connections=True,
        use_similarity_regularization=True,
        use_mcr=True,
    )

    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            exp_config = json.load(f)
        model_cfg = exp_config.get("model", {})
        hidden_dim = model_cfg.get("hidden_dim", hidden_dim)
        num_layers = model_cfg.get("num_layers", num_layers)
        dropout = model_cfg.get("dropout", dropout)
        use_flux_messages = model_cfg.get("use_flux_messages", True)
        enhanced_cfg_data = model_cfg.get("enhanced_config")
        if enhanced_cfg_data:
            config = EnhancedModelConfig.from_dict(enhanced_cfg_data)
    else:
        print(f"Warning: No config at {config_path}, using defaults")
        if "reaction_encoder.weight" in checkpoint:
            hidden_dim = checkpoint["reaction_encoder.weight"].shape[0]

    model = EnhancedConcatGNNWithSMILES(
        embedding_dim=embedding_dim,
        gene_dim=gene_dim,
        smiles_dim=smiles_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        config=config,
        use_flux_messages=use_flux_messages,
    )

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded: hidden_dim={hidden_dim}, num_layers={num_layers}")
    return model


def get_metabolite_info(dataset):
    """Build a mapping from graph node index to metabolite identifiers."""
    graph = dataset.graph
    info = {}
    for idx in dataset.metabolite_indices:
        hmdb = graph.hmdb_ids[idx] if hasattr(graph, "hmdb_ids") else ""
        kegg = graph.kegg_ids[idx] if hasattr(graph, "kegg_ids") else ""
        name = graph.node_names[idx] if hasattr(graph, "node_names") else f"node_{idx}"
        info[idx] = {"name": name, "hmdb": hmdb, "kegg": kegg}
    return info


def run_inference(model, dataset, device, batch_size=32):
    """Run model inference on all samples, returning predictions and targets per metabolite."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_metabolic_data,
        num_workers=0,
    )

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, "num_nodes"):
                batch.num_nodes = batch.x_embedding.size(1)

            out, *_ = model(batch, return_embeddings=False, compute_physics=False)
            # out: [B, N]

            targets = batch.y
            if targets.dim() > 2:
                targets = targets.squeeze(-1)

            all_preds.append(out.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(batch.mask.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()     # [N_samples, N_nodes]
    targets = torch.cat(all_targets, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()

    return preds, targets, masks


def compute_per_metabolite_metrics(preds, targets, masks, metabolite_indices, met_info):
    """Compute Spearman, Pearson, R², RMSE per metabolite."""
    results = []
    for met_idx in metabolite_indices:
        mask = masks[:, met_idx].astype(bool)
        n_valid = mask.sum()
        if n_valid < 3:
            continue

        p = preds[mask, met_idx]
        t = targets[mask, met_idx]

        # Filter NaN
        valid = ~(np.isnan(p) | np.isnan(t))
        if valid.sum() < 3:
            continue
        p, t = p[valid], t[valid]

        spearman_rho, spearman_p = spearmanr(t, p)
        pearson_r, pearson_p = pearsonr(t, p)
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = np.sqrt(np.mean((t - p) ** 2))

        info = met_info.get(met_idx, {})
        results.append({
            "node_idx": met_idx,
            "name": info.get("name", ""),
            "hmdb": info.get("hmdb", ""),
            "kegg": info.get("kegg", ""),
            "n_samples": int(p.shape[0]),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "r2": float(r2),
            "rmse": float(rmse),
        })
    return results


def compute_per_dataset_metrics(preds, targets, masks, metabolite_indices, dataset_labels, met_info):
    """Compute metrics broken down by dataset."""
    datasets = sorted(set(dataset_labels))
    per_dataset = {}

    for ds in datasets:
        ds_mask = np.array([d == ds for d in dataset_labels])
        ds_preds = preds[ds_mask]
        ds_targets = targets[ds_mask]
        ds_masks = masks[ds_mask]

        ds_results = compute_per_metabolite_metrics(
            ds_preds, ds_targets, ds_masks, metabolite_indices, met_info
        )
        if ds_results:
            df = pd.DataFrame(ds_results)
            per_dataset[ds] = {
                "n_metabolites": len(df),
                "n_samples": int(ds_mask.sum()),
                "mean_spearman": float(df["spearman_rho"].mean()),
                "median_spearman": float(df["spearman_rho"].median()),
                "mean_pearson": float(df["pearson_r"].mean()),
                "mean_r2": float(df["r2"].mean()),
                "mean_rmse": float(df["rmse"].mean()),
                "metabolite_results": ds_results,
            }
    return per_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAZE on external UnitedMet datasets")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mudata-path", type=str,
                        default=os.path.join(project_root, "data", "unitedmet", "unitedmet_ccRCC.h5mu"),
                        help="Path to UnitedMet MuData")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--graph-identifier", type=str, default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load dataset
    graph_path = get_graph_path(identifier=args.graph_identifier)
    gene_mapping_path = get_gene_mapping_path()

    print(f"Loading UnitedMet MuData from {args.mudata_path}")
    dataset = FiLMDatasetWithSMILES(
        mudata_path=args.mudata_path,
        embeddings_graph_path=graph_path,
        gene_mapping_path=gene_mapping_path,
    )

    # Load model
    model = load_model(args.model_path, dataset, device)

    # Get metabolite info
    met_info = get_metabolite_info(dataset)
    metabolite_indices = dataset.metabolite_indices

    # Get dataset labels from mudata
    dataset_labels = dataset.mudata["rna"].obs["dataset"].tolist()

    # Run inference
    print(f"\nRunning inference on {len(dataset)} samples...")
    preds, targets, masks = run_inference(model, dataset, device, args.batch_size)
    print(f"Predictions shape: {preds.shape}")

    # Per-metabolite metrics (all samples pooled)
    print("\nComputing per-metabolite metrics (all samples)...")
    all_results = compute_per_metabolite_metrics(
        preds, targets, masks, metabolite_indices, met_info
    )
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("spearman_rho", ascending=False)

    # Per-dataset breakdown
    print("Computing per-dataset metrics...")
    per_dataset = compute_per_dataset_metrics(
        preds, targets, masks, metabolite_indices, dataset_labels, met_info
    )

    # Summary
    print("\n" + "=" * 60)
    print("EXTERNAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total metabolites evaluated: {len(results_df)}")
    print(f"Mean Spearman rho: {results_df['spearman_rho'].mean():.4f}")
    print(f"Median Spearman rho: {results_df['spearman_rho'].median():.4f}")
    print(f"Mean Pearson r: {results_df['pearson_r'].mean():.4f}")
    print(f"Mean R²: {results_df['r2'].mean():.4f}")
    print(f"Mean RMSE: {results_df['rmse'].mean():.4f}")

    print(f"\nPer-dataset breakdown:")
    for ds, info in sorted(per_dataset.items()):
        print(f"  {ds}: {info['n_metabolites']} metabolites, {info['n_samples']} samples, "
              f"mean Spearman={info['mean_spearman']:.4f}, median={info['median_spearman']:.4f}")

    print(f"\nTop 10 metabolites by Spearman:")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['name']}: rho={row['spearman_rho']:.3f}, r={row['pearson_r']:.3f}, n={row['n_samples']}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results_df.to_csv(os.path.join(args.output_dir, "per_metabolite_results.csv"), index=False)

    # Per-dataset CSVs
    for ds, info in per_dataset.items():
        ds_df = pd.DataFrame(info["metabolite_results"])
        ds_df.to_csv(os.path.join(args.output_dir, f"per_metabolite_{ds}.csv"), index=False)

    # Summary JSON
    summary = {
        "n_samples": int(preds.shape[0]),
        "n_metabolites_evaluated": len(results_df),
        "overall": {
            "mean_spearman": float(results_df["spearman_rho"].mean()),
            "median_spearman": float(results_df["spearman_rho"].median()),
            "mean_pearson": float(results_df["pearson_r"].mean()),
            "mean_r2": float(results_df["r2"].mean()),
            "mean_rmse": float(results_df["rmse"].mean()),
        },
        "per_dataset": {
            ds: {k: v for k, v in info.items() if k != "metabolite_results"}
            for ds, info in per_dataset.items()
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
