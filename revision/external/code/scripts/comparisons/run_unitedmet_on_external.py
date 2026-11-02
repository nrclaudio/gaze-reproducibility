#!/usr/bin/env python3
"""
Run UnitedMet (Bayesian matrix factorization) on its own external ccRCC cohort.

UnitedMet is a supervised method that learns the joint gene-metabolite
distribution from paired measurements. This script runs metabolite-holdout
cross-validation on the UnitedMet ccRCC data, giving the method the best
possible advantage (its own data, its own evaluation protocol).

Usage:
    python scripts/comparisons/run_unitedmet_on_external.py \
        --mudata-path data/unitedmet/unitedmet_ccRCC.h5mu \
        --output-dir results/external_baselines/unitedmet_method/
        [--n-folds 10] [--n-steps 4000] [--fold-only N]
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from datetime import datetime

# Add UnitedMet to path
UNITEDMET_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "UnitedMet"
sys.path.insert(0, str(UNITEDMET_ROOT))

project_root = Path(__file__).resolve().parent.parent.parent


def load_external_data(mudata_path, max_genes=5000, min_coverage=0.3, datasets=None):
    """Load UnitedMet MuData and prepare matrices.

    Args:
        datasets: list of sub-cohort names to keep (e.g. ['RC18', 'RC20']).
                  If None, use all samples.
    """
    import mudata as mu

    mdata = mu.read(str(mudata_path))
    rna = mdata.mod["rna"]
    met = mdata.mod["metabolomics"]

    rna_X = rna.X.toarray() if hasattr(rna.X, "toarray") else np.array(rna.X)
    met_X = met.X.toarray() if hasattr(met.X, "toarray") else np.array(met.X)

    gene_names = list(rna.var_names)
    met_names = list(met.var_names)

    # Dataset labels
    dataset_labels = None
    if "dataset" in rna.obs.columns:
        dataset_labels = rna.obs["dataset"].tolist()

    print(f"Raw data: {rna_X.shape[0]} samples, {rna_X.shape[1]} genes, {met_X.shape[1]} metabolites")
    if dataset_labels:
        from collections import Counter
        print(f"  Datasets: {dict(Counter(dataset_labels))}")

    # Filter to requested sub-cohorts
    if datasets is not None and dataset_labels is not None:
        keep = np.array([d in datasets for d in dataset_labels])
        rna_X = rna_X[keep]
        met_X = met_X[keep]
        dataset_labels = [d for d, k in zip(dataset_labels, keep) if k]
        print(f"After filtering to {datasets}: {rna_X.shape[0]} samples")

    # Filter genes by coverage
    gene_coverage = np.mean(~np.isnan(rna_X) & (rna_X != 0), axis=0)
    gene_mask = gene_coverage >= min_coverage
    rna_X = rna_X[:, gene_mask]
    gene_names = [g for g, m in zip(gene_names, gene_mask) if m]
    print(f"After gene coverage filter (>={min_coverage:.0%}): {rna_X.shape[1]} genes")

    # Filter genes by variance
    gene_var = np.nanvar(rna_X, axis=0)
    var_mask = gene_var > 0
    rna_X = rna_X[:, var_mask]
    gene_names = [g for g, m in zip(gene_names, var_mask) if m]

    # Top N most variable
    if max_genes and rna_X.shape[1] > max_genes:
        gene_var = np.nanvar(rna_X, axis=0)
        top_idx = np.argsort(gene_var)[::-1][:max_genes]
        top_idx = np.sort(top_idx)
        rna_X = rna_X[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]
        print(f"After top-{max_genes} selection: {rna_X.shape[1]} genes")

    # Filter metabolites by coverage
    met_coverage = np.mean(~np.isnan(met_X), axis=0)
    met_mask = met_coverage >= 0.3
    met_X = met_X[:, met_mask]
    met_names = [n for n, m in zip(met_names, met_mask) if m]
    print(f"Metabolites after coverage filter: {met_X.shape[1]}")

    return rna_X, met_X, gene_names, met_names, dataset_labels


def tic_normalize(met_data):
    """TIC normalization for metabolomics data (per UnitedMet paper)."""
    normalized = np.copy(met_data)
    nan_mask = np.isnan(normalized)
    min_val = np.nanmin(normalized)
    for row in range(normalized.shape[0]):
        n_censored = np.sum(nan_mask[row])
        row_tic = np.nansum(normalized[row]) + 0.5 * n_censored * min_val
        if row_tic > 0:
            normalized[row] = normalized[row] / row_tic
    row_sums = np.nansum(normalized, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = normalized / row_sums
    return normalized


def prepare_unitedmet_data(rna_X, met_X, n_dims=30):
    """Prepare data in UnitedMet format (concatenated, ranked)."""
    met_norm = tic_normalize(met_X)
    data = np.concatenate([met_norm, rna_X], axis=1)

    N, J = data.shape
    K = n_dims
    n_batch = 1
    n_mets = met_norm.shape[1]

    n_obs = np.zeros((n_batch, J), dtype=int)
    n_obs[0, :] = np.sum(~np.isnan(data), axis=0)

    orders = np.zeros((N, J), dtype=int)
    ranks = np.zeros((N, J), dtype=int)

    batch_data = np.where(np.isnan(data), 0, data)
    indices = batch_data.argsort(axis=0, kind="stable")
    orders[:] = indices[::-1]
    ranks_temp = indices[::-1].argsort(axis=0, kind="stable")
    ranks[:] = np.where(ranks_temp > n_obs[0, :], n_obs[0, :], ranks_temp)

    censor_indicator = np.where(np.isnan(data), 1, 0)

    return {
        "data": data,
        "met_data": met_norm,
        "N": N, "J": J, "K": K,
        "n_batch": n_batch,
        "n_mets": n_mets,
        "n_obs": n_obs,
        "orders": orders,
        "ranks": ranks,
        "censor_indicator": censor_indicator,
        "batch_index_vector": np.zeros(N, dtype=int),
        "start_row": np.array([0]),
        "stop_row": np.array([N]),
    }


def run_fold(pyro_data, test_met_indices, n_steps=4000, lr=0.001, seed=42, n_posterior=500):
    """Run UnitedMet SVI with held-out metabolite features."""
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer.autoguide import AutoNormal
    from pyro.optim import Adam
    from pyro.infer import SVI, Trace_ELBO
    import pyro.poutine as poutine

    from Performance_Benchmarking.scripts.pyro_model import PlackettLuce_2D
    from Performance_Benchmarking.scripts.utils import gumbel_sampling_3D

    N = pyro_data["N"]
    J = pyro_data["J"]
    K = pyro_data["K"]
    n_batch = pyro_data["n_batch"]
    start_row = pyro_data["start_row"]
    stop_row = pyro_data["stop_row"]
    ranks = pyro_data["ranks"]

    n_obs = np.copy(pyro_data["n_obs"])
    for midx in test_met_indices:
        n_obs[0, midx] = 0

    orders_t = torch.tensor(pyro_data["orders"])
    n_obs_t = torch.tensor(n_obs)

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    def model(N, J, K, n_batch, start_row, stop_row, n_obs, orders):
        W = pyro.sample("W", dist.Normal(torch.zeros(N, K), torch.ones(N, K)).to_event(2))
        H = pyro.sample("H", dist.Normal(torch.zeros(K, J), torch.ones(K, J)).to_event(2))
        X = torch.mm(W, H)
        for b in range(n_batch):
            X_temp = X[start_row[b]:stop_row[b], :]
            temp_order = orders[start_row[b]:stop_row[b], :]
            pyro.sample(f"R_{b}", PlackettLuce_2D(X_temp, n_obs[b, :]), obs=temp_order)

    guide = AutoNormal(poutine.block(model, expose=["W", "H"]))
    optimizer = Adam({"lr": lr, "betas": [0.95, 0.999]})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    tol_e = 0.01
    loss_list = []
    for step in range(n_steps):
        loss = svi.step(N, J, K, n_batch, start_row, stop_row, n_obs_t, orders_t)
        loss_list.append(loss)
        if step % 500 == 0:
            print(f"  step {step}: loss = {loss:.1f}")
        if step > 0 and abs(loss - loss_list[-2]) <= tol_e:
            print(f"  Converged at step {step}")
            break

    W_loc = pyro.param("AutoNormal.locs.W").detach().numpy()
    W_scale = pyro.param("AutoNormal.scales.W").detach().numpy()
    H_loc = pyro.param("AutoNormal.locs.H").detach().numpy()
    H_scale = pyro.param("AutoNormal.scales.H").detach().numpy()

    np.random.seed(seed)
    W_draws = np.random.normal(W_loc, W_scale, size=(n_posterior, W_loc.shape[0], W_loc.shape[1]))
    np.random.seed(seed + 1)
    H_draws = np.random.normal(H_loc, H_scale, size=(n_posterior, H_loc.shape[0], H_loc.shape[1]))

    H_met_draws = H_draws[:, :, test_met_indices]
    X_met_draws = np.matmul(W_draws, H_met_draws)
    rank_hat_draws = gumbel_sampling_3D(X_met_draws)
    rank_hat_mean = np.mean(rank_hat_draws.argsort(axis=1, kind="stable"), axis=0)

    results = []
    for i, midx in enumerate(test_met_indices):
        true_ranks = ranks[:, midx]
        pred_ranks = rank_hat_mean[:, i]

        non_censor = pyro_data["censor_indicator"][:, midx] == 0
        if non_censor.sum() < 5:
            continue

        true_r = true_ranks[non_censor]
        pred_r = pred_ranks[non_censor]
        rho, pval = spearmanr(true_r, pred_r)

        met_true = pyro_data["met_data"][:, midx]
        valid = ~np.isnan(met_true)
        if valid.sum() >= 5:
            pr, pp = pearsonr(pred_ranks[valid], met_true[valid])
        else:
            pr, pp = np.nan, np.nan

        results.append({
            "metabolite_idx": int(midx),
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "n_samples": int(non_censor.sum()),
        })

    return results, loss_list[-1]


def main():
    parser = argparse.ArgumentParser(description="Run UnitedMet on external ccRCC data")
    parser.add_argument("--mudata-path", type=str,
                        default=str(project_root / "data/unitedmet/unitedmet_ccRCC.h5mu"))
    parser.add_argument("--output-dir", type=str,
                        default=str(project_root / "results/external_baselines/unitedmet_method"))
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--n-dims", type=int, default=30)
    parser.add_argument("--n-posterior", type=int, default=500)
    parser.add_argument("--max-genes", type=int, default=5000)
    parser.add_argument("--fold-only", type=int, default=None, help="Run single fold (for SLURM array)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Sub-cohorts to use (e.g. RC18 RC20). Default: all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"UnitedMet on External ccRCC Cohort")
    print(f"{'=' * 70}")
    print(f"  Data: {args.mudata_path}")
    print(f"  Datasets: {args.datasets or 'all'}")
    print(f"  Folds: {args.n_folds}, Steps: {args.n_steps}, K: {args.n_dims}")
    print(f"  Fold only: {args.fold_only}")
    print()

    # Load external data
    rna_X, met_X, gene_names, met_names, dataset_labels = load_external_data(
        args.mudata_path, max_genes=args.max_genes, datasets=args.datasets
    )

    # Prepare UnitedMet format
    pyro_data = prepare_unitedmet_data(rna_X, met_X, n_dims=args.n_dims)
    n_mets = pyro_data["n_mets"]
    print(f"\nPrepared: N={pyro_data['N']}, J={pyro_data['J']}, K={pyro_data['K']}")
    print(f"  Metabolites: {n_mets}, Genes: {pyro_data['J'] - n_mets}")

    # Create metabolite folds
    np.random.seed(args.seed)
    available_mets = []
    for m in range(n_mets):
        n_valid = pyro_data["n_obs"][0, m]
        if n_valid >= 20:
            available_mets.append(m)
    available_mets = np.array(available_mets)
    np.random.shuffle(available_mets)
    folds = np.array_split(available_mets, args.n_folds)
    print(f"  Available metabolites: {len(available_mets)} (in {args.n_folds} folds)")

    # Save fold assignments
    fold_info = {}
    for i, fold in enumerate(folds):
        fold_info[f"fold_{i}"] = {
            "metabolite_indices": fold.tolist(),
            "metabolite_names": [met_names[j] for j in fold],
            "n_metabolites": len(fold),
        }
    with open(out_dir / "fold_assignments.json", "w") as f:
        json.dump(fold_info, f, indent=2)

    # Run folds
    fold_range = [args.fold_only] if args.fold_only is not None else range(args.n_folds)

    all_results = []
    for fold_idx in fold_range:
        test_indices = folds[fold_idx]
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx}: holding out {len(test_indices)} metabolites")
        print(f"  e.g.: {[met_names[j] for j in test_indices[:3]]}...")
        print(f"{'=' * 60}")
        t0 = datetime.now()

        fold_results, final_loss = run_fold(
            pyro_data,
            test_met_indices=test_indices,
            n_steps=args.n_steps,
            seed=args.seed + fold_idx,
            n_posterior=args.n_posterior,
        )

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Fold {fold_idx} done in {elapsed:.0f}s, loss={final_loss:.1f}")

        for r in fold_results:
            r["metabolite_name"] = met_names[r["metabolite_idx"]]
            r["fold"] = fold_idx
        all_results.extend(fold_results)

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(out_dir / f"fold_{fold_idx}_results.csv", index=False)
        if len(fold_df) > 0:
            print(f"  Median rho: {fold_df['spearman_rho'].median():.3f}, "
                  f">0: {100 * (fold_df['spearman_rho'] > 0).mean():.0f}%")

    # Aggregate
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(out_dir / "unitedmet_external_results.csv", index=False)

        print(f"\n{'=' * 70}")
        print(f"OVERALL ({len(results_df)} metabolites)")
        print(f"{'=' * 70}")
        print(f"  Median Spearman rho: {results_df['spearman_rho'].median():.3f}")
        print(f"  Mean Spearman rho:   {results_df['spearman_rho'].mean():.3f}")
        print(f"  % positive rho:      {100 * (results_df['spearman_rho'] > 0).mean():.0f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
