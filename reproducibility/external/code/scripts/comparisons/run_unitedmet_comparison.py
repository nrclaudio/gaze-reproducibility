#!/usr/bin/env python3
"""
Run UnitedMet metabolite-holdout evaluation on CAMP data.

UnitedMet (Xie et al., 2025) uses Bayesian matrix factorization with a
Plackett-Luce ranking likelihood to impute metabolite levels from paired
RNA-seq + metabolomics data.

Evaluation protocol:
  - 10-fold CV over metabolite features (hold out ~115 metabolites per fold)
  - Train on remaining metabolites + all genes
  - Predict held-out metabolite ranks
  - Evaluate Spearman correlation between predicted and true ranks

Usage:
  python scripts/comparisons/run_unitedmet_comparison.py [--n-folds 10] [--n-steps 4000] [--n-dims 30] [--fold-only N]
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


def load_camp_data(max_genes=5000, min_coverage=0.5):
    """Load CAMP MuData and prepare for UnitedMet."""
    import mudata as mu

    mdata = mu.read(str(project_root / "data/raw/mudata_correct_HMDB_new_no_negatives.h5mu"))
    rna = mdata.mod["rna"]
    met = mdata.mod["metabolomics"]

    # Extract dense matrices
    rna_X = rna.X.toarray() if hasattr(rna.X, "toarray") else np.array(rna.X)
    met_X = met.X.toarray() if hasattr(met.X, "toarray") else np.array(met.X)

    gene_names = list(rna.var_names)
    met_names = list(met.var_names)

    print(f"Raw data: {rna_X.shape[0]} samples, {rna_X.shape[1]} genes, {met_X.shape[1]} metabolites")

    # Filter genes: require non-NaN in >= min_coverage of samples
    gene_coverage = np.mean(~np.isnan(rna_X), axis=0)
    gene_mask = gene_coverage >= min_coverage
    rna_X = rna_X[:, gene_mask]
    gene_names = [g for g, m in zip(gene_names, gene_mask) if m]
    print(f"After coverage filter (>={min_coverage:.0%}): {rna_X.shape[1]} genes")

    # Filter genes: require non-zero variance
    gene_var = np.nanvar(rna_X, axis=0)
    var_mask = gene_var > 0
    rna_X = rna_X[:, var_mask]
    gene_names = [g for g, m in zip(gene_names, var_mask) if m]
    print(f"After variance filter: {rna_X.shape[1]} genes")

    # Select top N most variable genes
    if max_genes and rna_X.shape[1] > max_genes:
        gene_var = np.nanvar(rna_X, axis=0)
        top_idx = np.argsort(gene_var)[::-1][:max_genes]
        top_idx = np.sort(top_idx)
        rna_X = rna_X[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]
        print(f"After top-{max_genes} selection: {rna_X.shape[1]} genes")

    # Filter metabolites: require non-NaN in >= 50% of samples
    met_coverage = np.mean(~np.isnan(met_X), axis=0)
    met_mask = met_coverage >= 0.5
    met_X = met_X[:, met_mask]
    met_names = [n for n, m in zip(met_names, met_mask) if m]
    print(f"Metabolites after coverage filter: {met_X.shape[1]}")

    return rna_X, met_X, gene_names, met_names, met.var


def tic_normalize(met_data):
    """TIC normalization for metabolomics data (per UnitedMet)."""
    normalized = np.copy(met_data)
    nan_mask = np.isnan(normalized)
    min_val = np.nanmin(normalized)
    for row in range(normalized.shape[0]):
        n_censored = np.sum(nan_mask[row])
        row_tic = np.nansum(normalized[row]) + 0.5 * n_censored * min_val
        if row_tic > 0:
            normalized[row] = normalized[row] / row_tic
    # Re-normalize so each row sums to 1
    row_sums = np.nansum(normalized, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = normalized / row_sums
    return normalized


def prepare_unitedmet_data(rna_X, met_X, n_dims=30):
    """Prepare single-batch CAMP data for UnitedMet SVI."""
    # TIC normalize metabolomics
    met_norm = tic_normalize(met_X)

    # Concatenate [metabolites | genes]
    data = np.concatenate([met_norm, rna_X], axis=1)

    N = data.shape[0]  # samples
    J = data.shape[1]  # features (mets + genes)
    K = n_dims
    n_batch = 1
    n_mets = met_norm.shape[1]
    batch_index_vector = np.zeros(N, dtype=int)

    # Count observations per feature
    n_obs = np.zeros((n_batch, J), dtype=int)
    n_obs[0, :] = np.sum(~np.isnan(data), axis=0)

    # Order and rank (within batch)
    orders = np.zeros((N, J), dtype=int)
    ranks = np.zeros((N, J), dtype=int)

    batch_data = np.where(np.isnan(data), 0, data)
    indices = batch_data.argsort(axis=0, kind="stable")
    orders[:] = indices[::-1]  # descending order
    ranks_temp = indices[::-1].argsort(axis=0, kind="stable")
    ranks[:] = np.where(ranks_temp > n_obs[0, :], n_obs[0, :], ranks_temp)

    censor_indicator = np.where(np.isnan(data), 1, 0)

    start_row = np.array([0])
    stop_row = np.array([N])

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
        "batch_index_vector": batch_index_vector,
        "start_row": start_row,
        "stop_row": stop_row,
    }


def run_fold(pyro_data, test_met_indices, n_steps=4000, lr=0.001, seed=42, n_posterior=500):
    """Run UnitedMet SVI with held-out metabolite features, return Spearman rho per metabolite."""
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer.autoguide import AutoNormal
    from pyro.optim import Adam
    from pyro.infer import SVI, Trace_ELBO
    import pyro.poutine as poutine

    # Import UnitedMet's Plackett-Luce distribution
    from Performance_Benchmarking.scripts.pyro_model import PlackettLuce_2D
    from Performance_Benchmarking.scripts.utils import gumbel_sampling_3D

    N = pyro_data["N"]
    J = pyro_data["J"]
    K = pyro_data["K"]
    n_batch = pyro_data["n_batch"]
    start_row = pyro_data["start_row"]
    stop_row = pyro_data["stop_row"]
    ranks = pyro_data["ranks"]
    n_mets = pyro_data["n_mets"]

    # Copy n_obs and mask held-out metabolites
    n_obs = np.copy(pyro_data["n_obs"])
    n_obs_orig = np.copy(pyro_data["n_obs"])
    for midx in test_met_indices:
        n_obs[0, midx] = 0  # mask this metabolite for batch 0

    orders_t = torch.tensor(pyro_data["orders"])
    n_obs_t = torch.tensor(n_obs)

    # Run SVI
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

    # Extract posterior parameters
    W_loc = pyro.param("AutoNormal.locs.W").detach().numpy()
    W_scale = pyro.param("AutoNormal.scales.W").detach().numpy()
    H_loc = pyro.param("AutoNormal.locs.H").detach().numpy()
    H_scale = pyro.param("AutoNormal.scales.H").detach().numpy()

    # Posterior prediction for held-out metabolites only
    np.random.seed(seed)
    W_draws = np.random.normal(W_loc, W_scale, size=(n_posterior, W_loc.shape[0], W_loc.shape[1]))
    np.random.seed(seed + 1)
    H_draws = np.random.normal(H_loc, H_scale, size=(n_posterior, H_loc.shape[0], H_loc.shape[1]))

    # Compute X_draws for held-out metabolites only
    H_met_draws = H_draws[:, :, test_met_indices]  # (n_posterior, K, n_test_mets)
    X_met_draws = np.matmul(W_draws, H_met_draws)  # (n_posterior, N, n_test_mets)

    # Gumbel sampling to get rank predictions
    rank_hat_draws = gumbel_sampling_3D(X_met_draws)
    rank_hat_mean = np.mean(rank_hat_draws.argsort(axis=1, kind="stable"), axis=0)

    # Evaluate: Spearman correlation for each held-out metabolite
    results = []
    for i, midx in enumerate(test_met_indices):
        # True ranks (from original data before masking)
        true_ranks = ranks[:, midx]
        pred_ranks = rank_hat_mean[:, i]

        # Only evaluate on non-censored samples
        non_censor = pyro_data["censor_indicator"][:, midx] == 0
        if non_censor.sum() < 10:
            continue

        true_r = true_ranks[non_censor]
        pred_r = pred_ranks[non_censor]

        rho, pval = spearmanr(true_r, pred_r)

        # Also compute Pearson r on the original (non-rank) metabolite values
        met_true = pyro_data["met_data"][:, midx]
        valid = ~np.isnan(met_true)
        if valid.sum() >= 10:
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
    parser = argparse.ArgumentParser(description="UnitedMet LOMO evaluation on CAMP")
    parser.add_argument("--n-folds", type=int, default=10, help="Number of CV folds over metabolites")
    parser.add_argument("--n-steps", type=int, default=4000, help="SVI optimization steps")
    parser.add_argument("--n-dims", type=int, default=30, help="Latent dimensions")
    parser.add_argument("--n-posterior", type=int, default=500, help="Posterior draws")
    parser.add_argument("--max-genes", type=int, default=5000, help="Max genes to use")
    parser.add_argument("--fold-only", type=int, default=None, help="Run only this fold (for SLURM array)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"UnitedMet LOMO Evaluation on CAMP")
    print(f"{'=' * 70}")
    print(f"  Folds: {args.n_folds}")
    print(f"  SVI steps: {args.n_steps}")
    print(f"  Latent dims: {args.n_dims}")
    print(f"  Max genes: {args.max_genes}")
    print(f"  Fold only: {args.fold_only}")
    print()

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else project_root / "comparison_results" / "unitedmet_lomo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    rna_X, met_X, gene_names, met_names, met_var = load_camp_data(max_genes=args.max_genes)

    # Prepare UnitedMet format
    pyro_data = prepare_unitedmet_data(rna_X, met_X, n_dims=args.n_dims)
    n_mets = pyro_data["n_mets"]
    print(f"\nPrepared UnitedMet data: N={pyro_data['N']}, J={pyro_data['J']}, K={pyro_data['K']}")
    print(f"  Metabolites: {n_mets}, Genes: {pyro_data['J'] - n_mets}")

    # Create metabolite folds
    np.random.seed(args.seed)
    # Only include metabolites with enough non-NaN samples
    available_mets = []
    for m in range(n_mets):
        n_valid = pyro_data["n_obs"][0, m]
        if n_valid >= 50:  # require at least 50 observed values
            available_mets.append(m)
    available_mets = np.array(available_mets)
    np.random.shuffle(available_mets)
    folds = np.array_split(available_mets, args.n_folds)
    print(f"  Available metabolites for CV: {len(available_mets)} (in {args.n_folds} folds)")

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
    if args.fold_only is not None:
        fold_range = [args.fold_only]
    else:
        fold_range = range(args.n_folds)

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
            lr=0.001,
            seed=args.seed + fold_idx,
            n_posterior=args.n_posterior,
        )

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Fold {fold_idx} done in {elapsed:.0f}s, final loss={final_loss:.1f}")

        # Add metabolite names
        for r in fold_results:
            r["metabolite_name"] = met_names[r["metabolite_idx"]]
            r["fold"] = fold_idx

        all_results.extend(fold_results)

        # Save per-fold results
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(out_dir / f"fold_{fold_idx}_results.csv", index=False)
        if len(fold_df) > 0:
            pos_rho = (fold_df["spearman_rho"] > 0).mean()
            print(f"  Median Spearman rho: {fold_df['spearman_rho'].median():.3f}")
            print(f"  % positive rho: {100 * pos_rho:.0f}%")

    # Aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(out_dir / "unitedmet_lomo_results.csv", index=False)

        print(f"\n{'=' * 70}")
        print(f"OVERALL RESULTS ({len(results_df)} metabolites)")
        print(f"{'=' * 70}")
        print(f"  Median Spearman rho: {results_df['spearman_rho'].median():.3f}")
        print(f"  Mean Spearman rho:   {results_df['spearman_rho'].mean():.3f}")
        print(f"  % positive rho:      {100 * (results_df['spearman_rho'] > 0).mean():.0f}%")
        print(f"  % significant (p<0.05): {100 * (results_df['spearman_p'] < 0.05).mean():.0f}%")
        print(f"  Median Pearson r:    {results_df['pearson_r'].median():.3f}")
        print(f"  % positive Pearson:  {100 * (results_df['pearson_r'] > 0).mean():.0f}%")

        # Top and bottom metabolites
        print(f"\nTop 10 metabolites by Spearman rho:")
        for _, r in results_df.nlargest(10, "spearman_rho").iterrows():
            sig = "***" if r["spearman_p"] < 0.001 else "**" if r["spearman_p"] < 0.01 else "*" if r["spearman_p"] < 0.05 else ""
            print(f"  {r['metabolite_name']:<50} rho={r['spearman_rho']:+.3f}{sig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
