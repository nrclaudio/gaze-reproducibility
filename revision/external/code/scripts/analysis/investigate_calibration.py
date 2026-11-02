#!/usr/bin/env python3
"""
Investigate whether negative-R² metabolites actually correlate with truth.

For each held-out metabolite, computes:
  - R² (as reported)
  - Pearson correlation between predictions and targets
  - Mean bias (predicted mean - true mean)
  - Scale ratio (predicted std / true std)
  - "Calibrated R²": R² after z-scoring predictions to match target mean/std

If many metabolites have positive correlation but negative R², the model
captures the signal but at the wrong scale — a fixable calibration problem.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

project_root = Path(__file__).resolve().parent.parent.parent

# Load all available LOMO runs with raw predictions
lomo_dirs = {
    "MCR + flux + mass_bal (≈ GAZE+physics)": "lomo_results/mcr_flux_mass_balance/baseline_mcr_20250929_064841",
    "MCR + flux (≈ GAZE+flux)": "lomo_results/mcr_flux/baseline_mcr_20250928_224206",
    "MCR + mass_bal": "lomo_results/mcr_mass_balance/baseline_mcr_20250929_205625",
    "MCR only (≈ GAZE)": "lomo_results/mcr/baseline_mcr_20250924_190535",
    "Baseline (no MCR)": "lomo_results/baseline_layers3/architecture_baseline_20250924_190535",
    "Cross-modal attention": "lomo_results/cross_modal/architecture_cross_modal_attention_20250924_190535",
    "Residual connections": "lomo_results/residual_layers4/architecture_residual_connections_20250924_190535",
}

for arch_name, rel_dir in lomo_dirs.items():
    json_path = project_root / rel_dir / "lomo_cross_validation/lomo_intermediate.json"
    if not json_path.exists():
        continue

    with open(json_path) as f:
        folds = json.load(f)

    print(f"\n{'='*80}")
    print(f"ARCHITECTURE: {arch_name}")
    print(f"  {len(folds)} metabolites, file: {rel_dir}")
    print(f"{'='*80}")

    results = []
    for fold in folds:
        preds = np.array(fold["predictions"])
        tgts = np.array(fold["targets"])
        met_id = fold["metabolite_id"]
        r2_reported = fold["r2"]

        # Compute correlation
        if len(preds) < 5 or np.std(tgts) < 1e-10 or np.std(preds) < 1e-10:
            continue

        pearson_r, pearson_p = stats.pearsonr(preds, tgts)
        spearman_r, spearman_p = stats.spearmanr(preds, tgts)

        # Bias and scale
        mean_bias = preds.mean() - tgts.mean()
        scale_ratio = preds.std() / tgts.std() if tgts.std() > 0 else np.nan

        # Calibrated R²: z-score predictions to match target distribution
        preds_cal = (preds - preds.mean()) / preds.std() * tgts.std() + tgts.mean()
        ss_res_cal = np.sum((tgts - preds_cal) ** 2)
        ss_tot = np.sum((tgts - tgts.mean()) ** 2)
        r2_calibrated = 1 - ss_res_cal / ss_tot if ss_tot > 0 else np.nan

        results.append({
            "met_id": met_id,
            "r2": r2_reported,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "mean_bias": mean_bias,
            "scale_ratio": scale_ratio,
            "r2_calibrated": r2_calibrated,
            "target_std": tgts.std(),
            "n_samples": len(tgts),
        })

    if not results:
        continue

    # Summary
    r2s = np.array([r["r2"] for r in results])
    corrs = np.array([r["pearson_r"] for r in results])
    r2_cals = np.array([r["r2_calibrated"] for r in results])
    biases = np.array([r["mean_bias"] for r in results])

    n_neg_r2 = (r2s < 0).sum()
    n_pos_corr = (corrs > 0).sum()
    n_neg_r2_pos_corr = ((r2s < 0) & (corrs > 0)).sum()
    n_neg_r2_sig_corr = ((r2s < 0) & (corrs > 0) & (np.array([r["pearson_p"] for r in results]) < 0.05)).sum()

    print(f"\n  SUMMARY:")
    print(f"  Total metabolites:                {len(results)}")
    print(f"  Negative R²:                      {n_neg_r2}/{len(results)}")
    print(f"  Positive correlation:              {n_pos_corr}/{len(results)}")
    print(f"  Negative R² BUT positive corr:     {n_neg_r2_pos_corr}/{n_neg_r2}")
    print(f"  Negative R² BUT significant corr:  {n_neg_r2_sig_corr}/{n_neg_r2} (p < 0.05)")
    print(f"")
    print(f"  Median R² (raw):        {np.median(r2s):>+.4f}")
    print(f"  Median R² (calibrated): {np.median(r2_cals):>+.4f}")
    print(f"  Median Pearson r:       {np.median(corrs):>+.4f}")
    print(f"  Median |bias|:          {np.median(np.abs(biases)):>.4f}")
    print(f"  % positive R² (raw):    {100*(r2s > 0).mean():.0f}%")
    print(f"  % positive R² (cal):    {100*(r2_cals > 0).mean():.0f}%")

    # Per-metabolite detail
    print(f"\n  {'Met_ID':<20} {'R²':>8} {'R²_cal':>8} {'Pearson':>8} {'p-val':>8} {'Bias':>8} {'Scale':>6} {'Tgt_std':>8}")
    print(f"  {'-'*78}")
    for r in sorted(results, key=lambda x: x["r2"]):
        sig = "*" if r["pearson_p"] < 0.05 else " "
        sig = "**" if r["pearson_p"] < 0.01 else sig
        sig = "***" if r["pearson_p"] < 0.001 else sig
        print(f"  {r['met_id']:<20} {r['r2']:>+8.3f} {r['r2_calibrated']:>+8.3f} {r['pearson_r']:>+7.3f}{sig} {r['pearson_p']:>8.4f} {r['mean_bias']:>+8.2f} {r['scale_ratio']:>6.2f} {r['target_std']:>8.3f}")

# Cross-architecture comparison for the physics model
print(f"\n\n{'='*80}")
print("CALIBRATION ANALYSIS: Can post-hoc scaling fix zero-shot predictions?")
print(f"{'='*80}")

json_path = project_root / "lomo_results/mcr_flux_mass_balance/baseline_mcr_20250929_064841/lomo_cross_validation/lomo_intermediate.json"
with open(json_path) as f:
    folds = json.load(f)

print(f"\nUsing MCR + flux + mass_balance ({len(folds)} metabolites)")
print(f"\nKey question: Among metabolites with NEGATIVE R², how many have")
print(f"POSITIVE correlation? If many, the model captures signal but at wrong scale.\n")

neg_r2_pos_corr = []
neg_r2_neg_corr = []

for fold in folds:
    preds = np.array(fold["predictions"])
    tgts = np.array(fold["targets"])
    if len(preds) < 5 or np.std(tgts) < 1e-10 or np.std(preds) < 1e-10:
        continue

    r2 = fold["r2"]
    pr, pp = stats.pearsonr(preds, tgts)

    if r2 < 0 and pr > 0:
        neg_r2_pos_corr.append((fold["metabolite_id"], r2, pr, pp))
    elif r2 < 0 and pr <= 0:
        neg_r2_neg_corr.append((fold["metabolite_id"], r2, pr, pp))

print(f"Negative R² with POSITIVE correlation ({len(neg_r2_pos_corr)}):")
print(f"  → These metabolites have a calibration problem (fixable)")
for mid, r2, pr, pp in sorted(neg_r2_pos_corr, key=lambda x: -x[2]):
    sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
    print(f"    {mid:<20} R²={r2:>+.3f}  r={pr:>+.3f}{sig}")

print(f"\nNegative R² with NEGATIVE correlation ({len(neg_r2_neg_corr)}):")
print(f"  → These metabolites are genuinely unpredictable")
for mid, r2, pr, pp in sorted(neg_r2_neg_corr, key=lambda x: x[2]):
    sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
    print(f"    {mid:<20} R²={r2:>+.3f}  r={pr:>+.3f}{sig}")

print("\nDone.")
