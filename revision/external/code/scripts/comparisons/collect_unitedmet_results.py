#!/usr/bin/env python3
"""
Collect UnitedMet fold results and compare against MEBOCOST, scCellFie, and GAZE.
Run this after all 10 folds complete.

Usage:
  python scripts/comparisons/collect_unitedmet_results.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

project_root = Path(__file__).resolve().parent.parent.parent
results_dir = project_root / "comparison_results" / "unitedmet_lomo"

# Collect all fold results
dfs = []
for f in sorted(results_dir.glob("fold_*_results.csv")):
    df = pd.read_csv(f)
    dfs.append(df)
    print(f"  {f.name}: {len(df)} metabolites, median rho={df['spearman_rho'].median():.3f}")

if not dfs:
    print("No fold results found. Run the UnitedMet comparison first.")
    exit(1)

results = pd.concat(dfs, ignore_index=True)
results.to_csv(results_dir / "unitedmet_lomo_results.csv", index=False)

print(f"\n{'=' * 70}")
print(f"UnitedMet RESULTS ({len(results)} metabolites)")
print(f"{'=' * 70}")
print(f"  Median Spearman rho:    {results['spearman_rho'].median():.3f}")
print(f"  Mean Spearman rho:      {results['spearman_rho'].mean():.3f}")
print(f"  % positive rho:         {100 * (results['spearman_rho'] > 0).mean():.0f}%")
print(f"  % sig (p<0.05):         {100 * (results['spearman_p'] < 0.05).mean():.0f}%")
print(f"  Median Pearson r:       {results['pearson_r'].dropna().median():.3f}")

# Load MEBOCOST and scCellFie for comparison
comp_dir = project_root / "comparison_results"
print(f"\n{'=' * 70}")
print(f"COMPARISON")
print(f"{'=' * 70}")

if (comp_dir / "mebocost_vs_camp.csv").exists():
    mebo = pd.read_csv(comp_dir / "mebocost_vs_camp.csv")
    print(f"\nMEBOCOST ({len(mebo)} metabolites):")
    print(f"  Median Pearson r:  {mebo['pearson_r'].median():.3f}")
    print(f"  % positive:        {100 * (mebo['pearson_r'] > 0).mean():.0f}%")

if (comp_dir / "sccellfie_vs_camp.csv").exists():
    scf = pd.read_csv(comp_dir / "sccellfie_vs_camp.csv")
    print(f"\nscCellFie ({len(scf)} metabolites):")
    print(f"  Median Pearson r:  {scf['pearson_r'].median():.3f}")
    print(f"  % positive:        {100 * (scf['pearson_r'] > 0).mean():.0f}%")

print(f"\nUnitedMet ({len(results)} metabolites):")
print(f"  Median Spearman rho: {results['spearman_rho'].median():.3f}")
print(f"  Median Pearson r:    {results['pearson_r'].dropna().median():.3f}")
print(f"  % positive rho:      {100 * (results['spearman_rho'] > 0).mean():.0f}%")

# Find overlapping metabolites between UnitedMet and MEBOCOST
if (comp_dir / "mebocost_vs_camp.csv").exists():
    mebo = pd.read_csv(comp_dir / "mebocost_vs_camp.csv")
    mebo_by_name = {r["camp_name"].lower(): r for _, r in mebo.iterrows()}
    um_by_name = {r["metabolite_name"].lower(): r for _, r in results.iterrows()}
    overlap = set(mebo_by_name.keys()) & set(um_by_name.keys())

    if overlap:
        print(f"\n{'=' * 70}")
        print(f"UnitedMet vs MEBOCOST overlap: {len(overlap)} metabolites")
        print(f"{'=' * 70}")
        rows = []
        for name in sorted(overlap):
            m = mebo_by_name[name]
            u = um_by_name[name]
            rows.append({
                "metabolite": name,
                "mebocost_r": m["pearson_r"],
                "unitedmet_rho": u["spearman_rho"],
                "unitedmet_r": u["pearson_r"],
            })
        ov_df = pd.DataFrame(rows)
        print(f"  MEBOCOST median |r|:    {ov_df['mebocost_r'].abs().median():.3f}")
        print(f"  UnitedMet median |rho|: {ov_df['unitedmet_rho'].abs().median():.3f}")
        um_wins = (ov_df["unitedmet_rho"].abs() > ov_df["mebocost_r"].abs()).sum()
        print(f"  UnitedMet wins: {um_wins}/{len(ov_df)}")

print("\nDone.")
