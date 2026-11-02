#!/usr/bin/env python3
"""
Investigate what makes a metabolite predictable in zero-shot (LOMO) mode.

Merges per-metabolite features from multiple sources:
  - R² from LOMO evaluation (best model: GAZE + physics)
  - Chemical_Support, Enzyme_Signal (from advanced_metrics.csv)
  - Degree, Carbon_Count (from prediction_success_factors.csv)
  - Target variance across cell lines (from MuData)
  - Cross-architecture consistency (from lomo_comparison.csv)

Computes correlations and identifies which features actually predict R².
"""

import os
import sys
import csv
import ast
import numpy as np
from pathlib import Path
from scipy import stats

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# 1. Load LOMO per-metabolite data (GAZE + physics)
# ---------------------------------------------------------------------------
base = project_root / "lomo_analysis"
adv_csv = base / "advanced_mechanisms/REAL_relational_full_L3_full_1013911/advanced_metrics.csv"
mech_csv = base / "mechanisms/REAL_relational_full_L3_full_1013911/prediction_success_factors.csv"

# Parse advanced metrics
adv = {}
with open(adv_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["met_idx"])
        adv[idx] = {
            "R2": float(row["R2"]),
            "True_ID": row["True_ID"],
            "Chemical_Support": float(row["Chemical_Support"]),
            "Enzyme_Signal": float(row["Enzyme_Signal"]) if row["Enzyme_Signal"] else np.nan,
        }

# Parse mechanism features
with open(mech_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["metabolite_idx"])
        if idx in adv:
            adv[idx]["Degree"] = int(row["Degree"])
            adv[idx]["Carbon_Count"] = int(row["Carbon_Count"])

# Build arrays
met_indices = sorted(adv.keys())
n = len(met_indices)
r2 = np.array([adv[i]["R2"] for i in met_indices])
chem_support = np.array([adv[i]["Chemical_Support"] for i in met_indices])
enzyme_signal = np.array([adv[i].get("Enzyme_Signal", np.nan) for i in met_indices])
degree = np.array([adv[i].get("Degree", np.nan) for i in met_indices])
carbon_count = np.array([adv[i].get("Carbon_Count", np.nan) for i in met_indices])
has_smiles = (chem_support > 0).astype(float)
true_ids = [adv[i]["True_ID"] for i in met_indices]

print(f"Loaded {n} metabolites from LOMO (GAZE + physics)")
print(f"  R² range: [{r2.min():.2f}, {r2.max():.2f}], median={np.median(r2):.3f}")
print(f"  Positive R²: {(r2 > 0).sum()}/{n} ({100*(r2>0).mean():.0f}%)")
print()

# ---------------------------------------------------------------------------
# 2. Load target variance from MuData
# ---------------------------------------------------------------------------
try:
    import mudata as mu
    mudata_path = project_root / "data/raw/mudata_correct_HMDB_new_no_negatives.h5mu"
    mdata = mu.read(str(mudata_path))

    # Get metabolite expression matrix
    met_adata = mdata.mod["metabolomics"]
    met_var_names = list(met_adata.var_names)
    met_X = met_adata.X
    if hasattr(met_X, "toarray"):
        met_X = met_X.toarray()

    # Build HMDB -> column index lookup from MuData var metadata
    hmdb_col = met_adata.var["HMDB"] if "HMDB" in met_adata.var.columns else None
    hmdb_tz_col = met_adata.var["HMDB_trailing_zeros"] if "HMDB_trailing_zeros" in met_adata.var.columns else None

    hmdb_to_colidx = {}
    name_to_colidx = {}
    for k in range(len(met_var_names)):
        name_to_colidx[met_var_names[k].lower()] = k
        if hmdb_col is not None:
            hid = str(hmdb_col.iloc[k])
            if hid and hid != "nan":
                hmdb_to_colidx[hid] = k
                # Also normalize to 7-digit format (HMDB0000XXX)
                if hid.startswith("HMDB") and len(hid) < 13:
                    digits = hid[4:]
                    normalized = "HMDB" + digits.zfill(7)
                    hmdb_to_colidx[normalized] = k
        if hmdb_tz_col is not None:
            hid_tz = str(hmdb_tz_col.iloc[k])
            if hid_tz and hid_tz != "nan":
                hmdb_to_colidx[hid_tz] = k

    print(f"  Built HMDB lookup: {len(hmdb_to_colidx)} entries, name lookup: {len(name_to_colidx)} entries")

    # Map True_ID to metabolomics var_names
    target_var = np.full(n, np.nan)
    target_cv = np.full(n, np.nan)
    target_mean = np.full(n, np.nan)
    target_iqr = np.full(n, np.nan)
    n_samples = np.full(n, np.nan)
    matched = 0

    for j, tid in enumerate(true_ids):
        col_idx = None

        # Try HMDB ID match
        if tid in hmdb_to_colidx:
            col_idx = hmdb_to_colidx[tid]
        else:
            # Normalize and try again
            if tid.startswith("HMDB"):
                digits = tid[4:]
                normalized = "HMDB" + digits.zfill(7)
                if normalized in hmdb_to_colidx:
                    col_idx = hmdb_to_colidx[normalized]

        # Try name match (for non-HMDB True_IDs like "creatine-phosphate")
        if col_idx is None:
            tid_lower = tid.lower().replace("_", " ").replace("-", " ")
            for vn_lower, k in name_to_colidx.items():
                vn_clean = vn_lower.replace("-", " ")
                if tid_lower in vn_clean or vn_clean in tid_lower:
                    col_idx = k
                    break

        if col_idx is not None:
            vals = met_X[:, col_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 10:
                target_var[j] = np.var(vals)
                target_mean[j] = np.mean(vals)
                n_samples[j] = len(vals)
                if target_mean[j] != 0:
                    target_cv[j] = np.std(vals) / abs(target_mean[j])
                target_iqr[j] = np.percentile(vals, 75) - np.percentile(vals, 25)
                matched += 1

    print(f"  Matched {matched}/{n} metabolites to expression data")
    if matched > 0:
        print(f"  Target variance range: [{np.nanmin(target_var):.4f}, {np.nanmax(target_var):.4f}]")
        print(f"  Target CV range: [{np.nanmin(target_cv):.4f}, {np.nanmax(target_cv):.4f}]")
    print()

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Could not load MuData: {e}")
    target_var = np.full(n, np.nan)
    target_cv = np.full(n, np.nan)
    target_mean = np.full(n, np.nan)
    target_iqr = np.full(n, np.nan)

# ---------------------------------------------------------------------------
# 3. Cross-architecture consistency
# ---------------------------------------------------------------------------
comp_csv = base / "lomo_comparison.csv"
lines = comp_csv.read_text().strip().split("\n")
header = lines[0].split(",")
name_idx = header.index("config_name")

arch_r2s = {}
for line in lines[1:]:
    bracket_start = line.index("[")
    prefix = line[:bracket_start]
    r2_str = line[bracket_start:].strip().strip('"')
    fields = prefix.rstrip(",").split(",")
    config = fields[name_idx]
    arch_r2s[config] = np.array(ast.literal_eval(r2_str))

# For each metabolite, compute variance of R² across architectures
# (all 10 architectures have same 50 metabolites in same order)
all_archs = np.array(list(arch_r2s.values()))  # (10, 50)
cross_arch_mean = all_archs.mean(axis=0)
cross_arch_std = all_archs.std(axis=0)
# How consistently positive is this metabolite across architectures?
cross_arch_pos_frac = (all_archs > 0).mean(axis=0)

print(f"Cross-architecture consistency ({all_archs.shape[0]} architectures, {all_archs.shape[1]} metabolites):")
print(f"  Mean cross-arch positive fraction: {cross_arch_pos_frac.mean():.2f}")
always_positive = (cross_arch_pos_frac == 1.0).sum()
always_negative = (cross_arch_pos_frac == 0.0).sum()
print(f"  Always positive (all 10 archs): {always_positive}")
print(f"  Always negative (all 10 archs): {always_negative}")
print()

# ---------------------------------------------------------------------------
# 4. Correlation analysis
# ---------------------------------------------------------------------------
print("=" * 70)
print("CORRELATION ANALYSIS: What predicts LOMO R²?")
print("=" * 70)

features = {
    "Chemical_Support": chem_support,
    "Has_SMILES (binary)": has_smiles,
    "Degree": degree,
    "Carbon_Count": carbon_count,
    "Enzyme_Signal": enzyme_signal,
    "log(Enzyme_Signal)": np.log1p(enzyme_signal),
    "Target_Variance": target_var,
    "Target_CV": target_cv,
    "Target_IQR": target_iqr,
    "Cross_Arch_Mean_R2": cross_arch_mean,
    "Cross_Arch_Pos_Frac": cross_arch_pos_frac,
}

print(f"\n{'Feature':<25} {'r (Pearson)':>12} {'p-value':>10} {'r (Spearman)':>14} {'p-value':>10} {'n':>5}")
print("-" * 80)

for name, vals in features.items():
    if name.startswith("Cross_Arch"):
        continue  # These are circular
    mask = np.isfinite(vals) & np.isfinite(r2)
    nn = mask.sum()
    if nn < 5:
        print(f"{name:<25} {'n/a':>12} {'':>10} {'n/a':>14} {'':>10} {nn:>5}")
        continue
    rp, pp = stats.pearsonr(vals[mask], r2[mask])
    rs, ps = stats.spearmanr(vals[mask], r2[mask])
    sig_p = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
    sig_s = "***" if ps < 0.001 else "**" if ps < 0.01 else "*" if ps < 0.05 else ""
    print(f"{name:<25} {rp:>+10.3f}{sig_p:<2} {pp:>10.4f} {rs:>+12.3f}{sig_s:<2} {ps:>10.4f} {nn:>5}")

# ---------------------------------------------------------------------------
# 5. Subgroup analysis: resolved SMILES only
# ---------------------------------------------------------------------------
resolved = chem_support > 0
n_res = resolved.sum()
print(f"\n{'='*70}")
print(f"RESOLVED SMILES ONLY ({n_res} metabolites)")
print(f"{'='*70}")

print(f"\n{'Feature':<25} {'r (Pearson)':>12} {'p-value':>10} {'r (Spearman)':>14} {'p-value':>10} {'n':>5}")
print("-" * 80)

for name, vals in features.items():
    if name.startswith("Cross_Arch"):
        continue
    mask = resolved & np.isfinite(vals) & np.isfinite(r2)
    nn = mask.sum()
    if nn < 5:
        print(f"{name:<25} {'n/a':>12} {'':>10} {'n/a':>14} {'':>10} {nn:>5}")
        continue
    rp, pp = stats.pearsonr(vals[mask], r2[mask])
    rs, ps = stats.spearmanr(vals[mask], r2[mask])
    sig_p = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
    sig_s = "***" if ps < 0.001 else "**" if ps < 0.01 else "*" if ps < 0.05 else ""
    print(f"{name:<25} {rp:>+10.3f}{sig_p:<2} {pp:>10.4f} {rs:>+12.3f}{sig_s:<2} {ps:>10.4f} {nn:>5}")

# ---------------------------------------------------------------------------
# 6. Profile the best and worst metabolites
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("TOP 10 BEST zero-shot metabolites (by R²)")
print(f"{'='*70}")
order = np.argsort(-r2)
print(f"{'Rank':<5} {'R²':>7} {'True_ID':<35} {'CS':>5} {'Deg':>4} {'CC':>4} {'TVar':>10} {'TCV':>8}")
print("-" * 85)
for rank, j in enumerate(order[:10], 1):
    tv = f"{target_var[j]:.4f}" if np.isfinite(target_var[j]) else "n/a"
    tc = f"{target_cv[j]:.3f}" if np.isfinite(target_cv[j]) else "n/a"
    print(f"{rank:<5} {r2[j]:>+7.3f} {true_ids[j]:<35} {chem_support[j]:>5.2f} {degree[j]:>4.0f} {carbon_count[j]:>4.0f} {tv:>10} {tc:>8}")

print(f"\n{'='*70}")
print("TOP 10 WORST zero-shot metabolites (by R²)")
print(f"{'='*70}")
print(f"{'Rank':<5} {'R²':>7} {'True_ID':<35} {'CS':>5} {'Deg':>4} {'CC':>4} {'TVar':>10} {'TCV':>8}")
print("-" * 85)
for rank, j in enumerate(order[-10:][::-1], 1):
    tv = f"{target_var[j]:.4f}" if np.isfinite(target_var[j]) else "n/a"
    tc = f"{target_cv[j]:.3f}" if np.isfinite(target_cv[j]) else "n/a"
    print(f"{rank:<5} {r2[j]:>+7.3f} {true_ids[j]:<35} {chem_support[j]:>5.2f} {degree[j]:>4.0f} {carbon_count[j]:>4.0f} {tv:>10} {tc:>8}")

# ---------------------------------------------------------------------------
# 7. Mann-Whitney U: positive R² vs negative R²
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("MANN-WHITNEY U: Positive R² group vs Negative R² group")
print(f"{'='*70}")

pos = r2 > 0
neg = r2 <= 0

for name, vals in features.items():
    if name.startswith("Cross_Arch"):
        continue
    mask = np.isfinite(vals)
    g_pos = vals[pos & mask]
    g_neg = vals[neg & mask]
    if len(g_pos) < 3 or len(g_neg) < 3:
        continue
    u_stat, p_val = stats.mannwhitneyu(g_pos, g_neg, alternative='two-sided')
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{name:<25} pos_median={np.median(g_pos):>8.3f}  neg_median={np.median(g_neg):>8.3f}  U={u_stat:>6.0f}  p={p_val:.4f} {sig}")

# ---------------------------------------------------------------------------
# 8. Which metabolites are ALWAYS predictable / unpredictable?
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("CROSS-ARCHITECTURE STABILITY")
print(f"{'='*70}")

print(f"\nMetabolites positive in ALL 10 architectures:")
for j in np.where(cross_arch_pos_frac == 1.0)[0]:
    print(f"  {true_ids[j]:<35} mean_R²={cross_arch_mean[j]:>+.3f}  CS={chem_support[j]:.2f}  Deg={degree[j]:.0f}")

print(f"\nMetabolites positive in 0 architectures:")
for j in np.where(cross_arch_pos_frac == 0.0)[0]:
    print(f"  {true_ids[j]:<35} mean_R²={cross_arch_mean[j]:>+.3f}  CS={chem_support[j]:.2f}  Deg={degree[j]:.0f}")

print(f"\nMetabolites with MIXED results (positive in 3-7 of 10 architectures):")
mixed = (cross_arch_pos_frac >= 0.3) & (cross_arch_pos_frac <= 0.7)
for j in np.where(mixed)[0]:
    print(f"  {true_ids[j]:<35} pos_frac={cross_arch_pos_frac[j]:.1f}  mean_R²={cross_arch_mean[j]:>+.3f}  CS={chem_support[j]:.2f}  Deg={degree[j]:.0f}")

print("\nDone.")
