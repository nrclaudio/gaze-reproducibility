#!/usr/bin/env python3
"""
Generate the mechanism figure for the manuscript.
Three-panel figure:
  (A) Chemical support vs R² — shows SMILES prerequisite (no SMILES = failure)
  (B) Target IQR vs R² (resolved SMILES only) — shows dominant predictor
  (C) Carbon count vs R² (resolved SMILES only) — shows molecular size effect
"""

import csv
import sys
import numpy as np
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
adv_csv = project_root / "lomo_analysis/advanced_mechanisms/REAL_relational_full_L3_full_1013911/advanced_metrics.csv"
mech_csv = project_root / "lomo_analysis/mechanisms/REAL_relational_full_L3_full_1013911/prediction_success_factors.csv"

adv = {}
with open(adv_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["met_idx"])
        adv[idx] = {
            "R2": float(row["R2"]),
            "True_ID": row["True_ID"],
            "Chemical_Support": float(row["Chemical_Support"]),
        }

with open(mech_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["metabolite_idx"])
        if idx in adv:
            adv[idx]["Carbon_Count"] = int(row["Carbon_Count"])

met_indices = sorted(adv.keys())
n = len(met_indices)
r2 = np.array([adv[i]["R2"] for i in met_indices])
cs = np.array([adv[i]["Chemical_Support"] for i in met_indices])
cc = np.array([adv[i].get("Carbon_Count", 0) for i in met_indices])
true_ids = [adv[i]["True_ID"] for i in met_indices]
resolved = cs > 0

# Load target IQR from MuData
import mudata as mu
mdata = mu.read(str(project_root / "data/raw/mudata_correct_HMDB_new_no_negatives.h5mu"))
met_adata = mdata.mod["metabolomics"]
met_var_names = list(met_adata.var_names)
met_X = met_adata.X
if hasattr(met_X, "toarray"):
    met_X = met_X.toarray()

hmdb_col = met_adata.var["HMDB"] if "HMDB" in met_adata.var.columns else None
hmdb_to_colidx = {}
name_to_colidx = {}
for k in range(len(met_var_names)):
    name_to_colidx[met_var_names[k].lower()] = k
    if hmdb_col is not None:
        hid = str(hmdb_col.iloc[k])
        if hid and hid != "nan":
            hmdb_to_colidx[hid] = k
            if hid.startswith("HMDB") and len(hid) < 13:
                normalized = "HMDB" + hid[4:].zfill(7)
                hmdb_to_colidx[normalized] = k

target_iqr = np.full(n, np.nan)
for j, tid in enumerate(true_ids):
    col_idx = hmdb_to_colidx.get(tid)
    if col_idx is None and tid.startswith("HMDB"):
        col_idx = hmdb_to_colidx.get("HMDB" + tid[4:].zfill(7))
    if col_idx is None:
        tid_lower = tid.lower().replace("_", " ").replace("-", " ")
        for vn_lower, k in name_to_colidx.items():
            if tid_lower in vn_lower.replace("-", " ") or vn_lower.replace("-", " ") in tid_lower:
                col_idx = k
                break
    if col_idx is not None:
        vals = met_X[:, col_idx]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 10:
            target_iqr[j] = np.percentile(vals, 75) - np.percentile(vals, 25)

print(f"Matched target IQR for {np.isfinite(target_iqr).sum()}/{n} metabolites")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
clip = -2.5
r2_clip = np.clip(r2, clip, None)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: Chemical support vs R² (all 50 metabolites)
ax = axes[0]
no_smiles = ~resolved
ax.scatter(cs[no_smiles], r2_clip[no_smiles], color="#999999", edgecolor="black",
           linewidth=0.5, s=45, alpha=0.7, zorder=3, label=f"No SMILES (n={no_smiles.sum()})")
ax.scatter(cs[resolved], r2_clip[resolved], color="#2171b5", edgecolor="black",
           linewidth=0.5, s=45, alpha=0.7, zorder=3, label=f"Resolved (n={resolved.sum()})")
ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax.set_xlabel("Chemical support (Tanimoto)", fontsize=10)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("(A) Chemical support", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.2)
ax.set_ylim(clip - 0.2, 1.1)

# Panel B: Target IQR vs R² (resolved only)
ax = axes[1]
mask = resolved & np.isfinite(target_iqr)
ax.scatter(target_iqr[mask], r2_clip[mask], color="#2171b5", edgecolor="black",
           linewidth=0.5, s=45, alpha=0.7, zorder=3)
# Regression line
if mask.sum() >= 3:
    slope, intercept, rval, pval, stderr = stats.linregress(target_iqr[mask], r2[mask])
    x_fit = np.linspace(target_iqr[mask].min(), target_iqr[mask].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, np.clip(y_fit, clip, None), color="#2171b5", linewidth=1.5,
            linestyle="--", alpha=0.6)
    # Spearman annotation
    rs, ps = stats.spearmanr(target_iqr[mask], r2[mask])
    ax.text(0.05, 0.95, f"$\\rho$ = {rs:.2f}\np < {ps:.0e}" if ps < 0.001 else f"$\\rho$ = {rs:.2f}\np = {ps:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax.set_xlabel("Target IQR (concentration variability)", fontsize=10)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("(B) Target variability", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.set_ylim(clip - 0.2, 1.1)

# Panel C: Carbon count vs R² (resolved only)
ax = axes[2]
mask_cc = resolved & (cc > 0)
ax.scatter(cc[mask_cc], r2_clip[mask_cc], color="#2171b5", edgecolor="black",
           linewidth=0.5, s=45, alpha=0.7, zorder=3)
if mask_cc.sum() >= 3:
    slope, intercept, rval, pval, stderr = stats.linregress(cc[mask_cc], r2[mask_cc])
    x_fit = np.linspace(cc[mask_cc].min(), cc[mask_cc].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, np.clip(y_fit, clip, None), color="#2171b5", linewidth=1.5,
            linestyle="--", alpha=0.6)
    rs, ps = stats.spearmanr(cc[mask_cc], r2[mask_cc])
    ax.text(0.95, 0.95, f"$\\rho$ = {rs:.2f}\np = {ps:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax.set_xlabel("Carbon count", fontsize=10)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("(C) Molecular size", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.set_ylim(clip - 0.2, 1.1)

fig.tight_layout()
out_dir = project_root / "lomo_analysis"
fig.savefig(out_dir / "mechanism_figure.pdf", bbox_inches="tight")
fig.savefig(out_dir / "mechanism_figure.png", dpi=300, bbox_inches="tight")
print(f"Saved {out_dir / 'mechanism_figure.pdf'}")
plt.close(fig)
