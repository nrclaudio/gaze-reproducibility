#!/usr/bin/env python3
"""
Clean chemical support vs R² plot for the manuscript.
Separates metabolites with no resolved SMILES (Chemical_Support=0)
from those with valid Tanimoto similarity.
"""

import ast
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

csv_path = Path("lomo_analysis/advanced_mechanisms/REAL_relational_full_L3_full_1013911/advanced_metrics.csv")
output_dir = Path("lomo_analysis")

# Parse CSV (handle quoted fields with commas)
import csv
r2s, css = [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            r2s.append(float(row["R2"]))
            css.append(float(row["Chemical_Support"]))
        except (ValueError, KeyError):
            continue

r2s = np.array(r2s)
css = np.array(css)

# Split into resolved (CS > 0) and unresolved (CS == 0)
resolved = css > 0
n_unresolved = (~resolved).sum()
n_resolved = resolved.sum()

fig = plt.figure(figsize=(8, 5.5))
gs = GridSpec(1, 2, width_ratios=[1, 4], wspace=0.05)

# Left panel: strip plot for unresolved (CS = 0)
ax_left = fig.add_subplot(gs[0])
ax_left.scatter(
    np.random.uniform(-0.15, 0.15, n_unresolved),
    np.clip(r2s[~resolved], -2.5, None),
    color="#999999", edgecolor="black", linewidth=0.5, s=40, alpha=0.7, zorder=3
)
ax_left.set_xlim(-0.4, 0.4)
ax_left.set_ylim(-2.7, 1.1)
ax_left.set_xticks([0])
ax_left.set_xticklabels(["No\nSMILES"], fontsize=8)
ax_left.set_ylabel("R²", fontsize=11)
ax_left.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax_left.text(0, 1.0, f"n={n_unresolved}", ha="center", fontsize=8, color="#666666")
ax_left.spines["right"].set_visible(False)
ax_left.grid(axis="y", alpha=0.2)

# Right panel: scatter for resolved
ax_right = fig.add_subplot(gs[1], sharey=ax_left)
ax_right.scatter(
    css[resolved], np.clip(r2s[resolved], -2.5, None),
    color="#2171b5", edgecolor="black", linewidth=0.5, s=40, alpha=0.7, zorder=3
)

# Linear fit for resolved only
mask = resolved & np.isfinite(r2s)
if mask.sum() >= 2:
    coeffs = np.polyfit(css[mask], r2s[mask], 1)
    fit_x = np.linspace(css[resolved].min(), css[resolved].max(), 100)
    fit_y = np.polyval(coeffs, fit_x)
    ax_right.plot(fit_x, np.clip(fit_y, -2.5, None), color="#2171b5", linewidth=1.5,
                  linestyle="--", alpha=0.6)

ax_right.set_xlim(0.15, 1.0)
ax_right.set_xlabel("Chemical support (Tanimoto similarity)", fontsize=11)
ax_right.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax_right.text(0.95, 1.0, f"n={n_resolved}", ha="right", fontsize=8, color="#666666")
plt.setp(ax_right.get_yticklabels(), visible=False)
ax_right.spines["left"].set_visible(False)
ax_right.grid(True, alpha=0.2)

fig.suptitle("Chemical Support vs LOMO R² (GAZE + physics)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(output_dir / "chemical_support.pdf", bbox_inches="tight")
fig.savefig(output_dir / "chemical_support.png", dpi=300, bbox_inches="tight")
print(f"Saved (n_unresolved={n_unresolved}, n_resolved={n_resolved})")
plt.close(fig)
