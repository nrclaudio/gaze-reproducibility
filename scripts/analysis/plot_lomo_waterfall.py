#!/usr/bin/env python3
"""Generate a clean LOMO architectural waterfall figure for the manuscript."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from lomo_comparison.csv — sorted by median R² (best first)
architectures = [
    ("GAZE + physics",                    -0.34,  30),
    ("GAZE",                              -0.35,  24),
    ("GAZE + multi-hop readout",          -0.41,  22),
    ("GAZE + multi-hop readout + GPE",    -0.41,  22),
    ("+ residual connections",            -0.52,  22),
    ("+ bipartite msg. passing + MCR",    -0.53,  18),
    ("GAZE + physics + multi-hop + GPE",  -0.60,  18),
    ("Bipartite GNN + GPE (no MCR)",      -0.63,  14),
    ("GATv2 (homogeneous baseline)",      -0.72,  16),
    ("+ cross-modal attention",           -0.82,  20),
]

labels   = [a[0] for a in architectures]
medians  = [a[1] for a in architectures]
pct_pos  = [a[2] for a in architectures]

fig, ax = plt.subplots(figsize=(8, 5.5))

y_pos = np.arange(len(labels))[::-1]
bars = ax.barh(y_pos, medians, color="#7fb3d8", edgecolor="black", linewidth=0.5, height=0.7)

# Annotate with % positive R²
for yp, med, pct in zip(y_pos, medians, pct_pos):
    ax.text(med - 0.02, yp, f"{med:.2f}", va="center", ha="right",
            fontsize=8, fontweight="bold", color="white")
    ax.text(0.01, yp, f"{pct}% pos.", va="center", ha="left",
            fontsize=7.5, color="#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("LOMO Median R²", fontsize=11)
ax.set_title("Zero-Shot Performance: Architectural Waterfall", fontsize=12, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_xlim(-0.95, 0.12)
ax.grid(axis="x", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("lomo_analysis/lomo_waterfall.png", dpi=300, bbox_inches="tight")
fig.savefig("lomo_analysis/lomo_waterfall.pdf", bbox_inches="tight")
print("Saved lomo_analysis/lomo_waterfall.png and .pdf")
plt.close(fig)
