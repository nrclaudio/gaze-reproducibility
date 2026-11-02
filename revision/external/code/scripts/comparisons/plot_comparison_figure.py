#!/usr/bin/env python3
"""
Generate comparison figure: GAZE vs scCellFie vs MEBOCOST vs UnitedMet.

Four panels:
  (A) Distribution of per-metabolite Spearman rho for each method (violin/box)
  (B) Head-to-head scatter: GAZE vs scCellFie for overlapping metabolites
  (C) Head-to-head scatter: GAZE vs MEBOCOST for overlapping LOMO metabolites
  (D) Head-to-head scatter: GAZE vs UnitedMet for overlapping metabolites
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
mebo = pd.read_csv(project_root / "comparison_results/mebocost_vs_camp.csv")
scf = pd.read_csv(project_root / "comparison_results/sccellfie_vs_camp.csv")

# Load UnitedMet results
um_dir = project_root / "comparison_results/unitedmed_lomo"
um_dfs = []
for f in sorted(um_dir.glob("fold_*_results.csv")):
    um_dfs.append(pd.read_csv(f))
um_df = pd.concat(um_dfs, ignore_index=True) if um_dfs else pd.DataFrame()

# Load GAZE LOMO predictions from individual fold files
lomo_dir = project_root / "lomo_results_best_with_predictions/REAL_relational_full_L3_full_1039043/lomo_cross_validation"
lomo_folds = []
for fold_file in sorted(lomo_dir.glob("fold_*_result.json")):
    with open(fold_file) as f:
        lomo_folds.append(json.load(f))

gaze_results = []
for fold in lomo_folds:
    preds = np.array(fold["predictions"])
    tgts = np.array(fold["targets"])
    if len(preds) >= 5 and np.std(tgts) > 1e-10 and np.std(preds) > 1e-10:
        sr, sp = stats.spearmanr(preds, tgts)
        pr, pp = stats.pearsonr(preds, tgts)
        gaze_results.append({
            "metabolite_id": fold["metabolite_id"],
            "spearman_r": sr,
            "pearson_r": pr,
            "r2": fold["r2"],
        })
gaze_df = pd.DataFrame(gaze_results)

# ---------------------------------------------------------------------------
# Find overlaps
# ---------------------------------------------------------------------------
mebo_by_name = {r["camp_name"].lower(): r for _, r in mebo.iterrows()}
scf_by_name = {r["camp_metabolite"].lower(): r for _, r in scf.iterrows()}

# GAZE ∩ MEBOCOST (LOMO)
import mudata as mu
mdata = mu.read(str(project_root / "data/raw/mudata_correct_HMDB_new_no_negatives.h5mu"))
met = mdata.mod["metabolomics"]
hmdb_col = met.var["HMDB"]
met_names = list(met.var_names)

# HMDB -> CAMP name
hmdb_to_camp = {}
for k in range(len(met_names)):
    hid = str(hmdb_col.iloc[k])
    if hid and hid != "nan":
        hmdb_to_camp[hid] = met_names[k].lower()
        if hid.startswith("HMDB") and len(hid) < 13:
            hmdb_to_camp["HMDB" + hid[4:].zfill(7)] = met_names[k].lower()

gaze_mebo_overlap = []
for _, g in gaze_df.iterrows():
    camp_name = hmdb_to_camp.get(g["metabolite_id"], "").lower()
    if camp_name in mebo_by_name:
        m = mebo_by_name[camp_name]
        if np.isfinite(m["spearman_r"]):
            gaze_mebo_overlap.append({
                "metabolite": camp_name,
                "gaze_rho": g["spearman_r"],
                "mebocost_rho": m["spearman_r"],
            })
gaze_mebo_df = pd.DataFrame(gaze_mebo_overlap)

# GAZE ∩ scCellFie
gaze_scf_overlap = []
for _, g in gaze_df.iterrows():
    camp_name = hmdb_to_camp.get(g["metabolite_id"], "").lower()
    if camp_name in scf_by_name:
        s = scf_by_name[camp_name]
        if np.isfinite(s["spearman_r"]):
            gaze_scf_overlap.append({
                "metabolite": camp_name,
                "gaze_rho": g["spearman_r"],
                "sccellfie_rho": s["spearman_r"],
            })
gaze_scf_df = pd.DataFrame(gaze_scf_overlap)

# GAZE ∩ UnitedMet
gaze_um_overlap = []
if len(um_df) > 0:
    um_by_name = {r["metabolite_name"].lower(): r for _, r in um_df.iterrows()}
    for _, g in gaze_df.iterrows():
        camp_name = hmdb_to_camp.get(g["metabolite_id"], "").lower()
        if camp_name in um_by_name:
            u = um_by_name[camp_name]
            if np.isfinite(u["spearman_rho"]):
                gaze_um_overlap.append({
                    "metabolite": camp_name,
                    "gaze_rho": g["spearman_r"],
                    "unitedmet_rho": u["spearman_rho"],
                })
gaze_um_df = pd.DataFrame(gaze_um_overlap)

print(f"MEBOCOST metabolites: {len(mebo)}")
print(f"scCellFie metabolites: {len(scf)}")
print(f"GAZE LOMO metabolites: {len(gaze_df)}")
print(f"UnitedMet metabolites: {len(um_df)}")
print(f"GAZE ∩ MEBOCOST: {len(gaze_mebo_df)}")
print(f"GAZE ∩ scCellFie: {len(gaze_scf_df)}")
print(f"GAZE ∩ UnitedMet: {len(gaze_um_df)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

# Panel A: Distribution of Spearman rho
ax = axes[0]
data_for_violin = [
    mebo["spearman_r"].dropna().values,
    um_df["spearman_rho"].values if len(um_df) > 0 else np.array([]),
    scf["spearman_r"].dropna().values,
    gaze_df["spearman_r"].values,
]
labels = [
    f"MEBOCOST\n(n={len(data_for_violin[0])})",
    f"UnitedMet\n(n={len(data_for_violin[1])})",
    f"scCellFie\n(n={len(data_for_violin[2])})",
    f"GAZE\n(n={len(data_for_violin[3])})",
]
colors = ["#999999", "#66c2a5", "#fb8072", "#2171b5"]

# Only plot non-empty violins
positions = [1, 2, 3, 4]
non_empty = [(d, p) for d, p in zip(data_for_violin, positions) if len(d) > 0]
violin_data = [d for d, _ in non_empty]
violin_pos = [p for _, p in non_empty]

parts = ax.violinplot(violin_data, positions=violin_pos, showmeans=False,
                       showmedians=False, showextrema=False)
color_idx = 0
for i, pc in enumerate(parts["bodies"]):
    while len(data_for_violin[color_idx]) == 0:
        color_idx += 1
    pc.set_facecolor(colors[color_idx])
    pc.set_edgecolor("black")
    pc.set_linewidth(0.5)
    pc.set_alpha(0.7)
    color_idx += 1

# Add box plots inside
bp = ax.boxplot(violin_data, positions=violin_pos, widths=0.15,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8))
color_idx = 0
for i, patch in enumerate(bp["boxes"]):
    while len(data_for_violin[color_idx]) == 0:
        color_idx += 1
    patch.set_facecolor(colors[color_idx])
    patch.set_alpha(0.9)
    color_idx += 1

# Add individual points (jittered) for GAZE (small n)
jitter = np.random.uniform(-0.08, 0.08, len(data_for_violin[3]))
ax.scatter(4 + jitter, data_for_violin[3], color="#2171b5", edgecolor="black",
           linewidth=0.5, s=25, alpha=0.7, zorder=4)

ax.axhline(0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Spearman $\\rho$ with measured concentration", fontsize=9)
ax.set_title("(A) Correlation distributions", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.2)

# Add median annotations
for i, d in enumerate(data_for_violin):
    if len(d) > 0:
        med = np.median(d)
        ax.text(i + 1, ax.get_ylim()[1] - 0.05, f"med={med:+.2f}",
                ha="center", fontsize=6, color=colors[i], fontweight="bold")

# Panel B: GAZE vs scCellFie (head-to-head)
ax = axes[1]
if len(gaze_scf_df) > 0:
    ax.scatter(gaze_scf_df["sccellfie_rho"], gaze_scf_df["gaze_rho"],
               color="#fb8072", edgecolor="black", linewidth=0.5, s=50, alpha=0.7, zorder=3)

    for _, row in gaze_scf_df.iterrows():
        name = row["metabolite"]
        if len(name) > 15:
            name = name[:15] + "..."
        ax.annotate(name, (row["sccellfie_rho"], row["gaze_rho"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")

    lim = max(abs(gaze_scf_df["sccellfie_rho"]).max(), abs(gaze_scf_df["gaze_rho"]).max()) + 0.15
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.3)

    gaze_wins = (gaze_scf_df["gaze_rho"].abs() > gaze_scf_df["sccellfie_rho"].abs()).sum()
    print(f"GAZE vs scCellFie: GAZE wins {gaze_wins}/{len(gaze_scf_df)}")

ax.set_xlabel("scCellFie Spearman $\\rho$", fontsize=9)
ax.set_ylabel("GAZE (LOMO) Spearman $\\rho$", fontsize=9)
ax.set_title("(B) GAZE vs scCellFie", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.set_aspect("equal")

# Panel C: MEBOCOST vs GAZE (head-to-head on LOMO metabolites)
ax = axes[2]
if len(gaze_mebo_df) > 0:
    ax.scatter(gaze_mebo_df["mebocost_rho"], gaze_mebo_df["gaze_rho"],
               color="#2171b5", edgecolor="black", linewidth=0.5, s=50, alpha=0.7, zorder=3)

    # Only label notable outliers (far from diagonal or high absolute rho)
    for _, row in gaze_mebo_df.iterrows():
        diff = abs(row["gaze_rho"] - row["mebocost_rho"])
        mag = max(abs(row["gaze_rho"]), abs(row["mebocost_rho"]))
        if diff > 0.4 or mag > 0.6:
            name = row["metabolite"]
            if len(name) > 15:
                name = name[:15] + "..."
            ax.annotate(name, (row["mebocost_rho"], row["gaze_rho"]),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")

    lim = max(abs(gaze_mebo_df["mebocost_rho"]).max(), abs(gaze_mebo_df["gaze_rho"]).max()) + 0.15
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.3)

    gaze_wins = (gaze_mebo_df["gaze_rho"].abs() > gaze_mebo_df["mebocost_rho"].abs()).sum()
    print(f"GAZE vs MEBOCOST: GAZE wins {gaze_wins}/{len(gaze_mebo_df)}")

ax.set_xlabel("MEBOCOST Spearman $\\rho$", fontsize=9)
ax.set_ylabel("GAZE (LOMO) Spearman $\\rho$", fontsize=9)
ax.set_title("(C) GAZE vs MEBOCOST", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.set_aspect("equal")

# Panel D: GAZE vs UnitedMet (head-to-head)
ax = axes[3]
if len(gaze_um_df) > 0:
    ax.scatter(gaze_um_df["unitedmet_rho"], gaze_um_df["gaze_rho"],
               color="#66c2a5", edgecolor="black", linewidth=0.5, s=50, alpha=0.7, zorder=3)

    for _, row in gaze_um_df.iterrows():
        name = row["metabolite"]
        if len(name) > 15:
            name = name[:15] + "..."
        ax.annotate(name, (row["unitedmet_rho"], row["gaze_rho"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")

    lim = max(abs(gaze_um_df["unitedmet_rho"]).max(), abs(gaze_um_df["gaze_rho"]).max()) + 0.15
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.3)

    gaze_wins = (gaze_um_df["gaze_rho"].abs() > gaze_um_df["unitedmet_rho"].abs()).sum()
    print(f"GAZE vs UnitedMet: GAZE wins {gaze_wins}/{len(gaze_um_df)}")

ax.set_xlabel("UnitedMet Spearman $\\rho$", fontsize=9)
ax.set_ylabel("GAZE (LOMO) Spearman $\\rho$", fontsize=9)
ax.set_title("(D) GAZE vs UnitedMet", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2)
ax.set_aspect("equal")

fig.tight_layout()
out_dir = project_root / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "comparison_figure.pdf", bbox_inches="tight")
fig.savefig(out_dir / "comparison_figure.png", dpi=300, bbox_inches="tight")
print(f"Saved {out_dir / 'comparison_figure.pdf'}")
plt.close(fig)
