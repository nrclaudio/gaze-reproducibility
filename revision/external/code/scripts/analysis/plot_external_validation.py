#!/usr/bin/env python3
"""
Generate the external validation figure for the manuscript.

Style matches plot_comparison_figure.py (Figure 6):
  - Same color scheme: grey=MEBOCOST, teal=UnitedMet, salmon=scCellFie, blue=GAZE
  - Same violin+boxplot style, same font sizes, same grid
  - 1x4 horizontal layout

Four panels:
  (A) Method comparison on external ccRCC tissue (violin/box)
  (B) GAZE per-cohort breakdown (box + jitter)
  (C) LOMO vs external concordance scatter
  (D) Top 20 external metabolites (horizontal bar)

Usage:
    python scripts/analysis/plot_external_validation.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).resolve().parent.parent.parent

# Colors matching plot_comparison_figure.py exactly
COLORS = {
    "MEBOCOST": "#999999",
    "UnitedMet": "#66c2a5",
    "scCellFie": "#fb8072",
    "GAZE": "#2171b5",
}

# Cohort colors (shades of the GAZE blue)
COHORT_COLORS = {
    "RC18": "#4292c6",
    "RC20": "#2171b5",
    "CPTAC": "#6baed6",
    "CPTAC_val": "#9ecae1",
}

# Pathway colors for bar chart
PATHWAY_COLORS = {
    "Amino acid": "#1b9e77",
    "Lipid": "#d95f02",
    "Carbohydrate": "#7570b3",
    "Nucleotide": "#e7298a",
    "TCA / Energy": "#66a61e",
    "Vitamin / Cofactor": "#e6ab02",
    "Other": "#bdbdbd",
}

METABOLITE_PATHWAYS = {
    "glucose": "Carbohydrate", "mannose-6-phosphate": "Carbohydrate",
    "alpha-d-galactose-1-phosphate": "Carbohydrate", "ribitol": "Carbohydrate",
    "sedoheptulose 7-phosphate": "Carbohydrate", "cmp-n-acetylneuraminate": "Carbohydrate",
    "fumarate": "TCA / Energy", "malate": "TCA / Energy", "succinate": "TCA / Energy",
    "citrate": "TCA / Energy", "methylmalonate": "TCA / Energy",
    "tryptophan": "Amino acid", "isoleucine": "Amino acid", "threonine": "Amino acid",
    "serine": "Amino acid", "histidine": "Amino acid", "glutamine": "Amino acid",
    "asparagine": "Amino acid", "cystine": "Amino acid", "alanine": "Amino acid",
    "valine": "Amino acid", "leucine": "Amino acid", "phenylalanine": "Amino acid",
    "tyrosine": "Amino acid", "proline": "Amino acid",
    "arachidonate": "Lipid", "retinol": "Vitamin / Cofactor",
    "hypoxanthine": "Nucleotide", "inosine": "Nucleotide",
    "1-methylnicotinamide": "Vitamin / Cofactor", "nag": "Carbohydrate",
}


def get_pathway(name):
    n = name.lower().strip()
    if n in METABOLITE_PATHWAYS:
        return METABOLITE_PATHWAYS[n]
    for key, pw in METABOLITE_PATHWAYS.items():
        if key in n or n in key:
            return pw
    return "Other"


def load_results(path, name):
    if not os.path.exists(path):
        print(f"  {name}: not found ({path})")
        return None
    df = pd.read_csv(path)
    for col in ["spearman_rho", "spearman_r", "rho"]:
        if col in df.columns:
            df["spearman_rho"] = df[col]
            break
    if "spearman_rho" not in df.columns:
        return None
    print(f"  {name}: {len(df)} metabolites, median rho = {df['spearman_rho'].median():.3f}")
    return df


def load_gaze_per_dataset(results_dir):
    per_ds = {}
    for f in Path(results_dir).glob("per_metabolite_*.csv"):
        ds = f.stem.replace("per_metabolite_", "")
        if ds == "results":
            continue
        df = pd.read_csv(f)
        if "spearman_rho" in df.columns:
            per_ds[ds] = df
    return per_ds


def load_lomo():
    candidates = [
        project_root / "lomo_results_best_with_predictions/REAL_relational_full_L3_full_1039043/lomo_cross_validation",
        project_root / "lomo_results/mcr_flux_mass_balance/baseline_mcr_20250929_064841/lomo_cross_validation",
    ]
    for cand in candidates:
        # Try individual fold JSONs first (more complete)
        fold_files = sorted(cand.glob("fold_*_result.json"))
        if fold_files:
            rows = []
            for ff in fold_files:
                with open(ff) as f:
                    fold = json.load(f)
                preds = np.array(fold["predictions"])
                tgts = np.array(fold["targets"])
                if len(preds) >= 5 and np.std(tgts) > 1e-10 and np.std(preds) > 1e-10:
                    sr, _ = stats.spearmanr(preds, tgts)
                    rows.append({
                        "metabolite_id": fold.get("metabolite_id", ""),
                        "metabolite_name": fold.get("metabolite_name", ""),
                        "spearman_rho": sr,
                    })
            if rows:
                print(f"  LOMO: {len(rows)} metabolites from {cand}")
                return pd.DataFrame(rows)

        # Fallback to intermediate JSON
        jp = cand / "lomo_intermediate.json"
        if jp.exists():
            with open(jp) as f:
                folds = json.load(f)
            rows = []
            for fold in folds:
                preds = np.array(fold["predictions"])
                tgts = np.array(fold["targets"])
                if len(preds) >= 5 and np.std(tgts) > 1e-10 and np.std(preds) > 1e-10:
                    sr, _ = stats.spearmanr(preds, tgts)
                    rows.append({
                        "metabolite_id": fold.get("metabolite_id", ""),
                        "metabolite_name": fold.get("metabolite_name", ""),
                        "spearman_rho": sr,
                    })
            if rows:
                print(f"  LOMO: {len(rows)} metabolites from {jp}")
                return pd.DataFrame(rows)
    print("  LOMO: not found")
    return None


# ── Panel A: Method comparison ──────────────────────────────────────────

def plot_panel_a(ax, method_dfs):
    order = ["MEBOCOST", "UnitedMet", "scCellFie", "GAZE"]
    data, positions, colors, labels = [], [], [], []
    pos = 1
    for method in order:
        df = method_dfs.get(method)
        if df is not None and len(df) > 0:
            vals = df["spearman_rho"].dropna().values
            data.append(vals)
            positions.append(pos)
            colors.append(COLORS[method])
            labels.append(f"{method}\n(n={len(vals)})")
        pos += 1

    if not data:
        return

    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
        pc.set_alpha(0.7)

    bp = ax.boxplot(data, positions=positions, widths=0.15,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.9)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Spearman $\\rho$ with measured concentration", fontsize=9)
    ax.set_title("(A) External: correlation distributions", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    for i, vals in enumerate(data):
        med = np.median(vals)
        ax.text(positions[i], ax.get_ylim()[1] - 0.05, f"med={med:+.2f}",
                ha="center", fontsize=6, color=colors[i], fontweight="bold")


# ── Panel B: GAZE per-cohort ───────────────────────────────────────────

def plot_panel_b(ax, gaze_per_dataset):
    cohort_order = ["RC18", "RC20"]
    data, positions, colors, labels = [], [], [], []
    pos = 1
    for cohort in cohort_order:
        if cohort in gaze_per_dataset:
            df = gaze_per_dataset[cohort]
            vals = df["spearman_rho"].dropna().values
            data.append(vals)
            positions.append(pos)
            colors.append(COHORT_COLORS.get(cohort, "#666666"))
            labels.append(f"{cohort}\n(n={len(vals)})")
        pos += 1

    if not data:
        ax.text(0.5, 0.5, "No per-cohort data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("(B) GAZE per cohort", fontsize=11, fontweight="bold")
        return

    bp = ax.boxplot(data, positions=positions, widths=0.4,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)

    for i, vals in enumerate(data):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(positions[i] + jitter, vals, s=12, alpha=0.35,
                   color=colors[i], edgecolors="none", zorder=3)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Spearman $\\rho$", fontsize=9)
    ax.set_title("(B) GAZE per cohort", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)


# ── Panel C: LOMO vs external concordance ──────────────────────────────

def plot_panel_c(ax, gaze_ext_df, lomo_df):
    if gaze_ext_df is None or lomo_df is None:
        ax.text(0.5, 0.5, "Missing data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("(C) LOMO vs external", fontsize=11, fontweight="bold")
        return

    # Match by name
    name_col = "name" if "name" in gaze_ext_df.columns else "metabolite_name"
    ext_lookup = {}
    for _, row in gaze_ext_df.iterrows():
        ext_lookup[str(row[name_col]).lower().strip()] = row["spearman_rho"]

    lomo_name_col = "metabolite_name" if "metabolite_name" in lomo_df.columns else "name"
    matched = []
    for _, row in lomo_df.iterrows():
        name = str(row.get(lomo_name_col, "")).lower().strip()
        if name in ext_lookup:
            matched.append({"name": name, "lomo": row["spearman_rho"], "ext": ext_lookup[name]})

    # Also try HMDB matching
    if len(matched) < 5 and "hmdb" in gaze_ext_df.columns and "metabolite_id" in lomo_df.columns:
        ext_hmdb = {}
        for _, row in gaze_ext_df.iterrows():
            h = str(row.get("hmdb", "")).strip()
            if h and h != "nan":
                ext_hmdb[h] = {"rho": row["spearman_rho"], "name": str(row[name_col])}
        lomo_name_col2 = "metabolite_name" if "metabolite_name" in lomo_df.columns else "name"
        for _, row in lomo_df.iterrows():
            mid = str(row.get("metabolite_id", "")).strip()
            lomo_name = str(row.get(lomo_name_col2, mid))
            if mid in ext_hmdb:
                ext_name = ext_hmdb[mid]["name"]
                display = lomo_name if lomo_name and lomo_name != mid else ext_name
                if not any(m["name"] == display for m in matched):
                    matched.append({"name": display, "lomo": row["spearman_rho"], "ext": ext_hmdb[mid]["rho"]})

    if len(matched) < 3:
        ax.text(0.5, 0.5, f"Only {len(matched)} overlapping\nmetabolites",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("(C) LOMO vs external", fontsize=11, fontweight="bold")
        return

    mdf = pd.DataFrame(matched)
    x, y = mdf["lomo"].values, mdf["ext"].values

    ax.scatter(x, y, color=COLORS["GAZE"], edgecolor="black", linewidth=0.5,
               s=50, alpha=0.7, zorder=3)

    rho, pval = stats.spearmanr(x, y)
    ax.text(0.05, 0.95, f"$\\rho$ = {rho:.2f}\np = {pval:.2f}\nn = {len(matched)}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.3)

    for _, row in mdf.iterrows():
        if abs(row["ext"]) > 0.4 or abs(row["lomo"]) > 0.4:
            name = row["name"]
            if len(name) > 15:
                name = name[:15] + "..."
            ax.annotate(name, (row["lomo"], row["ext"]),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("LOMO Spearman $\\rho$ (CAMP)", fontsize=9)
    ax.set_ylabel("External Spearman $\\rho$ (ccRCC)", fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("(C) LOMO vs external", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)


# ── Panel D: Top metabolites ───────────────────────────────────────────

def plot_panel_d(ax, gaze_ext_df, n_top=20):
    if gaze_ext_df is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("(D) Top external metabolites", fontsize=11, fontweight="bold")
        return

    name_col = "name" if "name" in gaze_ext_df.columns else "metabolite_name"
    df = gaze_ext_df.nlargest(n_top, "spearman_rho").iloc[::-1].copy()

    names = df[name_col].values
    rhos = df["spearman_rho"].values
    pathways = [get_pathway(n) for n in names]
    bar_colors = [PATHWAY_COLORS.get(p, "#bdbdbd") for p in pathways]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, rhos, color=bar_colors, edgecolor="white", linewidth=0.3, height=0.7)

    display = [n[:32] + "..." if len(n) > 35 else n for n in names]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display, fontsize=6.5)
    ax.set_xlabel("Spearman $\\rho$", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("(D) Top external metabolites", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    for i, rho in enumerate(rhos):
        ax.text(rho + 0.01, i, f"{rho:.2f}", va="center", fontsize=6, color="#333333")

    # Pathway legend
    from matplotlib.patches import Patch
    unique_pw = sorted(set(pathways))
    legend_patches = [Patch(facecolor=PATHWAY_COLORS.get(p, "#bdbdbd"), label=p) for p in unique_pw]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=6,
              framealpha=0.9, title="Pathway", title_fontsize=7)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze-results", default=str(project_root / "results/external_unitedmet/per_metabolite_results_rc.csv"))
    parser.add_argument("--gaze-results-dir", default=str(project_root / "results/external_unitedmet"))
    parser.add_argument("--mebocost-results", default=str(project_root / "results/external_baselines/mebocost_external_rc.csv"))
    parser.add_argument("--sccellfie-results", default=str(project_root / "results/external_baselines/sccellfie_external_rc.csv"))
    parser.add_argument("--unitedmet-results", default=str(project_root / "results/external_baselines/unitedmet_method/unitedmet_external_results.csv"))
    parser.add_argument("--output", default=str(project_root / "figures/external_validation.pdf"))
    args = parser.parse_args()

    print("Loading results...")
    gaze_df = load_results(args.gaze_results, "GAZE")
    mebocost_df = load_results(args.mebocost_results, "MEBOCOST")
    sccellfie_df = load_results(args.sccellfie_results, "scCellFie")
    unitedmet_df = load_results(args.unitedmet_results, "UnitedMet")

    method_dfs = {"GAZE": gaze_df, "MEBOCOST": mebocost_df, "scCellFie": sccellfie_df, "UnitedMet": unitedmet_df}
    gaze_per_ds = load_gaze_per_dataset(args.gaze_results_dir)
    print(f"  GAZE per-dataset: {list(gaze_per_ds.keys())}")
    lomo_df = load_lomo()

    # ── Figure: 1x4 matching comparison_figure.py ───────────────────
    print("\nPlotting...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    plot_panel_a(axes[0], method_dfs)
    plot_panel_b(axes[1], gaze_per_ds)
    plot_panel_c(axes[2], gaze_df, lomo_df)
    plot_panel_d(axes[3], gaze_df)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    png = args.output.replace(".pdf", ".png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    print(f"Saved: {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
