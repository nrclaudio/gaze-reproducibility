#!/usr/bin/env python3
"""
Regenerate LOMO figures with clean labels for manuscript.

Figure 3: Per-metabolite R2 cascade (GAZE + physics, 50 metabolites)
Figure 4: Summary bar chart of median R2 + positive R2 % across architectures
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent

# Clean labels ordered by architectural waterfall
LABEL_MAP = {
    "REAL_baseline_legacy_L3_none_908364": "GATv2 baseline",
    "REAL_relational_core_L3_none_908365": "Relational HeteroConv",
    "REAL_relational_residuals_L3_none_908366": "+ Residuals",
    "REAL_relational_attention_L3_none_908367": "+ Cross-modal attention",
    "REAL_relational_mcr_L3_none_908368": "GAZE",
    "REAL_relational_full_L3_full_908369": "GAZE + physics",
}

WATERFALL_ORDER = [
    "GATv2 baseline",
    "Relational HeteroConv",
    "+ Residuals",
    "+ Cross-modal attention",
    "GAZE",
    "GAZE + physics",
]

COLORS = {
    "GATv2 baseline": "#bdbdbd",
    "Relational HeteroConv": "#c6dbef",
    "+ Residuals": "#9ecae1",
    "+ Cross-modal attention": "#6baed6",
    "GAZE": "#3182bd",
    "GAZE + physics": "#08519c",
}


def load_summary():
    csv_path = project_root / "results/lomo_analysis_summary/relational/lomo_comparison.csv"
    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["label"] = df["config_name"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    print(f"Loaded {len(df)} configurations")
    return df


def load_best_folds():
    """Load per-metabolite R2 from the best config (GAZE + physics)."""
    lomo_dir = project_root / "lomo_results_best_with_predictions/REAL_relational_full_L3_full_1039043/lomo_cross_validation"
    if not lomo_dir.exists():
        return None
    r2s, names = [], []
    for fold_file in sorted(lomo_dir.glob("fold_*_result.json")):
        with open(fold_file) as f:
            fold = json.load(f)
        r2s.append(fold.get("r2", float("nan")))
        names.append(fold.get("metabolite_name", ""))
    if r2s:
        print(f"  Loaded {len(r2s)} fold results")
        return np.array(r2s), names
    return None


def plot_cascade(r2s, met_names, output_path):
    """Figure 3: Per-metabolite LOMO R2 for GAZE + physics, sorted by R2."""
    sort_idx = np.argsort(r2s)[::-1]
    r2_sorted = np.clip(r2s[sort_idx], -1.5, None)
    names_sorted = [met_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(r2_sorted))

    # Fill area
    ax.fill_between(x, 0, r2_sorted, where=(r2_sorted >= 0),
                     color="#08519c", alpha=0.3)
    ax.fill_between(x, 0, r2_sorted, where=(r2_sorted < 0),
                     color="#ef6548", alpha=0.2)

    ax.plot(x, r2_sorted, color="#08519c", linewidth=1.8)
    ax.scatter(x, r2_sorted, color="#08519c", s=20, edgecolor="white",
               linewidth=0.5, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Label top and bottom metabolites
    for i in range(min(5, len(names_sorted))):
        if r2_sorted[i] > 0.3:
            ax.annotate(names_sorted[i], (x[i], r2_sorted[i]),
                        fontsize=6, rotation=45, ha="left", va="bottom",
                        xytext=(2, 4), textcoords="offset points")

    n_pos = (r2s > 0).sum()
    ax.text(0.98, 0.95,
            f"Median R$^2$ = {np.median(r2s):.2f}\n{n_pos}/50 positive",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Metabolites (sorted by R$^2$)", fontsize=10)
    ax.set_ylabel("R$^2$ (clipped at $-1.5$)", fontsize=10)
    ax.set_xlim(-0.5, len(r2_sorted) - 0.5)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_summary_bars(df, output_path):
    """Figure 4: Median R2 + positive R2% across architectures."""
    # Order by waterfall
    df["_order"] = df["label"].apply(lambda x: WATERFALL_ORDER.index(x) if x in WATERFALL_ORDER else 99)
    df = df.sort_values("_order")

    labels = df["label"].values
    medians = df["lomo_median_r2"].values
    pos_pct = df["positive_r2_pct"].values
    above_05 = df["metabolites_above_0_5"].values
    bar_colors = [COLORS.get(l, "#999999") for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Left: Median R2
    y = np.arange(len(labels))
    bars = ax1.barh(y, medians, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Median R$^2$", fontsize=10)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.2)

    for i, (med, pct) in enumerate(zip(medians, above_05)):
        ax1.text(med - 0.02, i, f"{med:.2f}", va="center", ha="right", fontsize=8, color="white", fontweight="bold")

    # Right: % positive R2
    ax2.barh(y, pos_pct, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([], fontsize=9)
    ax2.set_xlabel("% metabolites with R$^2 > 0$", fontsize=10)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.2)

    for i, pct in enumerate(pos_pct):
        ax2.text(pct + 0.5, i, f"{pct:.0f}%", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    print("Loading data...")
    df = load_summary()
    fold_data = load_best_folds()

    out_dir = project_root / "lomo_analysis"
    out_dir.mkdir(exist_ok=True)

    if fold_data is not None:
        r2s, names = fold_data
        plot_cascade(r2s, names, out_dir / "lomo_cascade_clean.pdf")

    if df is not None:
        plot_summary_bars(df, out_dir / "r2_distribution_boxplot.pdf")


if __name__ == "__main__":
    main()
