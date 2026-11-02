#!/usr/bin/env python3
"""
Modality ablation analysis for manuscript Section 3.1.

Produces:
  - Incremental contribution waterfall chart
  - Topology gap (Real vs Random) paired bar chart
  - GPE-specific analysis (additive benefit + parameter control)
  - Ranked bar chart with XGBoost reference line
  - Welch's t-tests between key configuration pairs
  - Auto-extracted manuscript numbers
  - LaTeX table ready for inclusion
"""

import os
import json
import glob
import argparse
import textwrap
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir):
    """Load every *_results.json, returning a DataFrame with per-fold arrays."""
    files = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))
    rows = []
    for fpath in files:
        with open(fpath) as fh:
            res = json.load(fh)

        config = res["config_name"]
        fold_r2 = [f["val_r2"] for f in res.get("fold_results", [])]
        fold_rmse = [f["val_rmse"] for f in res.get("fold_results", [])]

        graph = "Random" if "Random" in config else "Real"
        norm = "MeanSub" if "Hard" in config else "Standard"

        # Clean modality label
        mod = config
        for prefix in ("RealGraph_", "RandomGraph_", "Hard_", "Arch_"):
            mod = mod.replace(prefix, "")
        mod = mod.replace("_", " ")

        rows.append({
            "config": config,
            "graph": graph,
            "norm": norm,
            "modality": mod,
            "mean_r2": res["mean_r2"],
            "std_r2": res["std_r2"],
            "mean_rmse": res["mean_rmse"],
            "std_rmse": res["std_rmse"],
            "fold_r2": fold_r2,
            "fold_rmse": fold_rmse,
        })

    return pd.DataFrame(rows)


def _lookup(df, config_name):
    """Return the row for a given config name, or None."""
    match = df[df["config"] == config_name]
    return match.iloc[0] if len(match) else None


# ---------------------------------------------------------------------------
# 2. Incremental contribution analysis
# ---------------------------------------------------------------------------

# Logical build-up order: each step adds one signal on top of the previous.
# Configs are identified by their JSON config_name.
INCREMENTAL_CHAIN = [
    ("RandomGraph_Graph_Only",    "Random graph\n(GNN capacity)"),
    ("RealGraph_Graph_Only",      "+Bio topology"),
    ("RealGraph_Genes_Only",      "+Gene expression"),
    ("RealGraph_Enzyme_Genes",    "+EC2Vec embeddings"),
    ("RealGraph_Full_Model",      "+SMILES\n(Full Model)"),
    ("RealGraph_Full_with_GPE",   "+GPE"),
]


def compute_incremental(df):
    """Return a list of dicts describing each incremental step."""
    steps = []
    prev_r2 = 0.0
    for cfg, label in INCREMENTAL_CHAIN:
        row = _lookup(df, cfg)
        if row is None:
            continue
        r2 = row["mean_r2"]
        delta = r2 - prev_r2 if steps else 0.0
        pct = (delta / prev_r2 * 100) if prev_r2 > 0 else 0.0
        steps.append({
            "config": cfg,
            "label": label,
            "r2": r2,
            "std": row["std_r2"],
            "delta": delta,
            "pct_change": pct,
        })
        prev_r2 = r2
    return steps


# ---------------------------------------------------------------------------
# 3. Topology gap
# ---------------------------------------------------------------------------

def compute_topology_gaps(df):
    """For each modality present in both Real and Random, compute the gap."""
    std = df[df["norm"] == "Standard"]
    real = std[std["graph"] == "Real"].set_index("modality")
    rand = std[std["graph"] == "Random"].set_index("modality")
    shared = sorted(set(real.index) & set(rand.index))

    rows = []
    for mod in shared:
        r_real, r_rand = real.loc[mod, "mean_r2"], rand.loc[mod, "mean_r2"]
        gap = r_real - r_rand
        pct = (gap / r_rand * 100) if r_rand != 0 else 0.0
        rows.append({
            "modality": mod,
            "real_r2": r_real,
            "random_r2": r_rand,
            "gap": gap,
            "gap_pct": pct,
        })
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


# ---------------------------------------------------------------------------
# 4. GPE analysis
# ---------------------------------------------------------------------------

def compute_gpe_analysis(df):
    """Compare Full_Model, Full_with_GPE, and Full_randomGPE."""
    keys = {
        "base": "RealGraph_Full_Model",
        "gpe": "RealGraph_Full_with_GPE",
        "rand_gpe": "RealGraph_Full_randomGPE",
        "gpe_only": "RealGraph_GPE_Only",
    }
    found = {}
    for tag, cfg in keys.items():
        row = _lookup(df, cfg)
        if row is not None:
            found[tag] = row

    if not found:
        return None

    report = {}
    if "base" in found and "gpe" in found:
        delta = found["gpe"]["mean_r2"] - found["base"]["mean_r2"]
        report["additive_benefit"] = delta
        report["base_r2"] = found["base"]["mean_r2"]
        report["gpe_r2"] = found["gpe"]["mean_r2"]
    if "gpe" in found and "rand_gpe" in found:
        report["gpe_vs_random_gpe"] = (
            found["gpe"]["mean_r2"] - found["rand_gpe"]["mean_r2"]
        )
        report["rand_gpe_r2"] = found["rand_gpe"]["mean_r2"]
    if "gpe_only" in found:
        report["gpe_only_r2"] = found["gpe_only"]["mean_r2"]

    # Statistical test (GPE vs base)
    if "base" in found and "gpe" in found:
        a, b = found["base"]["fold_r2"], found["gpe"]["fold_r2"]
        if len(a) >= 2 and len(b) >= 2:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            report["t_stat"] = t
            report["p_value"] = p

    return report


# ---------------------------------------------------------------------------
# 5. Statistical tests between key pairs
# ---------------------------------------------------------------------------

# Pairs of (config_A, config_B, description)
KEY_PAIRS = [
    ("RealGraph_Graph_Only", "RandomGraph_Graph_Only", "Bio topology vs random"),
    ("RealGraph_Enzyme_Genes", "RealGraph_Genes_Only", "EC2Vec additive effect"),
    ("RealGraph_Full_Model", "RealGraph_Enzyme_Genes", "SMILES additive effect"),
    ("RealGraph_Full_with_GPE", "RealGraph_Full_Model", "GPE additive effect"),
    ("RealGraph_Full_with_GPE", "RealGraph_Full_randomGPE", "GPE real vs random"),
]


def run_statistical_tests(df):
    """Welch's t-test for pre-defined key configuration pairs."""
    results = []
    for cfg_a, cfg_b, desc in KEY_PAIRS:
        row_a = _lookup(df, cfg_a)
        row_b = _lookup(df, cfg_b)
        if row_a is None or row_b is None:
            continue
        a, b = row_a["fold_r2"], row_b["fold_r2"]
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results.append({
            "comparison": desc,
            "config_a": cfg_a,
            "config_b": cfg_b,
            "mean_a": np.mean(a),
            "mean_b": np.mean(b),
            "delta": np.mean(a) - np.mean(b),
            "t_stat": t,
            "p_value": p,
            "significant": p < 0.05,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 6. Manuscript numbers
# ---------------------------------------------------------------------------

def print_manuscript_numbers(df, xgb_r2, xgb_std):
    """Print a copy-paste-ready block of numbers for the manuscript."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  MANUSCRIPT NUMBERS  (Section 3.1)")
    print(sep)

    def _num(cfg):
        row = _lookup(df, cfg)
        if row is None:
            return None, None
        return row["mean_r2"], row["std_r2"]

    # Total configs & runs
    n_configs = len(df)
    n_folds = df["fold_r2"].apply(len).max() if len(df) else 0
    n_runs = sum(len(r) for r in df["fold_r2"])
    print(f"\nConfigurations evaluated:  {n_configs}")
    print(f"Seeds per config:          {n_folds}")
    print(f"Total training runs:       {n_runs}")

    # XGBoost baseline
    print(f"\nXGBoost baseline:          R² = {xgb_r2:.3f} ± {xgb_std:.3f}")

    # Key configs
    labels = [
        ("RandomGraph_Graph_Only",    "Standard GNN (random graph)"),
        ("RealGraph_Graph_Only",      "Graph-only (bio topology)"),
        ("RealGraph_Genes_Only",      "Gene expression only"),
        ("RealGraph_Enzyme_Only",     "Enzyme embeddings only"),
        ("RealGraph_Enzyme_Genes",    "EC2Vec + genes"),
        ("RealGraph_Full_Baseline",   "Full Baseline"),
        ("RealGraph_Full_Model",      "Full Model"),
        ("RealGraph_Full_with_GPE",   "Full Model + GPE"),
        ("RealGraph_Full_randomGPE",  "Full Model + random GPE"),
        ("RealGraph_GPE_Only",        "GPE only"),
    ]
    print("\nKey R² values (Real graph, Standard norm unless noted):")
    for cfg, label in labels:
        r2, std = _num(cfg)
        if r2 is not None:
            print(f"  {label:38s}  R² = {r2:.4f} ± {std:.4f}")

    # Deltas
    print("\nIncremental improvements:")
    steps = compute_incremental(df)
    for i, s in enumerate(steps):
        if i == 0:
            print(f"  {s['label']:38s}  R² = {s['r2']:.4f}  (baseline)")
        else:
            sign = "+" if s["delta"] >= 0 else ""
            print(
                f"  {s['label']:38s}  R² = {s['r2']:.4f}  "
                f"({sign}{s['delta']:.4f}, {sign}{s['pct_change']:.1f}%)"
            )

    # Topology gap
    base_real, _ = _num("RealGraph_Graph_Only")
    base_rand, _ = _num("RandomGraph_Graph_Only")
    if base_real is not None and base_rand is not None:
        gap = base_real - base_rand
        pct = gap / base_rand * 100 if base_rand else 0
        print(
            f"\nTopology gap (Graph-Only):  "
            f"{base_real:.3f} vs {base_rand:.3f}  "
            f"(+{gap:.3f}, +{pct:.1f}%)"
        )

    # Best config
    best = df.loc[df["mean_r2"].idxmax()]
    print(f"\nBest configuration:        {best['config']}")
    print(f"                           R² = {best['mean_r2']:.4f} ± {best['std_r2']:.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# 7. LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(df, xgb_r2, xgb_std, output_path,
                         xgb_rmse=None, xgb_rmse_std=None):
    """Write a publication-ready LaTeX table to disk."""
    # Select Real-graph Standard configs + MeanSub Full Model
    display_configs = {
        "RandomGraph_Graph_Only":        "Random Graph Only",
        "RealGraph_Graph_Only":          "Graph Only (bio topology)",
        "RealGraph_Enzyme_Only":         "Enzyme embeddings only",
        "RealGraph_Genes_Only":          "Gene expression only",
        "RealGraph_Enzyme_Genes":        "EC2Vec + gene expression",
        "RealGraph_SMILES_Only":         "SMILES only",
        "RealGraph_Full_Baseline":       "Full Baseline",
        "RealGraph_Full_Model":          "Full Model",
        "RealGraph_GPE_Only":            "GPE only",
        "RealGraph_Full_with_GPE":       "Full Model + GPE",
        "RealGraph_Full_randomGPE":      "Full Model + random GPE",
        "RealGraph_Hard_Full_Model":     "Full Model (mean-sub)",
        "RealGraph_Hard_Graph_Only":     "Graph Only (mean-sub)",
    }

    # Build rows and sort by R² descending
    table_rows = []
    for cfg, label in display_configs.items():
        row = _lookup(df, cfg)
        if row is None:
            continue
        table_rows.append((label, row["mean_r2"], row["std_r2"],
                           row["mean_rmse"], row["std_rmse"]))
    table_rows.sort(key=lambda x: x[1], reverse=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Modality ablation results. All configurations use the biological graph"
        r" unless noted. R$^2$ and RMSE are averaged over 5 folds ($\pm$ std).}",
        r"\label{tab:modality_ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Configuration & R$^2$ ($\uparrow$) & RMSE ($\downarrow$) \\",
        r"\midrule",
        rf"XGBoost (per-metabolite) & {xgb_r2:.3f} \tiny{{$\pm$ {xgb_std:.3f}}} & "
        + (rf"{xgb_rmse:.3f} \tiny{{$\pm$ {xgb_rmse_std:.3f}}}" if xgb_rmse is not None else "---")
        + r" \\",
        r"\midrule",
    ]

    for label, mean_r2, std_r2, mean_rmse, std_rmse in table_rows:
        r2_str = f"{mean_r2:.3f} \\tiny{{$\\pm$ {std_r2:.3f}}}"
        rmse_str = f"{mean_rmse:.3f} \\tiny{{$\\pm$ {std_rmse:.3f}}}"
        lines.append(f"{label} & {r2_str} & {rmse_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    with open(output_path, "w") as fh:
        fh.write(latex)
    print(f"\nLaTeX table saved to {output_path}")
    return latex


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

FIGSIZE_SINGLE = (7, 4.5)
FIGSIZE_WIDE = (10, 5)
BAR_COLOR = "#4C72B0"
BAR_COLOR_ALT = "#DD8452"
REF_COLOR = "#C44E52"


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_incremental_waterfall(df, xgb_r2, output_dir):
    """Bar chart showing the incremental R² as modalities are added."""
    steps = compute_incremental(df)
    if len(steps) < 2:
        print("  Skipping waterfall: fewer than 2 incremental steps found.")
        return

    labels = [s["label"] for s in steps]
    r2_vals = [s["r2"] for s in steps]
    stds = [s["std"] for s in steps]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(labels))
    bars = ax.bar(x, r2_vals, yerr=stds, capsize=3, color=BAR_COLOR,
                  edgecolor="white", linewidth=0.5)

    # XGBoost reference
    ax.axhline(xgb_r2, ls="--", color=REF_COLOR, lw=1.2, label=f"XGBoost R² = {xgb_r2:.3f}")

    # Delta annotations
    for i in range(1, len(steps)):
        delta = steps[i]["delta"]
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(x[i], r2_vals[i] + stds[i] + 0.008),
            ha="center", va="bottom", fontsize=7.5, color="0.25",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("R²")
    ax.set_title("Incremental Modality Contributions")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, min(1.0, max(r2_vals) + 0.08))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _savefig(fig, os.path.join(output_dir, "fig_incremental_waterfall.png"))
    _savefig(fig, os.path.join(output_dir, "fig_incremental_waterfall.pdf"))


def plot_topology_gap(df, output_dir):
    """Paired bar chart: Real vs Random graph per modality."""
    gaps = compute_topology_gaps(df)
    if gaps.empty:
        print("  Skipping topology gap plot: no paired configs found.")
        return

    gaps = gaps.sort_values("real_r2", ascending=True)
    y = np.arange(len(gaps))
    h = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.barh(y + h / 2, gaps["real_r2"], h, label="Bio graph", color=BAR_COLOR)
    ax.barh(y - h / 2, gaps["random_r2"], h, label="Random graph", color=BAR_COLOR_ALT)

    # Gap annotations
    for i, row in gaps.iterrows():
        idx = list(gaps.index).index(i)
        ax.annotate(
            f"+{row['gap']:.3f}",
            xy=(max(row["real_r2"], row["random_r2"]) + 0.005, y[idx]),
            va="center", fontsize=7.5, color="0.3",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(gaps["modality"], fontsize=8)
    ax.set_xlabel("R²")
    ax.set_title("Topology Contribution: Biological vs Random Graph")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(output_dir, "fig_topology_gap.png"))
    _savefig(fig, os.path.join(output_dir, "fig_topology_gap.pdf"))


def plot_ranked_bar(df, xgb_r2, output_dir):
    """All configs ranked by R² with XGBoost reference line."""
    # Filter to Real-graph Standard configs (+ MeanSub Full for completeness)
    subset = df[df["graph"] == "Real"].copy()
    subset = subset.sort_values("mean_r2", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(subset) * 0.4)))
    y = np.arange(len(subset))
    ax.barh(y, subset["mean_r2"], xerr=subset["std_r2"], capsize=2,
            color=BAR_COLOR, edgecolor="white", linewidth=0.5)
    ax.axvline(xgb_r2, ls="--", color=REF_COLOR, lw=1.2,
               label=f"XGBoost R² = {xgb_r2:.3f}")

    ax.set_yticks(y)
    ax.set_yticklabels(subset["modality"], fontsize=8)
    ax.set_xlabel("R²")
    ax.set_title("All Configurations Ranked by R² (Real Graph)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(output_dir, "fig_ranked_configs.png"))
    _savefig(fig, os.path.join(output_dir, "fig_ranked_configs.pdf"))


def plot_gpe_comparison(df, output_dir):
    """Three-bar comparison: Full Model vs +GPE vs +randomGPE."""
    configs = [
        ("RealGraph_Full_Model",      "Full Model"),
        ("RealGraph_Full_with_GPE",   "Full + GPE"),
        ("RealGraph_Full_randomGPE",  "Full + rand GPE"),
        ("RealGraph_GPE_Only",        "GPE Only"),
    ]
    items = []
    for cfg, label in configs:
        row = _lookup(df, cfg)
        if row is not None:
            items.append((label, row["mean_r2"], row["std_r2"]))

    if len(items) < 2:
        print("  Skipping GPE comparison: fewer than 2 GPE configs found.")
        return

    labels, vals, errs = zip(*items)
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(items))
    ax.bar(x, vals, yerr=errs, capsize=4, color=BAR_COLOR, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("GPE Contribution Analysis")

    # Annotate values
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _savefig(fig, os.path.join(output_dir, "fig_gpe_comparison.png"))
    _savefig(fig, os.path.join(output_dir, "fig_gpe_comparison.pdf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Modality ablation analysis (manuscript Section 3.1)"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory with *_results.json files")
    parser.add_argument("--xgb-r2", type=float, default=0.835,
                        help="XGBoost baseline R² (default: 0.835)")
    parser.add_argument("--xgb-std", type=float, default=0.003,
                        help="XGBoost baseline std (default: 0.003)")
    args = parser.parse_args()

    results_dir = args.results_dir
    df = load_all_results(results_dir)
    if df.empty:
        print("No results found.")
        return

    # Exclude PNA configs (not part of the ablation narrative)
    df = df[~df["config"].str.contains("PNA")].reset_index(drop=True)


    # Auto-detect XGBoost from the same results directory if available
    xgb_row = _lookup(df, "XGBoost_Baseline")
    if xgb_row is not None:
        args.xgb_r2 = xgb_row["mean_r2"]
        args.xgb_std = xgb_row["std_r2"]
        args.xgb_rmse = xgb_row["mean_rmse"]
        args.xgb_rmse_std = xgb_row["std_rmse"]
        # Remove XGBoost from the main dataframe (it's shown separately)
        df = df[df["config"] != "XGBoost_Baseline"].reset_index(drop=True)
        print(f"Auto-detected XGBoost baseline: R² = {args.xgb_r2:.4f} ± {args.xgb_std:.4f}")

    print(f"Loaded {len(df)} configurations from {results_dir}\n")

    # ---- Manuscript numbers ----
    print_manuscript_numbers(df, args.xgb_r2, args.xgb_std)

    # ---- Topology gap table ----
    gaps = compute_topology_gaps(df)
    if not gaps.empty:
        print("\n=== Topology Gap (Real − Random) ===")
        print(gaps.to_string(index=False, float_format="%.4f"))

    # ---- GPE analysis ----
    gpe = compute_gpe_analysis(df)
    if gpe:
        print("\n=== GPE Analysis ===")
        for k, v in gpe.items():
            if isinstance(v, float):
                print(f"  {k:30s}  {v:.4f}")
            else:
                print(f"  {k:30s}  {v}")

    # ---- Statistical tests ----
    tests = run_statistical_tests(df)
    if not tests.empty:
        print("\n=== Statistical Tests (Welch's t-test, α = 0.05) ===")
        for _, row in tests.iterrows():
            sig = "*" if row["significant"] else "ns"
            print(
                f"  {row['comparison']:35s}  "
                f"Δ = {row['delta']:+.4f}  "
                f"t = {row['t_stat']:+.3f}  "
                f"p = {row['p_value']:.4f}  [{sig}]"
            )
        tests.to_csv(os.path.join(results_dir, "statistical_tests.csv"), index=False)

    # ---- LaTeX table ----
    generate_latex_table(
        df, args.xgb_r2, args.xgb_std,
        os.path.join(results_dir, "table_modality_ablation.tex"),
        xgb_rmse=getattr(args, "xgb_rmse", None),
        xgb_rmse_std=getattr(args, "xgb_rmse_std", None),
    )

    # ---- CSV (full) ----
    csv_path = os.path.join(results_dir, "modality_comparison_detailed.csv")
    df.drop(columns=["fold_r2", "fold_rmse"]).to_csv(csv_path, index=False)
    print(f"\nFull comparison CSV saved to {csv_path}")

    # ---- Figures ----
    print("\nGenerating figures...")
    plot_incremental_waterfall(df, args.xgb_r2, results_dir)
    plot_topology_gap(df, results_dir)
    plot_ranked_bar(df, args.xgb_r2, results_dir)
    plot_gpe_comparison(df, results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
