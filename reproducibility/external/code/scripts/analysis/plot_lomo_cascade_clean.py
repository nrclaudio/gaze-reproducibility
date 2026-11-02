#!/usr/bin/env python3
"""
Generate a clean LOMO cascade plot with only the key architectures.
Reads raw R² lists from lomo_comparison.csv (no fold JSONs needed).
"""

import ast
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Architecture keys to show (substring match on config_name → label, color, linestyle, linewidth)
KEY_ARCHS = [
    ("baseline_legacy",          "GATv2 (homogeneous baseline)",    "#999999", "--", 1.8),
    ("relational_core_L",       "+ bipartite msg. passing + MCR",   "#7fb3d8", "-",  1.5),
    ("relational_mcr_L",        "GAZE",                             "#2171b5", "-",  2.2),
    ("relational_full_L3_full", "GAZE + physics",                   "#08306b", "-",  2.2),
]


def main():
    csv_path = Path("lomo_analysis/lomo_comparison.csv")
    output_dir = Path("lomo_analysis")

    # Parse CSV manually (avoid pandas dependency)
    lines = csv_path.read_text().strip().split("\n")
    header = lines[0].split(",")
    name_idx = header.index("config_name")

    # Find the raw_r2s column — it's the last column, quoted with brackets
    rows = {}
    for line in lines[1:]:
        # Split carefully: raw_r2s is quoted and contains commas
        # Format: ...,\"[val, val, ...]\"\n
        bracket_start = line.index("[")
        prefix = line[:bracket_start]
        r2_str = line[bracket_start:].strip().strip('"')
        fields = prefix.rstrip(",").split(",")
        config_name = fields[name_idx]
        r2_list = ast.literal_eval(r2_str)
        rows[config_name] = r2_list

    # Match key architectures
    arch_data = {}
    for key, label, color, ls, lw in KEY_ARCHS:
        match = [name for name in rows if key in name]
        if not match:
            print(f"  SKIP {key}: no match in CSV")
            continue
        config = match[0]
        r2s = rows[config]
        arch_data[key] = (label, color, ls, lw, r2s)
        print(f"  {label}: {len(r2s)} metabolites, median R² = {np.median(r2s):.3f}")

    if not arch_data:
        print("No data loaded.")
        return

    # All architectures share the same 50 metabolites in the same order.
    # Sort by baseline difficulty (descending R²).
    baseline_key = "baseline_legacy"
    if baseline_key in arch_data:
        baseline_r2s = np.array(arch_data[baseline_key][4])
        sort_idx = np.argsort(-baseline_r2s)
    else:
        sort_idx = np.arange(len(next(iter(arch_data.values()))[4]))

    n = len(sort_idx)
    clip = -1.5
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot baseline filled area
    if baseline_key in arch_data:
        base_y = np.clip(np.array(arch_data[baseline_key][4])[sort_idx], clip, None)
        ax.fill_between(x, base_y, color="#999999", alpha=0.10)

    # Plot each architecture
    for key, label, color, ls, lw in KEY_ARCHS:
        if key not in arch_data:
            continue
        _, _, _, _, r2s = arch_data[key]
        y = np.clip(np.array(r2s)[sort_idx], clip, None)
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("R² (clipped at −1.5)", fontsize=11)
    ax.set_xlabel("Metabolites (sorted by baseline difficulty)", fontsize=11)
    ax.set_title("LOMO Performance Cascade", fontsize=13, fontweight="bold")
    ax.set_ylim(clip - 0.1, 1.05)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "lomo_cascade_clean.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "lomo_cascade_clean.pdf", bbox_inches="tight")
    print(f"\nSaved lomo_analysis/lomo_cascade_clean.png and .pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
