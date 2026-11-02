#!/usr/bin/env python3
"""Produce a scatter plot of LOMO R² scores versus neighbour coverage.

The script expects the standard `lomo_results.csv` that is emitted by
`run_lomo_cross_validation`.  It uses the direct neighbour coverage fraction
when present and falls back to the combined coverage score otherwise.  The
resulting figure highlights whether performance correlates with the proximity
of metabolites that already have experimental measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_from_fold_jsons(lomo_cv_dir: Path) -> pd.DataFrame:
    """Build a results DataFrame from individual fold_*_result.json files."""
    import json

    rows = []
    for f in sorted(lomo_cv_dir.glob("fold_*_result.json")):
        try:
            with open(f) as fh:
                rows.append(json.load(fh))
        except Exception:
            continue
    if not rows:
        raise FileNotFoundError(f"No fold_*_result.json files found in {lomo_cv_dir}")
    return pd.DataFrame(rows)


def _resolve_results(path_str: str) -> pd.DataFrame:
    """Return a results DataFrame given a file, lomo_cv dir, or arch dir.

    Priority:
      1. Direct path to a CSV file.
      2. lomo_results.csv inside the given directory (or its lomo_cross_validation/ subdir).
      3. fold_*_result.json files inside lomo_cross_validation/ (recovered runs).
    """
    candidate = Path(path_str).expanduser().resolve()

    if candidate.is_file():
        return pd.read_csv(candidate)

    for csv_candidate in [
        candidate / "lomo_results.csv",
        candidate / "lomo_cross_validation" / "lomo_results.csv",
    ]:
        if csv_candidate.exists():
            return pd.read_csv(csv_candidate)

    # Fall back to fold JSONs
    for json_dir in [candidate, candidate / "lomo_cross_validation"]:
        if json_dir.is_dir() and list(json_dir.glob("fold_*_result.json")):
            return _load_from_fold_jsons(json_dir)

    raise FileNotFoundError(
        f"Could not locate lomo_results.csv or fold_*_result.json within {candidate}."
    )


def _load_results(path_str: str):
    """Load results and pick an appropriate proximity metric."""

    df = _resolve_results(path_str)

    if "r2" not in df.columns:
        raise ValueError("Results must contain an 'r2' column")

    proximity_columns = ["direct_coverage", "coverage_score", "far_coverage"]
    available = [col for col in proximity_columns if col in df.columns
                 and df[col].notna().any() and df[col].ne(0).any()]
    if not available:
        raise ValueError(
            "Results do not include non-zero coverage metrics; cannot plot proximity."
        )

    chosen_metric = available[0]
    df = df.dropna(subset=["r2", chosen_metric])
    df["proximity_metric"] = df[chosen_metric]
    return df, chosen_metric


def plot_proximity_vs_r2(df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    """Create and save the proximity versus R² scatter plot."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.scatter(df["proximity_metric"], df["r2"], s=70, alpha=0.8, edgecolor="black")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Proximity of measured metabolites")
    ax.set_ylabel("R² score (LOMO)")
    ax.set_title("LOMO performance versus neighbour coverage")
    ax.grid(True, alpha=0.25)

    if len(df) >= 2:
        coeffs = np.polyfit(df["proximity_metric"], df["r2"], deg=1)
        line_x = np.linspace(df["proximity_metric"].min(), df["proximity_metric"].max(), 100)
        line_y = np.poly1d(coeffs)(line_x)
        ax.plot(line_x, line_y, "#2ca02c", linestyle="--", linewidth=1.2, label="Linear trend")

        correlation = df[["proximity_metric", "r2"]].corr().iloc[0, 1]
        ax.text(
            0.02,
            0.95,
            f"Pearson r = {correlation:.2f}\nMetric = {metric_name}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.legend(loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot R² scores against metabolite proximity metrics from LOMO runs"
    )
    parser.add_argument(
        "results",
        nargs="?",
        type=str,
        help="Path to lomo_results.csv or a single architecture directory. "
             "Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        metavar="RESULTS_DIR",
        type=str,
        help="Top-level LOMO results directory; produces one plot per architecture.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lomo_proximity_vs_r2.png",
        help="Output PNG path for single-arch mode (default: lomo_proximity_vs_r2.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lomo_analysis",
        help="Output directory for --all mode (default: lomo_analysis)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        # Multi-architecture mode: one plot per subdirectory
        top_dir = Path(args.all).expanduser().resolve()
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        arch_dirs = sorted(d for d in top_dir.iterdir() if d.is_dir())
        print(f"Found {len(arch_dirs)} architecture directories in {top_dir}")

        for arch_dir in arch_dirs:
            arch_name = arch_dir.name
            try:
                df, metric_name = _load_results(str(arch_dir))
            except (FileNotFoundError, ValueError) as e:
                print(f"  SKIP {arch_name}: {e}")
                continue

            out_path = out_dir / f"proximity_vs_r2_{arch_name}.png"
            plot_proximity_vs_r2(df, metric_name, out_path)
            print(f"  {arch_name} → {out_path.name}")

        print(f"\nDone. Plots saved to {out_dir}")

    else:
        if not args.results:
            print("ERROR: provide a results path or use --all <dir>")
            return
        df, metric_name = _load_results(args.results)
        output_path = Path(args.output).expanduser().resolve()
        plot_proximity_vs_r2(df, metric_name, output_path)
        print(f"Saved proximity plot to {output_path}")


if __name__ == "__main__":
    main()
