#!/usr/bin/env python3
"""
Patch fold_*_result.json files with neighbour coverage scores.

When LOMO results are recovered from SLURM logs the coverage fields
(coverage_score, direct_coverage, far_coverage) are all 0.  This script
recomputes them from the graph + dataset and writes the values back into
every fold JSON it finds.

Usage:
    python scripts/utils/patch_coverage_scores.py \
        --dir lomo_results_final_waterfall_N50
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Project root on path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.utils.data_paths import get_standard_data_paths
from src.evaluation.lomo_cross_validation import (
    _build_metabolite_neighbor_maps,
    _compute_metabolite_statistics,
)


def compute_coverage(dataset):
    """Return (coverage_scores, direct_scores, far_scores) dicts keyed by metabolite_idx."""
    _, coverage_scores, direct_scores, far_scores = _compute_metabolite_statistics(dataset)
    return coverage_scores, direct_scores, far_scores


def patch_dir(results_dir: Path, coverage_scores, direct_scores, far_scores):
    fold_files = list(results_dir.rglob('fold_*_result.json'))
    if not fold_files:
        print(f"  No fold JSONs found in {results_dir}")
        return 0

    patched = 0
    for fp in fold_files:
        with open(fp) as f:
            data = json.load(f)

        met_idx = data.get('metabolite_idx')
        if met_idx is None:
            continue

        # Only patch if coverage is 0 (i.e. recovered from logs)
        if data.get('coverage_score', 0) == 0 and data.get('direct_coverage', 0) == 0:
            data['coverage_score']  = float(coverage_scores.get(met_idx, 0.0))
            data['direct_coverage'] = float(direct_scores.get(met_idx, 0.0))
            data['far_coverage']    = float(far_scores.get(met_idx, 0.0))
            with open(fp, 'w') as f:
                json.dump(data, f, indent=2)
            patched += 1

    return patched


def main():
    parser = argparse.ArgumentParser(description="Patch recovered LOMO fold JSONs with coverage scores.")
    parser.add_argument('--dir', required=True,
                        help='Top-level LOMO results directory (e.g. lomo_results_final_waterfall_N50)')
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    print("Loading dataset to compute coverage scores...")
    mudata_path, graph_path, gene_mapping_path = get_standard_data_paths()
    dataset = FiLMDatasetWithSMILES(
        mudata_path=mudata_path,
        embeddings_graph_path=graph_path,
        gene_mapping_path=gene_mapping_path,
    )
    print(f"Dataset loaded ({len(dataset)} samples)")

    coverage_scores, direct_scores, far_scores = compute_coverage(dataset)
    print(f"Computed coverage for {len(coverage_scores)} metabolites")

    total = 0
    for arch_dir in sorted(results_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        n = patch_dir(arch_dir, coverage_scores, direct_scores, far_scores)
        if n:
            print(f"  {arch_dir.name}: patched {n} fold files")
        total += n

    print(f"\nDone. Patched {total} fold result files.")
    print("Re-run plot_lomo_proximity_vs_r2.py --all to generate plots.")


if __name__ == '__main__':
    main()
