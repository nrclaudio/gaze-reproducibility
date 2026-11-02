#!/usr/bin/env python3
"""
Recover lomo_results_final_waterfall_N50 from SLURM log files.

Log files (logs/lomo_worker_{JOB_ID}_{TASK_ID}.out) contain per-fold R² and
RMSE values printed by lomo_cross_validation.py.  This script parses them and
reconstructs the fold_{idx}_result.json files that collect_lomo_results.py needs.

Usage:
    python scripts/analysis/recover_lomo_from_logs.py
    python scripts/analysis/recover_lomo_from_logs.py --logs-dir logs --out-dir lomo_results_final_waterfall_N50
"""

import re
import json
import argparse
from pathlib import Path


# ── regex patterns matching what the worker / lomo_cv prints ──────────────────
RE_ARCH        = re.compile(r'Architecture:\s+(\S+)')
RE_PHYSICS     = re.compile(r'Physics Mode:\s+(\S+)')
RE_LAYERS      = re.compile(r'Layers to run:\s+(\S+)')
RE_FOLD_HEADER = re.compile(r'LOMO FOLD \d+: Held-out metabolite index (\d+)')
RE_R2          = re.compile(r'•\s+R²:\s+([-\d.]+|N/A)')
RE_RMSE        = re.compile(r'•\s+RMSE:\s+([-\d.]+|N/A)')
# ─────────────────────────────────────────────────────────────────────────────


def parse_log(log_path: Path):
    """
    Return a dict with keys:
        arch, physics_mode, layers (int), job_id (int),
        folds: list of {metabolite_idx, r2, rmse}
    Returns None if the log looks incomplete / wrong format.
    """
    text = log_path.read_text(errors='replace')

    arch_m   = RE_ARCH.search(text)
    phys_m   = RE_PHYSICS.search(text)
    layer_m  = RE_LAYERS.search(text)

    if not (arch_m and phys_m and layer_m):
        return None  # header info missing — skip

    arch        = arch_m.group(1)
    physics     = phys_m.group(1)
    layers_str  = layer_m.group(1)
    layers      = int(layers_str) if layers_str.isdigit() else None
    if layers is None:
        return None

    # Job ID is the second-to-last part of the stem: lomo_worker_{JOB}_{TASK}
    parts = log_path.stem.split('_')   # ['lomo', 'worker', JOB, TASK]
    if len(parts) < 4 or not parts[-2].isdigit():
        return None
    job_id = int(parts[-2])

    # Walk line-by-line to pair fold headers with their R²/RMSE.
    # R² and RMSE are printed on consecutive separate lines, so we buffer the
    # R² value and emit a result when the RMSE line follows.
    _MISSING = object()
    folds = []
    current_met = None
    pending_r2  = _MISSING   # set when we see the R² line, consumed by RMSE line

    for line in text.splitlines():
        fh = RE_FOLD_HEADER.search(line)
        if fh:
            current_met = int(fh.group(1))
            pending_r2  = _MISSING
            continue

        if current_met is None:
            continue

        r2_m = RE_R2.search(line)
        if r2_m:
            pending_r2 = None if r2_m.group(1) == 'N/A' else float(r2_m.group(1))
            continue

        rmse_m = RE_RMSE.search(line)
        if rmse_m and pending_r2 is not _MISSING:
            rmse_val = None if rmse_m.group(1) == 'N/A' else float(rmse_m.group(1))
            folds.append({
                'metabolite_idx': current_met,
                'r2':   pending_r2,
                'rmse': rmse_val,
                'mae':  0.0,    # not printed in logs — placeholder
                'mse':  0.0,
                'n_test_samples': 0,
            })
            current_met = None
            pending_r2  = _MISSING

    if not folds:
        return None   # header found but no completed folds

    return dict(arch=arch, physics_mode=physics, layers=layers,
                job_id=job_id, folds=folds)


def recover(logs_dir: Path, out_dir: Path, dry_run: bool = False):
    log_files = sorted(logs_dir.glob('lomo_worker_*.out'))
    print(f"Found {len(log_files)} lomo_worker log files in {logs_dir}")

    written = 0
    skipped = 0

    for lf in log_files:
        info = parse_log(lf)
        if info is None:
            # Check whether the header was at least found to distinguish
            # "job not started / wrong file" from "job ran but no folds completed"
            text = lf.read_text(errors='replace')
            has_header = bool(RE_ARCH.search(text) and RE_PHYSICS.search(text) and RE_LAYERS.search(text))
            has_fold   = bool(RE_FOLD_HEADER.search(text))
            reason = "no header" if not has_header else ("no completed folds" if not has_fold else "folds found but R²/RMSE not matched")
            print(f"  SKIP {lf.name} — {reason}")
            skipped += 1
            continue

        prefix     = 'REAL'   # all our runs are non-synthetic
        dir_name   = f"{prefix}_{info['arch']}_L{info['layers']}_{info['physics_mode']}_{info['job_id']}"
        lomo_cv    = out_dir / dir_name / 'lomo_cross_validation'

        print(f"  {lf.name} → {dir_name}  ({len(info['folds'])} folds)")

        if not dry_run:
            lomo_cv.mkdir(parents=True, exist_ok=True)

        for fold in info['folds']:
            fold_file = lomo_cv / f"fold_{fold['metabolite_idx']}_result.json"
            if not dry_run:
                with open(fold_file, 'w') as f:
                    json.dump(fold, f, indent=2)
            written += 1

    print(f"\nDone. Written {written} fold result files across {len(log_files) - skipped} experiments.")
    if skipped:
        print(f"Skipped {skipped} log files (incomplete or non-LOMO).")


def main():
    parser = argparse.ArgumentParser(description="Recover LOMO results from SLURM log files.")
    parser.add_argument('--logs-dir', default='logs',
                        help='Directory containing lomo_worker_*.out files (default: logs)')
    parser.add_argument('--out-dir', default='lomo_results_final_waterfall_N50',
                        help='Output directory to (re)create (default: lomo_results_final_waterfall_N50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be written without creating any files')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir  = Path(args.out_dir)

    if not logs_dir.exists():
        print(f"ERROR: logs directory not found: {logs_dir}")
        return

    if args.dry_run:
        print("=== DRY RUN — no files will be written ===\n")

    recover(logs_dir, out_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
