#!/usr/bin/env python3
"""
Visualize LOMO results as a 'Cascade Plot' to demonstrate robustness.
Compares the Best Baseline vs. Best MCR/Attention architecture.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to import tqdm, but fallback if missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

def load_per_metabolite_results(experiment_dir):
    """
    Load individual fold results to get per-metabolite performance.
    Returns a dict mapping metabolite_idx -> results_dict
    """
    is_synth = "SYNTH" in os.path.basename(experiment_dir)
    results = {}
    
    # Prioritize zero-shot for SYNTH
    if is_synth:
        synth_json = os.path.join(experiment_dir, 'zero_shot_evaluation', 'lomo_full_results.json')
        synth_csv = os.path.join(experiment_dir, 'zero_shot_evaluation', 'lomo_results.csv')
        
        loaded = False
        if os.path.exists(synth_json):
            try:
                with open(synth_json, 'r') as f:
                    data_list = json.load(f)
                    for data in data_list:
                        met_idx = data.get('metabolite_idx', data.get('fold_idx'))
                        results[met_idx] = data
                loaded = True
            except Exception: pass
            
        if not loaded and os.path.exists(synth_csv):
            try:
                df = pd.read_csv(synth_csv)
                for _, row in df.iterrows():
                    data = row.to_dict()
                    met_idx = data.get('metabolite_idx', data.get('fold_idx'))
                    results[met_idx] = data
                loaded = True
            except Exception: pass
            
        if loaded:
            return results

    # Fallback/Default: Standard folds
    lomo_cv_dir = os.path.join(experiment_dir, 'lomo_cross_validation')
    fold_files = glob.glob(os.path.join(lomo_cv_dir, 'fold_*_result.json'))
    
    # Filter out universal marker 999 if there are real folds
    if len(fold_files) > 1:
        fold_files = [f for f in fold_files if "fold_999_result.json" not in f]

    for f in fold_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                met_idx = data.get('metabolite_idx', data.get('fold_idx'))
                # Skip marker index
                if met_idx == -1 or met_idx == 999:
                    if len(fold_files) > 1: continue
                results[met_idx] = data
        except Exception:
            continue
            
    return results

def parse_folder_name(folder_name):
    """Parse metadata from folder name."""
    parts = folder_name.split('_')
    layers = "Unknown"
    arch = "Unknown"
    
    # Try to find L#
    for i, p in enumerate(parts):
        if p.startswith('L') and p[1:].isdigit():
            layers = int(p[1:])
            # Architecture is roughly everything before the layer part (excluding PREFIX)
            # PREFIX (REAL/SYNTH) is parts[0]
            arch = "_".join(parts[1:i])
            break
            
    if arch == "Unknown":
        # Fallback
        arch = folder_name
        
    return arch, layers

def process_dataset_cascade(dataset_name, candidates, base_dir, output_dir):
    """Generate cascade plot for a specific dataset (REAL or SYNTH)."""
    print(f"\nProcessing {dataset_name} results...")
    
    # Group all runs by Architecture
    arch_runs = {}
    
    for d in tqdm(candidates, desc=f"Scanning {dataset_name} experiments"):
        if not d.startswith(dataset_name): continue
        
        # Parse Arch and Layers
        arch, layers = parse_folder_name(d)
        
        # Load results to get Median R2
        try:
            res = load_per_metabolite_results(os.path.join(base_dir, d))
            if not res: continue
            
            r2s = [r['r2'] for r in res.values() if r.get('r2') is not None]
            if not r2s: continue
            
            median_r2 = np.median([x for x in r2s if not np.isnan(x)])
            
            if arch not in arch_runs:
                arch_runs[arch] = []
            
            arch_runs[arch].append({
                'dir': d,
                'median_r2': median_r2,
                'layers': layers,
                'results': res
            })
        except Exception as e:
            print(f"Skipping {d}: {e}")
            continue

    if not arch_runs:
        print(f"No valid results found for {dataset_name}.")
        return

    # Select Best Run for each Architecture
    print(f"\nBest {dataset_name} runs per architecture:")
    best_runs = {} # arch -> {dir, scores, layers}
    all_metabolites = set()
    
    for arch, runs in arch_runs.items():
        # Sort by Median R2 descending
        runs.sort(key=lambda x: x['median_r2'], reverse=True)
        best = runs[0]
        
        print(f"  {arch:<25}: Median R2 = {best['median_r2']:.4f} (Layers: {best['layers']})")
        
        # Extract per-metabolite scores
        scores = {}
        for mid, data in best['results'].items():
            if data.get('r2') is not None and not np.isnan(data['r2']):
                scores[mid] = data['r2']
                all_metabolites.add(mid)
                
        best_runs[arch] = {
            'dir': best['dir'],
            'layers': best['layers'],
            'scores': scores
        }
    
    # --- PLOT: The "Rescue" Cascade ---
    print(f"Plotting cascade for {len(best_runs)} architectures...")
    
    # Sort metabolites by Baseline difficulty (descending R2)
    # Search for any arch that is 'baseline' or starts with 'baseline'
    baseline_arch = next((a for a in best_runs.keys() if a == 'baseline' or a == 'baseline_legacy'), None)
    
    if baseline_arch:
        print(f"Using {baseline_arch} as sorting reference.")
        base_scores = best_runs[baseline_arch]['scores']
        sorted_metabolites = sorted(all_metabolites, key=lambda m: base_scores.get(m, -999), reverse=True)
    else:
        # Fallback: sort by average score across all models
        print("No baseline found. Sorting by average performance.")
        avg_scores = {}
        for m in all_metabolites:
            scores = [best_runs[arch]['scores'].get(m, -999) for arch in best_runs]
            avg_scores[m] = np.mean([s for s in scores if s > -999])
        sorted_metabolites = sorted(all_metabolites, key=lambda m: avg_scores.get(m, -999), reverse=True)

    plt.figure(figsize=(16, 9))
    
    x_indices = range(len(sorted_metabolites))
    
    # Plot Baseline first (as a filled area if present)
    if baseline_arch:
        base_scores_plot = [max(-1.5, best_runs[baseline_arch]['scores'].get(m, -1.5)) for m in sorted_metabolites]
        plt.fill_between(x_indices, base_scores_plot, color='gray', alpha=0.2, label=f'{baseline_arch} (Reference)')
        plt.plot(x_indices, base_scores_plot, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Plot other architectures
    for idx, (arch, data) in enumerate(best_runs.items()):
        if arch == 'baseline': continue # Already plotted
        
        y_scores = [max(-1.5, data['scores'].get(m, -1.5)) for m in sorted_metabolites]
        
        # Highlight MCR variants with thicker lines or specific markers
        if 'mcr' in arch:
            plt.plot(x_indices, y_scores, label=f"{arch} (L{data['layers']})", 
                     linewidth=2.5, alpha=0.9, marker='o', markersize=4)
        else:
            plt.plot(x_indices, y_scores, label=f"{arch} (L{data['layers']})", 
                     linewidth=1.5, alpha=0.6, linestyle='-')

    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel('R² Score (Clipped at -1.5)')
    plt.xlabel('Metabolites (Sorted by Baseline Difficulty)')
    plt.title(f'Performance Cascade: Comparison of All Architectures ({dataset_name})', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plot_filename = f'lomo_cascade_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    print(f"Saved {dataset_name} Cascade to {output_dir}/{plot_filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize LOMO results as a Cascade Plot.")
    parser.add_argument("--dir", type=str, default="lomo_results_array_50", help="Directory containing experiment results.")
    parser.add_argument("--output", type=str, default="results/visualizations", help="Directory to save plots.")
    args = parser.parse_args()
    
    base_dir = args.dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"LOMO Cascade Visualization")
    print(f"Source Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")
    print("="*50)
    
    if not os.path.exists(base_dir):
        print(f"Error: Source directory {base_dir} does not exist.")
        return

    candidates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not candidates:
        print(f"No experiment directories found in {base_dir}.")
        return

    # Process REAL
    process_dataset_cascade("REAL", candidates, base_dir, output_dir)
    
    # Process SYNTH
    process_dataset_cascade("SYNTH", candidates, base_dir, output_dir)

if __name__ == "__main__":
    main()
