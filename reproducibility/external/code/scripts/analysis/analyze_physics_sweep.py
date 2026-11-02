#!/usr/bin/env python3
"""
Aggregate and analyze results from the Relational Physics Hyperparameter Sweep.
Identifies the optimal balance between predictive accuracy (R2) and biophysical consistency.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
import glob
from pathlib import Path

def analyze_physics_sweep(base_dir):
    print(f"Scanning for physics sweep results in: {base_dir}")
    
    # Path pattern: base_dir/seed_42/phys_bal*_cor*_var*/test_results.json
    result_files = glob.glob(os.path.join(base_dir, "seed_*", "phys_bal*", "test_results.json"))
    
    if not result_files:
        print("No result files found. Check your base_dir and folder structure.")
        return

    records = []
    for res_path in result_files:
        try:
            folder = os.path.dirname(res_path)
            config_path = os.path.join(folder, "experiment_config.json")
            
            with open(res_path, 'r') as f:
                results = json.load(f)
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract Lambdas
            phys = config.get('physics', {})
            l_bal = phys.get('lambda_balance', 0.0)
            l_corr = phys.get('lambda_correlation', 0.0)
            l_var = phys.get('lambda_variance', 0.0)
            
            # Extract Metrics
            r2 = results.get('test_r2', -1.0)
            rmse = results.get('test_rmse', 100.0)
            
            records.append({
                'Config': config.get('config_name', 'unknown'),
                'λ_bal': l_bal,
                'λ_corr': l_corr,
                'λ_var': l_var,
                'R2': r2,
                'RMSE': rmse,
                'Path': folder
            })
        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    df = pd.DataFrame(records)
    
    # Sort by R2 descending
    df = df.sort_values('R2', ascending=False)
    
    # Save CSV
    output_path = os.path.join(os.getcwd(), "physics_sweep_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to {output_path}")
    
    # Print Top 10
    print("\nTop 10 Configurations by R2:")
    print(df.head(10)[['λ_bal', 'λ_corr', 'λ_var', 'R2', 'RMSE']].to_markdown(index=False))
    
    # Find "The Sweet Spot" 
    # (High R2 but has at least some physics weighting)
    physics_active = df[(df['λ_bal'] > 0) & (df['λ_corr'] > 0)]
    if not physics_active.empty:
        best_phys = physics_active.head(1)
        print("\nOptimal Physics-Informed Configuration (The Sweet Spot):")
        print(best_phys[['λ_bal', 'λ_corr', 'λ_var', 'R2', 'RMSE']].to_markdown(index=False))
        
        # Output values for LOMO Level 5
        row = best_phys.iloc[0]
        print(f"\nRecommended flags for LOMO Level 5:")
        print(f"--lambda-bal {row['λ_bal']} --lambda-corr {row['λ_corr']} --lambda-var {row['λ_var']}")
    else:
        print("\nNo configurations found with both λ_bal and λ_corr > 0.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, 
                        default="trained_models/concat_smiles_physics",
                        help='Base directory containing sweep results')
    args = parser.parse_args()
    
    analyze_physics_sweep(args.dir)
