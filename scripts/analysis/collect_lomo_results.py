#!/usr/bin/env python3
"""
Collect and analyze LOMO results from parallel HPC jobs.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Headless support
import matplotlib.pyplot as plt
import seaborn as sns


def _latest_dirs(all_dirs):
    """
    For directories named REAL_{arch}_{depth}_{physics}_{JOBID}, keep only the
    most recent run (highest numeric job ID) per unique arch+depth+physics prefix.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for d in all_dirs:
        parts = d.name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
            groups[prefix].append((int(parts[1]), d))
        else:
            groups[d.name].append((0, d))  # no job ID suffix — keep as-is
    return [max(entries, key=lambda x: x[0])[1] for entries in groups.values()]


def collect_lomo_results(results_dir='lomo_results_hpc'):
    """Collect LOMO results from all completed experiments."""

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    # Find all experiment directories, then deduplicate to latest run per architecture
    all_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    experiment_dirs = _latest_dirs(all_dirs)

    print(f"Found {len(all_dirs)} experiment directories, "
          f"deduplicated to {len(experiment_dirs)} (latest run per architecture)")
    
    collected_results = []
    
    for exp_dir in experiment_dirs:
        summary_file = exp_dir / 'summary.json'
        lomo_cv_dir = exp_dir / 'lomo_cross_validation'
        
        # Scenario A: Main summary exists
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                if summary.get('success'):
                    print(f"✓ Collected results for {summary['config_name']}")
                    collected_results.append(summary)
                    continue
            except (OSError, json.JSONDecodeError) as e:
                print(f"Error reading {summary_file}: {e}")

        # Scenario B: Aggregating from individual folds (Standard Waterfall Output)
        if lomo_cv_dir.exists():
            fold_files = list(lomo_cv_dir.glob('fold_*_result.json'))
            if not fold_files:
                continue
                
            print(f"⟳ Aggregating {len(fold_files)} folds for {exp_dir.name}...")
            
            fold_results = []
            for f in fold_files:
                try:
                    with open(f, 'r') as file:
                        fold_results.append(json.load(file))
                except Exception:
                    continue

            if not fold_results:
                continue
            
            # Extract R2s and compute metrics
            r2s = [r['r2'] for r in fold_results if r.get('r2') is not None and not np.isnan(r['r2'])]
            rmses = [r['rmse'] for r in fold_results if r.get('rmse') is not None]
            
            if not r2s:
                continue
            
            # Reconstruct a summary-like object
            config_name = exp_dir.name
            l_bal, l_corr, l_var = 0.0, 0.0, 0.0
            
            try:
                config_path = exp_dir / 'config.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)

                    # Waterfall runs nest values in 'lomo_parameters',
                    # standard runs use 'physics' or top-level keys.
                    lomo_params = config_data.get('lomo_parameters', {})
                    physics = config_data.get('physics', {})

                    def _get_lambda(lomo_key, physics_key, top_key):
                        return lomo_params.get(lomo_key,
                               physics.get(physics_key,
                               config_data.get(top_key, 0.0)))

                    l_bal = _get_lambda('lambda_bal', 'lambda_balance', 'lambda_bal')
                    l_corr = _get_lambda('lambda_corr', 'lambda_correlation', 'lambda_corr')
                    l_var = _get_lambda('lambda_var', 'lambda_variance', 'lambda_var')

                    description = config_data.get('description', 'Waterfall Run')
                else:
                    description = "Reconstructed from folds"
            except Exception:
                description = "Reconstructed from folds"

            reconstructed_summary = {
                'config_name': config_name,
                'config': {
                    'description': description,
                    'lambda_bal': l_bal,
                    'lambda_corr': l_corr,
                    'lambda_var': l_var,
                },
                'lomo_summary': {
                    'mean_r2': np.mean(r2s),
                    'median_r2': np.median(r2s),
                    'std_r2': np.std(r2s),
                    'n_metabolites_tested': len(fold_results),
                    'n_metabolites_with_valid_r2': len(r2s),
                    'percentage_positive_r2': (sum(1 for x in r2s if x > 0) / len(r2s)) * 100,
                    'metabolites_with_r2_above_0.5': sum(1 for x in r2s if x > 0.5),
                    'mean_rmse': np.mean(rmses) if rmses else 0.0,
                    'median_rmse': np.median(rmses) if rmses else 0.0,
                    'raw_r2s': r2s # Store for distribution plots
                },
                'success': True
            }
            collected_results.append(reconstructed_summary)
            print(f"✓ Successfully aggregated {config_name} (λ_bal={l_bal}, λ_corr={l_corr})")
            
    print(f"\nCollected {len(collected_results)} successful experiments")
    return collected_results


def analyze_collected_results(results, output_dir='lomo_analysis'):
    """Analyze and compare collected LOMO results."""
    
    if not results:
        print("No results to analyze!")
        return
    
    # Create output directory
    analysis_dir = Path(output_dir)
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nAnalyzing {len(results)} LOMO experiments...")
    
    # Extract comparison data
    comparison_data = []
    for result in results:
        config = result['config']
        analysis = result['lomo_summary']
        
        comparison_data.append({
            'config_name': result['config_name'],
            'description': config['description'],
            'lambda_bal': config['lambda_bal'],
            'lambda_corr': config['lambda_corr'],
            'lambda_var': config['lambda_var'],
            'total_physics_weight': float(config['lambda_bal']) + float(config['lambda_corr']) + float(config['lambda_var']),
            'val_r2': config.get('val_r2', 'N/A'),
            'priority': config.get('priority', 'N/A'),
            'lomo_mean_r2': analysis['mean_r2'],
            'lomo_median_r2': analysis['median_r2'],
            'lomo_std_r2': analysis['std_r2'],
            'n_metabolites_tested': analysis['n_metabolites_tested'],
            'n_valid_results': analysis.get('n_metabolites_with_valid_r2', analysis.get('n_valid_results', 0)),
            'positive_r2_pct': analysis['percentage_positive_r2'],
            'metabolites_above_0_5': analysis['metabolites_with_r2_above_0.5'],
            'mean_rmse': analysis['mean_rmse'],
            'median_rmse': analysis['median_rmse'],
            'raw_r2s': analysis.get('raw_r2s', [])
        })
    
    # Sort by LOMO Median R² (more robust than Mean)
    comparison_data.sort(key=lambda x: x['lomo_median_r2'], reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print(f"\n{'='*120}")
    print("LOMO RESULTS COMPARISON (sorted by Median R²)")
    print(f"{'='*120}")
    print(f"{'Config':<35} {'Med R²':<8} {'Pos R²%':<8} {'LOMO R²':<10} {'RMSE':<8}")
    print(f"{'-'*120}")
    
    for data in comparison_data:
        print(f"{data['config_name']:<35} {data['lomo_median_r2']:<8.4f} {data['positive_r2_pct']:<8.1f} "
              f"{data['lomo_mean_r2']:<10.4f} {data['mean_rmse']:<8.4f}")
    
    # Save detailed comparison
    df.to_csv(analysis_dir / 'lomo_comparison.csv', index=False)
    
    # Key insights analysis
    # Find baseline and physics models using flexible naming
    baseline = next((d for d in comparison_data if 'baseline' in d['config_name']), None)
    physics_results = [d for d in comparison_data if d['total_physics_weight'] > 0]
    
    insights = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_configs_tested': len(results),
        'performance_ranking': comparison_data,
        'key_insights': []
    }
    
    print(f"\n{'='*120}")
    print("KEY INSIGHTS")
    print(f"{'='*120}")
    
    if baseline and physics_results:
        base_performance = baseline['lomo_median_r2']
        best_physics = max(physics_results, key=lambda x: x['lomo_median_r2'])
        
        improvement = best_physics['lomo_median_r2'] - base_performance
        
        insight = {
            'type': 'Baseline vs Physics Comparison',
            'baseline_median_r2': base_performance,
            'best_physics_median_r2': best_physics['lomo_median_r2'],
            'physics_improvement': improvement,
            'best_physics_config': best_physics['config_name']
        }
        insights['key_insights'].append(insight)
        
        print(f"Baseline Median R²: {base_performance:.4f}")
        print(f"Best physics model Median R²: {best_physics['lomo_median_r2']:.4f} ({best_physics['config_name']})")
        print(f"Improvement in Median R²: {improvement:+.4f}")
        
        # Add insight about % positives
        best_positives = max(comparison_data, key=lambda x: x['positive_r2_pct'])
        print(f"Highest % Positive R²: {best_positives['positive_r2_pct']:.1f}% ({best_positives['config_name']})")
    
    # Save insights
    with open(analysis_dir / 'analysis_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    # Create visualizations
    create_lomo_visualizations(df, analysis_dir)
    
    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*120}")
    print(f"Results saved to: {analysis_dir}")
    print(f"Key files:")
    print(f"  - lomo_comparison.csv: Detailed comparison table")
    print(f"  - analysis_insights.json: Key insights and rankings")
    print(f"  - median_r2_comparison.png: Bar chart of median R2")
    print(f"  - positive_r2_percentage.png: Bar chart of % positives")
    print(f"  - r2_distribution_boxplot.png: Distribution of scores")
    
    return insights


def create_lomo_visualizations(df, output_dir):
    """Create visualizations of LOMO results."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Median R² comparison (Bar Chart)
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('lomo_median_r2', ascending=True)
    plt.barh(df_sorted['config_name'], df_sorted['lomo_median_r2'], color='skyblue', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('LOMO Median R²')
    plt.title('Metabolite Prediction Performance (Median R²)')
    plt.tight_layout()
    plt.savefig(output_dir / 'median_r2_comparison.png', dpi=300)
    plt.close()
    
    # 2. Percentage of Positive R² (Bar Chart)
    plt.figure(figsize=(12, 8))
    df_sorted_pos = df.sort_values('positive_r2_pct', ascending=True)
    plt.barh(df_sorted_pos['config_name'], df_sorted_pos['positive_r2_pct'], color='salmon', alpha=0.8)
    plt.xlabel('% of Metabolites with Positive R²')
    plt.title('Generalization Success Rate')
    plt.tight_layout()
    plt.savefig(output_dir / 'positive_r2_percentage.png', dpi=300)
    plt.close()

    # 3. Distribution of R² (Boxplot)
    # Prepare long-form data for seaborn
    plot_rows = []
    for _, row in df.iterrows():
        r2s = row['raw_r2s']
        if not r2s: continue
        for val in r2s:
            # Clip very negative values for better visualization
            clipped_val = max(-2.0, val)
            plot_rows.append({'Config': row['config_name'], 'R2': clipped_val})
    
    if plot_rows:
        dist_df = pd.DataFrame(plot_rows)
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=dist_df, x='R2', y='Config', palette='viridis')
        plt.axvline(0, color='red', linestyle='--', alpha=0.6)
        plt.title('Distribution of Metabolite R² Scores (Clipped at -2.0)')
        plt.tight_layout()
        plt.savefig(output_dir / 'r2_distribution_boxplot.png', dpi=300)
        plt.close()

    # 4. Physics vs Performance
    physics_models = df[df['total_physics_weight'] > 0]
    if not physics_models.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(physics_models['total_physics_weight'], physics_models['lomo_median_r2'], s=100)
        for _, row in physics_models.iterrows():
            # Label with short name
            label = row['config_name'].replace('REAL_', '').replace('relational_', '')
            plt.annotate(label, (row['total_physics_weight'], row['lomo_median_r2']), 
                         xytext=(5,5), textcoords='offset points')
        plt.xlabel('Total Physics Weight')
        plt.ylabel('LOMO Median R²')
        plt.title('Physics Regularization Impact')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'physics_impact_scatter.png', dpi=300)
        plt.close()
    
    print("✓ Additional visualizations created")


def main():
    """Main function to collect and analyze LOMO results."""
    import argparse
    parser = argparse.ArgumentParser(description="Collect and analyze LOMO results.")
    parser.add_argument("--dir", type=str, default="lomo_results_hpc", help="Directory containing experiment results.")
    parser.add_argument("--output", type=str, default="lomo_analysis", help="Directory to save analysis results.")
    args = parser.parse_args()
    
    print(f"LOMO Results Collection and Analysis")
    print(f"Source Directory: {args.dir}")
    print(f"Output Directory: {args.output}")
    print("="*50)
    
    # Collect results
    results = collect_lomo_results(args.dir)
    
    if not results:
        print(f"No results found in {args.dir}. Make sure LOMO experiments have completed.")
        return
    
    # Analyze results
    insights = analyze_collected_results(results, output_dir=args.output)
    
    print("\n🎉 LOMO analysis complete!")


if __name__ == "__main__":
    main()
