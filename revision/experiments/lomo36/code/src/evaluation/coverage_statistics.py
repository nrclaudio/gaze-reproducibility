"""Coverage statistics for metabolite predictions to assess evaluation completeness."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict


def compute_coverage_statistics(dataset, test_indices, model_dir):
    """
    Compute comprehensive coverage statistics for test set evaluation.
    
    Args:
        dataset: Full dataset
        test_indices: Indices of test samples
        model_dir: Directory to save results
    
    Returns:
        Dictionary with coverage statistics
    """
    
    # Create output directory
    coverage_dir = os.path.join(model_dir, 'coverage_statistics')
    os.makedirs(coverage_dir, exist_ok=True)
    
    # Collect metabolite statistics across entire dataset
    all_metabolite_presence = defaultdict(list)  # metabolite_idx -> list of sample indices
    test_metabolite_presence = defaultdict(list)
    train_metabolite_presence = defaultdict(list)
    
    # Total number of metabolites in the graph
    total_metabolites = 0
    
    # Analyze entire dataset
    for idx in range(len(dataset)):
        data = dataset[idx]
        mask = data.mask.cpu().numpy() if hasattr(data, 'mask') else np.zeros(len(data.y), dtype=bool)
        
        # Update total metabolites count
        if total_metabolites == 0:
            # Get total number of metabolites from the graph structure
            if hasattr(dataset, 'metabolite_indices'):
                total_metabolites = len(dataset.metabolite_indices)
            else:
                # Fallback: estimate from mask size
                total_metabolites = len(mask)
        
        # Track which metabolites are present in this sample
        for met_idx in np.where(mask)[0]:
            all_metabolite_presence[met_idx].append(idx)
            
            if idx in test_indices:
                test_metabolite_presence[met_idx].append(idx)
            else:
                train_metabolite_presence[met_idx].append(idx)
    
    # Calculate coverage statistics
    stats = {
        'total_metabolites_in_graph': total_metabolites,
        'metabolites_with_any_measurement': len(all_metabolite_presence),
        'metabolites_in_test_set': len(test_metabolite_presence),
        'metabolites_only_in_train': len(set(train_metabolite_presence.keys()) - set(test_metabolite_presence.keys())),
        'metabolites_only_in_test': len(set(test_metabolite_presence.keys()) - set(train_metabolite_presence.keys())),
        'metabolites_never_measured': total_metabolites - len(all_metabolite_presence),
        'test_coverage_percentage': (len(test_metabolite_presence) / total_metabolites * 100) if total_metabolites > 0 else 0,
        'train_coverage_percentage': (len(train_metabolite_presence) / total_metabolites * 100) if total_metabolites > 0 else 0,
    }
    
    # Sample count distribution
    test_sample_counts = [len(samples) for samples in test_metabolite_presence.values()]
    train_sample_counts = [len(samples) for samples in train_metabolite_presence.values()]
    all_sample_counts = [len(samples) for samples in all_metabolite_presence.values()]
    
    if test_sample_counts:
        stats['test_samples_per_metabolite'] = {
            'mean': np.mean(test_sample_counts),
            'median': np.median(test_sample_counts),
            'min': np.min(test_sample_counts),
            'max': np.max(test_sample_counts),
            'std': np.std(test_sample_counts)
        }
    
    if train_sample_counts:
        stats['train_samples_per_metabolite'] = {
            'mean': np.mean(train_sample_counts),
            'median': np.median(train_sample_counts),
            'min': np.min(train_sample_counts),
            'max': np.max(train_sample_counts),
            'std': np.std(train_sample_counts)
        }
    
    # Distribution of metabolites by sample frequency
    frequency_bins = [1, 5, 10, 20, 50, 100, 200, 500]
    test_frequency_dist = _compute_frequency_distribution(test_sample_counts, frequency_bins)
    train_frequency_dist = _compute_frequency_distribution(train_sample_counts, frequency_bins)
    all_frequency_dist = _compute_frequency_distribution(all_sample_counts, frequency_bins)
    
    stats['test_frequency_distribution'] = test_frequency_dist
    stats['train_frequency_distribution'] = train_frequency_dist
    stats['all_frequency_distribution'] = all_frequency_dist
    
    # Identify metabolites with poor test coverage
    poorly_covered = []
    well_covered = []
    never_tested = []
    
    for met_idx in all_metabolite_presence.keys():
        train_count = len(train_metabolite_presence.get(met_idx, []))
        test_count = len(test_metabolite_presence.get(met_idx, []))
        total_count = len(all_metabolite_presence[met_idx])
        
        if test_count == 0:
            never_tested.append({
                'metabolite_idx': met_idx,
                'train_samples': train_count,
                'total_samples': total_count
            })
        elif test_count < 5 and train_count >= 20:
            poorly_covered.append({
                'metabolite_idx': met_idx,
                'train_samples': train_count,
                'test_samples': test_count,
                'test_train_ratio': test_count / train_count
            })
        elif test_count >= 20:
            well_covered.append({
                'metabolite_idx': met_idx,
                'train_samples': train_count,
                'test_samples': test_count,
                'test_train_ratio': test_count / train_count if train_count > 0 else float('inf')
            })
    
    stats['never_tested_metabolites'] = len(never_tested)
    stats['poorly_covered_metabolites'] = len(poorly_covered)
    stats['well_covered_metabolites'] = len(well_covered)
    
    # Save detailed lists
    if never_tested:
        pd.DataFrame(never_tested).to_csv(
            os.path.join(coverage_dir, 'never_tested_metabolites.csv'), index=False
        )
    
    if poorly_covered:
        pd.DataFrame(poorly_covered).to_csv(
            os.path.join(coverage_dir, 'poorly_covered_metabolites.csv'), index=False
        )
    
    if well_covered:
        pd.DataFrame(well_covered).to_csv(
            os.path.join(coverage_dir, 'well_covered_metabolites.csv'), index=False
        )
    
    # Create visualizations
    _create_coverage_plots(stats, test_sample_counts, train_sample_counts, all_sample_counts, coverage_dir)
    
    # Save summary report
    _save_coverage_report(stats, coverage_dir)
    
    return stats


def _compute_frequency_distribution(sample_counts, bins):
    """Compute distribution of metabolites across frequency bins."""
    if not sample_counts:
        return {}
    
    distribution = {}
    for i in range(len(bins)):
        if i == 0:
            lower = 0
            upper = bins[i]
            label = f"1-{bins[i]}"
        elif i == len(bins) - 1:
            lower = bins[i-1]
            upper = float('inf')
            label = f"{bins[i-1]}+"
        else:
            lower = bins[i-1]
            upper = bins[i]
            label = f"{bins[i-1]+1}-{bins[i]}"
        
        count = sum(1 for c in sample_counts if lower < c <= upper)
        distribution[label] = count
    
    return distribution


def _create_coverage_plots(stats, test_counts, train_counts, all_counts, output_dir):
    """Create comprehensive coverage visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Coverage overview pie chart
    coverage_data = [
        stats['metabolites_in_test_set'],
        stats['metabolites_only_in_train'],
        stats['metabolites_never_measured']
    ]
    coverage_labels = ['In Test Set', 'Only in Train', 'Never Measured']
    
    axes[0, 0].pie(coverage_data, labels=coverage_labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Metabolite Coverage Overview')
    
    # 2. Sample count distributions
    if test_counts:
        axes[0, 1].hist(test_counts, bins=30, alpha=0.7, label='Test', edgecolor='black')
    if train_counts:
        axes[0, 1].hist(train_counts, bins=30, alpha=0.7, label='Train', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Samples')
    axes[0, 1].set_ylabel('Number of Metabolites')
    axes[0, 1].set_title('Sample Count Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # 3. Frequency distribution comparison
    if stats['test_frequency_distribution'] and stats['train_frequency_distribution']:
        labels = list(stats['test_frequency_distribution'].keys())
        test_values = list(stats['test_frequency_distribution'].values())
        train_values = list(stats['train_frequency_distribution'].values())
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, test_values, width, label='Test', alpha=0.8)
        axes[0, 2].bar(x + width/2, train_values, width, label='Train', alpha=0.8)
        axes[0, 2].set_xlabel('Sample Count Range')
        axes[0, 2].set_ylabel('Number of Metabolites')
        axes[0, 2].set_title('Metabolite Frequency Distribution')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 2].legend()
    
    # 4. Coverage quality assessment
    coverage_quality = [
        stats['well_covered_metabolites'],
        stats['poorly_covered_metabolites'],
        stats['never_tested_metabolites']
    ]
    quality_labels = ['Well Covered\n(≥20 test samples)', 
                     'Poorly Covered\n(<5 test samples)', 
                     'Never Tested']
    
    axes[1, 0].bar(quality_labels, coverage_quality, color=['green', 'orange', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Number of Metabolites')
    axes[1, 0].set_title('Test Coverage Quality')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Cumulative coverage curve
    if all_counts:
        sorted_counts = sorted(all_counts, reverse=True)
        cumsum = np.cumsum(sorted_counts)
        total_samples = cumsum[-1]
        
        axes[1, 1].plot(range(len(sorted_counts)), cumsum / total_samples * 100)
        axes[1, 1].set_xlabel('Number of Metabolites')
        axes[1, 1].set_ylabel('Cumulative % of Samples')
        axes[1, 1].set_title('Cumulative Sample Coverage')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add reference lines
        axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
        axes[1, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80%')
        axes[1, 1].axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95%')
        axes[1, 1].legend()
    
    # 6. Test vs Train sample correlation
    if test_counts and train_counts:
        metabolite_comparison = []
        for met_idx in set(list(range(len(test_counts))) + list(range(len(train_counts)))):
            if met_idx < len(test_counts) and met_idx < len(train_counts):
                metabolite_comparison.append((train_counts[met_idx], test_counts[met_idx]))
        
        if metabolite_comparison:
            train_vals, test_vals = zip(*metabolite_comparison)
            axes[1, 2].scatter(train_vals, test_vals, alpha=0.5)
            axes[1, 2].set_xlabel('Train Samples')
            axes[1, 2].set_ylabel('Test Samples')
            axes[1, 2].set_title('Train vs Test Sample Distribution')
            
            # Add diagonal reference line
            max_val = max(max(train_vals), max(test_vals))
            axes[1, 2].plot([0, max_val], [0, max_val * 0.3], 'r--', alpha=0.5, label='30% ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _save_coverage_report(stats, output_dir):
    """Save comprehensive coverage report."""
    
    report_path = os.path.join(output_dir, 'coverage_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("METABOLITE COVERAGE STATISTICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL COVERAGE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total metabolites in graph: {stats['total_metabolites_in_graph']}\n")
        f.write(f"Metabolites with any measurement: {stats['metabolites_with_any_measurement']}\n")
        f.write(f"Metabolites in test set: {stats['metabolites_in_test_set']}\n")
        f.write(f"Test coverage: {stats['test_coverage_percentage']:.1f}%\n")
        f.write(f"Train coverage: {stats['train_coverage_percentage']:.1f}%\n\n")
        
        f.write("COVERAGE GAPS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Metabolites never measured: {stats['metabolites_never_measured']}\n")
        f.write(f"Metabolites only in train (never tested): {stats['metabolites_only_in_train']}\n")
        f.write(f"Metabolites only in test (never trained): {stats['metabolites_only_in_test']}\n\n")
        
        f.write("COVERAGE QUALITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Well covered metabolites (≥20 test samples): {stats['well_covered_metabolites']}\n")
        f.write(f"Poorly covered metabolites (<5 test samples): {stats['poorly_covered_metabolites']}\n")
        f.write(f"Never tested metabolites: {stats['never_tested_metabolites']}\n\n")
        
        if 'test_samples_per_metabolite' in stats:
            f.write("TEST SAMPLE STATISTICS\n")
            f.write("-" * 40 + "\n")
            test_stats = stats['test_samples_per_metabolite']
            f.write(f"Mean samples per metabolite: {test_stats['mean']:.1f}\n")
            f.write(f"Median samples per metabolite: {test_stats['median']:.1f}\n")
            f.write(f"Min samples: {test_stats['min']}\n")
            f.write(f"Max samples: {test_stats['max']}\n")
            f.write(f"Std dev: {test_stats['std']:.1f}\n\n")
        
        if 'test_frequency_distribution' in stats:
            f.write("TEST FREQUENCY DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for range_label, count in stats['test_frequency_distribution'].items():
                f.write(f"{range_label} samples: {count} metabolites\n")
            f.write("\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if stats['metabolites_never_measured'] > 0:
            f.write(f"⚠️  {stats['metabolites_never_measured']} metabolites have no measurements - "
                   "consider if these should be removed from the graph\n")
        
        if stats['metabolites_only_in_train'] > 10:
            f.write(f"⚠️  {stats['metabolites_only_in_train']} metabolites are never tested - "
                   "consider cross-metabolite validation for these\n")
        
        if stats['poorly_covered_metabolites'] > stats['well_covered_metabolites']:
            f.write("⚠️  More metabolites are poorly covered than well covered - "
                   "test metrics may not be representative\n")
        
        if stats['test_coverage_percentage'] < 50:
            f.write(f"⚠️  Only {stats['test_coverage_percentage']:.1f}% of metabolites are tested - "
                   "consider alternative evaluation strategies\n")