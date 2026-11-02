# Scripts Directory

This directory contains all executable scripts organized by purpose.

## Structure

- **`training/`** - Model training scripts
- **`analysis/`** - Data analysis and visualization scripts  
- **`experiments/`** - Experiment runners and configuration generators
- **`slurm/`** - HPC submission scripts

## Usage

All scripts should be run from the repository root directory to ensure proper imports:

```bash
# From repository root
python scripts/training/train_concat_smiles.py --args
python scripts/analysis/extract_metrics_from_logs.py --args
./scripts/slurm/submit_enhanced_ablation.sh
```

## Training Scripts

- **`train_concat_smiles.py`** - Main training script with full configuration options
- **`train_enhanced.py`** - Enhanced model training with architectural improvements
- **`train_baselines.py`** - Baseline model training for comparisons

## Analysis Scripts

- **`extract_metrics_from_logs.py`** - Extract performance metrics from log files
- **`aggregate_ablation_results.py`** - Combine results from ablation studies
- **`analyze_sample_size_stratified.py`** - Sample size analysis
- **`visualize_physics_sweep_3d.py`** - 3D visualization of physics loss sweeps
- **`collect_lomo_results.py`** - Collect LOMO optimization results

## Experiment Scripts

- **`run_ablation_config.py`** - Run individual ablation study configurations
- **`run_single_lomo_config.py`** - Run single LOMO optimization configuration
- **`generate_physics_loss_sweep_configs.py`** - Generate physics loss sweep configurations

## SLURM Scripts

- **`submit_enhanced_ablation.sh`** - Submit enhanced model ablation study
- **`submit_lomo_phase1.sh`** - Submit LOMO phase 1 jobs