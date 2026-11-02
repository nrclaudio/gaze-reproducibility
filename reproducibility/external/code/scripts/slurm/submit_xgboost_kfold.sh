#!/bin/bash
#SBATCH --job-name=xgb_kfold
#SBATCH --output=logs/xgboost_kfold_%j.out
#SBATCH --error=logs/xgboost_kfold_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu-zen4

module purge
module load ALICE/default
module load slurm
module load GCC/11.3.0

source ~/.bashrc
conda activate MetaboliteRNA

export PYTHONUNBUFFERED=1

PROJECT_ROOT="/home/cnovellarausell/data1/concat-smiles-model"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p logs

echo "Running XGBoost 5-fold CV baseline"
echo "Date: $(date)"

python scripts/experiments/run_kfold_xgboost_baseline.py \
    --output-dir results/modality_ablation_5fold \
    --seed 42

echo "Done: $(date)"
