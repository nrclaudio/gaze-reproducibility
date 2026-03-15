#!/bin/bash
#SBATCH --job-name=baseline_mcr_array
#SBATCH --output=logs/baseline_mcr_%A_%a.out
#SBATCH --error=logs/baseline_mcr_%A_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

# SLURM Job Array for Baseline vs MCR LOMO Comparison
# Array index 0: baseline
# Array index 1: mcr

set -euo pipefail

PROJECT_ROOT="/home/cnovellarausell/data1/concat-smiles-model"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load ALICE/default || true
  module load slurm || true
  module load CUDA/12.3.2 || true
  module load GCC/11.3.0 || true
fi

if [ -f "${HOME}/.bashrc" ]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

if command -v conda >/dev/null 2>&1; then
  conda activate MetaboliteRNA || true
fi

export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}" || exit 1
mkdir -p logs

echo "Running on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

echo "======================================================================"
echo "BASELINE vs MCR LOMO - JOB ARRAY"
echo "======================================================================"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Configuration parameters for 10-fold LOMO
N_METABOLITES=10
EPOCHS=300
LEARNING_RATE=0.00075
SEED=2

# Define configurations for each array index
declare -a CONFIGS=("baseline" "mcr")
declare -a DESCRIPTIONS=("Baseline Model (Traditional MLP Head)" "Metabolite-Conditioned Reader (Attention-Based Head)")

# Get configuration for this array task
CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
DESCRIPTION=${DESCRIPTIONS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "RUNNING: $DESCRIPTION"
echo "Config: $CONFIG_NAME"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "======================================================================"
echo "Configuration:"
echo "  N_METABOLITES: $N_METABOLITES"
echo "  EPOCHS: $EPOCHS"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  LAMBDA_BAL: 0.0 (pure readout head comparison)"
echo "  SEED: $SEED"
echo ""

# Run LOMO cross-validation
echo "Starting LOMO cross-validation for $CONFIG_NAME..."
echo "Start time: $(date)"

python scripts/experiments/run_lomo_ablation.py \
    --arch-config "$CONFIG_NAME" \
    --lambda-bal 0.0 \
    --lambda-corr 0.0 \
    --lambda-var 0.0 \
    --lambda-kl 0.0 \
    --output-dir lomo_results_array \
    --n-metabolites $N_METABOLITES \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --seed $SEED \
    --device auto

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "TASK COMPLETION SUMMARY"
echo "======================================================================"
echo "Config: $CONFIG_NAME ($DESCRIPTION)"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: $CONFIG_NAME completed successfully"
    echo ""
    echo "Results saved to: lomo_results_array/architecture_${CONFIG_NAME}_*/"
else
    echo "❌ FAILED: $CONFIG_NAME failed with exit code $EXIT_CODE"
fi

echo "======================================================================"

exit $EXIT_CODE
