#!/bin/bash
# Run UnitedMet metabolite-holdout evaluation on CAMP data.
# 10-fold CV over metabolite features, one fold per array task.
#
# Prerequisites:
#   1. Clone UnitedMet: git clone https://github.com/reznik-lab/UnitedMet.git
#      into the parent directory of concat-smiles-model
#   2. Install dependencies: pip install pyro-ppl torch
#
# Usage: sbatch scripts/slurm/submit_unitedmet_comparison.sh

#SBATCH --job-name=unitedmet_cv
#SBATCH --output=logs/unitedmet_%A_%a.out
#SBATCH --error=logs/unitedmet_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --partition=cpu-skylake,cpu-zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "UnitedMet LOMO: Fold ${SLURM_ARRAY_TASK_ID}"
echo "======================================================================"

# Setup
module purge
module load ALICE/default
module load slurm
module load GCC/11.3.0

source ~/.bashrc
conda activate MetaboliteRNA

# Navigate to project root
if [ -d "/home/$USER/concat-smiles-model" ]; then
    cd /home/$USER/concat-smiles-model
elif [ -d "/data1/$USER/concat-smiles-model" ]; then
    cd /data1/$USER/concat-smiles-model
else
    cd $SLURM_SUBMIT_DIR
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p logs

# Ensure pyro-ppl is installed
python -c "import pyro" 2>/dev/null || pip install pyro-ppl

# Ensure UnitedMet is cloned
UNITEDMET_DIR="$(dirname $(pwd))/UnitedMet"
if [ ! -d "$UNITEDMET_DIR" ]; then
    echo "Cloning UnitedMet..."
    git clone https://github.com/reznik-lab/UnitedMet.git "$UNITEDMET_DIR"
fi

OUTPUT_DIR="comparison_results/unitedmet_lomo"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Fold: ${SLURM_ARRAY_TASK_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo "  UnitedMet: ${UNITEDMET_DIR}"

python scripts/comparisons/run_unitedmet_comparison.py \
    --n-folds 10 \
    --n-steps 4000 \
    --n-dims 30 \
    --n-posterior 500 \
    --max-genes 5000 \
    --fold-only ${SLURM_ARRAY_TASK_ID} \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Fold ${SLURM_ARRAY_TASK_ID} complete."
