#!/bin/bash
# Run UnitedMet (Bayesian matrix factorization) on its own ccRCC data.
# This is CPU-heavy (Pyro SVI), ~30 min per fold, ~5 hours total.
#
# Usage:
#   sbatch scripts/slurm/submit_unitedmet_external.sh          # all 10 folds
#   sbatch --array=0-9 scripts/slurm/submit_unitedmet_external.sh  # parallel folds

#SBATCH --job-name=umet_ext
#SBATCH --output=logs/umet_ext_%j_%a.out
#SBATCH --error=logs/umet_ext_%j_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "UNITEDMET ON EXTERNAL ccRCC: Job ${SLURM_JOB_ID}, Array ${SLURM_ARRAY_TASK_ID:-all}"
echo "======================================================================"

# Setup
module purge
module load ALICE/default
module load slurm
module load CUDA/12.3.2
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

mkdir -p logs

OUTPUT_DIR="results/external_baselines/unitedmet_method_rc"

# If running as array job, do single fold
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    FOLD_ARG="--fold-only ${SLURM_ARRAY_TASK_ID}"
    echo "Running fold ${SLURM_ARRAY_TASK_ID} only"
else
    FOLD_ARG=""
    echo "Running all 10 folds sequentially"
fi

python scripts/comparisons/run_unitedmet_on_external.py \
    --mudata-path data/unitedmet/unitedmet_ccRCC.h5mu \
    --output-dir "${OUTPUT_DIR}" \
    --datasets RC18 RC20 \
    --n-folds 10 \
    --n-steps 4000 \
    --n-dims 30 \
    ${FOLD_ARG}

echo ""
echo "Done. Results: ${OUTPUT_DIR}/"
