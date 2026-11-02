#!/bin/bash
# Run MEBOCOST and scCellFie on external UnitedMet ccRCC cohort.
# CPU only (no GPU needed for knowledge-based methods).
#
# Usage: sbatch scripts/slurm/submit_external_baselines.sh

#SBATCH --job-name=ext_baselines
#SBATCH --output=logs/ext_baselines_%j.out
#SBATCH --error=logs/ext_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "EXTERNAL BASELINES (MEBOCOST + scCellFie on UnitedMet): Job ${SLURM_JOB_ID}"
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

mkdir -p logs

echo "Running MEBOCOST and scCellFie on UnitedMet external cohort..."
echo ""

python scripts/comparisons/run_baselines_on_external.py \
    --mudata-path data/unitedmet/unitedmet_ccRCC.h5mu \
    --output-dir results/external_baselines/

echo ""
echo "Done."
