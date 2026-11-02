#!/bin/bash
# Evaluate deployed GAZE model on external UnitedMet ccRCC cohort.
# Inference only (no training) -- fast, ~10 min.
#
# Usage: sbatch scripts/slurm/submit_external_eval.sh

#SBATCH --job-name=eval_external
#SBATCH --output=logs/eval_external_%j.out
#SBATCH --error=logs/eval_external_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "======================================================================"
echo "EXTERNAL VALIDATION (UnitedMet ccRCC): Job ${SLURM_JOB_ID}"
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

MODEL_PATH="deployed_models/relational_full/best_model.pt"
OUTPUT_DIR="results/external_unitedmet"

echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python scripts/experiments/evaluate_external.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "External evaluation complete. Results: ${OUTPUT_DIR}/"
