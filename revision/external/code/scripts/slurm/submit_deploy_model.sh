#!/bin/bash
# Two-phase deployment model training (relational_full + detection gate).
# Phase 1: find optimal epoch with 90/10 val split.
# Phase 2: retrain on ALL data for exactly that many epochs.
# Saves best_model.pt, gene_stats.pt, and experiment_config.json.
#
# Usage: sbatch scripts/slurm/submit_deploy_model.sh

#SBATCH --job-name=deploy_model
#SBATCH --output=logs/deploy_model_%j.out
#SBATCH --error=logs/deploy_model_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "DEPLOYMENT MODEL TRAINING: Job ${SLURM_JOB_ID}"
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

export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p logs

# Resolve physics weights from best_physics_weights.json (same pattern as LOMO scripts)
if [ -f "logs/best_physics_weights.json" ]; then
    L_BAL=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_bal'])")
    L_CORR=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_corr'])")
    L_VAR=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_var'])")
    echo "Physics weights: AUTO from logs/best_physics_weights.json (bal=${L_BAL}, corr=${L_CORR}, var=${L_VAR})"
else
    L_BAL=2.0
    L_CORR=0.5
    L_VAR=0.0
    echo "Physics weights: defaults (bal=2.0, corr=0.5, var=0.0)"
fi

OUTPUT_DIR="deployed_models/relational_full"

echo "Output directory: ${OUTPUT_DIR}"
echo ""

python scripts/experiments/train_deploy_model.py \
    --phase1-epochs 300 \
    --phase1-patience 20 \
    --learning-rate 0.001 \
    --seed 42 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --output-dir "${OUTPUT_DIR}" \
    --mask-target-smiles 0.5 \
    --lambda-bal ${L_BAL} \
    --lambda-corr ${L_CORR} \
    --lambda-var ${L_VAR} \
    --num-layers 3

echo ""
echo "Deployment training complete."
echo "Artifacts: ${OUTPUT_DIR}/{best_model.pt, gene_stats.pt, experiment_config.json}"
