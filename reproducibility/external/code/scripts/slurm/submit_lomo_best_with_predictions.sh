#!/bin/bash
# Re-run the best architecture (relational_full) with prediction saving enabled.
# This uses the current codebase which saves predictions + targets in fold results.
#
# Usage: sbatch scripts/slurm/submit_lomo_best_with_predictions.sh

#SBATCH --job-name=lomo_best_preds
#SBATCH --output=logs/lomo_best_preds_%A_%a.out
#SBATCH --error=logs/lomo_best_preds_%A_%a.err
#SBATCH --time=144:00:00
#SBATCH --array=0-9
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "LOMO BEST MODEL RE-RUN (with predictions): Task ${SLURM_ARRAY_TASK_ID}"
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

# Best architecture from N=50 waterfall: relational_full (30% positive R², median -0.34)
ARCH_CONFIG="relational_full"
N_METABOLITES=50
CHUNK_SIZE=5
TASK_ID=$SLURM_ARRAY_TASK_ID

# Physics weights (same as original run)
if [ -f "logs/best_physics_weights.json" ]; then
    L_BAL=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_bal'])")
    L_CORR=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_corr'])")
    L_VAR=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_var'])")
    echo "  Physics Params: AUTO from logs/best_physics_weights.json (Bal=${L_BAL}, Corr=${L_CORR}, Var=${L_VAR})"
else
    L_BAL=2.0
    L_CORR=0.5
    L_VAR=0.0
    echo "  Physics Params: Default (Bal=2.0, Corr=0.5)"
fi
L_KL=0.0

# Calculate metabolite range for this array task
START_IDX=$((TASK_ID * CHUNK_SIZE))
END_IDX=$((START_IDX + CHUNK_SIZE - 1))
MET_INDICES=$(seq $START_IDX $END_IDX)

OUTPUT_DIR="lomo_results_best_with_predictions/REAL_${ARCH_CONFIG}_L3_full_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Worker Configuration:"
echo "  Task ID: $TASK_ID"
echo "  Architecture: $ARCH_CONFIG"
echo "  Metabolite Indices: $MET_INDICES"
echo "  Output: $OUTPUT_DIR"

for MET_IDX in $MET_INDICES; do
    echo "  > Running Fold/Metabolite: $MET_IDX"

    python scripts/experiments/run_lomo_ablation.py \
        --arch-config "${ARCH_CONFIG}" \
        --num-layers 3 \
        --lambda-bal ${L_BAL} \
        --lambda-corr ${L_CORR} \
        --lambda-var ${L_VAR} \
        --lambda-kl ${L_KL} \
        --output-dir "${OUTPUT_DIR}" \
        --n-metabolites ${N_METABOLITES} \
        --epochs 300 \
        --learning-rate 0.001 \
        --seed 42 \
        --batch-size 4 \
        --gradient-accumulation-steps 4 \
        --fold-only ${MET_IDX}

done

echo ""
echo "All tasks in this worker complete."
