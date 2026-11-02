#!/bin/bash
#SBATCH --job-name=modality_gpe
#SBATCH --output=logs/modality_gpe_%A_%a.out
#SBATCH --error=logs/modality_gpe_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "GPE MODALITY ABLATION JOB: Task ${SLURM_ARRAY_TASK_ID}"
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
OUTPUT_DIR="results/modality_ablation_5fold"
mkdir -p "$OUTPUT_DIR"

# GPE Ablation Configs (3 runs):
#   0: GPE_Only        - GPE is the sole real signal (everything else randomized)
#   1: Full_with_GPE   - Full model with GPE enabled (measures GPE additive benefit)
#   2: Full_randomGPE  - Full model with GPE randomized (controls for added parameters)

case $SLURM_ARRAY_TASK_ID in
    0)
        FEATURE_ARGS="--use-gpe --randomize-smiles --randomize-genes --randomize-embeddings --blindfold-prob 1.0"
        CONFIG_NAME="RealGraph_GPE_Only"
        ;;
    1)
        FEATURE_ARGS="--use-gpe --blindfold-prob 0.5 --use-residuals --use-attention"
        CONFIG_NAME="RealGraph_Full_with_GPE"
        ;;
    2)
        FEATURE_ARGS="--use-gpe --randomize-gpe --blindfold-prob 0.5 --use-residuals --use-attention"
        CONFIG_NAME="RealGraph_Full_randomGPE"
        ;;
esac

echo "Running Configuration: $CONFIG_NAME"
echo "Feature Flags: $FEATURE_ARGS"

python scripts/experiments/run_kfold_modality_sweep.py \
    --config-name "${CONFIG_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 200 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 0.001 \
    --seed 42 \
    $FEATURE_ARGS

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete."
