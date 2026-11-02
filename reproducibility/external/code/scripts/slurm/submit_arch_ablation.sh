#!/bin/bash
#SBATCH --job-name=arch_ablation
#SBATCH --output=logs/arch_ab_%A_%a.out
#SBATCH --error=logs/arch_ab_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "ARCHITECTURE ABLATION JOB: Task ${SLURM_ARRAY_TASK_ID}"
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
OUTPUT_DIR="results/arch_ablation_5fold"
mkdir -p "$OUTPUT_DIR"

# Logic: 0-3 Real, 4-7 Random
if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    GRAPH_ARG=""
    GRAPH_NAME="RealGraph"
    CONFIG_IDX=$SLURM_ARRAY_TASK_ID
else
    GRAPH_ARG="--random-graph"
    GRAPH_NAME="RandomGraph"
    CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID - 4))
fi

# Determine Architectural Flags
# 0: Base, 1: +Res, 2: +Attn, 3: +Both
case $CONFIG_IDX in
    0)
        ARCH_FLAGS=""
        ARCH_NAME="Base"
        ;;
    1)
        ARCH_FLAGS="--use-residuals"
        ARCH_NAME="PlusRes"
        ;;
    2)
        ARCH_FLAGS="--use-attention"
        ARCH_NAME="PlusAttn"
        ;;
    3)
        ARCH_FLAGS="--use-residuals --use-attention"
        ARCH_NAME="PlusBoth"
        ;;
esac

# Common: Graph Only (100% Blindfold)
# We use standard mode (no MeanSub) to see the pure architectural effect
COMMON_ARGS="--blindfold-prob 1.0"

FULL_CONFIG_NAME="Arch_${GRAPH_NAME}_${ARCH_NAME}"

echo "Running Configuration: $FULL_CONFIG_NAME"
echo "Architecture: $ARCH_NAME"
echo "Flags: $ARCH_FLAGS $COMMON_ARGS"

python scripts/experiments/run_kfold_modality_sweep.py \
    --config-name "${FULL_CONFIG_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 200 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 0.001 \
    --seed 42 \
    $GRAPH_ARG \
    $COMMON_ARGS \
    $ARCH_FLAGS

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete."
