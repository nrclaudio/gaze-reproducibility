#!/bin/bash
#SBATCH --job-name=modality_ablation
#SBATCH --output=logs/modality_%A_%a.out
#SBATCH --error=logs/modality_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "MODALITY ABLATION JOB: Task ${SLURM_ARRAY_TASK_ID}"
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

# Define configurations
# 0-3: Real Graph (Standard)
# 4-7: Random Graph (Standard)
# 8-9: Real Graph (Hard Mode - Mean Subtraction)
# 10-11: Random Graph (Hard Mode - Mean Subtraction)

HARD_MODE=""
if [ $SLURM_ARRAY_TASK_ID -ge 8 ]; then
    HARD_MODE="--subtract-sample-mean"
fi

if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    GRAPH_ARG=""
    GRAPH_NAME="RealGraph"
    CONFIG_IDX=$SLURM_ARRAY_TASK_ID
elif [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    GRAPH_ARG="--random-graph"
    GRAPH_NAME="RandomGraph"
    CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID - 4))
elif [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
    # Real Hard Mode (Indices 8, 9 map to Configs 2, 3)
    GRAPH_ARG=""
    GRAPH_NAME="RealGraph_Hard"
    CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID - 6)) 
else
    # Random Hard Mode (Indices 10, 11 map to Configs 2, 3)
    GRAPH_ARG="--random-graph"
    GRAPH_NAME="RandomGraph_Hard"
    CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID - 8))
fi

# Determine Features based on CONFIG_IDX
case $CONFIG_IDX in
    0)
        FEATURE_ARGS="--randomize-genes --randomize-embeddings --blindfold-prob 0.0 --use-residuals --use-attention"
        CONFIG_NAME="SMILES_Only"
        ;;
    1)
        FEATURE_ARGS="--randomize-smiles --randomize-embeddings --blindfold-prob 0.0 --use-residuals --use-attention"
        CONFIG_NAME="Genes_Only"
        ;;
    2)
        FEATURE_ARGS="--blindfold-prob 1.0"
        CONFIG_NAME="Graph_Only"
        ;;
    3)
        FEATURE_ARGS="--blindfold-prob 0.5 --use-residuals --use-attention"
        CONFIG_NAME="Full_Model"
        ;;
esac

# Append Hard Mode flag
FEATURE_ARGS="$FEATURE_ARGS $HARD_MODE"
FULL_CONFIG_NAME="${GRAPH_NAME}_${CONFIG_NAME}"

echo "Running Configuration: $FULL_CONFIG_NAME"
echo "Graph: $GRAPH_NAME"
echo "Feature Flags: $FEATURE_ARGS"

python scripts/experiments/run_kfold_modality_sweep.py \
    --config-name "${FULL_CONFIG_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 200 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 0.001 \
    --seed 42 \
    $GRAPH_ARG \
    $FEATURE_ARGS

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete."
