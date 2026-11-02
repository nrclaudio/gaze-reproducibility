#!/bin/bash
#SBATCH --job-name=mod_masked
#SBATCH --output=logs_masked_topology/modality_%A_%a.out
#SBATCH --error=logs_masked_topology/modality_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "======================================================================"
echo "MODALITY ABLATION (MASKED TOPOLOGY) JOB: Task ${SLURM_ARRAY_TASK_ID}"
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
mkdir -p logs_masked_topology
OUTPUT_DIR="results_masked_topology/modality_ablation_5fold"
mkdir -p "$OUTPUT_DIR"

# Determine Graph Type
if [ $SLURM_ARRAY_TASK_ID -lt 6 ]; then
    GRAPH_ARG=""
    GRAPH_NAME="RealGraph"
    CONFIG_IDX=$SLURM_ARRAY_TASK_ID
else
    GRAPH_ARG="--random-graph"
    GRAPH_NAME="RandomGraph"
    CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID - 6))
fi

# Determine Features based on CONFIG_IDX (0-5)
case $CONFIG_IDX in
    0)
        FEATURE_ARGS="--randomize-genes --randomize-embeddings"
        CONFIG_NAME="SMILES_Only"
        ;;
    1)
        FEATURE_ARGS="--randomize-smiles --randomize-genes"
        CONFIG_NAME="Enzyme_Only"
        ;;
    2)
        FEATURE_ARGS="--randomize-smiles --randomize-embeddings"
        CONFIG_NAME="Genes_Only"
        ;;
    3)
        FEATURE_ARGS="--randomize-smiles"
        CONFIG_NAME="Enzyme_Genes"
        ;;
    4)
        FEATURE_ARGS=""
        CONFIG_NAME="Full_Baseline"
        ;;
    5)
        FEATURE_ARGS=""
        CONFIG_NAME="PNA_Baseline"
        ;;
esac

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
    --graph-identifier masked \
    $GRAPH_ARG \
    $FEATURE_ARGS

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete."
