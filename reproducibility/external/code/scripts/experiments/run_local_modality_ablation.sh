#!/bin/bash

# Local runner for Modality Ablation Study
# Simulates the SLURM array job locally

OUTPUT_DIR="results/modality_ablation_5fold"
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "STARTING LOCAL MODALITY ABLATION SWEEP (12 CONFIGURATIONS)"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================"

for TASK_ID in {0..11}; do
    echo "======================================================================"
    echo "RUNNING TASK ${TASK_ID}/11"
    echo "======================================================================"

    # Determine Graph Type
    if [ $TASK_ID -lt 6 ]; then
        GRAPH_ARG=""
        GRAPH_NAME="RealGraph"
        CONFIG_IDX=$TASK_ID
    else
        GRAPH_ARG="--random-graph"
        GRAPH_NAME="RandomGraph"
        CONFIG_IDX=$(($TASK_ID - 6))
    fi

    # Determine Features based on CONFIG_IDX (0-5)
    case $CONFIG_IDX in
        0) 
            # SMILES Only (Randomize Enz, Genes)
            FEATURE_ARGS="--randomize-genes --randomize-embeddings"
            CONFIG_NAME="SMILES_Only"
            ;;
        1) 
            # Enzyme Only (Randomize SMILES, Genes)
            FEATURE_ARGS="--randomize-smiles --randomize-genes"
            CONFIG_NAME="Enzyme_Only"
            ;;
        2) 
            # Genes Only (Randomize SMILES, Enz)
            FEATURE_ARGS="--randomize-smiles --randomize-embeddings"
            CONFIG_NAME="Genes_Only"
            ;;
        3) 
            # Enz + Genes (Randomize SMILES)
            FEATURE_ARGS="--randomize-smiles"
            CONFIG_NAME="Enzyme_Genes"
            ;;
        4) 
            # Full Baseline (All Real)
            FEATURE_ARGS=""
            CONFIG_NAME="Full_Baseline"
            ;;
        5) 
            # PNA Baseline (All Real + PNA Layer)
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
        $GRAPH_ARG \
        $FEATURE_ARGS

    echo ""
    echo "Task ${TASK_ID} complete."
    echo "----------------------------------------------------------------------"
done

echo "ALL TASKS COMPLETED."
