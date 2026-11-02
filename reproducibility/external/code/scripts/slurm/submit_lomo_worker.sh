#!/bin/bash
#SBATCH --job-name=lomo_worker
#SBATCH --output=logs/lomo_worker_%A_%a.out
#SBATCH --error=logs/lomo_worker_%A_%a.err
#SBATCH --time=144:00:00
#SBATCH --array=0-9
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ^^^ NOTE: --array=0-9 generates 10 jobs indices (CHUNK_SIZE=5 for 50 metabolites)

echo "======================================================================"
echo "LOMO WORKER: Task ${SLURM_ARRAY_TASK_ID}"
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

# Configuration
# Accept config name from command line argument, default to "all_features"
ARCH_CONFIG=${1:-"all_features"}
PHYSICS_MODE=${2:-"full"}
EXTRA_ARGS=${3:-""}
LAYER_INPUT=${4:-3}

# Parse Layers
if [ "$LAYER_INPUT" == "ALL" ]; then
    LAYERS=(3 4 5)
else
    # Split by space (treat as array)
    IFS=' ' read -r -a LAYERS <<< "$LAYER_INPUT"
fi

# Determine folder prefix based on dropout flag
if [[ "$EXTRA_ARGS" == *"--enable-metabolite-dropout"* ]]; then
    PREFIX="SYNTH"
else
    PREFIX="REAL"
fi

N_METABOLITES=50
CHUNK_SIZE=5
TASK_ID=$SLURM_ARRAY_TASK_ID

# Determine Metabolite Indices (Folds) to run
if [ "$TASK_ID" -eq 999 ]; then
    # Synthetic / Single run case
    MET_INDICES="999"
else
    # Calculate range
    START_IDX=$((TASK_ID * CHUNK_SIZE))
    END_IDX=$((START_IDX + CHUNK_SIZE - 1))
    # Generate sequence
    MET_INDICES=$(seq $START_IDX $END_IDX)
fi

echo "Worker Configuration:"
echo "  Task ID: $TASK_ID"
echo "  Architecture: $ARCH_CONFIG"
echo "  Physics Mode: $PHYSICS_MODE"
echo "  Layers to run: ${LAYERS[*]}"
echo "  Metabolite Indices: ${MET_INDICES[*]}"

# Set physics parameters based on mode
if [ "$PHYSICS_MODE" == "none" ]; then
    L_BAL=0.0
    L_CORR=0.0
    L_VAR=0.0
    L_KL=0.0
    echo "  Physics Params: NONE (MSE Only)"
else
    # Check if EXTRA_ARGS contains --override-physics-weights (set by orchestrator)
    if [[ "$EXTRA_ARGS" == *"--override-physics-weights"* ]]; then
        # Parse: --override-physics-weights <bal> <corr> <var>
        OPW=$(echo "$EXTRA_ARGS" | grep -oE '\-\-override-physics-weights [0-9.]+ [0-9.]+ [0-9.]+' | sed 's/--override-physics-weights //')
        L_BAL=$(echo "$OPW" | awk '{print $1}')
        L_CORR=$(echo "$OPW" | awk '{print $2}')
        L_VAR=$(echo "$OPW" | awk '{print $3}')
        L_KL=0.0
        # Strip the override flag from EXTRA_ARGS so it doesn't get passed downstream
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--override-physics-weights [0-9.]* [0-9.]* [0-9.]*//')
        echo "  Physics Params: OVERRIDE from sweep (Bal=${L_BAL}, Corr=${L_CORR}, Var=${L_VAR})"
    elif [ -f "logs/best_physics_weights.json" ]; then
        # Auto-read from physics sweep analysis output
        L_BAL=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_bal'])")
        L_CORR=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_corr'])")
        L_VAR=$(python -c "import json; print(json.load(open('logs/best_physics_weights.json'))['lambda_var'])")
        L_KL=0.0
        echo "  Physics Params: AUTO from logs/best_physics_weights.json (Bal=${L_BAL}, Corr=${L_CORR}, Var=${L_VAR})"
    else
        # Default: Values from Relational Physics Sweep Winner
        L_BAL=2.0
        L_CORR=0.5
        L_VAR=0.0
        L_KL=0.0
        echo "  Physics Params: FULL (Default Weights: Bal=2.0, Corr=0.5)"
    fi
fi

# Loop over Layers
for NUM_LAYERS in "${LAYERS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "PROCESSING LAYERS: $NUM_LAYERS"
    
    # Define local args for this layer count
    CURRENT_EXTRA_ARGS="$EXTRA_ARGS"
    
    # AUTOMATIC RESIDUALS FOR DEEP NETWORKS
    # If layers >= 4 and not explicitly "baseline", force residuals to prevent collapse
    if [ "$NUM_LAYERS" -ge 4 ] && [ "$ARCH_CONFIG" != "baseline" ]; then
        echo "  > Deep network detected ($NUM_LAYERS layers): Forcing Residuals + LayerNorm"
        CURRENT_EXTRA_ARGS="$CURRENT_EXTRA_ARGS --force-residuals"
    fi
    
    # Include layers in directory name for easy comparison, parent reflects count
    OUTPUT_DIR="lomo_results_final_waterfall_N${N_METABOLITES}/${PREFIX}_${ARCH_CONFIG}_L${NUM_LAYERS}_${PHYSICS_MODE}_${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$OUTPUT_DIR"
    
    # Loop over Metabolites (Folds)
    for MET_IDX in $MET_INDICES; do
        echo "  > Running Fold/Metabolite: $MET_IDX"

        python scripts/experiments/run_lomo_ablation.py \
            --arch-config "${ARCH_CONFIG}" \
            --num-layers ${NUM_LAYERS} \
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
            --fold-only ${MET_IDX} \
            ${CURRENT_EXTRA_ARGS}
            
        # === SYNTHETIC EVALUATION TRIGGER ===
        # If this is a synthetic run (Universal Model, Index 999), run the evaluation immediately
        if [ "$MET_IDX" -eq 999 ]; then
            echo "  > Synthetic LOMO Training Complete. Starting Evaluation..."
            
            # The model is saved in OUTPUT_DIR/lomo_cross_validation/best_model.pt
            # (Note: run_lomo_ablation -> run_lomo_cross_validation -> saves to lomo_dir)
            MODEL_PATH="${OUTPUT_DIR}/lomo_cross_validation/best_model.pt"
            EVAL_OUTPUT_DIR="${OUTPUT_DIR}/zero_shot_evaluation"
            
            if [ -f "$MODEL_PATH" ]; then
                python scripts/experiments/evaluate_synthetic_lomo.py \
                    --model-path "$MODEL_PATH" \
                    --output-dir "$EVAL_OUTPUT_DIR" \
                    --device auto \
                    --n-metabolites 50
                
                echo "  ✅ Zero-Shot Evaluation Complete. Results in: $EVAL_OUTPUT_DIR"
            else
                echo "  ❌ Model file not found at $MODEL_PATH. Evaluation skipped."
            fi
        fi
            
    done
done

echo ""
echo "All tasks in this worker complete."
