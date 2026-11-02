#!/bin/bash

# Submission script for ALL architectures using the corrected components:
# 1. ResidualGNNLayer (LayerNorm added)
# 2. CrossModalAttention (Local sparse masking added)
# 3. Automatic Residuals for layers 4 & 5 (except baseline)
#
# Usage: ./scripts/slurm/submit_fixed_architecture_jobs.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKER_SCRIPT="${SCRIPT_DIR}/submit_lomo_worker.sh"

# ALL model configurations (EXCLUDING baseline as it was already run and is unaffected by auto-residuals)
CONFIGS=(
    "mcr"
    "cross_modal_attention"
    "flux_messages"
    "residual_connections"
    "mass_balance_projection"
    "mcr_attention"
    "physics_only"
    "mcr_physics"
    "all_features"
)

# List of GNN hops (layers) to test
# Deep layers (4, 5) will now automatically use residuals (except for baseline)
LAYER_COUNTS=(3 4 5)

echo "=========================================================="
echo "LAUNCHING FULL SUITE LOMO JOBS (Fixed Architecture)"
echo "Architectures: ${#CONFIGS[@]} | Depths: 3, 4, 5"
echo "Worker Script: $WORKER_SCRIPT"
echo "=========================================================="

# Create distinct log directory for these runs
mkdir -p logs/lomo_fixed

for config in "${CONFIGS[@]}"; do
    
    # Determine physics mode based on architecture (same logic as master script)
    if [[ "$config" == *"physics"* || "$config" == "all_features" || "$config" == "flux_messages" || "$config" == "mass_balance_projection" ]]; then
        PHYSICS="full"
    else
        PHYSICS="none"
    fi

    # ------------------------------------------------------------------
    # SUB-EXPERIMENT A: REAL LOMO ARRAY (10 jobs x 5 metabolites x 3 layers)
    # ------------------------------------------------------------------
    echo "Submitting Real LOMO Array: $config | Layers: 3,4,5 | Physics: $PHYSICS"
    
    # Added "FIXED" prefix to job name to easily identify in squeue
    JOB_ID_REAL=$(sbatch --job-name="FIX_${config}" \
                        --output="logs/lomo_fixed/R_${config}_%A_%a.out" \
                        --error="logs/lomo_fixed/R_${config}_%A_%a.err" \
                        "$WORKER_SCRIPT" "$config" "$PHYSICS" "" "ALL" \
                 | awk '{print $4}')
                 
    echo "  -> Real Job ID: $JOB_ID_REAL"
    
    echo "----------------------------------------------------------"
    sleep 1
done

echo ""
echo "Full suite of fixed jobs submitted!"
echo "Monitor with: squeue -u $USER"