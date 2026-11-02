#!/bin/bash

# Master script to submit LOMO arrays for MCR-dependent configurations ONLY
# Usage: ./scripts/slurm/submit_mcr_ablation_arrays.sh

# Get the directory where this script is located to correctly find the worker script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKER_SCRIPT="${SCRIPT_DIR}/submit_lomo_worker.sh"

# List of configurations to test (from run_lomo_ablation.py)
CONFIGS=(
    "mcr"
    "mcr_attention"
    "mcr_physics"
    "all_features"
)

# List of GNN hops (layers) to test
LAYER_COUNTS=(3 4 5)

echo "=========================================================="
echo "LAUNCHING MCR LOMO ABLATION STUDY (50 Metabolites)"
echo "Architectures: ${#CONFIGS[@]} | Depths: 3, 4, 5 (Consolidated)"
echo "Script Directory: $SCRIPT_DIR"
echo "Worker Script: $WORKER_SCRIPT"
echo "=========================================================="

mkdir -p logs/lomo_50

for config in "${CONFIGS[@]}"; do
    
    # 1. Determine physics mode based on architecture
    if [[ "$config" == *"physics"* || "$config" == "all_features" || "$config" == "flux_messages" || "$config" == "mass_balance_projection" ]]; then
        PHYSICS="full"
    else
        PHYSICS="none"
    fi

    # ------------------------------------------------------------------
    # SUB-EXPERIMENT A: REAL LOMO ARRAY (10 jobs x 5 metabolites x 3 layers)
    # ------------------------------------------------------------------
    echo "Submitting Real LOMO Array: $config | Layers: 3,4,5 | Physics: $PHYSICS"
    
    JOB_ID_REAL=$(sbatch --job-name="R_${config}" \
                        --output="logs/lomo_50/R_${config}_%A_%a.out" \
                        --error="logs/lomo_50/R_${config}_%A_%a.err" \
                        "$WORKER_SCRIPT" "$config" "$PHYSICS" "" "ALL" \
                 | awk '{print $4}')
                 
    echo "  -> Real Job ID: $JOB_ID_REAL"

    # ------------------------------------------------------------------
    # SUB-EXPERIMENT B: SYNTHETIC LOMO (1 job x 3 layers)
    # ------------------------------------------------------------------
    echo "Submitting Synthetic LOMO: $config | Layers: 3,4,5 | Physics: $PHYSICS"
    
    JOB_ID_SYNTH=$(sbatch --job-name="S_${config}" \
                         --output="logs/lomo_50/S_${config}_%j.out" \
                         --error="logs/lomo_50/S_${config}_%j.err" \
                         --array=999-999 \
                         "$WORKER_SCRIPT" "$config" "$PHYSICS" "--enable-metabolite-dropout" "ALL" \
                  | awk '{print $4}')
                  
    echo "  -> Synthetic Job ID: $JOB_ID_SYNTH"
    
    echo "----------------------------------------------------------"
    sleep 1
done

echo ""
echo "All arrays submitted!"
echo "You can monitor them with: squeue -u $USER"
