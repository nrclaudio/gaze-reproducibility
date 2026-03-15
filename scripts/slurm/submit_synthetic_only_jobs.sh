#!/bin/bash

# Submission script for ONLY Synthetic LOMO jobs (Universal Training)
# for ALL model variants (fixes previous failure due to metabolite validation).
#
# Usage: ./scripts/slurm/submit_synthetic_only_jobs.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKER_SCRIPT="${SCRIPT_DIR}/submit_lomo_worker.sh"

# ALL model configurations
CONFIGS=(
    "baseline"
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
LAYER_COUNTS=(3 4 5)

echo "=========================================================="
echo "LAUNCHING SYNTHETIC LOMO JOBS (Universal Training)"
echo "Architectures: ${#CONFIGS[@]} | Depths: 3, 4, 5"
echo "Worker Script: $WORKER_SCRIPT"
echo "=========================================================="

# Create distinct log directory for these runs
mkdir -p logs/lomo_synth

for config in "${CONFIGS[@]}"; do
    
    # Determine physics mode based on architecture
    if [[ "$config" == *"physics"* || "$config" == "all_features" || "$config" == "flux_messages" || "$config" == "mass_balance_projection" ]]; then
        PHYSICS="full"
    else
        PHYSICS="none"
    fi

    # ------------------------------------------------------------------
    # SUB-EXPERIMENT B: SYNTHETIC LOMO (Universal mask)
    # ------------------------------------------------------------------
    echo "Submitting Synthetic LOMO: $config | Layers: 3,4,5 | Physics: $PHYSICS"
    
    JOB_ID_SYNTH=$(sbatch --job-name="S_${config}" \
                         --output="logs/lomo_synth/S_${config}_%j.out" \
                         --error="logs/lomo_synth/S_${config}_%j.err" \
                         --array=999-999 \
                         "$WORKER_SCRIPT" "$config" "$PHYSICS" "--enable-metabolite-dropout" "ALL" \
                  | awk '{print $4}')
                  
    echo "  -> Synthetic Job ID: $JOB_ID_SYNTH"
    
    echo "----------------------------------------------------------"
    sleep 1
done

echo ""
echo "All synthetic jobs submitted!"
echo "Monitor with: squeue -u $USER"