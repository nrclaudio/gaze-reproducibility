#!/bin/bash
# Submit all 10 LOMO architecture configs at depth 3.
# Workers with physics_mode=full auto-read logs/best_physics_weights.json.

CONFIGS=(
    "baseline_legacy:none:3"
    "relational_core:none:3"
    "relational_residuals:none:3"
    "relational_attention:none:3"
    "relational_mcr:none:3"
    "relational_full:full:3"
    "relational_core_gpe:none:3"
    "relational_mcr_multihop:none:3"
    "relational_mcr_multihop_gpe:none:3"
    "relational_full_plus:full:3"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS=':' read -r ARCH PHYS LAYERS <<< "$CONFIG"
    sbatch scripts/slurm/submit_lomo_worker.sh "$ARCH" "$PHYS" "" "$LAYERS"
done
