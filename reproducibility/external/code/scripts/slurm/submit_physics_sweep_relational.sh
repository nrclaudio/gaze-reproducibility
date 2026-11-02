#!/bin/bash
# Master script to run Physics Loss Hyperparameter Sweep using the new Relational Architecture.
# Tests combinations of λ_bal, λ_corr, λ_var to find the optimal bio-physical regularisation balance.

# Values to sweep
BAL_WEIGHTS=(0.0 0.1 0.5 1.0 2.0)
CORR_WEIGHTS=(0.0 0.1 0.5 1.0)
VAR_WEIGHTS=(0.0 1.0 2.0)

echo "Starting Relational Physics Sweep..."
PROJECT_ROOT=$(pwd)

for BAL in "${BAL_WEIGHTS[@]}"; do
    for CORR in "${CORR_WEIGHTS[@]}"; do
        for VAR in "${VAR_WEIGHTS[@]}"; do
            EXP_NAME="phys_bal${BAL}_cor${CORR}_var${VAR}"
            
            # Submit individual jobs using the standard training script
            # We must activate the environment and setup paths for every wrapped job
            sbatch --job-name="phys_${EXP_NAME}" \
                   --output="logs/phys_sweep_${EXP_NAME}_%j.out" \
                   --partition=gpu-l4-24g,gpu-mig-40g \
                   --gres=gpu:1 \
                   --time=24:00:00 \
                   --mem=32G \
                   --wrap="source ~/.bashrc && conda activate MetaboliteRNA && module load CUDA/12.3.2 && export PYTHONPATH=${PROJECT_ROOT}:\$PYTHONPATH && cd ${PROJECT_ROOT} && python scripts/training/train_concat_smiles.py \
                            --config_name ${EXP_NAME} \
                            --lambda_bal ${BAL} \
                            --lambda_corr ${CORR} \
                            --lambda_var ${VAR} \
                            --layer_type Relational \
                            --structural_loss_weight 1.0 \
                            --mask_target_smiles 0.5 \
                            --use_flux_messages \
                            --residual_connections \
                            --cross_modal_attention \
                            --similarity_loss_weight 0.0 \
                            --batch_size 4 \
                            --gradient_accumulation_steps 4 \
                            --learning_rate 0.001 \
                            --max_epochs 200 \
                            --seed 42"
            
            echo "Submitted: ${EXP_NAME}"
        done
    done
done

echo "All physics sweep jobs submitted."
