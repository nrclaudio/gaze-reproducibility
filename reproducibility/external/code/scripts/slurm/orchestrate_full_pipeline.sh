#!/bin/bash
# ==============================================================================
# FULL PIPELINE ORCHESTRATOR
# ==============================================================================
#
# Submits the complete experiment pipeline with correct SLURM dependency chains.
# ALL sbatch calls are made from the login node — no nested sbatch from compute.
#
#   1. Graph Regeneration  (CPU, ~24h)
#          |
#          +---> 2a. Modality 5-Fold CV       (GPU array, 12 tasks)
#          |
#          +---> 2b. XGBoost 5-Fold CV        (CPU, ~12h)
#          |
#          +---> 3.  Physics Loss Sweep        (GPU, 60 jobs, dep: modality)
#          |              |
#          |              v
#          |         3b. Analysis job          (CPU, dep: all 60 physics jobs)
#          |              |  -> writes logs/best_physics_weights.json
#          |              v
#          +--------> 4. LOMO Ablation         (GPU arrays, 10 levels, dep: analysis)
#
# Usage:
#   cd /path/to/concat-smiles-model
#   bash scripts/slurm/orchestrate_full_pipeline.sh [--dry-run]
# ==============================================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — no jobs will be submitted ==="
    echo ""
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "======================================================================"
echo "  FULL PIPELINE ORCHESTRATOR"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Date: $(date)"
echo "======================================================================"

mkdir -p logs

# Helper: submit a job and capture its ID
submit_job() {
    local desc="$1"
    shift
    if $DRY_RUN; then
        echo "  [DRY RUN] Would submit: $*" >&2
        echo "DRY_RUN_${RANDOM}"  # return a fake job ID
        return
    fi
    local output
    output=$("$@" 2>&1)
    local job_id
    job_id=$(echo "$output" | grep -oE '[0-9]+' | head -1)
    if [[ -z "$job_id" ]]; then
        echo "  ERROR: Failed to submit ${desc}" >&2
        echo "  Output: ${output}" >&2
        exit 1
    fi
    echo "  Submitted ${desc}: Job ${job_id}" >&2
    echo "$job_id"
}

# ==============================================================================
# STAGE 1: Graph Regeneration
# ==============================================================================
echo ""
echo ">>> STAGE 1: Graph Regeneration"
echo "    (MetabolicAtlas API -> SMILES -> EC2Vec -> Gene Features -> Merge)"

GRAPH_JOB_ID=$(submit_job "graph regeneration" \
    sbatch --parsable \
           --job-name=regen_graphs \
           --output=logs/regen_graphs_%j.out \
           --error=logs/regen_graphs_%j.err \
           --time=24:00:00 \
           --partition=cpu-zen4 \
           --cpus-per-task=8 \
           --mem=32G \
           scripts/slurm/regenerate_graphs.sh)

echo "    Graph regen job: ${GRAPH_JOB_ID}"

# ==============================================================================
# STAGE 2a: Modality 5-Fold Cross-Validation (depends on graphs)
# ==============================================================================
echo ""
echo ">>> STAGE 2a: Modality 5-Fold CV (12 configurations)"
echo "    Dependency: afterok:${GRAPH_JOB_ID}"

MODALITY_JOB_ID=$(submit_job "modality ablation" \
    sbatch --parsable \
           --dependency=afterok:${GRAPH_JOB_ID} \
           scripts/slurm/submit_modality_ablation.sh)

echo "    Modality CV job: ${MODALITY_JOB_ID}"

# ==============================================================================
# STAGE 2b: XGBoost 5-Fold Cross-Validation (depends on graphs, parallel w/ 2a)
# ==============================================================================
echo ""
echo ">>> STAGE 2b: XGBoost 5-Fold CV Baseline"
echo "    Dependency: afterok:${GRAPH_JOB_ID}"

XGBOOST_JOB_ID=$(submit_job "XGBoost baseline" \
    sbatch --parsable \
           --dependency=afterok:${GRAPH_JOB_ID} \
           scripts/slurm/submit_xgboost_kfold.sh)

echo "    XGBoost job: ${XGBOOST_JOB_ID}"

# ==============================================================================
# STAGE 3: Physics Loss Sweep (depends on modality CV)
# ==============================================================================
echo ""
echo ">>> STAGE 3: Physics Loss Sweep (5 x 4 x 3 = 60 configs)"
echo "    Dependency: afterok:${MODALITY_JOB_ID}"

BAL_WEIGHTS=(0.0 0.1 0.5 1.0 2.0)
CORR_WEIGHTS=(0.0 0.1 0.5 1.0)
VAR_WEIGHTS=(0.0 1.0 2.0)

PHYSICS_JOB_IDS=()
PHYSICS_JOB_IDS_FILE="${PROJECT_ROOT}/logs/physics_sweep_job_ids.txt"
> "${PHYSICS_JOB_IDS_FILE}"  # truncate

for BAL in "${BAL_WEIGHTS[@]}"; do
    for CORR in "${CORR_WEIGHTS[@]}"; do
        for VAR in "${VAR_WEIGHTS[@]}"; do
            EXP_NAME="phys_bal${BAL}_cor${CORR}_var${VAR}"

            JOB_ID=$(submit_job "physics ${EXP_NAME}" \
                sbatch --parsable \
                       --dependency=afterok:${MODALITY_JOB_ID} \
                       --job-name="phys_${EXP_NAME}" \
                       --output="logs/phys_sweep_${EXP_NAME}_%j.out" \
                       --error="logs/phys_sweep_${EXP_NAME}_%j.err" \
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
                                --seed 42")

            PHYSICS_JOB_IDS+=("$JOB_ID")
            echo "${JOB_ID}" >> "${PHYSICS_JOB_IDS_FILE}"
        done
    done
done

echo "    Total physics sweep jobs: ${#PHYSICS_JOB_IDS[@]}"

# Build colon-separated dependency string for all physics jobs
PHYSICS_DEP=$(IFS=:; echo "${PHYSICS_JOB_IDS[*]}")

# ==============================================================================
# STAGE 3b: Physics Sweep Analysis (depends on ALL physics sweep jobs)
# ==============================================================================
echo ""
echo ">>> STAGE 3b: Physics Sweep Analysis"
echo "    Dependency: afterok on all ${#PHYSICS_JOB_IDS[@]} physics jobs"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/logs/analyze_physics_sweep_wrapper.sh"
cat > "${ANALYSIS_SCRIPT}" << ANALYSIS_EOF
#!/bin/bash

source ~/.bashrc
conda activate MetaboliteRNA

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"

echo "Physics Sweep Analysis started at \$(date)"

SWEEP_DIR="trained_models/concat_smiles_physics"
BEST_WEIGHTS_FILE="${PROJECT_ROOT}/logs/best_physics_weights.json"

python -c "
import json, glob, os

base_dir = '\${SWEEP_DIR}'
result_files = glob.glob(os.path.join(base_dir, 'seed_*', 'phys_bal*', 'test_results.json'))

if not result_files:
    result_files = glob.glob(os.path.join(base_dir, 'phys_bal*', 'test_results.json'))

if not result_files:
    print('WARNING: No physics sweep results found. Using default weights.')
    best = {'lambda_bal': 2.0, 'lambda_corr': 0.5, 'lambda_var': 0.0, 'r2': -1, 'source': 'default'}
else:
    best_r2 = -999
    best = None
    for res_path in result_files:
        try:
            folder = os.path.dirname(res_path)
            config_path = os.path.join(folder, 'experiment_config.json')
            with open(res_path) as f:
                results = json.load(f)
            with open(config_path) as f:
                config = json.load(f)
            phys = config.get('physics', {})
            l_bal = phys.get('lambda_balance', 0.0)
            l_corr = phys.get('lambda_correlation', 0.0)
            l_var = phys.get('lambda_variance', 0.0)
            r2 = results.get('test_r2', -999)
            if (l_bal > 0 or l_corr > 0) and r2 > best_r2:
                best_r2 = r2
                best = {'lambda_bal': l_bal, 'lambda_corr': l_corr, 'lambda_var': l_var, 'r2': r2, 'source': 'sweep'}
        except Exception as e:
            continue
    if best is None:
        print('WARNING: No valid physics sweep results. Using defaults.')
        best = {'lambda_bal': 2.0, 'lambda_corr': 0.5, 'lambda_var': 0.0, 'r2': -1, 'source': 'default'}
    else:
        print(f'Best physics config: bal={best[\"lambda_bal\"]}, corr={best[\"lambda_corr\"]}, var={best[\"lambda_var\"]} (R2={best[\"r2\"]:.4f})')

with open('\${BEST_WEIGHTS_FILE}', 'w') as f:
    json.dump(best, f, indent=2)
print(f'Saved best weights to \${BEST_WEIGHTS_FILE}')
"

# Also run the full analysis script for a report
python scripts/analysis/analyze_physics_sweep.py --dir "\${SWEEP_DIR}" || true

echo "Physics Sweep Analysis finished at \$(date)"
ANALYSIS_EOF
chmod +x "${ANALYSIS_SCRIPT}"

ANALYSIS_JOB_ID=$(submit_job "physics analysis" \
    sbatch --parsable \
           --dependency=afterok:${PHYSICS_DEP} \
           --job-name=phys_analysis \
           --output=logs/phys_analysis_%j.out \
           --error=logs/phys_analysis_%j.err \
           --time=01:00:00 \
           --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g \
           --gres=gpu:1 \
           --cpus-per-task=4 \
           --mem=32G \
           "${ANALYSIS_SCRIPT}")

echo "    Analysis job: ${ANALYSIS_JOB_ID}"

# ==============================================================================
# STAGE 4: LOMO Architecture Ablation (depends on analysis)
# ==============================================================================
echo ""
echo ">>> STAGE 4: LOMO Architecture Ablation (10 configurations)"
echo "    Dependency: afterok:${ANALYSIS_JOB_ID}"
echo "    Workers auto-read best weights from logs/best_physics_weights.json"

LOMO_CONFIGS=(
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

LOMO_JOB_IDS=()
for CONFIG in "${LOMO_CONFIGS[@]}"; do
    IFS=':' read -r ARCH_CONFIG PHYSICS_MODE NUM_LAYERS <<< "$CONFIG"

    # submit_lomo_worker.sh has #SBATCH directives for GPU partition, array, etc.
    # Workers with physics_mode=full auto-read logs/best_physics_weights.json at runtime.
    JOB_ID=$(submit_job "LOMO ${ARCH_CONFIG}" \
        sbatch --parsable \
               --dependency=afterok:${ANALYSIS_JOB_ID} \
               scripts/slurm/submit_lomo_worker.sh \
               "${ARCH_CONFIG}" "${PHYSICS_MODE}" "" "${NUM_LAYERS}")

    LOMO_JOB_IDS+=("$JOB_ID")
done

echo "    Total LOMO jobs: ${#LOMO_JOB_IDS[@]}"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "======================================================================"
echo "  PIPELINE SUBMITTED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "  Job Dependency Chain:"
echo "  ---------------------"
echo "  Stage 1:  Graph Regeneration     ${GRAPH_JOB_ID}"
echo "      |"
echo "      +-> Stage 2a: Modality CV    ${MODALITY_JOB_ID}  (afterok:${GRAPH_JOB_ID})"
echo "      +-> Stage 2b: XGBoost CV     ${XGBOOST_JOB_ID}  (afterok:${GRAPH_JOB_ID})"
echo "      |"
echo "      +-> Stage 3:  Physics Sweep  ${#PHYSICS_JOB_IDS[@]} jobs  (afterok:${MODALITY_JOB_ID})"
echo "              |"
echo "              +-> Stage 3b: Analysis  ${ANALYSIS_JOB_ID}  (afterok:all physics)"
echo "                      |"
echo "                      +-> Stage 4: LOMO  ${#LOMO_JOB_IDS[@]} jobs  (afterok:${ANALYSIS_JOB_ID})"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo ""
echo "  Key output locations:"
echo "    Modality results:  results/modality_ablation_5fold/"
echo "    XGBoost results:   results/modality_ablation_5fold/"
echo "    Physics sweep:     trained_models/concat_smiles_physics/"
echo "    Best weights:      logs/best_physics_weights.json"
echo "    LOMO results:      lomo_results_final_waterfall_N50/"
echo ""

# Save pipeline metadata
PIPELINE_META="${PROJECT_ROOT}/logs/pipeline_$(date +%Y%m%d_%H%M%S).json"
if ! $DRY_RUN; then
    python -c "
import json
meta = {
    'graph_regen_job': '${GRAPH_JOB_ID}',
    'modality_cv_job': '${MODALITY_JOB_ID}',
    'xgboost_cv_job': '${XGBOOST_JOB_ID}',
    'physics_sweep_jobs': ${#PHYSICS_JOB_IDS[@]},
    'analysis_job': '${ANALYSIS_JOB_ID}',
    'lomo_jobs': ${#LOMO_JOB_IDS[@]},
    'submitted_at': '$(date -Iseconds)',
    'project_root': '${PROJECT_ROOT}'
}
with open('${PIPELINE_META}', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Pipeline metadata saved to ${PIPELINE_META}')
"
fi
