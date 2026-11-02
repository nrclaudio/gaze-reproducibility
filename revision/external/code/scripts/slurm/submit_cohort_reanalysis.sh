#!/bin/bash
# Submit from the bundle root: sbatch code/scripts/slurm/submit_cohort_reanalysis.sh
# Two experiments, one GPU at a time. Seed 42 is fixed before scoring.
#SBATCH --job-name=gaze_rc_holdout
#SBATCH --output=logs/rc_holdout_%A_%a.out
#SBATCH --error=logs/rc_holdout_%A_%a.err
#SBATCH --array=0-1%1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-l4-24g,gpu-mig-40g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -eo pipefail
module purge
module load ALICE/default slurm CUDA/12.3.2 GCC/11.3.0
source ~/.bashrc
conda activate MetaboliteRNA
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
TASK_MODES=(cohort cohort_plus_targets)
TASK_MODE="${TASK_MODES[${SLURM_ARRAY_TASK_ID:-0}]}"
TASK_OUTPUT="runs/${TASK_MODE}_seed42_${SLURM_ARRAY_JOB_ID:-manual}"
python -c 'import sys, torch, torch_geometric, mudata, pandas, scipy; assert sys.version_info >= (3,11); assert torch.cuda.is_available(), "GPU is required for the full job"'
python -u code/scripts/experiments/train_cohort_reanalysis.py \
  --artifacts artifacts --output "${TASK_OUTPUT}" --mode "${TASK_MODE}" \
  --seed 42 --epochs 300 --patience 20 --batch-size 4 --accumulation 4 --threads 4
python -u code/scripts/experiments/evaluate_cohort_reanalysis.py \
  --artifacts artifacts --run "${TASK_OUTPUT}" --output "${TASK_OUTPUT}/evaluation_primary"
python -u code/scripts/experiments/evaluate_cohort_reanalysis.py \
  --artifacts artifacts --run "${TASK_OUTPUT}" --output "${TASK_OUTPUT}/evaluation_processing_sensitivity" --secondary
