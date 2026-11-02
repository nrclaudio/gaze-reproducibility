#!/bin/bash
set -eo pipefail
case "$1" in
  cpu_preflight|gpu_preflight|core|lomo_ablation|aggregate) ;;
  *) echo 'Unknown rebuttal stage' >&2; exit 2 ;;
esac
module purge
case "$1" in
  gpu_preflight|core|lomo_ablation) module load ALICE/default slurm CUDA/12.3.2 GCC/11.3.0 ;;
  *) module load ALICE/default slurm ;;
esac
source ~/.bashrc
conda activate "${GAZE_CONDA_ENV:-MetaboliteRNA}"
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg PYTHONHASHSEED=42
case "$1" in
  cpu_preflight) python -u preflight.py ;;
  gpu_preflight) python -u preflight.py --gpu ;;
  core) python -u worker.py --task "${SLURM_ARRAY_TASK_ID}" ;;
  lomo_ablation) python -u worker.py --task "$((57 + SLURM_ARRAY_TASK_ID))" ;;
  aggregate) python -u aggregate.py ;;
esac
