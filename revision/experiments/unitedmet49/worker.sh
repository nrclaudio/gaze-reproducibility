#!/bin/bash
set -eo pipefail
module purge
module load ALICE/default slurm
source ~/.bashrc
conda activate "${GAZE_CONDA_ENV:-MetaboliteRNA}"
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg PYTHONHASHSEED=42
case "$1" in
  preflight) python -u worker.py --smoke ;;
  fits) python -u worker.py --task "${SLURM_ARRAY_TASK_ID}" ;;
  review) python -u review.py ;;
  *) echo 'Unknown UnitedMet LOMO stage' >&2; exit 2 ;;
esac
