#!/bin/bash
set -eo pipefail
case "$1" in preflight|train|aggregate) ;; *) exit 2 ;; esac
module purge
if [[ "$1" == train ]]; then
  module load ALICE/default slurm CUDA/12.3.2 GCC/11.3.0
else
  module load ALICE/default slurm
fi
source ~/.bashrc
conda activate "${GAZE_CONDA_ENV:-MetaboliteRNA}"
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg PYTHONHASHSEED=42
if [[ "$1" == train ]]; then
  python -u restored_panel.py train --task "${SLURM_ARRAY_TASK_ID}"
elif [[ "$1" == aggregate ]]; then
  python -u restored_panel.py aggregate --require-checkpoints
else
  python -u restored_panel.py preflight
fi
