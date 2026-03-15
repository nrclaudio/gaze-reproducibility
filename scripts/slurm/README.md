# SLURM Job Helpers

This folder now contains a minimal set of SLURM launchers that wrap the
configuration-driven training pipeline. Each script hard-codes the project
root (`/home/cnovellarausell/data1/concat-smiles-model`) so they keep working
when Slurm copies them to its spool directory.

- `submit_baseline_mcr_array.sh` runs the baseline vs. MCR LOMO comparison via a
  job array and prepares the runtime environment before launching the Python
  entry point.
- `submit_lomo_suite.sh` launches the LOMO experiment sweep.
- `submit_lomo_suite.sh` automatically runs contrastive pretraining with
  `pretrain_contrastive.py` before the contrastive LOMO evaluation task.
- `submit_core_experiments.sh` covers the broader contrastive / baseline suite.

## Writing New Jobs

1. Start from one of the existing submission scripts or copy the pattern below
   (replace the command at the end with your experiment):

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --output=logs/my_experiment_%j.out
#SBATCH --error=logs/my_experiment_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-medium
#SBATCH --gres=gpu:l4:1

set -euo pipefail

PROJECT_ROOT="/home/cnovellarausell/data1/concat-smiles-model"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load ALICE/default || true
  module load slurm || true
  module load CUDA/12.3.2 || true
  module load GCC/11.3.0 || true
fi

if [ -f "${HOME}/.bashrc" ]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

if command -v conda >/dev/null 2>&1; then
  conda activate MetaboliteRNA || true
fi

export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}" || exit 1
mkdir -p logs

echo "Running on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python scripts/training/train_concat_smiles.py \
  --config configs/experiments/baseline.yaml
```

2. Submit with `sbatch my_job.sh`. The inline helpers take care of environment
   activation and logging.

The legacy multi-script launchers (`submit_enhanced_ablation.sh`,
`submit_lomo_ablation.sh`, `submit_model_comparison.sh`) have been removed in
favour of this streamlined approach. Convert any remaining bespoke workflows to
`ExperimentConfig` files and submit them using the pattern above.
