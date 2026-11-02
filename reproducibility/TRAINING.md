# Optional: training models again

Read [the getting-started tutorial](TUTORIAL.md) first. The commands below are for someone who wants to repeat model fitting. Checking the released results and using the supplied external-validation models do not require this step.

Run commands from the `gaze-reproducibility` directory with the environment from the main tutorial activated. Each new run needs a new output directory; an existing run is never silently overwritten.

## A short training check, then an optional full LOMO fit

Create a separate output bundle so the archived results remain intact:

```bash
python reproducibility/new_run.py --experiment lomo36 --output work/lomo36
python work/lomo36/worker.py --root work/lomo36 --task 0 --smoke
```

The smoke run fits two tiny development-only phases for one epoch and checks finite predictions. It is not a scientific performance estimate. For a full fit, omit `--smoke`; `tasks.json` enumerates all experiment indices. The worker selects the epoch using other-metabolite validation MSE and freshly initializes the final fit; held target labels remain masked. Full fits require substantial compute, normally a GPU. `lomo36` contains the 157 core/panel/ablation tasks; `lomo10` and `lomo3` add 13 full-model targets. Full-panel results use seed 42; ablations use three seeds and five predefined LOMO targets.

UnitedMet can be retrained using the same `new_run.py --experiment unitedmet49` command and its `worker.py --task N` (or `--smoke`). All target labels are removed before ranking. In this LOMO setting, the held target has no likelihood training signal; its output is based on the fitted variational approximation and posterior sampling. This is distinct from usual reference-supported metabolite imputation.

## Train or evaluate the external models

```bash
python reproducibility/external/code/scripts/experiments/train_cohort_reanalysis.py \
  --artifacts reproducibility/external/artifacts --output work/cohort_smoke --mode cohort --smoke
```

For full training use a new output directory and omit `--smoke`. `--mode cohort_plus_targets` adds the 20 target-label exclusions. The saved configurations and sample indices describe both phases.

To evaluate the supplied full checkpoint on all external samples:

```bash
python reproducibility/external/code/scripts/experiments/evaluate_cohort_reanalysis.py \
  --artifacts reproducibility/external/artifacts --run reproducibility/external/runs/cohort \
  --output work/external_evaluation
```

The split manifests define cohort identities and exclusions. These commands accept local paths and make no SSH or Slurm connections.


## Which experiment folder means what?

| Folder in `reproducibility/experiments/` | Contents |
|---|---|
| `lomo36` | The first 36 full-model LOMO targets plus the component and loss ablations: 157 fits in total. |
| `lomo10` | Ten further full-model LOMO targets. |
| `lomo3` | The final three full-model LOMO targets. |
| `unitedmet49` | The matched UnitedMet comparisons for all 49 targets. |

The three GAZE folders together contain 170 fits. The full-model LOMO panel has 49 targets at seed 42. The LOMO ablations cover five predefined targets and three seeds; they are not three-seed fits of all 49 targets.

For UnitedMet, create a separate run and pass its location explicitly:

```bash
python reproducibility/new_run.py --experiment unitedmet49 --output work/unitedmet49
python work/unitedmet49/worker.py --root work/unitedmet49 --task 0 --smoke
```

For a full fit, omit `--smoke`. Each task number is an index in that run's `tasks.json`.

## Data and preprocessing details

The processed training file is `reproducibility/external/artifacts/camp_reanalysis.h5mu`. It contains RNA and metabolite measurements for 867 samples. Gene identifiers, metabolite identifiers and missing measurements are included. The graph and gene mapping supplied beside it are the ones used for the saved checkpoints.

Use this processed input as supplied. Applying another log transform or normalization changes the experiment. Assessment outcomes stay on their observed source-workbook scale, and missing metabolite measurements are masked rather than imputed. RNA-to-graph mapping is handled by the supplied data loader.

The original experiment records retain additional diagnostic summaries. The manuscript's ablation tables use `reproducibility/manuscript_data/ablation_seed_metrics.csv`: for each seed, the median of the per-metabolite Spearman correlations, then the mean paired ablation-minus-full difference across seeds. Other diagnostics are not interchangeable with that summary.

The original GPU/library versions and run configurations are retained with the results. The pinned Python environment in this repository was tested for CPU verification and inference. Retraining on different hardware or library versions may not reproduce identical weights.

The per-target GAZE LOMO checkpoints are not included in the public archive. Their predictions, recorded checkpoint hashes, configurations, training histories and retraining code are included. The two final external-validation checkpoints are included.
