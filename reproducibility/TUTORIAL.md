# Worked reproduction and inference tutorial

## 1. Install and retrieve the fixed inputs

Follow the checkout and installation commands in the repository README. Use Python 3.12 and `requirements-release.txt` for the tested CPU environment. CUDA training versions are recorded in the original run status files; exact training trajectories may differ across devices and library versions.

`python reproducibility/download.py` retrieves `gaze-artifacts-v2026.11.02.tar.gz`, checks its SHA-256 and extracts it into this directory. `release_manifest.json` lists every released experiment file with its checksum. No cluster access or private author paths are required.

The main data are `external/artifacts/camp_reanalysis.h5mu`, `graph.pt`, `gene_mapping.json`, `phase1_manifest.csv` and `phase2_manifest.csv`. Frozen LOMO inputs and splits are in `experiments/lomo36/inputs`; `lomo10` and `lomo3` hold the additional target lists and assessment annotations. The 867-sample MuData has `rna` and `metabolomics` modalities. RNA columns carry `ensembl_gene_id`; metabolite columns carry source identifiers and explicit graph-node mappings. Preserve their recorded ordering. The graph has 3,687 nodes and 6,760 directed edges.

The released MuData is the processed training input. Use it directly: do not apply another log transform, impute assessment outcomes, renormalize across metabolites, or fit scaling on assessment labels. The original source-workbook intensity scale and missing values are preserved. Missing metabolite outcomes are masked by the loaders. `gene_mapping.json` and gene–reaction rules place RNA values into graph features. This release does not claim that arbitrary raw RNA counts are already compatible.

## 2. Recompute reported summaries without training

```bash
python reproducibility/verify.py --output numerical_verification.json
```

This verifies all released file hashes and experiment manifests, checks the 170 completed GAZE result records, and recalculates per-target correlations/R² and all 42 ablation seed summaries from predictions and measured outcomes. It also recomputes the matched UnitedMet LOMO and per-cohort external correlations. Expected values are in the repository README. Numerical source tables, mappings and per-cohort activity-score comparisons are in `manuscript_data/`.

The original frozen aggregate scripts additionally compute diagnostic summaries. The manuscript uses the pooled per-target correlations documented in `manuscript_data/ablation_seed_metrics.csv`; do not substitute the within-study diagnostic from an older aggregate report. Each ablation table uses the median over its complete target set for each seed and the average paired difference across seeds 42, 43 and 44.

## 3. Replay saved model weights

```bash
python reproducibility/verify.py --replay --output checkpoint_verification.json
```

This strictly loads both `external/runs/*/phase2/best_model.pt` checkpoints and predicts the first, middle and last sample in each cohort. All metabolite labels are hidden during prediction. Twelve sample/checkpoint combinations must agree with the saved GPU predictions within `rtol=1e-4, atol=1e-3`. This is an inference check, not a new model fit.

`cohort` was refitted on all 615 development samples for 59 selected epochs. `cohort_plus_targets` additionally withholds the fixed 20 target label sets and uses 35 selected epochs. Both exclude RC18 and RC20 throughout training and epoch selection. Do not substitute the older 867-sample deployment checkpoint for these experiments.

## 4. Run a short training check or a full LOMO fit

Create a separate output bundle so the archived results remain intact:

```bash
python reproducibility/new_run.py --experiment lomo36 --output work/lomo36
python work/lomo36/worker.py --root work/lomo36 --task 0 --smoke
```

The smoke run fits two tiny development-only phases for one epoch and checks finite predictions. It is not a scientific performance estimate. For a full fit, omit `--smoke`; `tasks.json` enumerates all experiment indices. The worker selects the epoch using other-metabolite validation MSE and freshly initializes the final fit; held target labels remain masked. Full fits require substantial compute, normally a GPU. `lomo36` contains the 157 core/panel/ablation tasks; `lomo10` and `lomo3` add 13 full-model targets. Full-panel results use seed 42; ablations use three seeds and five predefined LOMO targets.

UnitedMet can be retrained using the same `new_run.py --experiment unitedmet49` command and its `worker.py --task N` (or `--smoke`). All target labels are removed before ranking. In this LOMO setting, the held target has no likelihood training signal; its output is based on the fitted variational approximation and posterior sampling. This is distinct from usual reference-supported metabolite imputation.

## 5. Train or evaluate the external model

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

## 6. Predict compatible RNA-only inputs

Install GAZE v0.2.1 and use `GAZEPredictor.from_release("reproducibility")`; see the runnable `examples/replay_release.py` in the GAZE repository. `predict_rna(frame)` accepts a samples-by-genes pandas DataFrame with Ensembl gene IDs, unique sample IDs and numeric values. NaN marks a missing measurement and leaves its graph feature at zero, as in training; infinite values and wholly unmeasured samples are rejected. Supply every gene present in the released RNA schema in training-compatible units and preprocessing; missing columns raise an error. No metabolite measurements are required. Output columns are graph node indices; names and identifiers are in `graph.pt`.

The default checkpoint is `cohort`. All model parameters are frozen at inference. A prediction is a relative abundance estimate on the training scale, not a calibrated molar concentration. Extending to new graph nodes or adapting a new assay's preprocessing is a separate modelling task.
