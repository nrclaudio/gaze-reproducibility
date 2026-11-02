# GAZE reproducibility

GAZE (Graph Attention for Zero-shot metabolite Estimation) predicts relative metabolite abundances from RNA using a metabolic graph, molecular descriptors and biochemical regularization. Zero-shot evaluation withholds a metabolite's abundance labels while retaining its node and descriptors in the graph.

## Versioned manuscript release

Use [v2026.11.02.1](https://github.com/nrclaudio/gaze-reproducibility/releases/tag/v2026.11.02.1) and the [worked tutorial](reproducibility/TUTORIAL.md). It provides the compatible processed CAMP inputs, graph, gene mapping, sample splits, target lists, configurations, saved predictions, and the two external-validation checkpoints. It includes the exact training implementations for the 49-metabolite LOMO panel, three-seed component ablations and RC18/RC20 evaluation, plus the matched UnitedMet LOMO code and posterior outputs.

```bash
git clone https://github.com/nrclaudio/gaze-reproducibility.git
cd gaze-reproducibility
git checkout v2026.11.02.1
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-release.txt
python reproducibility/download.py
python reproducibility/verify.py --replay --output verification.json
```

Expected main checks: 170 completed GAZE fits; 49 LOMO targets, median Spearman 0.5227504783; UnitedMet median 0.1153929657, GAZE higher on 40/49 targets; 42 per-seed ablation summaries; CPU replay of 12 sample/checkpoint combinations. The replay checks the two released external checkpoints, not all LOMO weights. GAZE's per-target LOMO checkpoints are not included in this release; the corresponding predictions, hashes, configurations and retraining code are included.

The [GAZE inference package](https://github.com/nrclaudio/GAZE) supplies `GAZEPredictor`, using the same implementation and preprocessing as these checkpoints.

## Repository layout

- `reproducibility/`: frozen experiment code, download/verification tools and tutorial for the manuscript analyses. The tutorial is the entry point for current results.
- `src/`, `scripts/`, `configs/`: original architecture, modality and analysis code. `src/data` is tracked. Data defaults to `data/`; `GAZE_DATA_DIR` overrides that location, including graph merging.
- `comparison_results/`: original cached comparator outputs.
- `tests/`: original implementation tests. Focused release checks are described in the tutorial.

The retained 50-target architecture comparison and five-fold modality analysis use their original protocols. They are distinct from the 49-target full-model evaluation. Original graph-building scripts require upstream pathway and embedding inputs; the release supplies the exact graph directly, so reproducing the released evaluation does not depend on a changing external API.

## Data and attribution

CAMP includes paired measurements from multiple public studies, including tissue samples. See [Benedetti et al.](https://doi.org/10.1038/s41467-023-40217-2) and [Zenodo record 7874228](https://doi.org/10.5281/zenodo.7874228), licensed CC BY 4.0. Processed inputs are derivatives of those source data; cite the original authors. See `reproducibility/DATA_SOURCES.md` for transformations, graph provenance and comparator attribution.

RC18/RC20 are part of the assembled CAMP collection; both cohorts and 32 additional renal samples were excluded from training and epoch selection for the released external models. The 615 development samples and 220 assessment samples are explicitly identified in the split manifests. Predictions concern relative abundances; auxiliary reaction outputs are regularizers rather than independently validated biological fluxes.

Code is distributed under the MIT license. Upstream code and data retain their respective licenses.

## Saved model files

The checkpoints are in the downloadable artifact archive, rather than the Git source tree. After running `python reproducibility/download.py`, the final external-validation models are:

| Model | Checkpoint path relative to this repository |
|---|---|
| CAMP model excluding RC18/RC20 | `reproducibility/external/runs/cohort/phase2/best_model.pt` |
| Same cohort exclusions plus 20 withheld metabolite label sets | `reproducibility/external/runs/cohort_plus_targets/phase2/best_model.pt` |

The adjacent `experiment_config.json` records each configuration. The `phase1` checkpoints select training duration; `phase2` checkpoints are the final refitted models used for evaluation. The compatible graph and gene mapping are in `reproducibility/external/artifacts/`.
