# GAZE — Reproducibility

Reproducibility code for **"Zero-Shot Metabolite Prediction from Gene Expression via Physics-Informed Graph Neural Networks"** (Novella-Rausell, Rabelink & Mahfouz).

GAZE (Graph Attention for Zero-shot metabolite Estimation) is a physics-informed graph neural network that predicts metabolite concentrations from gene expression, including for metabolites never seen during training. This repository contains the full training, evaluation, and analysis code used to produce every figure and table in the paper.

For the standalone, pip-installable inference package with pretrained weights, see **[github.com/nrclaudio/GAZE](https://github.com/nrclaudio/GAZE)**.

## Repository layout

```
src/                  Library code
  models/               GAZE architecture (relational convolutions, cross-modal attention, MCR, flux head)
  losses/               Physics-informed losses (mass balance, gene-flux, flux variance) + structural alignment
  data/                 Dataset construction and metabolic-graph handling
  evaluation/           Metrics and evaluation utilities
  inference/            Forward-pass / deployment helpers
  training/             Trainer and training loops
scripts/              Executable pipelines (see scripts/README.md)
  preprocessing/        CAMP and ccRCC data preparation
  graph_generation/     Build the bipartite metabolic graph and node features
  training/             Model training entry point
  experiments/          Modality ablation, physics sweep, LOMO, external evaluation
  analysis/             Result collection, statistics, figures
  comparisons/          scCellFie / MEBOCOST / UnitedMet baselines
  slurm/                HPC submission scripts
configs/              Experiment configurations (JSON)
tests/                Unit tests (pytest)
docs/                 Architecture and implementation notes
comparison_results/   Cached baseline comparison outputs
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.10 with PyTorch >= 2.0 and PyTorch Geometric >= 2.3.

## Data

- **CAMP** (Cancer Atlas of Metabolic Profiles) — paired transcriptomics and untargeted metabolomics across cancer cell lines, used for training and the in-distribution / LOMO evaluations. [Benedetti et al. 2023](https://doi.org/10.1038/s41467-023-40217-2).
- **UnitedMet ccRCC cohort** (RC18 + RC20) — independent clear cell renal cell carcinoma tissue, used for external validation. Xie et al. 2025.

## Reproducing the paper

All scripts are run from the repository root so that `src/` imports resolve. Most accept `--help`; `scripts/README.md` lists the full set. The pipelines map onto the paper as follows.

**1. Graph construction** (Methods 2.3)
```bash
python scripts/graph_generation/create_all_merged_graphs.py
python scripts/graph_generation/initialize_enzyme_embeddings.py
python scripts/graph_generation/generate_metabolite_embeddings.py
python scripts/graph_generation/normalize_edge_features.py
```

**2. Modality ablation** — Table 1, Supplementary Figs. topology/GPE (Results 3.2)
```bash
python scripts/experiments/run_kfold_modality_sweep.py
python scripts/experiments/run_kfold_xgboost_baseline.py
python scripts/analysis/analyze_modality_results.py
```

**3. Physics-loss calibration** — 60-point grid (Results 3.2)
```bash
python scripts/analysis/analyze_physics_sweep.py
```

**4. Leave-One-Metabolite-Out (LOMO)** — Table 2, Fig. cascade, mechanism analysis (Results 3.3)
```bash
python scripts/experiments/run_lomo_ablation.py
python scripts/analysis/collect_lomo_results.py
python scripts/analysis/analyze_lomo_mechanisms.py
python scripts/analysis/investigate_zeroshot_predictors.py
python scripts/analysis/plot_lomo_cascade_clean.py
```

**5. Baseline comparisons** — Fig. comparison (Results 3.4)
```bash
python scripts/comparisons/run_sccellfie_comparison.py
python scripts/comparisons/run_mebocost_comparison.py
python scripts/comparisons/run_unitedmet_comparison.py
python scripts/comparisons/plot_comparison_figure.py
```

**6. External validation on ccRCC** — Fig. external (Results 3.5)
```bash
python scripts/preprocessing/prepare_unitedmet_mudata.py
python scripts/experiments/evaluate_external.py
python scripts/comparisons/run_baselines_on_external.py
python scripts/analysis/plot_external_validation.py
```

**Deployment weights.** `scripts/experiments/train_deploy_model.py` trains the single full-data model whose weights ship with the [GAZE package](https://github.com/nrclaudio/GAZE).

HPC users: `scripts/slurm/` contains SLURM submission wrappers for the longer sweeps.

## Tests

```bash
pytest tests/
```

## Citation

```bibtex
@article{novellarausell2026gaze,
  title={Zero-Shot Metabolite Prediction from Gene Expression via Physics-Informed Graph Neural Networks},
  author={Novella-Rausell, Claudio and Rabelink, Ton and Mahfouz, Ahmed},
  year={2026}
}
```
