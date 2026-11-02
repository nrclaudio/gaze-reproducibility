# Get started with GAZE

This guide answers two questions:

1. **Can I reproduce the reported numbers from the supplied results?** Follow steps 1–3.
2. **Can I use a trained GAZE model to make predictions?** Continue to step 4.

Steps 1–3 run on a computer with Python 3.12 and Git. They do not require ALICE, a GPU or another training run. The commands below use a macOS/Linux terminal.

## 1. Download the code and install its Python environment

For a new copy, run:

```bash
git clone --branch v2026.11.02.1 https://github.com/nrclaudio/gaze-reproducibility.git
cd gaze-reproducibility
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-release.txt
```

The first two commands download the code and enter its folder. The next three create a separate Python environment and install the required libraries.

**Run all remaining commands from this `gaze-reproducibility` folder.** If you already have this version and its environment, enter the folder and activate the environment; you can skip the installation commands.

## 2. Download the data, saved predictions and trained models

```bash
python reproducibility/download.py
```

This downloads about **215 MB** and places the files under `reproducibility/`. It also checks that the download is intact. It finishes with:

```text
Artifacts extracted. Run: python reproducibility/verify.py --replay
```

You do not need to find or download CAMP, the graph or the external-validation models separately. If you have already extracted these files, skip this step.

## 3. Check that the released results match the paper

```bash
python reproducibility/verify.py --replay --output verification.json
```

This command does two things:

- Reads the saved predictions and measured metabolite values, then recalculates the reported results.
- Loads both supplied trained models and checks a small set of predictions against the saved outputs. Here, `--replay` means “run the trained models again”; it does not train them.

At the end, it prints the checks and saves them in **`verification.json`**, in the current folder. The main expected results are:

| Check | Expected result |
|---|---:|
| GAZE completed fit records checked | 170 |
| Metabolites in the full-model LOMO evaluation | 49 |
| GAZE median LOMO Spearman correlation | approximately 0.523 |
| UnitedMet median correlation on the same 49 targets | approximately 0.115 |
| Targets where GAZE has higher correlation | 40 |
| Ablation summaries checked, counting each seed separately | 42 |
| Sample/model combinations checked using the supplied weights | 12 |

A successful run reaches the final report without a traceback. The calculation uses the supplied predictions; it does not repeat all 170 training fits. The model check uses the two released external-validation models. Individual GAZE LOMO weights are not in the archive.

**You can stop here if your aim is to check the reported results.** The numerical tables behind the figures are in `reproducibility/manuscript_data/`.

## 4. Optional: make predictions using a trained model

First install the GAZE inference package in the same activated environment:

```bash
python -m pip install --no-deps 'git+https://github.com/nrclaudio/GAZE.git@v0.2.1'
```

The following example uses RNA from the first two samples in the supplied data. It loads the trained model, makes predictions and writes **`example_predictions.csv`**. Metabolite measurements are not passed to the predictor. This demonstrates how to use the model; the external-validation check is performed in step 3.

Copy this whole block into the terminal:

```bash
python - <<'PY'
import mudata as mu
import pandas as pd
import torch
from gaze import GAZEPredictor

torch.set_num_threads(2)
data = mu.read_h5mu("reproducibility/external/artifacts/camp_reanalysis.h5mu")
rna = data["rna"][:2]
values = rna.X.toarray() if hasattr(rna.X, "toarray") else rna.X
rna_table = pd.DataFrame(
    values,
    index=rna.obs_names,
    columns=rna.var["ensembl_gene_id"].astype(str),
)

model = GAZEPredictor.from_release("reproducibility", mode="cohort")
predictions = model.predict_rna(rna_table)
predictions.to_csv("example_predictions.csv")
print(f"Saved predictions for {len(predictions)} samples to example_predictions.csv")
PY
```

Each output row is a sample. Each output column is a metabolite's node number in the supplied graph. Predictions are relative metabolite abundances on the training scale.

For your own RNA data, replace `rna_table` with a pandas table containing samples as rows and Ensembl gene IDs as columns. Supply all 2,184 genes in the released RNA schema, using the same preprocessing and units. NaN can mark an unmeasured value; the loader handles it as it did during training. Raw RNA counts need appropriate preprocessing before this step.

### Which model does this use?

The example explicitly selects `cohort`: the CAMP model trained on 615 samples, excluding RC18 and RC20. It demonstrates a model used in external validation; it is not a selection of the best model across experiments. These are the two final external-validation model files, relative to the repository folder:

| Model | File |
|---|---|
| Standard external-validation model | `reproducibility/external/runs/cohort/phase2/best_model.pt` |
| Model also withholding 20 metabolite label sets | `reproducibility/external/runs/cohort_plus_targets/phase2/best_model.pt` |

The original all-867-sample deployment weights are also public in [GAZE/weights](https://github.com/nrclaudio/GAZE/tree/v0.2.1/weights). They use the earlier implementation and have not been validated with the current `GAZEPredictor` loading workflow.

A `.pt` checkpoint contains the trained model weights. Use **`phase2`** for final predictions; `phase1` was used to select training duration. The loading function finds the matching configuration, graph and gene mapping automatically.

To select the second model:

```python
model = GAZEPredictor.from_release("reproducibility", mode="cohort_plus_targets")
```

## Optional: train the models again

If you specifically want to repeat model fitting, follow [the training instructions](https://github.com/nrclaudio/gaze-reproducibility/blob/main/reproducibility/TRAINING.md). They cover a small training check, individual LOMO fits, UnitedMet and the external-validation models. Full fits take substantially more compute than the checks above.
