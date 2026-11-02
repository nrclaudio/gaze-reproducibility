#!/usr/bin/env python3
"""
Compare scCellFie metabolic task scores against measured metabolite concentrations
and GAZE predictions.

scCellFie scores metabolic tasks (e.g., "Alanine synthesis", "Glutamine degradation")
based on gene expression using gene-protein-reaction rules. Tasks map to specific
metabolites via their names. This script:

1. Runs scCellFie on CAMP gene expression data
2. Maps metabolic task names to CAMP metabolites
3. Correlates task scores with measured concentrations
4. Compares against GAZE predictions where available
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
import re

project_root = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# 1. Load CAMP data
# ---------------------------------------------------------------------------
import mudata as mu

mdata = mu.read(str(project_root / "data/raw/mudata_correct_HMDB_new_no_negatives.h5mu"))
rna = mdata.mod["rna"]
met = mdata.mod["metabolomics"]

rna_X = rna.X
if hasattr(rna_X, "toarray"):
    rna_X = rna_X.toarray()
met_X = met.X
if hasattr(met_X, "toarray"):
    met_X = met_X.toarray()

met_names = list(met.var_names)
met_names_lower = [n.lower() for n in met_names]

print(f"CAMP: {rna_X.shape[0]} samples, {rna_X.shape[1]} genes, {met_X.shape[1]} metabolites")

# ---------------------------------------------------------------------------
# 2. Run scCellFie
# ---------------------------------------------------------------------------
import anndata as ad
import sccellfie

# scCellFie expects raw counts. CAMP data is TPM-like (normalized).
# We'll run it anyway — the thresholding mechanism in scCellFie
# compares gene expression to global/local thresholds, so the
# relative ranking should be preserved with normalized data.
# We note this caveat in the comparison.

# Create AnnData for scCellFie
adata = ad.AnnData(
    X=rna_X.copy(),
    obs=rna.obs.copy(),
    var=rna.var.copy(),
)
adata.var_names = list(rna.var_names)
adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]

print("Running scCellFie pipeline...")
try:
    results = sccellfie.run_sccellfie_pipeline(
        adata,
        organism='human',
        smooth_cells=False,  # Bulk data, no smoothing
        n_counts_col=None,
        disable_pbar=True,
    )
    task_adata = results['adata'].metabolic_tasks
    task_scores = task_adata.X
    if hasattr(task_scores, "toarray"):
        task_scores = task_scores.toarray()
    task_names = list(task_adata.var_names)
    print(f"scCellFie completed: {task_scores.shape[1]} metabolic tasks scored")
except Exception as e:
    print(f"scCellFie pipeline failed: {e}")
    print("Falling back to manual task scoring...")

    # Manual fallback: load database and compute scores directly
    from sccellfie.datasets.database import load_sccellfie_database
    from sccellfie.preprocessing.gpr_rules import find_genes_gpr

    db = load_sccellfie_database(organism='human')
    task_info = db['task_info']
    task_by_rxn = db['task_by_rxn']
    rxn_by_gene = db['rxn_by_gene']

    # Get gene expression as DataFrame
    gene_df = pd.DataFrame(rna_X, columns=rna.var_names)

    # For each reaction, score = mean of constituent gene expression
    rxn_genes = db['rxn_by_gene']  # (reactions x genes) binary matrix
    gene_overlap = [g for g in rxn_genes.columns if g in gene_df.columns]
    print(f"  Gene overlap: {len(gene_overlap)}/{len(rxn_genes.columns)} scCellFie genes in CAMP")

    # Score reactions as mean expression of their genes
    rxn_scores = np.zeros((rna_X.shape[0], len(rxn_genes.index)))
    for j, rxn in enumerate(rxn_genes.index):
        rxn_gene_list = [g for g in rxn_genes.columns[rxn_genes.loc[rxn] > 0] if g in gene_df.columns]
        if rxn_gene_list:
            rxn_scores[:, j] = gene_df[rxn_gene_list].values.mean(axis=1)

    # Score tasks as min reaction score (all reactions must be active)
    task_by_rxn_mat = task_by_rxn.values  # (tasks x reactions)
    task_scores = np.zeros((rna_X.shape[0], len(task_info)))
    for t in range(len(task_info)):
        active_rxns = np.where(task_by_rxn_mat[t] > 0)[0]
        if len(active_rxns) > 0:
            task_scores[:, t] = rxn_scores[:, active_rxns].min(axis=1)

    task_names = list(task_info['Task'])
    print(f"  Scored {len(task_names)} metabolic tasks")

# ---------------------------------------------------------------------------
# 3. Map task names to CAMP metabolites
# ---------------------------------------------------------------------------
# Extract metabolite name from task name patterns like:
#   "Alanine synthesis" -> "alanine"
#   "Glutamine degradation" -> "glutamine"
#   "(R)-3-Hydroxybutanoate synthesis" -> "(R)-3-hydroxybutanoate"

def extract_metabolite_from_task(task_name):
    """Extract metabolite name from scCellFie task name."""
    task_lower = task_name.lower()
    # Remove common suffixes
    for suffix in [" synthesis", " degradation", " to ", " from ", " salvage"]:
        if suffix in task_lower:
            idx = task_lower.index(suffix)
            return task_lower[:idx].strip()
    return task_lower.strip()

# Build flexible metabolite name matching
def normalize_name(name):
    """Normalize metabolite name for fuzzy matching."""
    n = name.lower().strip()
    n = re.sub(r'[_\-\s]+', ' ', n)
    n = re.sub(r'\(.*?\)', '', n).strip()
    return n

camp_name_to_idx = {}
for i, name in enumerate(met_names):
    norm = normalize_name(name)
    camp_name_to_idx[norm] = i
    # Also store the raw lowercase
    camp_name_to_idx[name.lower()] = i

# Try to match each task to a CAMP metabolite
task_met_matches = []
for t_idx, task_name in enumerate(task_names):
    met_name = extract_metabolite_from_task(task_name)
    met_norm = normalize_name(met_name)

    # Direct match
    camp_idx = camp_name_to_idx.get(met_norm)

    # Try substring matching if direct match fails
    if camp_idx is None:
        for camp_name_lower, idx in camp_name_to_idx.items():
            if met_norm in camp_name_lower or camp_name_lower in met_norm:
                camp_idx = idx
                break

    if camp_idx is not None:
        task_met_matches.append({
            "task_idx": t_idx,
            "task_name": task_name,
            "camp_idx": camp_idx,
            "camp_name": met_names[camp_idx],
        })

print(f"\n{'='*80}")
print(f"SCCELLFIE COMPARISON RESULTS")
print(f"{'='*80}")
print(f"Matched tasks to CAMP metabolites: {len(task_met_matches)}/{len(task_names)}")

# ---------------------------------------------------------------------------
# 4. Correlate task scores with measured concentrations
# ---------------------------------------------------------------------------
comparison_results = []
for match in task_met_matches:
    t_scores = task_scores[:, match["task_idx"]]
    m_values = met_X[:, match["camp_idx"]]

    valid = np.isfinite(t_scores) & np.isfinite(m_values) & (np.std(t_scores) > 1e-10)
    if valid.sum() < 20:
        continue

    # Check if task scores have any variance
    if np.std(t_scores[valid]) < 1e-10:
        continue

    pr, pp = stats.pearsonr(t_scores[valid], m_values[valid])
    sr, sp = stats.spearmanr(t_scores[valid], m_values[valid])

    comparison_results.append({
        "task_name": match["task_name"],
        "camp_metabolite": match["camp_name"],
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
        "n_samples": valid.sum(),
    })

if comparison_results:
    comp_df = pd.DataFrame(comparison_results)
    print(f"Tasks with sufficient data: {len(comp_df)}")
    print(f"\nCorrelation with measured concentrations:")
    print(f"  Median Pearson r:  {comp_df['pearson_r'].median():+.3f}")
    print(f"  Median Spearman r: {comp_df['spearman_r'].median():+.3f}")
    print(f"  Mean Pearson r:    {comp_df['pearson_r'].mean():+.3f}")
    print(f"  % positive Pearson: {100*(comp_df['pearson_r'] > 0).mean():.0f}%")
    print(f"  % significant (p<0.05): {100*(comp_df['pearson_p'] < 0.05).mean():.0f}%")

    print(f"\n  {'Task':<45} {'Met':<30} {'Pearson_r':>10} {'p-val':>10}")
    print(f"  {'-'*97}")
    for _, r in comp_df.sort_values("pearson_r", ascending=False).iterrows():
        sig = "***" if r["pearson_p"] < 0.001 else "**" if r["pearson_p"] < 0.01 else "*" if r["pearson_p"] < 0.05 else ""
        print(f"  {r['task_name']:<45} {r['camp_metabolite']:<30} {r['pearson_r']:>+10.3f}{sig} {r['pearson_p']:>10.4f}")

    # Save
    out_dir = project_root / "comparison_results"
    out_dir.mkdir(exist_ok=True)
    comp_df.to_csv(out_dir / "sccellfie_vs_camp.csv", index=False)
    print(f"\nSaved to {out_dir / 'sccellfie_vs_camp.csv'}")
else:
    print("No valid comparisons could be made")

# ---------------------------------------------------------------------------
# 5. Compare against GAZE LOMO predictions
# ---------------------------------------------------------------------------
lomo_json = project_root / "lomo_results/mcr_flux_mass_balance/baseline_mcr_20250929_064841/lomo_cross_validation/lomo_intermediate.json"
if lomo_json.exists() and comparison_results:
    with open(lomo_json) as f:
        lomo_folds = json.load(f)

    hmdb_col = met.var["HMDB"]
    camp_name_to_hmdb = {}
    for k in range(len(met_names)):
        hid = str(hmdb_col.iloc[k])
        if hid and hid != "nan":
            camp_name_to_hmdb[met_names[k]] = hid

    lomo_lookup = {}
    for fold in lomo_folds:
        mid = fold["metabolite_id"]
        preds = np.array(fold["predictions"])
        tgts = np.array(fold["targets"])
        if len(preds) >= 5 and np.std(tgts) > 1e-10 and np.std(preds) > 1e-10:
            pr, pp = stats.pearsonr(preds, tgts)
            lomo_lookup[mid] = {"r2": fold["r2"], "pearson_r": pr}

    print(f"\n{'='*80}")
    print(f"COMPARISON: scCellFie vs GAZE (LOMO metabolites)")
    print(f"{'='*80}")

    matched = []
    for r in comparison_results:
        camp_name = r["camp_metabolite"]
        hmdb = camp_name_to_hmdb.get(camp_name, "")
        candidates = [hmdb]
        if hmdb.startswith("HMDB"):
            candidates.append("HMDB" + hmdb[4:].zfill(7))
        for cand in candidates:
            if cand in lomo_lookup:
                matched.append({
                    "metabolite": camp_name,
                    "task": r["task_name"],
                    "sccellfie_r": r["pearson_r"],
                    "gaze_r": lomo_lookup[cand]["pearson_r"],
                    "gaze_r2": lomo_lookup[cand]["r2"],
                })
                break

    if matched:
        match_df = pd.DataFrame(matched)
        print(f"Matched {len(match_df)} metabolites")
        print(f"\n  {'Task':<40} {'scCellFie_r':>12} {'GAZE_r':>10} {'GAZE_R²':>10}")
        print(f"  {'-'*74}")
        for _, r in match_df.sort_values("gaze_r2", ascending=False).iterrows():
            print(f"  {r['task']:<40} {r['sccellfie_r']:>+12.3f} {r['gaze_r']:>+10.3f} {r['gaze_r2']:>+10.3f}")
    else:
        print("No overlapping metabolites found between scCellFie and GAZE LOMO")
