#!/usr/bin/env python3
"""
Compare MEBOCOST metabolite activity estimates against measured concentrations
and GAZE LOMO predictions.

MEBOCOST estimates metabolite activity as:
  activity = mean(expression of producing enzymes) - mean(expression of consuming enzymes)

This script:
1. Loads the MEBOCOST enzyme-metabolite database
2. Loads CAMP gene expression and metabolomics from MuData
3. Computes MEBOCOST-style metabolite activity for each cell line
4. Correlates activity estimates with measured concentrations
5. Compares against GAZE LOMO predictions where available
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import re
import json

project_root = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# 1. Load MEBOCOST enzyme-metabolite database
# ---------------------------------------------------------------------------
mebocost_db_path = project_root.parent / "MEBOCOST_official/data/mebocost_db/human/metabolite_associated_gene_reaction_HMDB_summary.tsv"
if not mebocost_db_path.exists():
    raise FileNotFoundError(f"MEBOCOST database not found at {mebocost_db_path}")

met_enzyme_db = pd.read_csv(mebocost_db_path, sep="\t")
print(f"MEBOCOST database: {len(met_enzyme_db)} enzyme-metabolite entries")
print(f"  Unique metabolites: {met_enzyme_db['HMDB_ID'].nunique()}")
print(f"  Columns: {met_enzyme_db.columns.tolist()}")

# Parse gene names: "GENE1[Type]; GENE2[Type]" -> ["GENE1", "GENE2"]
def parse_genes(gene_str):
    if pd.isna(gene_str):
        return []
    genes = []
    for g in gene_str.split(";"):
        g = g.strip()
        match = re.match(r"(\w+)\[", g)
        if match:
            genes.append(match.group(1))
        elif g:
            genes.append(g.split("[")[0].strip())
    return genes

# Build per-metabolite producing/consuming gene sets
met_genes = {}  # HMDB_ID -> {"producing": set(), "consuming": set(), "name": str}
for _, row in met_enzyme_db.iterrows():
    hmdb = row["HMDB_ID"]
    if hmdb not in met_genes:
        met_genes[hmdb] = {"producing": set(), "consuming": set(), "name": row["metabolite"]}
    genes = parse_genes(row["gene"])
    if row["direction"] == "product":
        met_genes[hmdb]["producing"].update(genes)
    elif row["direction"] == "substrate":
        met_genes[hmdb]["consuming"].update(genes)

print(f"Parsed {len(met_genes)} metabolites with enzyme mappings")

# ---------------------------------------------------------------------------
# 2. Load CAMP data from MuData
# ---------------------------------------------------------------------------
import mudata as mu

mdata = mu.read(str(project_root / "data/raw/mudata_correct_HMDB_new_no_negatives.h5mu"))
rna = mdata.mod["rna"]
met = mdata.mod["metabolomics"]

# Gene expression matrix
gene_names = list(rna.var_names)  # Gene symbols
gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
rna_X = rna.X
if hasattr(rna_X, "toarray"):
    rna_X = rna_X.toarray()

# Metabolomics matrix
met_names = list(met.var_names)
met_X = met.X
if hasattr(met_X, "toarray"):
    met_X = met_X.toarray()

# Build HMDB lookup for CAMP metabolites
hmdb_col = met.var["HMDB"] if "HMDB" in met.var.columns else None
camp_hmdb_to_idx = {}
if hmdb_col is not None:
    for k in range(len(met_names)):
        hid = str(hmdb_col.iloc[k])
        if hid and hid != "nan":
            camp_hmdb_to_idx[hid] = k
            # Normalize to 7-digit HMDB format
            if hid.startswith("HMDB") and len(hid) < 13:
                normalized = "HMDB" + hid[4:].zfill(7)
                camp_hmdb_to_idx[normalized] = k

print(f"\nCAMP data: {rna_X.shape[0]} cell lines, {rna_X.shape[1]} genes, {met_X.shape[1]} metabolites")
print(f"CAMP HMDB-mapped metabolites: {len(camp_hmdb_to_idx)}")

# ---------------------------------------------------------------------------
# 3. Compute MEBOCOST metabolite activity estimates
# ---------------------------------------------------------------------------
n_samples = rna_X.shape[0]
results = []

for hmdb_id, info in met_genes.items():
    # Check if this metabolite is in CAMP
    camp_idx = camp_hmdb_to_idx.get(hmdb_id)
    if camp_idx is None:
        # Try normalized format
        if hmdb_id.startswith("HMDB"):
            normalized = "HMDB" + hmdb_id[4:].zfill(7)
            camp_idx = camp_hmdb_to_idx.get(normalized)
    if camp_idx is None:
        continue

    # Find matching genes in expression data
    prod_indices = [gene_name_to_idx[g] for g in info["producing"] if g in gene_name_to_idx]
    cons_indices = [gene_name_to_idx[g] for g in info["consuming"] if g in gene_name_to_idx]

    if not prod_indices and not cons_indices:
        continue

    # Compute activity per sample
    activity = np.zeros(n_samples)
    if prod_indices:
        activity += rna_X[:, prod_indices].mean(axis=1)
    if cons_indices:
        activity -= rna_X[:, cons_indices].mean(axis=1)

    # Get measured concentrations
    measured = met_X[:, camp_idx]
    valid = np.isfinite(measured) & np.isfinite(activity)
    if valid.sum() < 20:
        continue

    # Correlate
    pr, pp = stats.pearsonr(activity[valid], measured[valid])
    sr, sp = stats.spearmanr(activity[valid], measured[valid])

    results.append({
        "hmdb_id": hmdb_id,
        "metabolite_name": info["name"],
        "camp_name": met_names[camp_idx],
        "n_producing": len(prod_indices),
        "n_consuming": len(cons_indices),
        "n_samples": valid.sum(),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    })

results_df = pd.DataFrame(results)
print(f"\n{'='*80}")
print(f"MEBOCOST COMPARISON RESULTS")
print(f"{'='*80}")
print(f"Matched metabolites (MEBOCOST ∩ CAMP): {len(results_df)}")
print(f"")
print(f"Correlation with measured concentrations:")
print(f"  Median Pearson r:  {results_df['pearson_r'].median():+.3f}")
print(f"  Median Spearman r: {results_df['spearman_r'].median():+.3f}")
print(f"  Mean Pearson r:    {results_df['pearson_r'].mean():+.3f}")
print(f"  % positive Pearson: {100*(results_df['pearson_r'] > 0).mean():.0f}%")
print(f"  % significant (p<0.05): {100*(results_df['pearson_p'] < 0.05).mean():.0f}%")

# ---------------------------------------------------------------------------
# 4. Compare against GAZE LOMO predictions
# ---------------------------------------------------------------------------
lomo_json = project_root / "lomo_results/mcr_flux_mass_balance/baseline_mcr_20250929_064841/lomo_cross_validation/lomo_intermediate.json"
if lomo_json.exists():
    with open(lomo_json) as f:
        lomo_folds = json.load(f)

    print(f"\n{'='*80}")
    print(f"COMPARISON: MEBOCOST vs GAZE (LOMO metabolites)")
    print(f"{'='*80}")

    # Build LOMO lookup by metabolite_id
    lomo_lookup = {}
    for fold in lomo_folds:
        mid = fold["metabolite_id"]
        preds = np.array(fold["predictions"])
        tgts = np.array(fold["targets"])
        if len(preds) >= 5 and np.std(tgts) > 1e-10 and np.std(preds) > 1e-10:
            pr, pp = stats.pearsonr(preds, tgts)
            lomo_lookup[mid] = {"r2": fold["r2"], "pearson_r": pr, "pearson_p": pp}

    # Match MEBOCOST results to LOMO metabolites
    matched = []
    for _, row in results_df.iterrows():
        hmdb = row["hmdb_id"]
        # Try various HMDB normalizations
        candidates = [hmdb]
        if hmdb.startswith("HMDB"):
            candidates.append("HMDB" + hmdb[4:].zfill(7))
            candidates.append("HMDB" + hmdb[4:].lstrip("0"))

        for cand in candidates:
            if cand in lomo_lookup:
                matched.append({
                    "metabolite": row["metabolite_name"],
                    "hmdb_id": hmdb,
                    "mebocost_pearson_r": row["pearson_r"],
                    "gaze_pearson_r": lomo_lookup[cand]["pearson_r"],
                    "gaze_r2": lomo_lookup[cand]["r2"],
                })
                break

    if matched:
        match_df = pd.DataFrame(matched)
        print(f"Matched {len(match_df)} metabolites between MEBOCOST and GAZE LOMO")
        print(f"\n  {'Metabolite':<30} {'MEBOCOST_r':>10} {'GAZE_r':>10} {'GAZE_R²':>10}")
        print(f"  {'-'*62}")
        for _, r in match_df.sort_values("gaze_r2", ascending=False).iterrows():
            print(f"  {r['metabolite']:<30} {r['mebocost_pearson_r']:>+10.3f} {r['gaze_pearson_r']:>+10.3f} {r['gaze_r2']:>+10.3f}")
        print(f"\n  MEBOCOST median |r|: {match_df['mebocost_pearson_r'].abs().median():.3f}")
        print(f"  GAZE median |r|:     {match_df['gaze_pearson_r'].abs().median():.3f}")
    else:
        print("No overlapping metabolites found between MEBOCOST and GAZE LOMO")

# ---------------------------------------------------------------------------
# 5. Save all results
# ---------------------------------------------------------------------------
out_dir = project_root / "comparison_results"
out_dir.mkdir(exist_ok=True)
results_df.to_csv(out_dir / "mebocost_vs_camp.csv", index=False)
print(f"\nSaved results to {out_dir / 'mebocost_vs_camp.csv'}")

# Summary table for all MEBOCOST-matched metabolites (top and bottom)
print(f"\nTop 15 MEBOCOST correlations:")
print(f"  {'Metabolite':<40} {'Pearson_r':>10} {'p-val':>10} {'n_prod':>6} {'n_cons':>6}")
print(f"  {'-'*74}")
for _, r in results_df.nlargest(15, "pearson_r").iterrows():
    sig = "***" if r["pearson_p"] < 0.001 else "**" if r["pearson_p"] < 0.01 else "*" if r["pearson_p"] < 0.05 else ""
    print(f"  {r['metabolite_name']:<40} {r['pearson_r']:>+10.3f}{sig} {r['pearson_p']:>10.4f} {r['n_producing']:>6} {r['n_consuming']:>6}")

print(f"\nBottom 15 MEBOCOST correlations:")
for _, r in results_df.nsmallest(15, "pearson_r").iterrows():
    sig = "***" if r["pearson_p"] < 0.001 else "**" if r["pearson_p"] < 0.01 else "*" if r["pearson_p"] < 0.05 else ""
    print(f"  {r['metabolite_name']:<40} {r['pearson_r']:>+10.3f}{sig} {r['pearson_p']:>10.4f} {r['n_producing']:>6} {r['n_consuming']:>6}")
