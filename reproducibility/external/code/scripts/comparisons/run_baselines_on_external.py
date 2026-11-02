#!/usr/bin/env python3
"""
Run MEBOCOST and scCellFie on the external UnitedMet ccRCC cohort.

Computes per-metabolite Spearman/Pearson correlations between predicted
activity scores and measured metabolite concentrations, enabling direct
comparison against GAZE external evaluation results.

Usage:
    python scripts/comparisons/run_baselines_on_external.py \
        --mudata-path data/unitedmet/unitedmet_ccRCC.h5mu \
        --output-dir results/external_baselines/
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).resolve().parent.parent.parent


def load_external_data(mudata_path):
    """Load UnitedMet MuData and return RNA expression + metabolomics matrices."""
    import mudata as mu

    mdata = mu.read(str(mudata_path))
    rna = mdata.mod["rna"]
    met = mdata.mod["metabolomics"]

    rna_X = rna.X
    if hasattr(rna_X, "toarray"):
        rna_X = rna_X.toarray()

    met_X = met.X
    if hasattr(met_X, "toarray"):
        met_X = met_X.toarray()

    gene_names = list(rna.var_names)
    met_names = list(met.var_names)

    # Build HMDB lookup
    hmdb_col = met.var.get("HMDB", met.var.get("hmdb", None))
    hmdb_to_idx = {}
    if hmdb_col is not None:
        for k in range(len(met_names)):
            hid = str(hmdb_col.iloc[k])
            if hid and hid != "nan":
                hmdb_to_idx[hid] = k
                if hid.startswith("HMDB") and len(hid) < 13:
                    hmdb_to_idx["HMDB" + hid[4:].zfill(7)] = k

    # Dataset labels per sample
    dataset_labels = None
    if "dataset" in rna.obs.columns:
        dataset_labels = rna.obs["dataset"].tolist()

    print(f"External data: {rna_X.shape[0]} samples, {rna_X.shape[1]} genes, {met_X.shape[1]} metabolites")
    print(f"HMDB-mapped metabolites: {len(hmdb_to_idx)}")

    return rna_X, met_X, gene_names, met_names, hmdb_to_idx, dataset_labels


# ---------------------------------------------------------------------------
# MEBOCOST
# ---------------------------------------------------------------------------

def run_mebocost(rna_X, met_X, gene_names, met_names, hmdb_to_idx):
    """Run MEBOCOST on external data and correlate with measured concentrations."""
    # Load MEBOCOST database
    mebocost_db_path = project_root.parent / "MEBOCOST_official/data/mebocost_db/human/metabolite_associated_gene_reaction_HMDB_summary.tsv"
    if not mebocost_db_path.exists():
        print(f"MEBOCOST database not found at {mebocost_db_path}")
        print("Skipping MEBOCOST comparison.")
        return None

    met_enzyme_db = pd.read_csv(mebocost_db_path, sep="\t")
    print(f"\nMEBOCOST database: {len(met_enzyme_db)} entries, {met_enzyme_db['HMDB_ID'].nunique()} metabolites")

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

    # Build per-metabolite gene sets
    met_genes = {}
    for _, row in met_enzyme_db.iterrows():
        hmdb = row["HMDB_ID"]
        if hmdb not in met_genes:
            met_genes[hmdb] = {"producing": set(), "consuming": set(), "name": row["metabolite"]}
        genes = parse_genes(row["gene"])
        if row["direction"] == "product":
            met_genes[hmdb]["producing"].update(genes)
        elif row["direction"] == "substrate":
            met_genes[hmdb]["consuming"].update(genes)

    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_samples = rna_X.shape[0]
    results = []

    for hmdb_id, info in met_genes.items():
        # Find in external metabolomics
        met_idx = hmdb_to_idx.get(hmdb_id)
        if met_idx is None and hmdb_id.startswith("HMDB"):
            met_idx = hmdb_to_idx.get("HMDB" + hmdb_id[4:].zfill(7))
        if met_idx is None:
            continue

        prod_indices = [gene_name_to_idx[g] for g in info["producing"] if g in gene_name_to_idx]
        cons_indices = [gene_name_to_idx[g] for g in info["consuming"] if g in gene_name_to_idx]
        if not prod_indices and not cons_indices:
            continue

        activity = np.zeros(n_samples)
        if prod_indices:
            activity += rna_X[:, prod_indices].mean(axis=1)
        if cons_indices:
            activity -= rna_X[:, cons_indices].mean(axis=1)

        measured = met_X[:, met_idx]
        valid = np.isfinite(measured) & np.isfinite(activity)
        if valid.sum() < 3:
            continue

        pr, pp = stats.pearsonr(activity[valid], measured[valid])
        sr, sp = stats.spearmanr(activity[valid], measured[valid])

        results.append({
            "hmdb_id": hmdb_id,
            "metabolite_name": info["name"],
            "external_name": met_names[met_idx],
            "n_producing": len(prod_indices),
            "n_consuming": len(cons_indices),
            "n_samples": int(valid.sum()),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        })

    if not results:
        print("MEBOCOST: no matching metabolites found")
        return None

    df = pd.DataFrame(results)
    print(f"\nMEBOCOST on external data:")
    print(f"  Metabolites evaluated: {len(df)}")
    print(f"  Median Spearman rho: {df['spearman_rho'].median():.4f}")
    print(f"  Mean Spearman rho:   {df['spearman_rho'].mean():.4f}")
    print(f"  Median Pearson r:    {df['pearson_r'].median():.4f}")
    print(f"  % positive rho:     {100 * (df['spearman_rho'] > 0).mean():.0f}%")
    return df


# ---------------------------------------------------------------------------
# scCellFie
# ---------------------------------------------------------------------------

def run_sccellfie(rna_X, met_X, gene_names, met_names):
    """Run scCellFie on external data and correlate with measured concentrations."""
    try:
        import anndata as ad
        import sccellfie
    except ImportError:
        print("scCellFie not installed. Skipping.")
        return None

    adata = ad.AnnData(
        X=rna_X.copy(),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs_names = [f"sample_{i}" for i in range(adata.n_obs)]

    print("\nRunning scCellFie pipeline on external data...")
    try:
        results = sccellfie.run_sccellfie_pipeline(
            adata,
            organism='human',
            smooth_cells=False,
            n_counts_col=None,
            disable_pbar=True,
        )
        task_adata = results['adata'].metabolic_tasks
        task_scores = task_adata.X
        if hasattr(task_scores, "toarray"):
            task_scores = task_scores.toarray()
        task_names = list(task_adata.var_names)
        print(f"scCellFie completed: {task_scores.shape[1]} tasks scored")
    except Exception as e:
        print(f"scCellFie pipeline failed: {e}")
        print("Falling back to manual task scoring...")

        from sccellfie.datasets.database import load_sccellfie_database
        db = load_sccellfie_database(organism='human')
        task_info = db['task_info']
        task_by_rxn = db['task_by_rxn']
        rxn_by_gene = db['rxn_by_gene']

        gene_df = pd.DataFrame(rna_X, columns=gene_names)
        gene_overlap = [g for g in rxn_by_gene.columns if g in gene_df.columns]
        print(f"  Gene overlap: {len(gene_overlap)}/{len(rxn_by_gene.columns)} scCellFie genes")

        rxn_scores = np.zeros((rna_X.shape[0], len(rxn_by_gene.index)))
        for j, rxn in enumerate(rxn_by_gene.index):
            rxn_gene_list = [g for g in rxn_by_gene.columns[rxn_by_gene.loc[rxn] > 0] if g in gene_df.columns]
            if rxn_gene_list:
                rxn_scores[:, j] = gene_df[rxn_gene_list].values.mean(axis=1)

        task_by_rxn_mat = task_by_rxn.values
        task_scores = np.zeros((rna_X.shape[0], len(task_info)))
        for t in range(len(task_info)):
            active_rxns = np.where(task_by_rxn_mat[t] > 0)[0]
            if len(active_rxns) > 0:
                task_scores[:, t] = rxn_scores[:, active_rxns].min(axis=1)

        task_names = list(task_info['Task'])
        print(f"  Scored {len(task_names)} tasks")

    # Map task names to external metabolites
    def extract_metabolite(task_name):
        task_lower = task_name.lower()
        for suffix in [" synthesis", " degradation", " to ", " from ", " salvage"]:
            if suffix in task_lower:
                return task_lower[:task_lower.index(suffix)].strip()
        return task_lower.strip()

    def normalize_name(name):
        n = name.lower().strip()
        n = re.sub(r'[_\-\s]+', ' ', n)
        n = re.sub(r'\(.*?\)', '', n).strip()
        return n

    met_names_lower = {normalize_name(n): i for i, n in enumerate(met_names)}
    for i, n in enumerate(met_names):
        met_names_lower[n.lower()] = i

    comparison_results = []
    for t_idx, task_name in enumerate(task_names):
        met_name = extract_metabolite(task_name)
        met_norm = normalize_name(met_name)

        met_idx = met_names_lower.get(met_norm)
        if met_idx is None:
            for camp_name_lower, idx in met_names_lower.items():
                if met_norm in camp_name_lower or camp_name_lower in met_norm:
                    met_idx = idx
                    break
        if met_idx is None:
            continue

        t_scores = task_scores[:, t_idx]
        m_values = met_X[:, met_idx]
        valid = np.isfinite(t_scores) & np.isfinite(m_values)
        if valid.sum() < 3 or np.std(t_scores[valid]) < 1e-10:
            continue

        pr, pp = stats.pearsonr(t_scores[valid], m_values[valid])
        sr, sp = stats.spearmanr(t_scores[valid], m_values[valid])

        comparison_results.append({
            "task_name": task_name,
            "external_metabolite": met_names[met_idx],
            "n_samples": int(valid.sum()),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        })

    if not comparison_results:
        print("scCellFie: no matching metabolites found in external data")
        return None

    df = pd.DataFrame(comparison_results)
    print(f"\nscCellFie on external data:")
    print(f"  Task-metabolite pairs: {len(df)}")
    print(f"  Median Spearman rho: {df['spearman_rho'].median():.4f}")
    print(f"  Mean Spearman rho:   {df['spearman_rho'].mean():.4f}")
    print(f"  Median Pearson r:    {df['pearson_r'].median():.4f}")
    print(f"  % positive rho:     {100 * (df['spearman_rho'] > 0).mean():.0f}%")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run baseline methods on external UnitedMet data")
    parser.add_argument("--mudata-path", type=str,
                        default=str(project_root / "data/unitedmet/unitedmet_ccRCC.h5mu"))
    parser.add_argument("--output-dir", type=str,
                        default=str(project_root / "results/external_baselines"))
    parser.add_argument("--skip-mebocost", action="store_true")
    parser.add_argument("--skip-sccellfie", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load external data
    rna_X, met_X, gene_names, met_names, hmdb_to_idx, dataset_labels = load_external_data(args.mudata_path)

    summary = {"n_samples": int(rna_X.shape[0]), "n_metabolites": int(met_X.shape[1])}

    # MEBOCOST
    mebocost_df = None
    if not args.skip_mebocost:
        mebocost_df = run_mebocost(rna_X, met_X, gene_names, met_names, hmdb_to_idx)
        if mebocost_df is not None:
            mebocost_df.to_csv(os.path.join(args.output_dir, "mebocost_external.csv"), index=False)
            summary["mebocost"] = {
                "n_metabolites": len(mebocost_df),
                "median_spearman": float(mebocost_df["spearman_rho"].median()),
                "mean_spearman": float(mebocost_df["spearman_rho"].mean()),
                "pct_positive": float((mebocost_df["spearman_rho"] > 0).mean()),
            }

    # scCellFie
    sccellfie_df = None
    if not args.skip_sccellfie:
        sccellfie_df = run_sccellfie(rna_X, met_X, gene_names, met_names)
        if sccellfie_df is not None:
            sccellfie_df.to_csv(os.path.join(args.output_dir, "sccellfie_external.csv"), index=False)
            summary["sccellfie"] = {
                "n_metabolites": len(sccellfie_df),
                "median_spearman": float(sccellfie_df["spearman_rho"].median()),
                "mean_spearman": float(sccellfie_df["spearman_rho"].mean()),
                "pct_positive": float((sccellfie_df["spearman_rho"] > 0).mean()),
            }

    # Print comparison summary
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Dataset: {args.mudata_path}")
    print(f"Samples: {rna_X.shape[0]}")
    print()

    # Load GAZE results if available
    gaze_results = os.path.join(project_root, "results/external_unitedmet/per_metabolite_results.csv")
    if os.path.exists(gaze_results):
        gaze_df = pd.read_csv(gaze_results)
        print(f"  GAZE:      n={len(gaze_df):>3}, median rho={gaze_df['spearman_rho'].median():+.3f}, "
              f">0: {100 * (gaze_df['spearman_rho'] > 0).mean():.0f}%")
    if mebocost_df is not None:
        print(f"  MEBOCOST:  n={len(mebocost_df):>3}, median rho={mebocost_df['spearman_rho'].median():+.3f}, "
              f">0: {100 * (mebocost_df['spearman_rho'] > 0).mean():.0f}%")
    if sccellfie_df is not None:
        print(f"  scCellFie: n={len(sccellfie_df):>3}, median rho={sccellfie_df['spearman_rho'].median():+.3f}, "
              f">0: {100 * (sccellfie_df['spearman_rho'] > 0).mean():.0f}%")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
