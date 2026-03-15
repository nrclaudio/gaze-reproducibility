"""
Prepare UnitedMet ccRCC datasets as MuData for GAZE external validation.

Converts 4 UnitedMet CSV datasets (RC18, RC20, CPTAC, CPTAC_val) into a single
MuData file matching CAMP's structure, so FiLMDatasetWithSMILES can load it
directly.

Usage:
    conda run -n MetaboliteRNA python scripts/preprocessing/prepare_unitedmet_mudata.py
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import mudata as mu

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.utils.data_paths import get_mudata_path

# === Configuration ===
UNITEDMET_DIR = os.path.join(project_root, "data", "unitedmet")
OUTPUT_PATH = os.path.join(UNITEDMET_DIR, "unitedmet_ccRCC.h5mu")

DATASETS = {
    "RC18": {
        "rna": "matched_tpm_RC18.csv",
        "met": "matched_Harmonized_RawData_RC18.csv",
    },
    "RC20": {
        "rna": "matched_tpm_RC20.csv",
        "met": "matched_Harmonized_RawData_RC20.csv",
    },
    "CPTAC": {
        "rna": "matched_tpm_CPTAC.csv",
        "met": "matched_Harmonized_Met_CPTAC.csv",
    },
    "CPTAC_val": {
        "rna": "matched_tpm_CPTAC_val.csv",
        "met": "matched_Harmonized_Met_CPTAC_val.csv",
    },
}


def load_camp_mappings():
    """Load gene symbol -> Ensembl ID and metabolite name -> HMDB/KEGG mappings from CAMP."""
    camp_path = get_mudata_path()
    print(f"Loading CAMP MuData from {camp_path}")
    camp = mu.read_h5mu(camp_path)

    # Gene symbol -> Ensembl ID mapping
    gene_map = camp["rna"].var["ensembl_gene_id"].to_dict()
    print(f"  CAMP gene mapping: {len(gene_map)} gene symbols -> Ensembl IDs")

    # Metabolite name -> HMDB/KEGG mapping
    met_var = camp["metabolomics"].var[["HMDB", "KEGG", "HMDB_trailing_zeros"]].copy()
    print(f"  CAMP metabolite mapping: {len(met_var)} metabolites")

    return gene_map, met_var


def load_dataset(name, files):
    """Load a single UnitedMet dataset (RNA + metabolomics CSVs).

    RNA and metabolomics use different sample ID schemes (RNAID vs MetabID)
    but files are positionally matched (row i in RNA = row i in metabolomics).
    We assign shared sample IDs: {dataset}_{0..N}.
    """
    rna_path = os.path.join(UNITEDMET_DIR, files["rna"])
    met_path = os.path.join(UNITEDMET_DIR, files["met"])

    rna_df = pd.read_csv(rna_path, index_col=0)
    met_df = pd.read_csv(met_path, index_col=0)

    assert len(rna_df) == len(met_df), \
        f"{name}: RNA ({len(rna_df)}) and metabolomics ({len(met_df)}) have different sample counts!"

    # Create shared sample IDs
    shared_ids = [f"{name}_{i}" for i in range(len(rna_df))]
    rna_df.index = shared_ids
    met_df.index = shared_ids

    print(f"  {name}: RNA {rna_df.shape}, Metabolomics {met_df.shape}")
    return rna_df, met_df


def build_rna_anndata(rna_df, gene_map):
    """Build RNA AnnData with Ensembl IDs mapped from CAMP.

    Only keeps genes present in CAMP's gene mapping.
    """
    # Find overlapping genes
    common_genes = [g for g in rna_df.columns if g in gene_map]
    missing = len(rna_df.columns) - len(common_genes)

    # Subset to common genes
    rna_subset = rna_df[common_genes].copy()

    # Build var with ensembl_gene_id column (matching CAMP structure)
    var = pd.DataFrame(
        {"ensembl_gene_id": [gene_map[g] for g in common_genes]},
        index=pd.Index(common_genes, name="gene_symbol"),
    )

    # Store as dense float64 numpy to match CAMP's X format
    X = rna_subset.values.astype(np.float64)

    adata = ad.AnnData(X=X, var=var)
    adata.obs_names = rna_subset.index.astype(str)

    print(f"    RNA: {adata.shape[0]} samples x {adata.shape[1]} genes "
          f"(mapped {len(common_genes)}, dropped {missing})")
    return adata


def build_met_anndata(met_df, camp_met_var):
    """Build metabolomics AnnData with HMDB/KEGG IDs mapped from CAMP.

    Only keeps metabolites that exist in CAMP's metabolomics panel (have HMDB/KEGG).
    """
    # Match UnitedMet metabolite names to CAMP metabolite names
    camp_met_names = set(camp_met_var.index)
    common_mets = [m for m in met_df.columns if m in camp_met_names]
    missing = len(met_df.columns) - len(common_mets)

    # Subset
    met_subset = met_df[common_mets].copy()

    # Build var with HMDB/KEGG columns from CAMP
    var = camp_met_var.loc[common_mets].copy()

    # Log-transform: UnitedMet raw intensities are large positive values
    # CAMP metabolomics are log-transformed. Apply log2(1+x) to match.
    values = met_subset.values.astype(np.float64)
    # Replace NaN with 0 before log transform, then restore NaN
    nan_mask = np.isnan(values)
    values[nan_mask] = 0
    values = np.log2(1 + values).astype(np.float32)
    values[nan_mask] = np.nan

    # Store as dense float64 numpy to match CAMP's X format
    X = values.astype(np.float64)

    adata = ad.AnnData(X=X, var=var)
    adata.obs_names = met_subset.index.astype(str)

    # Count metabolites that have valid HMDB or KEGG IDs (graph-mapped)
    has_hmdb = var["HMDB"].notna().sum()
    has_kegg = var["KEGG"].notna().sum()
    has_either = (var["HMDB"].notna() | var["KEGG"].notna()).sum()

    print(f"    Metabolomics: {adata.shape[0]} samples x {adata.shape[1]} metabolites "
          f"(matched {len(common_mets)}, dropped {missing})")
    print(f"    Graph-mappable: {has_either} (HMDB: {has_hmdb}, KEGG: {has_kegg})")
    return adata


def main():
    print("=" * 60)
    print("Preparing UnitedMet ccRCC datasets for GAZE evaluation")
    print("=" * 60)

    # Load CAMP reference mappings
    gene_map, camp_met_var = load_camp_mappings()

    # Process each dataset
    all_rna = []
    all_met = []
    all_dataset_labels = []

    for name, files in DATASETS.items():
        print(f"\nProcessing {name}...")
        rna_df, met_df = load_dataset(name, files)

        rna_adata = build_rna_anndata(rna_df, gene_map)
        met_adata = build_met_anndata(met_df, camp_met_var)

        # Verify sample alignment (same sample IDs in both modalities)
        assert list(rna_adata.obs_names) == list(met_adata.obs_names), \
            f"{name}: RNA and metabolomics sample IDs don't match!"

        n_samples = rna_adata.shape[0]
        all_dataset_labels.extend([name] * n_samples)

        all_rna.append(rna_adata)
        all_met.append(met_adata)

    # Concatenate all datasets
    print("\nConcatenating datasets...")

    # Inner join for genes (keep genes present in ALL datasets).
    # Outer join for metabolites — each dataset has different coverage;
    # per-sample masks in the model handle which metabolites are available.
    # Use merge="first" so var columns (HMDB, KEGG, etc.) are preserved from
    # the first dataset that contains each metabolite.
    rna_combined = ad.concat(all_rna, join="inner", merge="same")
    met_combined = ad.concat(all_met, join="outer", merge="first")

    # Make obs_names unique (some sample IDs might overlap across datasets)
    rna_combined.obs_names_make_unique()
    met_combined.obs_names_make_unique()

    # Sync obs_names between modalities
    met_combined.obs_names = rna_combined.obs_names

    # Add dataset labels
    rna_combined.obs["dataset"] = all_dataset_labels
    met_combined.obs["dataset"] = all_dataset_labels

    # Add numerical_index and original_index to match CAMP structure
    rna_combined.obs["numerical_index"] = range(len(rna_combined))
    rna_combined.obs["original_index"] = rna_combined.obs_names.tolist()
    met_combined.obs["numerical_index"] = range(len(met_combined))
    met_combined.obs["original_index"] = met_combined.obs_names.tolist()

    print(f"\nCombined RNA: {rna_combined.shape}")
    print(f"Combined Metabolomics: {met_combined.shape}")
    print(f"Dataset distribution: {pd.Series(all_dataset_labels).value_counts().to_dict()}")

    # Create MuData
    mdata = mu.MuData({"rna": rna_combined, "metabolomics": met_combined})

    print(f"\nMuData: {mdata}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}")
    mdata.write(OUTPUT_PATH)
    print("Done!")

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    mdata_check = mu.read_h5mu(OUTPUT_PATH)
    print(f"Loaded MuData: {mdata_check.n_obs} samples x {mdata_check.n_vars} features")
    print(f"  RNA: {mdata_check['rna'].shape}")
    print(f"  Metabolomics: {mdata_check['metabolomics'].shape}")
    print(f"  RNA var columns: {mdata_check['rna'].var.columns.tolist()}")
    print(f"  Met var columns: {mdata_check['metabolomics'].var.columns.tolist()}")
    print(f"  Datasets: {mdata_check['rna'].obs['dataset'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
