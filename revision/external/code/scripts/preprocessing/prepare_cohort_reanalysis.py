"""Build explicit, study-grouped reanalysis artifacts from the ALICE snapshot.

Original inputs are read-only. Outcomes come from the source workbooks' `data`
sheet (before imputation); no outcome-dependent preprocessing is fitted here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import anndata as ad
import h5py
import mudata as mu
import numpy as np
import pandas as pd
from anndata.io import read_elem
import torch
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.metabolite_mapping import metabolite_feature_nodes


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    alice = workspace / 'cluster-downloads/ALICE/concat-smiles-model'
    source = workspace / 'cluster-downloads/ALICE/Metabolomics-RNA/source-data/geo_data/Tumour_metabolism/pancancer_metabolomics/data'
    camp_path = alice / 'data/raw/mudata_correct_HMDB_new_no_negatives.h5mu'
    graph_path = alice / 'data/graphs/PyG_merged_graph_with_SMILES_with_embeddings/merged_metabolic_network.pt'
    gene_path = alice / 'data/raw/gene_mapping.json'
    master_path = source / 'MasterMapping_MetImmune_03_16_2022_release.csv'
    inputs = {str(p.relative_to(workspace)): digest(p) for p in [camp_path, graph_path, gene_path, master_path]}
    original = pd.read_csv(workspace / 'external_validation_recovery/split_manifest.csv', dtype=str, keep_default_na=False)
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    manifest = original.merge(master, left_on=['rna_original_id', 'metabolomics_original_id'],
                              right_on=['RNAID', 'MetabID'], how='left', validate='one_to_one')
    if manifest.Dataset.isna().any():
        raise ValueError('Every sample must match the original source-study mapping.')
    manifest['source_group'] = manifest.RNAFile
    manifest.loc[manifest.partition == 'cohort_holdout', 'partition'] = 'holdout'
    # No claim that different kidney-study identifiers necessarily identify
    # different donors. Exclude the remaining kidney cohort conservatively.
    extra = manifest.Dataset.str.startswith('ccRCC') & manifest.partition.ne('holdout')
    manifest.loc[extra, 'partition'] = 'excluded'
    manifest['exclusion_reason'] = np.where(extra, 'additional_renal_cohort_precaution', '')
    development = manifest.partition.eq('development_candidate')
    development_rows = np.flatnonzero(development)
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
        development_rows, groups=manifest.loc[development, 'source_group']))
    manifest.loc[development_rows[train], 'partition'] = 'train'
    manifest.loc[development_rows[val], 'partition'] = 'validation'
    phase2 = manifest.copy()
    phase2.loc[phase2.partition.eq('validation'), 'partition'] = 'train'
    manifest.to_csv(out / 'phase1_manifest.csv', index=False)
    phase2.to_csv(out / 'phase2_manifest.csv', index=False)

    with h5py.File(camp_path, 'r') as handle:
        rna_obs = read_elem(handle['mod/rna/obs'])
        met_obs = read_elem(handle['mod/metabolomics/obs'])
        rna_var = read_elem(handle['mod/rna/var'])
        met_var = read_elem(handle['mod/metabolomics/var'])
        rna_values = read_elem(handle['mod/rna/X'])
    if rna_obs.index.astype(str).tolist() != manifest.camp_obs_id.tolist():
        raise ValueError('Manifest does not match the input row order.')
    # Match the historical Ensembl-to-symbol convention, freezing it from the
    # supplied annotation table rather than using test expression variability.
    mapping = json.loads(gene_path.read_text())['gene_to_index']
    genes = sorted(mapping, key=mapping.get)
    last_column = {str(gene): i for i, gene in enumerate(rna_var.ensembl_gene_id)}
    selected = [last_column[gene] for gene in genes]
    symbols = rna_var.index[selected].astype(str).tolist()
    fixed_var = pd.DataFrame({'ensembl_gene_id': genes, 'source_gene_symbol': symbols}, index=pd.Index(genes, name='gene_id'))
    rna_values = np.asarray(rna_values[:, selected], dtype=np.float32)

    # Restore measured-only outcomes using source harmonized feature names.
    outcomes = np.full((len(manifest), len(met_var)), np.nan, dtype=np.float32)
    workbook_summary = []
    for filename, rows in manifest.groupby('MetabFile', sort=True):
        workbook = source / 'metabolomics_processed' / filename
        values = pd.read_excel(workbook, sheet_name='data', index_col=0)
        if not values.index.is_unique:
            raise ValueError(f'Duplicate harmonized feature names in {workbook}.')
        missing = set(rows.MetabID) - set(values.columns)
        if missing:
            raise ValueError(f'Missing source assay sample IDs: {sorted(missing)}')
        block = values.reindex(index=met_var.index, columns=rows.MetabID).T.to_numpy(dtype=np.float32)
        outcomes[rows.index] = block
        inputs[str(workbook.relative_to(workspace))] = digest(workbook)
        workbook_summary.append({'file': filename, 'samples': len(rows), 'finite_observations': int(np.isfinite(block).sum()),
                                 'negative_observations_retained': int((block < 0).sum())})
        print(f'Restored measured values: {filename}, {len(rows)} samples', flush=True)

    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    hmdb = {x: i for i, x in enumerate(graph.hmdb_ids) if x and graph.node_types[i] == 'metabolite'}
    kegg = {x: i for i, x in enumerate(graph.kegg_ids) if x and graph.node_types[i] == 'metabolite'}
    candidates = defaultdict(list)
    for name, row in met_var.iterrows():
        node = hmdb.get(row.HMDB_trailing_zeros, kegg.get(row.KEGG))
        if node is not None:
            candidates[int(node)].append(str(name))
    ambiguous = {node: names for node, names in candidates.items() if len(names) > 1}
    excluded_features = {name for names in ambiguous.values() for name in names}
    met_var['exclude_from_prediction'] = met_var.index.isin(excluded_features)
    issues = [{'node_idx': node, 'feature': name, 'decision': 'exclude_unresolved_many_features_to_one_node',
               'other_features': ';'.join(x for x in names if x != name)}
              for node, names in sorted(ambiguous.items()) for name in names]
    pd.DataFrame(issues).to_csv(out / 'mapping_exclusions.csv', index=False)
    nodes = metabolite_feature_nodes(met_var, hmdb, kegg)
    gene_missing = np.sum(~np.isfinite(rna_values), axis=1)
    for obs, ids in [(rna_obs, manifest.rna_original_id), (met_obs, manifest.metabolomics_original_id)]:
        obs['original_index'] = ids.to_numpy()
        for column in ['Dataset', 'TN', 'source_group', 'holdout_cohort']:
            obs[column] = manifest[column].to_numpy()
        obs['missing_model_genes'] = gene_missing
    prepared = mu.MuData({'rna': ad.AnnData(rna_values, obs=rna_obs, var=fixed_var),
                          'metabolomics': ad.AnnData(outcomes, obs=met_obs, var=met_var)})
    prepared.write_h5mu(out / 'camp_reanalysis.h5mu')
    shutil.copy2(graph_path, out / 'graph.pt')
    shutil.copy2(gene_path, out / 'gene_mapping.json')

    # Joint holdout panel is chosen from development coverage only, independently
    # of held-out cohort values or performance. Assay labels are later masked in
    # both model fitting and validation used to choose epochs.
    eligible = []
    dev_values = outcomes[development.to_numpy()]
    for column, node in enumerate(nodes):
        if node >= 0 and np.isfinite(dev_values[:, column]).sum() >= 0.2 * len(dev_values):
            eligible.append(int(node))
    heldout_nodes = sorted(np.random.RandomState(101).choice(sorted(set(eligible)), size=min(20, len(set(eligible))), replace=False).tolist())
    (out / 'heldout_target_nodes.json').write_text(json.dumps(heldout_nodes, indent=2))

    # Secondary UnitedMet preparation: restore the full annotation table and a
    # fixed model-gene panel; preserve actual original IDs in both modalities.
    raw = workspace / 'concat-smiles-model_archive/data/unitedmet'
    rna_parts, met_parts = [], []
    for cohort, metfile in [('RC18', 'matched_Harmonized_RawData_RC18.csv'), ('RC20', 'matched_Harmonized_RawData_RC20.csv')]:
        rp, mp = raw / f'matched_tpm_{cohort}.csv', raw / metfile
        rr, mm = pd.read_csv(rp, index_col=0), pd.read_csv(mp, index_col=0)
        known = set(zip(manifest.loc[manifest.holdout_cohort.eq(cohort), 'rna_original_id'],
                        manifest.loc[manifest.holdout_cohort.eq(cohort), 'metabolomics_original_id']))
        if set(zip(rr.index.astype(str), mm.index.astype(str))) != known:
            raise ValueError('External CSV sample pairs differ from the verified manifest.')
        ids = pd.Index([f'{cohort}_{i}' for i in range(len(rr))])
        robs = pd.DataFrame({'original_index': rr.index.astype(str), 'dataset': cohort}, index=ids)
        mobs = pd.DataFrame({'original_index': mm.index.astype(str), 'dataset': cohort}, index=ids)
        rx = rr.reindex(columns=symbols).to_numpy(dtype=np.float32)
        mx = mm.reindex(columns=met_var.index).to_numpy(dtype=np.float64)
        if (mx[np.isfinite(mx)] < 0).any():
            raise ValueError('Unexpected negative raw external intensity; log2(1+x) is not appropriate without review.')
        mx = np.log2(1 + mx).astype(np.float32)
        rna_parts.append(ad.AnnData(rx, obs=robs, var=fixed_var.copy()))
        met_parts.append(ad.AnnData(mx, obs=mobs, var=met_var.copy()))
        inputs[str(rp.relative_to(workspace))] = digest(rp)
        inputs[str(mp.relative_to(workspace))] = digest(mp)
    secondary_rna, secondary_met = ad.concat(rna_parts, merge='same'), ad.concat(met_parts, merge='same')
    secondary_rna.var = fixed_var.copy()
    secondary_met.var = met_var.copy()
    mu.MuData({'rna': secondary_rna, 'metabolomics': secondary_met}).write_h5mu(out / 'unitedmet_rc_secondary.h5mu')
    summary = {'input_hashes': inputs, 'phase1_counts': manifest.partition.value_counts().to_dict(),
               'phase2_counts': phase2.partition.value_counts().to_dict(),
               'validation_source_groups': sorted(manifest.loc[manifest.partition.eq('validation'), 'source_group'].unique()),
               'holdout_tissue_counts': manifest.loc[manifest.partition.eq('holdout'), 'TN'].value_counts().to_dict(),
               'donor_policy': 'All kidney cohorts are held out or excluded; entire RNA source studies are kept together in development splitting. No universal cross-study patient-ID proof is claimed.',
               'targets': 'Publisher/source workbook data sheets before imputation; signed processed values retained, no new target imputation or clipping.',
               'source_processing': 'Existing source-study preprocessing is retained; groups sharing RNA inputs are kept in one partition. No model-level preprocessing statistics are fitted on holdout data.',
               'primary_evaluation': 'CAMP held-out rows on the measured-only source-workbook scale.',
               'secondary_evaluation': 'Same RC18/RC20 specimens from UnitedMet CSVs on log2(1+raw intensity); sensitivity to processing, not a second independent cohort.',
               'graph_nodes': len(graph.node_types), 'ambiguous_graph_targets_excluded': len(ambiguous),
               'ambiguous_source_features_excluded': len(excluded_features), 'joint_holdout_target_count': len(heldout_nodes),
               'workbooks': workbook_summary}
    summary['artifact_hashes'] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'preparation_summary.json'}
    (out / 'preparation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ['phase1_counts', 'phase2_counts', 'holdout_tissue_counts', 'ambiguous_graph_targets_excluded']}, indent=2))


if __name__ == '__main__':
    main()
