"""Score a frozen reanalysis checkpoint, preserving exact specimen/target identities."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import ExperimentConfig
from src.data.dataset_smiles import FiLMDatasetWithSMILES
from src.data.sample_splits import manifest_splits, verify_sha256
from src.models.factory import create_model
from scripts.experiments.train_cohort_reanalysis import verify_artifacts, code_hashes, digest


def target_metrics(predictions, targets, nodes, sample_metadata, target_metadata):
    records = []
    groups = [('pooled_secondary', np.ones(len(sample_metadata), dtype=bool))]
    for cohort in sorted(sample_metadata.holdout_cohort.unique()):
        cohort_mask = sample_metadata.holdout_cohort.eq(cohort).to_numpy()
        groups.append((str(cohort), cohort_mask))
        for tissue in ['Tumor', 'Normal']:
            groups.append((f'{cohort}_{tissue}_descriptive', cohort_mask & sample_metadata.TN.eq(tissue).to_numpy()))
    for group, samples in groups:
        for j, node in enumerate(nodes):
            true, pred = targets[samples, j], predictions[samples, j]
            observed = np.isfinite(true)
            if not np.isfinite(pred[observed]).all():
                raise ValueError('Non-finite predictions for observed outcomes; evaluation refused.')
            true, pred = true[observed], pred[observed]
            n = len(true)
            row = {'group': group, 'node_idx': int(node), **target_metadata[int(node)],
                   'n_observed': n, 'n_samples': int(samples.sum()), 'pearson_r': np.nan,
                   'spearman_r': np.nan, 'r2': np.nan, 'rmse': np.nan}
            if n >= 3:
                row['rmse'] = float(np.sqrt(np.mean((true - pred) ** 2)))
                if np.ptp(true) > 0:
                    row['r2'] = float(r2_score(true, pred))
                    if np.ptp(pred) > 0:
                        row['pearson_r'] = float(pearsonr(true, pred).statistic)
                        row['spearman_r'] = float(spearmanr(true, pred).statistic)
            records.append(row)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--secondary', action='store_true', help='Same specimens on the UnitedMet processing scale; sensitivity analysis only.')
    parser.add_argument('--smoke', action='store_true', help='Verify a smoke checkpoint on development rows only; never score holdout.')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    artifacts, run, out = args.artifacts.resolve(), args.run.resolve(), args.output.resolve()
    summary = verify_artifacts(artifacts)
    status = json.loads((run / 'run_status.json').read_text())
    required_state = 'smoke_complete' if args.smoke else 'training_complete'
    if status['state'] != required_state or status['smoke_only'] != args.smoke:
        raise ValueError('Only a completed, appropriately designated run can be evaluated.')
    if args.smoke and args.secondary:
        raise ValueError('Smoke checks cannot use external specimens.')
    if summary['artifact_hashes'] != status['artifact_hashes'] or code_hashes() != status['code_hashes']:
        raise ValueError('Evaluation code/data differ from the frozen training record.')
    checkpoint, config_path = run / 'phase2/best_model.pt', run / 'phase2/experiment_config.json'
    verify_sha256(checkpoint, status['final_checkpoint_sha256'])
    verify_sha256(config_path, status['final_config_sha256'])
    verify_sha256(run / 'phase1/experiment_config.json', status['phase1_config_sha256'])
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_file = 'unitedmet_rc_secondary.h5mu' if args.secondary else 'camp_reanalysis.h5mu'
    dataset = FiLMDatasetWithSMILES(str(artifacts / data_file), str(artifacts / 'graph.pt'),
                                  str(artifacts / 'gene_mapping.json'), device=device)
    manifest = pd.read_csv(artifacts / 'phase2_manifest.csv', dtype=str, keep_default_na=False)
    if args.secondary:
        sample_metadata = pd.DataFrame({
            'rna_original_id': dataset.mudata['rna'].obs.original_index.to_numpy(),
            'metabolomics_original_id': dataset.mudata['metabolomics'].obs.original_index.to_numpy()})
        sample_metadata = sample_metadata.merge(manifest, on=['rna_original_id', 'metabolomics_original_id'],
                                                how='left', validate='one_to_one')
        if not sample_metadata.partition.eq('holdout').all() or len(sample_metadata) != 220:
            raise ValueError('Secondary specimen identities do not match the frozen holdout.')
        indices = list(range(len(dataset)))
    else:
        splits = manifest_splits(artifacts / 'phase2_manifest.csv', dataset.mudata['rna'].obs,
                                 dataset.mudata['metabolomics'].obs)
        indices = splits['train'][:4] if args.smoke else splits['holdout']
        sample_metadata = manifest.set_index('camp_obs_id').loc[
            dataset.mudata['rna'].obs.index[indices].astype(str)].reset_index()
        if args.smoke:
            sample_metadata['holdout_cohort'] = 'development_smoke_only'
    config = ExperimentConfig.from_dict(json.loads(config_path.read_text()))
    model = create_model(config.model, embedding_dim=dataset.embedding_dim, gene_dim=dataset.gene_dim,
                         smiles_dim=dataset.smiles_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    # Evaluation obtains observed truth; the separately verified training config
    # masks the target panel in BOTH development fitting and epoch selection.
    nodes = sorted(set(dataset.feature_nodes[dataset.feature_nodes >= 0].tolist()))
    if status['mode'] == 'cohort_plus_targets':
        panel = json.loads((artifacts / 'heldout_target_nodes.json').read_text())
        if sorted(config.data.excluded_target_nodes) != panel:
            raise ValueError('Checkpoint was not configured to withhold the expected target panel.')
        phase1 = ExperimentConfig.from_dict(json.loads((run / 'phase1/experiment_config.json').read_text()))
        if sorted(phase1.data.excluded_target_nodes) != panel:
            raise ValueError('Epoch selection did not exclude the expected target panel.')
        nodes = panel
    target_metadata = {}
    for name, node in zip(dataset.mudata['metabolomics'].var.index, dataset.feature_nodes):
        if node >= 0:
            target_metadata[int(node)] = {'feature': str(name), 'hmdb': str(dataset.graph.hmdb_ids[node]),
                                          'kegg': str(dataset.graph.kegg_ids[node])}
    predictions, targets = [], []
    with torch.no_grad():
        for position, idx in enumerate(indices):
            data = dataset[idx].to(device)
            prediction = model(data, compute_physics=False)[0].reshape(-1)
            predictions.append(prediction[nodes].cpu().numpy())
            targets.append(data.y.reshape(-1)[nodes].cpu().numpy())
            if (position + 1) % 25 == 0:
                print(f'Predicted {position + 1}/{len(indices)} specimens', flush=True)
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    metrics = target_metrics(predictions, targets, nodes, sample_metadata, target_metadata)
    out.mkdir(parents=True, exist_ok=False)
    sample_metadata.to_csv(out / 'samples.csv', index=False)
    metrics.to_csv(out / 'per_target_metrics.csv', index=False)
    np.savez_compressed(out / 'predictions.npz', predictions=predictions, targets=targets,
                        node_indices=np.asarray(nodes), rna_ids=sample_metadata.rna_original_id.to_numpy(dtype=str),
                        metabolomics_ids=sample_metadata.metabolomics_original_id.to_numpy(dtype=str))
    aggregate = metrics.groupby('group').agg(targets_scored=('pearson_r', 'count'),
        median_pearson=('pearson_r', 'median'), median_spearman=('spearman_r', 'median'),
        median_r2=('r2', 'median'), median_rmse=('rmse', 'median')).reset_index()
    aggregate.to_csv(out / 'summary.csv', index=False)
    record = {'smoke_only': args.smoke, 'processing_sensitivity_only': args.secondary,
        'run_status_sha256': digest(run / 'run_status.json'), 'checkpoint_sha256': digest(checkpoint),
        'scoring': 'Per target: at least 3 observed outcomes. Undefined constant-target correlations retained as missing. Primary summaries are separate within-cohort medians; pooled correlation is explicitly secondary. No calibration.',
        'limitations': 'Retrospective holdout; source-study preprocessing retained. Universal donor identities are unavailable; no patient-independent confidence intervals are asserted.',
        'output_hashes': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    (out / 'evaluation_record.json').write_text(json.dumps(record, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == '__main__':
    main()
