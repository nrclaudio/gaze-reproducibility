"""Regression tests for the scientific failure modes found in the ALICE audit."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.data.metabolite_mapping import metabolite_feature_nodes
from src.data.sample_splits import manifest_splits, verify_sha256
from src.models.components import EnhancedModelConfig
from src.models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
from src.training.trainer_engine import Trainer
from src.data.film_dataset import FiLMDataset


def example_model_and_data():
    torch.manual_seed(11)
    torch.set_num_threads(2)
    config = EnhancedModelConfig.with_features(['cross_modal_attention', 'mcr', 'flux_messages'])
    config.reader_max_neighbors = 1
    model = EnhancedConcatGNNWithSMILES(embedding_dim=8, gene_dim=4, smiles_dim=6,
        hidden_dim=128, num_layers=2, dropout=0.0, config=config)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 3, 5],
                          [3, 4, 4, 5, 3, 5, 0, 0, 1, 1, 2, 2]])
    data = Data(x_embedding=torch.randn(6, 8), x_gene=torch.randn(6, 4),
                x_smiles=torch.randn(6, 6), edge_index=edges, edge_attr=torch.ones(12, 1),
                metabolite_indices=torch.arange(3), reaction_indices=torch.arange(3, 6),
                node_types=['metabolite'] * 3 + ['reaction'] * 3,
                y=torch.randn(6), mask=torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool), num_nodes=6)
    data.incidence_matrix = torch.tensor([[-1., 1., 0.], [0., -1., 1.], [1., 0., -1.]]).to_sparse()
    return model, data


def test_repeated_input_and_sample_order_do_not_change_prediction():
    model, data = example_model_and_data()
    model.eval()
    other = data.clone()
    other.x_gene += 3
    with torch.no_grad():
        first = model(data.clone(), compute_physics=False)[0]
        model(other, compute_physics=False)
        repeated = model(data.clone(), compute_physics=False)[0]
        model.clear_neighbor_cache()
        cleared = model(data.clone(), compute_physics=False)[0]
    torch.testing.assert_close(first, repeated, rtol=0, atol=1e-6)
    torch.testing.assert_close(first, cleared, rtol=0, atol=1e-6)


def test_auxiliary_outputs_and_target_values_do_not_change_predictions():
    model, data = example_model_and_data()
    model.eval()
    with torch.no_grad():
        without = model(data.clone(), compute_physics=False)[0]
        with_aux = model(data.clone(), compute_physics=True)[0]
        changed = data.clone()
        changed.y.fill_(12345.)
        changed.mask = ~changed.mask
        perturbed = model(changed, compute_physics=False)[0]
    torch.testing.assert_close(without, with_aux, rtol=0, atol=1e-6)
    torch.testing.assert_close(without, perturbed, rtol=0, atol=1e-6)


def test_corrected_prediction_path_supports_backward():
    model, data = example_model_and_data()
    model.train()
    prediction = model(data, compute_physics=True)[0]
    loss = prediction.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_ambiguous_metabolite_mapping_cannot_silently_overwrite():
    var = pd.DataFrame({'HMDB_trailing_zeros': ['H1', 'H1'], 'KEGG': ['K1', 'K1']}, index=['parent', 'sulfate'])
    with pytest.raises(ValueError, match='Ambiguous assay'):
        metabolite_feature_nodes(var, {'H1': 3}, {'K1': 3})
    var['exclude_from_prediction'] = [False, True]
    assert metabolite_feature_nodes(var, {'H1': 3}, {'K1': 3}).tolist() == [3, -1]


def manifest_example(tmp_path):
    frame = pd.DataFrame({'camp_obs_id': ['a', 'b', 'c', 'd'], 'rna_original_id': ['ra', 'rb', 'rc', 'rd'],
        'metabolomics_original_id': ['ma', 'mb', 'mc', 'md'], 'partition': ['train', 'train', 'validation', 'holdout'],
        'source_group': ['s1', 's1', 's2', 's3'], 'donor_id': ['', '', '', '']})
    rna = pd.DataFrame({'original_index': frame.rna_original_id.to_numpy()}, index=frame.camp_obs_id)
    met = pd.DataFrame({'original_index': frame.metabolomics_original_id.to_numpy()}, index=frame.camp_obs_id)
    path = tmp_path / 'manifest.csv'
    frame.to_csv(path, index=False)
    return path, frame, rna, met


def test_manifest_excludes_holdout_from_all_training_time_loaders(tmp_path):
    path, frame, rna, met = manifest_example(tmp_path)
    splits = manifest_splits(path, rna, met)
    assert splits == {'train': [0, 1], 'val': [2], 'test': [], 'holdout': [3], 'excluded': []}


@pytest.mark.parametrize('fault', ['id', 'duplicate', 'study', 'donor'])
def test_manifest_rejects_identity_and_group_leakage(tmp_path, fault):
    path, frame, rna, met = manifest_example(tmp_path)
    if fault == 'id':
        frame.loc[0, 'rna_original_id'] = 'wrong'
    elif fault == 'duplicate':
        frame.loc[0, 'camp_obs_id'] = 'b'
    elif fault == 'study':
        frame.loc[3, 'source_group'] = 's1'
    else:
        frame.loc[[0, 3], 'donor_id'] = 'same_patient'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        manifest_splits(path, rna, met)


def test_checksum_mismatch_stops_training(tmp_path):
    p = tmp_path / 'data'
    p.write_text('changed input')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        verify_sha256(p, '0' * 64)


def test_target_panel_has_neither_values_nor_loss_mask():
    dataset = SimpleNamespace(node_types=['metabolite'] * 5,
        feature_nodes=np.array([1, 2, 3]), excluded_target_nodes={2})
    y, mask = FiLMDataset._targets_for_values(dataset, np.array([4., 99., np.nan], dtype=np.float32))
    assert mask.tolist() == [False, True, False, False, False]
    assert y[1] == 4 and torch.isnan(y[2])
    changed, changed_mask = FiLMDataset._targets_for_values(dataset, np.array([4., -123., np.nan]))
    torch.testing.assert_close(y, changed, equal_nan=True)
    assert torch.equal(mask, changed_mask)


def test_metric_aggregation_keeps_cohorts_separate():
    from scripts.experiments.evaluate_cohort_reanalysis import target_metrics
    meta = pd.DataFrame({'holdout_cohort': ['RC18'] * 3 + ['RC20'] * 3, 'TN': ['Tumor'] * 6})
    truth = np.array([[0.], [1.], [2.], [100.], [101.], [102.]])
    prediction = np.array([[2.], [1.], [0.], [102.], [101.], [100.]])
    metrics = target_metrics(prediction, truth, [5], meta, {5: {'feature': 'test'}}).set_index('group')
    assert metrics.loc['RC18', 'pearson_r'] == pytest.approx(-1.)
    assert metrics.loc['RC20', 'pearson_r'] == pytest.approx(-1.)
    assert metrics.loc['pooled_secondary', 'pearson_r'] > 0.99


def test_best_validation_weights_are_a_snapshot(tmp_path):
    class FakeEpochTrainer(Trainer):
        def _run_training_epoch(self, loader):
            with torch.no_grad():
                self.model.weight.fill_(self.state.epoch + 1.)
            return {'train_loss': 1., 'learning_rate': 0.1}

        def _run_validation_epoch(self, loader, epoch):
            return {'val_loss': float(epoch + 1), 'val_r2': 0., 'val_rmse': 1.}

    model = torch.nn.Linear(1, 1, bias=False)
    trainer = FakeEpochTrainer(model=model, device=torch.device('cpu'), optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None, criterion=None, gradient_accumulation_steps=1, contrastive_weight=0., enable_contrastive=False)
    checkpoint = tmp_path / 'best.pt'
    trainer.fit([], [], epochs=3, patience=2, checkpoint_path=checkpoint)
    assert model.weight.item() == 1.
    assert torch.load(checkpoint, weights_only=True)['weight'].item() == 1.
