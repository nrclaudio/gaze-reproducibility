"""Tests for 5-fold cross-validation splitting and DataConfig fraction logic."""

import unittest
import numpy as np
from sklearn.model_selection import KFold

from src.config import DataConfig


class DataConfigFractionsTest(unittest.TestCase):
    """DataConfig should correctly compute train/val/test fractions."""

    def test_default_fractions(self) -> None:
        cfg = DataConfig()
        self.assertAlmostEqual(cfg.train_fraction, 0.7)
        self.assertAlmostEqual(cfg.val_fraction, 0.15)
        self.assertAlmostEqual(cfg.test_fraction, 0.15)

    def test_fractions_sum_to_one(self) -> None:
        cfg = DataConfig(train_fraction=0.6, val_fraction=0.2)
        total = cfg.train_fraction + cfg.val_fraction + cfg.test_fraction
        self.assertAlmostEqual(total, 1.0)

    def test_custom_fractions(self) -> None:
        cfg = DataConfig(train_fraction=0.8, val_fraction=0.1)
        self.assertAlmostEqual(cfg.test_fraction, 0.1)

    def test_all_train(self) -> None:
        cfg = DataConfig(train_fraction=1.0, val_fraction=0.0)
        self.assertAlmostEqual(cfg.test_fraction, 0.0)


class KFoldSplitPropertiesTest(unittest.TestCase):
    """Verify fundamental properties of 5-fold cross-validation splits.

    The k-fold modality sweep (run_kfold_modality_sweep.py) uses
    KFold(n_splits=5, shuffle=True, random_state=seed) from sklearn.
    """

    N_SAMPLES = 100
    N_SPLITS = 5
    SEED = 42

    def setUp(self) -> None:
        self.kf = KFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.SEED)
        self.indices = list(range(self.N_SAMPLES))

    def test_produces_correct_number_of_folds(self) -> None:
        folds = list(self.kf.split(self.indices))
        self.assertEqual(len(folds), self.N_SPLITS)

    def test_each_sample_appears_in_exactly_one_validation_fold(self) -> None:
        val_counts = np.zeros(self.N_SAMPLES, dtype=int)
        for _, val_idx in self.kf.split(self.indices):
            val_counts[val_idx] += 1
        self.assertTrue(np.all(val_counts == 1))

    def test_train_val_disjoint_per_fold(self) -> None:
        for train_idx, val_idx in self.kf.split(self.indices):
            overlap = set(train_idx) & set(val_idx)
            self.assertEqual(len(overlap), 0)

    def test_train_val_union_covers_all(self) -> None:
        for train_idx, val_idx in self.kf.split(self.indices):
            union = set(train_idx) | set(val_idx)
            self.assertEqual(union, set(self.indices))

    def test_val_size_approximately_equal(self) -> None:
        expected_val_size = self.N_SAMPLES // self.N_SPLITS
        for _, val_idx in self.kf.split(self.indices):
            self.assertEqual(len(val_idx), expected_val_size)

    def test_deterministic_with_seed(self) -> None:
        kf1 = KFold(n_splits=5, shuffle=True, random_state=self.SEED)
        kf2 = KFold(n_splits=5, shuffle=True, random_state=self.SEED)
        for (t1, v1), (t2, v2) in zip(kf1.split(self.indices), kf2.split(self.indices)):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)


class RandomizationFlagsTest(unittest.TestCase):
    """Modality ablation configs express randomization through DataConfig flags."""

    def test_smiles_only_randomizes_others(self) -> None:
        """SMILES_Only: randomize embeddings and genes, keep SMILES."""
        cfg = DataConfig(randomize_smiles=False, randomize_embeddings=True, randomize_genes=True)
        self.assertFalse(cfg.randomize_smiles)
        self.assertTrue(cfg.randomize_embeddings)
        self.assertTrue(cfg.randomize_genes)

    def test_genes_only_randomizes_others(self) -> None:
        cfg = DataConfig(randomize_smiles=True, randomize_embeddings=True, randomize_genes=False)
        self.assertTrue(cfg.randomize_smiles)
        self.assertTrue(cfg.randomize_embeddings)
        self.assertFalse(cfg.randomize_genes)

    def test_full_model_no_randomization(self) -> None:
        cfg = DataConfig()
        self.assertFalse(cfg.randomize_smiles)
        self.assertFalse(cfg.randomize_embeddings)
        self.assertFalse(cfg.randomize_genes)
        self.assertFalse(cfg.randomize_gpe)

    def test_random_graph_flag(self) -> None:
        cfg = DataConfig(random_graph=True)
        self.assertTrue(cfg.random_graph)

    def test_serialization_preserves_randomization(self) -> None:
        cfg = DataConfig(
            randomize_smiles=True, randomize_genes=True,
            randomize_embeddings=False, randomize_gpe=True,
            random_graph=True,
        )
        rebuilt = DataConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg.randomize_smiles, rebuilt.randomize_smiles)
        self.assertEqual(cfg.randomize_genes, rebuilt.randomize_genes)
        self.assertEqual(cfg.randomize_embeddings, rebuilt.randomize_embeddings)
        self.assertEqual(cfg.randomize_gpe, rebuilt.randomize_gpe)
        self.assertEqual(cfg.random_graph, rebuilt.random_graph)


if __name__ == '__main__':
    unittest.main()
