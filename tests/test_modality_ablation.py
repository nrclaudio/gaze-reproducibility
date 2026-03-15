"""Tests for modality ablation via ArchitecturalFeatureConfig."""

import unittest
from src.models.components.enhanced_config import ArchitecturalFeatureConfig


class BaselineConfigTest(unittest.TestCase):
    """The original_model_config should disable every enhancement."""

    def test_baseline_all_features_off(self) -> None:
        cfg = ArchitecturalFeatureConfig.original_model_config()
        self.assertFalse(cfg.use_cross_modal_attention)
        self.assertFalse(cfg.use_residual_connections)
        self.assertFalse(cfg.use_layer_normalization)
        self.assertFalse(cfg.use_similarity_regularization)
        self.assertFalse(cfg.use_flux_messages)
        self.assertFalse(cfg.use_mass_balance_projection)
        self.assertFalse(cfg.use_mcr)
        self.assertFalse(cfg.use_contrastive_pretraining)

    def test_baseline_str_is_baseline(self) -> None:
        cfg = ArchitecturalFeatureConfig.original_model_config()
        self.assertEqual(str(cfg), "BaselineModel")

    def test_baseline_enabled_features_empty(self) -> None:
        cfg = ArchitecturalFeatureConfig.original_model_config()
        enabled = cfg.get_enabled_features()
        # Only truly-empty baseline should have nothing
        self.assertEqual(len(enabled), 0)


class AblationConfigTest(unittest.TestCase):
    """ablation_config enables exactly one feature against baseline."""

    SINGLE_FEATURES = [
        'cross_modal_attention',
        'residual_connections',
        'similarity_regularization',
        'flux_messages',
        'mass_balance_projection',
        'mcr',
        'contrastive_pretraining',
        'graph_positional_encodings',
    ]

    def test_ablation_enables_single_feature(self) -> None:
        for feature in self.SINGLE_FEATURES:
            with self.subTest(feature=feature):
                cfg = ArchitecturalFeatureConfig.ablation_config(feature)
                enabled = cfg.get_enabled_features()
                # The feature (or its coupled friends) should be present
                self.assertGreaterEqual(len(enabled), 1, f"{feature} not enabled")

    def test_ablation_unknown_feature_raises(self) -> None:
        with self.assertRaises(ValueError):
            ArchitecturalFeatureConfig.ablation_config('nonexistent_feature')

    def test_residual_enables_layer_norm(self) -> None:
        """Residual connections are bundled with layer normalization."""
        cfg = ArchitecturalFeatureConfig.ablation_config('residual_connections')
        self.assertTrue(cfg.use_residual_connections)
        self.assertTrue(cfg.use_layer_normalization)


class WithFeaturesTest(unittest.TestCase):
    """with_features enables an arbitrary set of features."""

    def test_empty_returns_baseline(self) -> None:
        cfg = ArchitecturalFeatureConfig.with_features([])
        self.assertEqual(str(cfg), "BaselineModel")

    def test_multiple_features_compose(self) -> None:
        cfg = ArchitecturalFeatureConfig.with_features([
            'cross_modal_attention', 'residual_connections', 'mcr',
        ])
        self.assertTrue(cfg.use_cross_modal_attention)
        self.assertTrue(cfg.use_residual_connections)
        self.assertTrue(cfg.use_mcr)

    def test_unknown_feature_raises(self) -> None:
        with self.assertRaises(ValueError):
            ArchitecturalFeatureConfig.with_features(['mcr', 'bogus_feature'])

    def test_mcr_multihop_sets_max_hops(self) -> None:
        cfg = ArchitecturalFeatureConfig.with_features(['mcr_multihop'])
        self.assertTrue(cfg.use_mcr)
        self.assertEqual(cfg.reader_max_hops, 2)
        self.assertEqual(cfg.effective_max_hops, 2)


class ModalityAblationSLURMConfigsTest(unittest.TestCase):
    """Verify the modality ablation configs from the SLURM scripts are expressible."""

    def test_full_model_config(self) -> None:
        """Full model as used in modality ablation: residuals + attention."""
        cfg = ArchitecturalFeatureConfig.with_features([
            'residual_connections', 'cross_modal_attention',
        ])
        self.assertTrue(cfg.use_cross_modal_attention)
        self.assertTrue(cfg.use_residual_connections)
        self.assertFalse(cfg.use_mcr)

    def test_graph_only_baseline(self) -> None:
        """Graph-only config (all features disabled, full blindfold)."""
        cfg = ArchitecturalFeatureConfig.original_model_config()
        self.assertEqual(str(cfg), "BaselineModel")

    def test_architecture_ablation_configs(self) -> None:
        """The 4 architecture ablation variants from submit_arch_ablation.sh."""
        base = ArchitecturalFeatureConfig.original_model_config()
        plus_res = ArchitecturalFeatureConfig.with_features(['residual_connections'])
        plus_attn = ArchitecturalFeatureConfig.with_features(['cross_modal_attention'])
        plus_both = ArchitecturalFeatureConfig.with_features([
            'residual_connections', 'cross_modal_attention',
        ])
        self.assertEqual(str(base), "BaselineModel")
        self.assertIn("Residual", str(plus_res))
        self.assertIn("CrossModal", str(plus_attn))
        self.assertIn("Residual", str(plus_both))
        self.assertIn("CrossModal", str(plus_both))


class SerializationRoundtripTest(unittest.TestCase):
    """to_dict / from_dict should preserve all fields."""

    def test_roundtrip_default(self) -> None:
        cfg = ArchitecturalFeatureConfig()
        rebuilt = ArchitecturalFeatureConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg.to_dict(), rebuilt.to_dict())

    def test_roundtrip_ablation(self) -> None:
        cfg = ArchitecturalFeatureConfig.ablation_config('mcr')
        rebuilt = ArchitecturalFeatureConfig.from_dict(cfg.to_dict())
        self.assertTrue(rebuilt.use_mcr)


if __name__ == '__main__':
    unittest.main()
