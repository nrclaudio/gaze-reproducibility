"""Tests for the detection gate feature."""

import os
import unittest


if os.environ.get("RUN_TORCH_TESTS") != "1":

    class DetectionGateTests(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable detection gate tests.")
        def test_placeholder(self):
            pass

else:
    import torch
    from torch_geometric.data import Data

    from src.models.components.builders import build_detection_gate
    from src.models.components import EnhancedModelConfig
    from src.models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES

    class DetectionGateTests(unittest.TestCase):
        # ----------------------------------------------------------
        # Builder tests
        # ----------------------------------------------------------

        def test_build_detection_gate_shape(self):
            gate = build_detection_gate(hidden_dim=16, dropout=0.1)
            sample = torch.randn(5, 16)
            out = gate(sample)
            self.assertEqual(out.shape, (5, 1))

        def test_gate_output_bounded_after_sigmoid(self):
            gate = build_detection_gate(hidden_dim=16, dropout=0.0)
            gate.eval()
            sample = torch.randn(10, 16)
            logit = gate(sample)
            prob = torch.sigmoid(logit)
            self.assertTrue(torch.all(prob >= 0))
            self.assertTrue(torch.all(prob <= 1))

        # ----------------------------------------------------------
        # Config tests
        # ----------------------------------------------------------

        def test_gate_disabled_by_default(self):
            config = EnhancedModelConfig()
            self.assertFalse(config.use_detection_gate)

        def test_config_serialization_roundtrip(self):
            config = EnhancedModelConfig(use_detection_gate=True, detection_gate_weight=0.5)
            d = config.to_dict()
            self.assertTrue(d["features"]["detection_gate"])
            restored = EnhancedModelConfig.from_dict(d)
            self.assertTrue(restored.use_detection_gate)

        def test_feature_toggles(self):
            config = EnhancedModelConfig.with_features(["detection_gate"])
            self.assertTrue(config.use_detection_gate)

        def test_str_includes_detgate(self):
            config = EnhancedModelConfig(use_detection_gate=True)
            self.assertIn("DetGate", str(config))

        def test_get_enabled_features_includes_detection_gate(self):
            config = EnhancedModelConfig(use_detection_gate=True)
            self.assertIn("detection_gate", config.get_enabled_features())

        # ----------------------------------------------------------
        # Model integration tests
        # ----------------------------------------------------------

        def _make_model_and_data(self):
            config = EnhancedModelConfig(
                use_detection_gate=True,
                detection_gate_weight=1.0,
                use_cross_modal_attention=False,
                use_residual_connections=False,
                use_layer_normalization=False,
                use_similarity_regularization=False,
            )
            model = EnhancedConcatGNNWithSMILES(
                embedding_dim=8,
                gene_dim=4,
                smiles_dim=6,
                hidden_dim=16,
                num_layers=2,
                dropout=0.1,
                config=config,
            )

            num_nodes = 10
            data = Data()
            data.x_embedding = torch.randn(num_nodes, 8)
            data.x_gene = torch.randn(num_nodes, 4)
            data.x_smiles = torch.randn(num_nodes, 6)
            data.edge_index = torch.tensor(
                [[0, 1, 2, 3, 5, 6], [5, 6, 7, 8, 0, 1]], dtype=torch.long
            )
            data.edge_attr = torch.ones(6, 1)
            data.reaction_indices = torch.arange(5)
            data.metabolite_indices = torch.arange(5, 10)
            data.node_types = ["reaction"] * 5 + ["metabolite"] * 5
            data.y = torch.randn(num_nodes)
            data.mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.mask[5:8] = True  # Some metabolites measured
            data.num_nodes = num_nodes

            return model, data

        def test_gate_produces_bounded_scores(self):
            model, data = self._make_model_and_data()
            model.eval()
            with torch.no_grad():
                model(data, return_embeddings=False, compute_physics=False)
            scores = model.last_detection_scores
            self.assertIsNotNone(scores)
            self.assertTrue(torch.all(scores >= 0))
            self.assertTrue(torch.all(scores <= 1))

        def test_gate_loss_during_training(self):
            model, data = self._make_model_and_data()
            model.train()
            model(data, return_embeddings=False, compute_physics=False)
            self.assertIsNotNone(model.last_detection_loss)
            self.assertTrue(model.last_detection_loss.requires_grad)

        def test_no_gate_loss_during_eval(self):
            model, data = self._make_model_and_data()
            model.eval()
            with torch.no_grad():
                model(data, return_embeddings=False, compute_physics=False)
            self.assertIsNone(model.last_detection_loss)

        def test_measurable_mask_restricts_bce(self):
            """BCE loss should only use CAMP-measurable metabolites."""
            model, data = self._make_model_and_data()
            # Mark first 3 metabolites (indices 5,6,7) as measurable,
            # last 2 (indices 8,9) as zero-shot
            data.measurable_mask = torch.zeros(10, dtype=torch.bool)
            data.measurable_mask[5:8] = True
            model.train()
            model(data, return_embeddings=False, compute_physics=False)
            self.assertIsNotNone(model.last_detection_loss)
            self.assertTrue(model.last_detection_loss.requires_grad)

        def test_no_measurable_mask_uses_all_metabolites(self):
            """Without measurable_mask, BCE uses all metabolite nodes."""
            model, data = self._make_model_and_data()
            # Ensure no measurable_mask attribute
            if hasattr(data, 'measurable_mask'):
                del data.measurable_mask
            model.train()
            model(data, return_embeddings=False, compute_physics=False)
            self.assertIsNotNone(model.last_detection_loss)

        def test_gate_disabled_produces_no_scores(self):
            config = EnhancedModelConfig(
                use_detection_gate=False,
                use_cross_modal_attention=False,
                use_residual_connections=False,
                use_layer_normalization=False,
                use_similarity_regularization=False,
            )
            model = EnhancedConcatGNNWithSMILES(
                embedding_dim=8,
                gene_dim=4,
                smiles_dim=6,
                hidden_dim=16,
                num_layers=2,
                dropout=0.1,
                config=config,
            )
            _, data = self._make_model_and_data()
            model.eval()
            with torch.no_grad():
                model(data, return_embeddings=False, compute_physics=False)
            self.assertIsNone(model.last_detection_scores)
            self.assertIsNone(model.last_detection_loss)


if __name__ == "__main__":
    unittest.main()
