"""Forward-pass smoke tests for all EnhancedConcatGNNWithSMILES feature variants."""

import os
import unittest


if os.environ.get("RUN_TORCH_TESTS") != "1":

    class ModelVariantForwardTest(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable model variant tests.")
        def test_placeholder(self):
            pass

else:
    import torch
    from torch_geometric.data import Data

    from src.models.enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
    from src.models.components import EnhancedModelConfig

    # ----------------------------------------------------------------
    # Constants (match test_detection_gate.py dimensions)
    # ----------------------------------------------------------------
    NUM_NODES = 10
    NUM_REACTIONS = 5
    NUM_METABOLITES = 5
    EMBEDDING_DIM = 8
    GENE_DIM = 4
    SMILES_DIM = 6
    HIDDEN_DIM = 16
    NUM_LAYERS = 2

    # GPE output_dim = 4 + rwse_steps; default rwse_steps=16 → 20
    GPE_DIM = 20

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _make_data(with_gpe=False, with_incidence=False, with_mask=False):
        """Build a minimal Data object for forward-pass testing."""
        data = Data()
        data.x_embedding = torch.randn(NUM_NODES, EMBEDDING_DIM)
        data.x_gene = torch.randn(NUM_NODES, GENE_DIM)
        data.x_smiles = torch.randn(NUM_NODES, SMILES_DIM)
        data.edge_index = torch.tensor(
            [[0, 1, 2, 3, 5, 6], [5, 6, 7, 8, 0, 1]], dtype=torch.long
        )
        data.edge_attr = torch.ones(6, 1)
        data.reaction_indices = torch.arange(NUM_REACTIONS)
        data.metabolite_indices = torch.arange(NUM_REACTIONS, NUM_NODES)
        data.node_types = ["reaction"] * NUM_REACTIONS + ["metabolite"] * NUM_METABOLITES
        data.y = torch.randn(NUM_NODES)
        data.num_nodes = NUM_NODES

        if with_gpe:
            data.x_gpe = torch.randn(NUM_NODES, GPE_DIM)

        if with_incidence:
            # Sparse [NUM_METABOLITES, NUM_REACTIONS] incidence matrix
            rows = torch.arange(NUM_METABOLITES)
            cols = torch.arange(NUM_REACTIONS)
            vals = torch.ones(NUM_METABOLITES)
            data.incidence_matrix = torch.sparse_coo_tensor(
                torch.stack([rows, cols]), vals,
                (NUM_METABOLITES, NUM_REACTIONS),
            ).coalesce()

        if with_mask:
            data.mask = torch.zeros(NUM_NODES, dtype=torch.bool)
            data.mask[5:8] = True
            data.measurable_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
            data.measurable_mask[5:8] = True

        return data

    def _make_config(*features):
        """Build an all-off baseline config then enable the listed features."""
        if features:
            return EnhancedModelConfig.with_features(features)
        return EnhancedModelConfig.original_model_config()

    def _forward(model, data, compute_physics=False):
        """Run a single eval forward pass and return the 4-tuple."""
        model.eval()
        with torch.no_grad():
            return model(data, return_embeddings=False, compute_physics=compute_physics)

    def _make_model(config):
        return EnhancedConcatGNNWithSMILES(
            embedding_dim=EMBEDDING_DIM,
            gene_dim=GENE_DIM,
            smiles_dim=SMILES_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=0.1,
            config=config,
        )

    # ----------------------------------------------------------------
    # Forward-pass smoke tests
    # ----------------------------------------------------------------

    class ModelVariantForwardTest(unittest.TestCase):
        """Each test instantiates a variant, runs forward, checks pred shape."""

        def _assert_forward(self, config, data=None, compute_physics=False):
            model = _make_model(config)
            if data is None:
                data = _make_data()
            pred, delta, flux_vec, embeddings = _forward(model, data, compute_physics)
            self.assertEqual(pred.shape, (1, NUM_NODES))
            return model, pred, delta, flux_vec, embeddings

        # --- Individual feature variants ---

        def test_baseline(self):
            self._assert_forward(_make_config())

        def test_cross_modal_attention(self):
            self._assert_forward(_make_config("cross_modal_attention"))

        def test_residual_connections(self):
            self._assert_forward(_make_config("residual_connections"))

        def test_attention_plus_residuals(self):
            self._assert_forward(
                _make_config("cross_modal_attention", "residual_connections")
            )

        def test_mcr(self):
            self._assert_forward(_make_config("mcr"))

        def test_mcr_multihop(self):
            self._assert_forward(_make_config("mcr_multihop"))

        def test_gpe(self):
            self._assert_forward(
                _make_config("graph_positional_encodings"),
                data=_make_data(with_gpe=True),
            )

        def test_mcr_plus_gpe(self):
            self._assert_forward(
                _make_config("mcr_multihop", "graph_positional_encodings"),
                data=_make_data(with_gpe=True),
            )

        def test_gene_skip(self):
            self._assert_forward(_make_config("mcr", "gene_skip"))

        def test_gene_film(self):
            self._assert_forward(_make_config("gene_film"))

        def test_detection_gate(self):
            self._assert_forward(
                _make_config("detection_gate"),
                data=_make_data(with_mask=True),
            )

        def test_flux_messages(self):
            self._assert_forward(
                _make_config("flux_messages"),
                data=_make_data(with_incidence=True),
                compute_physics=True,
            )

        def test_similarity_regularization(self):
            self._assert_forward(_make_config("similarity_regularization"))

        # --- LOMO composite architectures ---

        def test_relational_mcr_gaze(self):
            """relational_mcr = full GAZE: residuals + attention + mcr."""
            self._assert_forward(
                _make_config("residual_connections", "cross_modal_attention", "mcr")
            )

        def test_relational_full(self):
            """relational_full: GAZE + flux_messages + mass_balance_projection."""
            self._assert_forward(
                _make_config(
                    "residual_connections", "cross_modal_attention", "mcr",
                    "flux_messages", "mass_balance_projection",
                ),
                data=_make_data(with_incidence=True),
                compute_physics=True,
            )

        def test_relational_full_plus(self):
            """relational_full_plus: full + mcr_multihop + gpe."""
            self._assert_forward(
                _make_config(
                    "residual_connections", "cross_modal_attention", "mcr_multihop",
                    "graph_positional_encodings", "flux_messages",
                    "mass_balance_projection",
                ),
                data=_make_data(with_gpe=True, with_incidence=True),
                compute_physics=True,
            )

    # ----------------------------------------------------------------
    # SMILES masking (Blindfold Test)
    # ----------------------------------------------------------------

    class SmilesMaskingTest(unittest.TestCase):
        """mask_target_smiles is used by every LOMO architecture."""

        def test_full_blindfold(self):
            """mask_target_smiles=True (100% masking) in train mode."""
            config = _make_config("residual_connections", "cross_modal_attention", "mcr")
            model = _make_model(config)
            model.train()
            data = _make_data()
            pred, *_ = model(data, return_embeddings=False, compute_physics=False,
                             mask_target_smiles=True)
            self.assertEqual(pred.shape, (1, NUM_NODES))

        def test_partial_blindfold(self):
            """mask_target_smiles=0.5 (50% probabilistic) in train mode."""
            config = _make_config("residual_connections", "cross_modal_attention", "mcr")
            model = _make_model(config)
            model.train()
            data = _make_data()
            pred, *_ = model(data, return_embeddings=False, compute_physics=False,
                             mask_target_smiles=0.5)
            self.assertEqual(pred.shape, (1, NUM_NODES))

        def test_blindfold_eval_noop(self):
            """Masking only applies during training; eval should be unaffected."""
            config = _make_config("mcr")
            model = _make_model(config)
            model.eval()
            data = _make_data()
            with torch.no_grad():
                pred_masked, *_ = model(data, return_embeddings=False,
                                        compute_physics=False, mask_target_smiles=True)
                pred_plain, *_ = model(data, return_embeddings=False,
                                       compute_physics=False, mask_target_smiles=False)
            self.assertTrue(torch.equal(pred_masked, pred_plain))

    # ----------------------------------------------------------------
    # Training (backward) smoke tests
    # ----------------------------------------------------------------

    class ModelVariantTrainingTest(unittest.TestCase):
        """Verify gradients flow in train mode for representative configs."""

        def test_baseline_backward(self):
            model = _make_model(_make_config())
            model.train()
            data = _make_data()
            pred, *_ = model(data, return_embeddings=False, compute_physics=False)
            loss = pred.sum()
            loss.backward()
            # At least one grad should be non-None
            has_grad = any(p.grad is not None for p in model.parameters())
            self.assertTrue(has_grad)

        def test_full_model_backward(self):
            config = _make_config(
                "cross_modal_attention", "residual_connections",
                "mcr", "detection_gate",
            )
            model = _make_model(config)
            model.train()
            data = _make_data(with_mask=True)
            pred, *_ = model(data, return_embeddings=False, compute_physics=False)
            loss = pred.sum()
            det_loss = getattr(model, "last_detection_loss", None)
            if det_loss is not None:
                loss = loss + det_loss
            loss.backward()
            has_grad = any(p.grad is not None for p in model.parameters())
            self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
