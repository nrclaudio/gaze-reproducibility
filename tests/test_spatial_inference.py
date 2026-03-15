"""Tests for the spatial inference pipeline."""

import os
import unittest


if os.environ.get("RUN_TORCH_TESTS") != "1":

    class SpatialInferenceSkip(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable spatial inference tests.")
        def test_placeholder(self):
            pass

else:
    import tempfile

    import numpy as np
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data

    from src.inference.normalization import (
        GeneExpressionNormalizer,
        GeneStatistics,
    )
    from src.inference.spatial import SpatialPrediction, SpatialRefiner

    # ----------------------------------------------------------------
    # Constants
    # ----------------------------------------------------------------

    NUM_METABOLIC_NODES = 10
    NUM_REACTIONS = 5
    NUM_METABOLITES = 5

    EMBEDDING_DIM = 8
    GENE_DIM = 4
    SMILES_DIM = 6
    HIDDEN_DIM = 16

    NUM_SPOTS = 20
    NUM_SPATIAL_EDGES = 30

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    class DummySpatialModel(nn.Module):
        """Minimal stand-in for EnhancedConcatGNNWithSMILES."""

        def __init__(self, num_metabolites=NUM_METABOLITES, with_detection=True):
            super().__init__()
            self.num_metabolites = num_metabolites
            self.with_detection = with_detection
            self.last_detection_scores = None
            # Need at least one parameter so .to(device) works
            self._dummy = nn.Linear(1, 1)

        def forward(self, data, return_embeddings=False, compute_physics=False):
            batch_sz = data.batch_size
            preds = torch.randn(batch_sz, self.num_metabolites)
            if self.with_detection:
                self.last_detection_scores = torch.sigmoid(
                    torch.randn(batch_sz, self.num_metabolites)
                )
            else:
                self.last_detection_scores = None
            delta = torch.zeros(1)
            flux_vec = torch.zeros(1)
            embeddings = torch.zeros(1)
            return preds, delta, flux_vec, embeddings

    def _make_spatial_data(with_reaction_gene_indices=True, with_xy=True):
        """Build a mock spatial Data object."""
        data = Data()

        # Metabolic graph features
        data.x_embedding = torch.randn(NUM_METABOLIC_NODES, EMBEDDING_DIM)
        data.x_smiles = torch.randn(NUM_METABOLIC_NODES, SMILES_DIM)
        data.edge_index = torch.tensor(
            [[0, 1, 2, 3, 5, 6], [5, 6, 7, 8, 0, 1]], dtype=torch.long
        )
        data.edge_attr = torch.ones(6, 1)
        data.reaction_indices = torch.arange(NUM_REACTIONS)
        data.metabolite_indices = torch.arange(NUM_REACTIONS, NUM_METABOLIC_NODES)
        data.node_types = (
            ["reaction"] * NUM_REACTIONS + ["metabolite"] * NUM_METABOLITES
        )
        # Mark first 3 metabolites as CAMP-measurable, last 2 as zero-shot
        data.measurable_mask = torch.tensor(
            [False] * NUM_REACTIONS + [True, True, True, False, False],
            dtype=torch.bool,
        )

        # Spatial fields
        data.x_gene = torch.randn(NUM_SPOTS, GENE_DIM).abs() + 0.1
        data.num_spots = NUM_SPOTS

        src = torch.randint(0, NUM_SPOTS, (NUM_SPATIAL_EDGES,))
        dst = torch.randint(0, NUM_SPOTS, (NUM_SPATIAL_EDGES,))
        data.edge_index_spatial = torch.stack([src, dst])

        if with_xy:
            data.xy = torch.randn(NUM_SPOTS, 2)

        if with_reaction_gene_indices:
            max_genes = 3
            indices = torch.full(
                (NUM_REACTIONS, max_genes), -1, dtype=torch.long
            )
            for r in range(NUM_REACTIONS):
                indices[r, 0] = r % GENE_DIM
                indices[r, 1] = (r + 1) % GENE_DIM
            data.reaction_gene_indices = indices

        return data

    class MockSpatialDataset:
        """Minimal dataset wrapper for SpatialRefiner.run_inference."""

        def __init__(self, **kwargs):
            self._data = _make_spatial_data(**kwargs)
            self.gene_dim = GENE_DIM
            self.metabolite_names = [f"met_{i}" for i in range(NUM_METABOLITES)]

        def __getitem__(self, idx):
            return self._data

        def __len__(self):
            return 1

    # ----------------------------------------------------------------
    # SpatialPrediction
    # ----------------------------------------------------------------

    class SpatialPredictionTest(unittest.TestCase):
        def test_construction(self):
            pred = SpatialPrediction(
                predictions=torch.randn(10, 5),
                coordinates=np.random.randn(10, 2),
                metabolite_names=["a", "b", "c", "d", "e"],
                detection_scores=torch.randn(10, 5),
            )
            self.assertEqual(pred.predictions.shape, (10, 5))

        def test_defaults(self):
            pred = SpatialPrediction(predictions=torch.randn(3, 2))
            self.assertIsNone(pred.coordinates)
            self.assertEqual(pred.metabolite_names, [])
            self.assertIsNone(pred.detection_scores)

        def test_type(self):
            pred = SpatialPrediction(predictions=torch.randn(3, 2))
            self.assertIsInstance(pred, SpatialPrediction)

    # ----------------------------------------------------------------
    # smooth_inputs
    # ----------------------------------------------------------------

    class SmoothInputsTest(unittest.TestCase):
        def test_shape_preserved(self):
            x = torch.randn(NUM_SPOTS, GENE_DIM)
            edges = torch.randint(0, NUM_SPOTS, (2, NUM_SPATIAL_EDGES))
            out = SpatialRefiner.smooth_inputs(x, edges, alpha=0.5)
            self.assertEqual(out.shape, x.shape)

        def test_alpha_zero_identity(self):
            x = torch.randn(NUM_SPOTS, GENE_DIM)
            edges = torch.randint(0, NUM_SPOTS, (2, NUM_SPATIAL_EDGES))
            out = SpatialRefiner.smooth_inputs(x, edges, alpha=0.0)
            self.assertTrue(torch.allclose(out, x))

        def test_alpha_one_three_node_chain(self):
            """3-node chain 0--1--2.  alpha=1 gives purely neighbour averages."""
            x = torch.tensor([[1.0, 0.0],
                              [0.0, 1.0],
                              [0.0, 0.0]])
            edges = torch.tensor([[0, 1, 1, 2],
                                  [1, 0, 2, 1]])
            out = SpatialRefiner.smooth_inputs(x, edges, alpha=1.0)
            # Node 0 (deg 1): avg = x[1] = [0, 1]
            # Node 1 (deg 2): avg = (x[0]+x[2])/2 = [0.5, 0]
            # Node 2 (deg 1): avg = x[1] = [0, 1]
            expected = torch.tensor([[0.0, 1.0],
                                     [0.5, 0.0],
                                     [0.0, 1.0]])
            self.assertTrue(torch.allclose(out, expected, atol=1e-6))

        def test_disconnected_nodes(self):
            """Disconnected nodes get (1-alpha)*x since no neighbours exist."""
            x = torch.tensor([[2.0], [3.0]])
            edges = torch.zeros(2, 0, dtype=torch.long)
            out = SpatialRefiner.smooth_inputs(x, edges, alpha=0.5)
            expected = 0.5 * x
            self.assertTrue(torch.allclose(out, expected))

        def test_self_loops(self):
            """Self-loop on node 0 counts it as its own neighbour."""
            x = torch.tensor([[4.0], [6.0]])
            # Node 0: self-loop + edge to 1 (deg 2).  Node 1: edge to 0 (deg 1).
            edges = torch.tensor([[0, 0, 1], [0, 1, 0]])
            out = SpatialRefiner.smooth_inputs(x, edges, alpha=1.0)
            # Node 0: (x[0]+x[1])/2 = 5.  Node 1: x[0] = 4.
            expected = torch.tensor([[5.0], [4.0]])
            self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    # ----------------------------------------------------------------
    # _build_reaction_mask
    # ----------------------------------------------------------------

    class BuildReactionMaskTest(unittest.TestCase):
        def _refiner(self):
            return SpatialRefiner(DummySpatialModel(), device="cpu")

        def test_shape(self):
            data = _make_spatial_data(with_reaction_gene_indices=True)
            mask = self._refiner()._build_reaction_mask(
                data, NUM_METABOLIC_NODES, GENE_DIM
            )
            self.assertEqual(mask.shape, (NUM_REACTIONS, GENE_DIM))

        def test_values_match_indices(self):
            data = _make_spatial_data(with_reaction_gene_indices=True)
            mask = self._refiner()._build_reaction_mask(
                data, NUM_METABOLIC_NODES, GENE_DIM
            )
            # Reaction 0 uses gene indices 0 and 1
            self.assertTrue(mask[0, 0].item())
            self.assertTrue(mask[0, 1].item())

        def test_padding_ignored(self):
            data = _make_spatial_data(with_reaction_gene_indices=True)
            mask = self._refiner()._build_reaction_mask(
                data, NUM_METABOLIC_NODES, GENE_DIM
            )
            # Each reaction has exactly 2 active genes (third slot is -1 padding)
            for r in range(NUM_REACTIONS):
                self.assertEqual(mask[r].sum().item(), 2)

        def test_missing_attr_returns_none(self):
            data = _make_spatial_data(with_reaction_gene_indices=False)
            mask = self._refiner()._build_reaction_mask(
                data, NUM_METABOLIC_NODES, GENE_DIM
            )
            self.assertIsNone(mask)

    # ----------------------------------------------------------------
    # _build_batch
    # ----------------------------------------------------------------

    class BuildBatchTest(unittest.TestCase):
        def _refiner(self):
            return SpatialRefiner(DummySpatialModel(), device="cpu")

        def _build(self, reaction_mask=None, batch_sz=4):
            data = _make_spatial_data()
            batch_genes = torch.randn(batch_sz, GENE_DIM)
            refiner = self._refiner()
            batch_data = refiner._build_batch(
                data, batch_genes, batch_sz,
                NUM_METABOLIC_NODES, GENE_DIM, reaction_mask,
            )
            return batch_data, batch_genes

        def test_output_shapes(self):
            bd, _ = self._build()
            self.assertEqual(
                bd.x_embedding.shape, (4, NUM_METABOLIC_NODES, EMBEDDING_DIM)
            )
            self.assertEqual(
                bd.x_gene.shape, (4, NUM_METABOLIC_NODES, GENE_DIM)
            )
            self.assertEqual(
                bd.x_smiles.shape, (4, NUM_METABOLIC_NODES, SMILES_DIM)
            )

        def test_genes_mapped_to_reaction_nodes(self):
            bd, _ = self._build()
            rxn_genes = bd.x_gene[:, torch.arange(NUM_REACTIONS), :]
            self.assertTrue(rxn_genes.abs().sum() > 0)

        def test_metabolite_slots_zero(self):
            bd, _ = self._build()
            met_idx = torch.arange(NUM_REACTIONS, NUM_METABOLIC_NODES)
            self.assertTrue(torch.all(bd.x_gene[:, met_idx, :] == 0))

        def test_batch_size_attr(self):
            bd, _ = self._build(batch_sz=7)
            self.assertEqual(bd.batch_size, 7)

        def test_mask_applied(self):
            data = _make_spatial_data(with_reaction_gene_indices=True)
            refiner = self._refiner()
            mask = refiner._build_reaction_mask(
                data, NUM_METABOLIC_NODES, GENE_DIM
            )
            batch_sz = 3
            batch_genes = torch.ones(batch_sz, GENE_DIM)
            bd = refiner._build_batch(
                data, batch_genes, batch_sz,
                NUM_METABOLIC_NODES, GENE_DIM, mask,
            )
            rxn_genes = bd.x_gene[:, torch.arange(NUM_REACTIONS), :]
            mask_expanded = mask.unsqueeze(0).expand(batch_sz, -1, -1)
            self.assertTrue(torch.all(rxn_genes[~mask_expanded] == 0))

    # ----------------------------------------------------------------
    # run_inference
    # ----------------------------------------------------------------

    class RunInferenceTest(unittest.TestCase):
        def _run(self, **kwargs):
            model = kwargs.pop("model", DummySpatialModel())
            normalizer = kwargs.pop("normalizer", None)
            ds_kwargs = {}
            for k in ("with_reaction_gene_indices", "with_xy"):
                if k in kwargs:
                    ds_kwargs[k] = kwargs.pop(k)
            refiner = SpatialRefiner(model, device="cpu", normalizer=normalizer)
            ds = MockSpatialDataset(**ds_kwargs)
            return refiner.run_inference(ds, **kwargs)

        def test_return_type(self):
            result = self._run(spot_batch_size=10)
            self.assertIsInstance(result, SpatialPrediction)

        def test_pred_shape(self):
            result = self._run(spot_batch_size=10)
            self.assertEqual(
                result.predictions.shape, (NUM_SPOTS, NUM_METABOLITES)
            )

        def test_coordinates(self):
            result = self._run(spot_batch_size=10)
            self.assertIsNotNone(result.coordinates)
            self.assertEqual(result.coordinates.shape, (NUM_SPOTS, 2))

        def test_metabolite_names(self):
            result = self._run(spot_batch_size=10)
            self.assertEqual(len(result.metabolite_names), NUM_METABOLITES)

        def test_detection_scores(self):
            result = self._run(
                model=DummySpatialModel(with_detection=True),
                spot_batch_size=10,
            )
            self.assertIsNotNone(result.detection_scores)
            self.assertEqual(
                result.detection_scores.shape, (NUM_SPOTS, NUM_METABOLITES)
            )

        def test_no_smoothing(self):
            result = self._run(smooth_input=False, spot_batch_size=10)
            self.assertEqual(
                result.predictions.shape, (NUM_SPOTS, NUM_METABOLITES)
            )

        def test_with_normalizer(self):
            stats = GeneStatistics(
                mean=torch.zeros(GENE_DIM), std=torch.ones(GENE_DIM),
            )
            result = self._run(
                normalizer=GeneExpressionNormalizer(stats),
                spot_batch_size=10,
            )
            self.assertEqual(
                result.predictions.shape, (NUM_SPOTS, NUM_METABOLITES)
            )

        def test_no_xy(self):
            result = self._run(with_xy=False, spot_batch_size=10)
            self.assertIsNone(result.coordinates)

        def test_no_gpr_indices(self):
            result = self._run(
                with_reaction_gene_indices=False, spot_batch_size=10,
            )
            self.assertEqual(
                result.predictions.shape, (NUM_SPOTS, NUM_METABOLITES)
            )

        def test_batching(self):
            """Different batch sizes produce the same output shape."""
            r1 = self._run(spot_batch_size=5)
            r2 = self._run(spot_batch_size=NUM_SPOTS)
            self.assertEqual(r1.predictions.shape, r2.predictions.shape)

    # ----------------------------------------------------------------
    # GeneStatistics
    # ----------------------------------------------------------------

    class GeneStatisticsTest(unittest.TestCase):
        def test_save_load_roundtrip(self):
            stats = GeneStatistics(
                mean=torch.randn(GENE_DIM),
                std=torch.rand(GENE_DIM) + 0.1,
            )
            with tempfile.NamedTemporaryFile(suffix=".pt") as f:
                stats.save(f.name)
                loaded = GeneStatistics.load(f.name)
            self.assertTrue(torch.allclose(stats.mean, loaded.mean))
            self.assertTrue(torch.allclose(stats.std, loaded.std))

        def test_shape_preserved(self):
            stats = GeneStatistics(
                mean=torch.randn(GENE_DIM),
                std=torch.rand(GENE_DIM) + 0.1,
            )
            with tempfile.NamedTemporaryFile(suffix=".pt") as f:
                stats.save(f.name)
                loaded = GeneStatistics.load(f.name)
            self.assertEqual(loaded.mean.shape, (GENE_DIM,))
            self.assertEqual(loaded.std.shape, (GENE_DIM,))

    # ----------------------------------------------------------------
    # GeneExpressionNormalizer
    # ----------------------------------------------------------------

    class GeneExpressionNormalizerTest(unittest.TestCase):
        def _make_normalizer(self):
            stats = GeneStatistics(
                mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                std=torch.tensor([0.5, 0.5, 0.5, 0.5]),
            )
            return GeneExpressionNormalizer(stats)

        def test_output_shape(self):
            norm = self._make_normalizer()
            x = torch.randn(10, GENE_DIM).abs() + 0.1
            self.assertEqual(norm.normalize(x).shape, x.shape)

        def test_zero_preservation(self):
            norm = self._make_normalizer()
            x = torch.randn(10, GENE_DIM).abs() + 0.1
            x[3, 2] = 0.0
            x[7, 0] = 0.0
            out = norm.normalize(x)
            self.assertEqual(out[3, 2].item(), 0.0)
            self.assertEqual(out[7, 0].item(), 0.0)

        def test_nonzero_shifted(self):
            norm = self._make_normalizer()
            x = torch.randn(10, GENE_DIM).abs() + 0.1
            out = norm.normalize(x)
            self.assertFalse(torch.allclose(out, x))

        def test_missing_checkpoint(self):
            with self.assertRaises(FileNotFoundError):
                GeneExpressionNormalizer.from_checkpoint("/nonexistent/path")

    # ----------------------------------------------------------------
    # SpatialRefiner.__init__
    # ----------------------------------------------------------------

    class SpatialRefinerInitTest(unittest.TestCase):
        def test_device_placement(self):
            refiner = SpatialRefiner(DummySpatialModel(), device="cpu")
            self.assertEqual(refiner.device, torch.device("cpu"))

        def test_eval_mode(self):
            refiner = SpatialRefiner(DummySpatialModel(), device="cpu")
            self.assertFalse(refiner.model.training)

        def test_normalizer_stored(self):
            stats = GeneStatistics(
                mean=torch.zeros(GENE_DIM), std=torch.ones(GENE_DIM),
            )
            norm = GeneExpressionNormalizer(stats)
            refiner = SpatialRefiner(
                DummySpatialModel(), device="cpu", normalizer=norm,
            )
            self.assertIs(refiner.normalizer, norm)


if __name__ == "__main__":
    unittest.main()
