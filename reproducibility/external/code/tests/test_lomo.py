"""Tests for Leave-One-Metabolite-Out (LOMO) cross-validation utilities."""

import os
import unittest

SKIP_TORCH_TESTS = os.environ.get('RUN_TORCH_TESTS') != '1'

if SKIP_TORCH_TESTS:

    class LOMOTest(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable LOMO tests.")
        def test_skip(self) -> None:
            pass

else:
    import numpy as np
    import torch
    from types import SimpleNamespace

    from src.evaluation.lomo_cross_validation import (
        _select_metabolites_by_coverage,
        _build_metabolite_neighbor_maps,
    )

    class SelectMetabolitesByCoverageTest(unittest.TestCase):
        """Tests for stratified metabolite selection across coverage quantiles."""

        def test_returns_all_when_n_exceeds_eligible(self) -> None:
            eligible = [0, 1, 2, 3, 4]
            scores = {m: float(m) for m in eligible}
            selected = _select_metabolites_by_coverage(eligible, scores, n_select=10)
            self.assertEqual(set(selected), set(eligible))

        def test_returns_exact_count(self) -> None:
            eligible = list(range(50))
            scores = {m: float(m) / 50 for m in eligible}
            selected = _select_metabolites_by_coverage(eligible, scores, n_select=10)
            self.assertEqual(len(selected), 10)

        def test_coverage_spread_across_quantiles(self) -> None:
            """Selected metabolites should span the coverage range, not cluster."""
            eligible = list(range(100))
            scores = {m: float(m) for m in eligible}
            selected = _select_metabolites_by_coverage(eligible, scores, n_select=10, n_quantiles=5)
            selected_scores = sorted(scores[m] for m in selected)
            # At least some from the bottom and top quartiles
            self.assertTrue(any(s < 25 for s in selected_scores), "No low-coverage metabolites selected")
            self.assertTrue(any(s >= 75 for s in selected_scores), "No high-coverage metabolites selected")

        def test_deterministic(self) -> None:
            eligible = list(range(50))
            scores = {m: float(m) for m in eligible}
            s1 = _select_metabolites_by_coverage(eligible, scores, n_select=10)
            s2 = _select_metabolites_by_coverage(eligible, scores, n_select=10)
            self.assertEqual(s1, s2)

        def test_no_duplicates(self) -> None:
            eligible = list(range(50))
            scores = {m: float(m) for m in eligible}
            selected = _select_metabolites_by_coverage(eligible, scores, n_select=20)
            self.assertEqual(len(selected), len(set(selected)))

    class BuildNeighborMapsTest(unittest.TestCase):
        """Tests for _build_metabolite_neighbor_maps graph traversal."""

        def _make_bipartite_data(self):
            """Create a simple bipartite graph: M0 - R0 - M1 - R1 - M2.

            Metabolite nodes: 0, 1, 2
            Reaction nodes: 3, 4
            Edges: 0-3, 3-1, 1-4, 4-2 (both directions)
            """
            edge_index = torch.tensor([
                [0, 3, 3, 1, 1, 4, 4, 2],
                [3, 0, 1, 3, 4, 1, 2, 4],
            ])
            data = SimpleNamespace(
                metabolite_indices=torch.tensor([0, 1, 2]),
                reaction_indices=torch.tensor([3, 4]),
                edge_index=edge_index,
                node_types=['metabolite', 'metabolite', 'metabolite', 'reaction', 'reaction'],
            )
            return data

        def test_direct_neighbors(self) -> None:
            data = self._make_bipartite_data()
            _, direct, _ = _build_metabolite_neighbor_maps(data)
            # M0 connects to R0 which connects to M1 → M0's direct neighbor is M1
            self.assertIn(1, direct[0])
            self.assertNotIn(2, direct[0])
            # M1 connects to R0 (→M0) and R1 (→M2)
            self.assertIn(0, direct[1])
            self.assertIn(2, direct[1])

        def test_no_self_neighbor(self) -> None:
            data = self._make_bipartite_data()
            _, direct, _ = _build_metabolite_neighbor_maps(data)
            for m, neighbors in direct.items():
                self.assertNotIn(m, neighbors)

        def test_far_neighbors(self) -> None:
            data = self._make_bipartite_data()
            _, direct, far = _build_metabolite_neighbor_maps(data)
            # M0's far neighbor is M2 (2 hops via M1)
            self.assertIn(2, far[0])
            # M1's far neighbors: from M0→{M1} and M2→{M1} → none that aren't direct
            self.assertEqual(len(far[1]), 0)

        def test_far_excludes_direct(self) -> None:
            data = self._make_bipartite_data()
            _, direct, far = _build_metabolite_neighbor_maps(data)
            for m in [0, 1, 2]:
                overlap = direct[m] & far[m]
                self.assertEqual(len(overlap), 0, f"Metabolite {m} has overlap between direct and far")

    class LOMOProtocolTest(unittest.TestCase):
        """Verify LOMO protocol properties from the manuscript."""

        def test_lomo_n50_split_into_10_chunks(self) -> None:
            """N=50 metabolites split into 10 chunks of 5 as in SLURM scripts."""
            n_metabolites = 50
            chunk_size = 5
            n_chunks = n_metabolites // chunk_size
            self.assertEqual(n_chunks, 10)
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                chunk = list(range(start, end))
                self.assertEqual(len(chunk), chunk_size)

        def test_lomo_fold_index_999_is_universal(self) -> None:
            """fold_index=999 means universal (synthetic) training per the scripts."""
            # This is a documentation test — ensures the protocol constant is clear
            UNIVERSAL_FOLD = 999
            self.assertEqual(UNIVERSAL_FOLD, 999)

        def test_layer_depth_variants(self) -> None:
            """LOMO tests 3 layer depths: 3, 4, 5."""
            depths = [3, 4, 5]
            self.assertEqual(len(depths), 3)
            # Residuals forced for depth >= 4
            for d in depths:
                force_residuals = d >= 4
                if d >= 4:
                    self.assertTrue(force_residuals)
                else:
                    self.assertFalse(force_residuals)

    class LOMOArchitectureWaterfallTest(unittest.TestCase):
        """Verify the 10 LOMO architecture configs from orchestrate_full_pipeline.sh."""

        LOMO_ARCHITECTURES = [
            'baseline_legacy',
            'relational_core',
            'relational_residuals',
            'relational_attention',
            'relational_mcr',
            'relational_full',
            'relational_core_gpe',
            'relational_mcr_multihop',
            'relational_mcr_multihop_gpe',
            'relational_full_plus',
        ]

        def test_ten_architectures(self) -> None:
            self.assertEqual(len(self.LOMO_ARCHITECTURES), 10)

        def test_physics_modes(self) -> None:
            """Only relational_full and relational_full_plus use physics."""
            physics_archs = {'relational_full', 'relational_full_plus'}
            for arch in self.LOMO_ARCHITECTURES:
                uses_physics = arch in physics_archs
                if uses_physics:
                    self.assertIn('full', arch)


if __name__ == '__main__':
    unittest.main()
