"""Tests for physics-informed loss functions and the loss factory."""

import os
import unittest

SKIP_TORCH_TESTS = os.environ.get('RUN_TORCH_TESTS') != '1'

if SKIP_TORCH_TESTS:

    class PhysicsLossTest(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable physics loss tests.")
        def test_skip(self) -> None:
            pass

else:
    import torch
    from src.losses.physics import PhysicsLoss, EnhancedPhysicsLoss
    from src.training.loss_factory import create_physics_loss
    from src.config import PhysicsConfig

    def _make_tensors(num_metabolites: int = 10, num_reactions: int = 8):
        """Create minimal tensors for loss computation."""
        c_hat = torch.randn(1, num_metabolites)
        c_true = torch.randn(1, num_metabolites)
        mask = torch.ones(1, num_metabolites, dtype=torch.bool)
        delta = torch.randn(1, num_metabolites)
        flux_vec = torch.randn(1, num_reactions).abs()  # non-negative
        return c_hat, c_true, mask, delta, flux_vec

    class PhysicsLossTest(unittest.TestCase):
        """Tests for the basic PhysicsLoss."""

        def test_forward_returns_scalar(self) -> None:
            loss_fn = PhysicsLoss(λ_mse=1.0, λ_bal=1.0)
            c_hat, c_true, mask, delta, flux_vec = _make_tensors()
            loss = loss_fn(c_hat, c_true, mask, delta, flux_vec, epoch=0)
            self.assertEqual(loss.shape, torch.Size([]))

        def test_zero_balance_weight_ignores_delta(self) -> None:
            loss_fn = PhysicsLoss(λ_mse=1.0, λ_bal=0.0)
            c_hat, c_true, mask, delta, flux_vec = _make_tensors()
            loss_a = loss_fn(c_hat, c_true, mask, delta, flux_vec, epoch=0)
            # Change delta — loss should stay the same since λ_bal=0
            loss_b = loss_fn(c_hat, c_true, mask, delta * 100, flux_vec, epoch=0)
            self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=5)

        def test_empty_mask_returns_balance_only(self) -> None:
            loss_fn = PhysicsLoss(λ_mse=1.0, λ_bal=1.0)
            c_hat, c_true, _, delta, flux_vec = _make_tensors()
            mask = torch.zeros(1, c_hat.size(1), dtype=torch.bool)
            loss = loss_fn(c_hat, c_true, mask, delta, flux_vec, epoch=0)
            # With no masked entries, MSE is 0 — loss equals balance only
            expected = torch.mean(delta ** 2)
            self.assertAlmostEqual(loss.item(), expected.item(), places=5)

        def test_unbatched_input_handled(self) -> None:
            """PhysicsLoss should accept 1-D (unbatched) tensors."""
            loss_fn = PhysicsLoss()
            n_met, n_rxn = 10, 8
            loss = loss_fn(
                torch.randn(n_met), torch.randn(n_met),
                torch.ones(n_met, dtype=torch.bool),
                torch.randn(n_met), torch.randn(n_rxn).abs(),
                epoch=0,
            )
            self.assertEqual(loss.shape, torch.Size([]))

    class EnhancedPhysicsLossTest(unittest.TestCase):
        """Tests for EnhancedPhysicsLoss."""

        def test_all_lambdas_zero_returns_mse_only(self) -> None:
            loss_fn = EnhancedPhysicsLoss(
                λ_mse=1.0, λ_bal=0.0, λ_corr=0.0, λ_var=0.0, λ_kl=0.0,
                use_listwise_ranking=False,
            )
            c_hat, c_true, mask, delta, flux_vec = _make_tensors()
            loss = loss_fn(c_hat, c_true, mask, delta, flux_vec, epoch=0)
            # Should equal pure MSE
            expected_mse = torch.nn.functional.mse_loss(c_hat[mask], c_true[mask])
            self.assertAlmostEqual(loss.item(), expected_mse.item(), places=4)

        def test_loss_components_logged(self) -> None:
            loss_fn = EnhancedPhysicsLoss(
                λ_mse=1.0, λ_bal=1.0, λ_corr=0.0, λ_var=1.0, λ_kl=0.0,
                use_listwise_ranking=False,
            )
            c_hat, c_true, mask, delta, flux_vec = _make_tensors()
            loss_fn(c_hat, c_true, mask, delta, flux_vec, epoch=0)
            components = loss_fn.last_loss_components
            self.assertIn('MSE', components)
            self.assertIn('Balance', components)
            self.assertIn('Variance', components)

        def test_variance_loss_penalises_constant_flux(self) -> None:
            """Constant flux (zero variance) should produce a larger variance loss."""
            loss_fn = EnhancedPhysicsLoss(
                λ_mse=0.0, λ_bal=0.0, λ_corr=0.0, λ_var=1.0, λ_kl=0.0,
                use_listwise_ranking=False,
            )
            c_hat, c_true, mask, delta, _ = _make_tensors()
            # Constant flux → high variance penalty
            flux_const = torch.ones(1, 8)
            loss_const = loss_fn(c_hat, c_true, mask, delta, flux_const, epoch=0)
            # Diverse flux → low variance penalty
            flux_diverse = torch.randn(1, 8) * 5
            loss_diverse = loss_fn(c_hat, c_true, mask, delta, flux_diverse, epoch=0)
            self.assertGreater(loss_const.item(), loss_diverse.item())

        def test_balance_loss_grows_with_delta(self) -> None:
            loss_fn = EnhancedPhysicsLoss(
                λ_mse=0.0, λ_bal=1.0, λ_corr=0.0, λ_var=0.0, λ_kl=0.0,
                use_listwise_ranking=False,
            )
            c_hat, c_true, mask, _, flux_vec = _make_tensors()
            small_delta = torch.zeros(1, 10)
            large_delta = torch.ones(1, 10) * 10.0
            loss_small = loss_fn(c_hat, c_true, mask, small_delta, flux_vec, epoch=0)
            loss_large = loss_fn(c_hat, c_true, mask, large_delta, flux_vec, epoch=0)
            self.assertGreater(loss_large.item(), loss_small.item())

    class LossFactoryTest(unittest.TestCase):
        """Tests for create_physics_loss factory."""

        def test_creates_enhanced_physics_loss(self) -> None:
            cfg = PhysicsConfig(lambda_balance=0.5, lambda_correlation=0.1, lambda_variance=1.0)
            loss, listwise_enabled = create_physics_loss(
                cfg, enzyme_confidence=3.0, flux_flexibility=0.8,
            )
            self.assertIsInstance(loss, EnhancedPhysicsLoss)
            self.assertFalse(listwise_enabled)

        def test_auto_enables_listwise_when_kl_positive(self) -> None:
            cfg = PhysicsConfig(lambda_kl=1.0)
            _, listwise_enabled = create_physics_loss(
                cfg, enzyme_confidence=3.0, flux_flexibility=0.8,
            )
            self.assertTrue(listwise_enabled)

        def test_lambda_weights_propagated(self) -> None:
            cfg = PhysicsConfig(
                lambda_balance=0.5, lambda_correlation=0.1,
                lambda_variance=2.0, lambda_kl=0.0,
            )
            loss, _ = create_physics_loss(cfg, enzyme_confidence=3.0, flux_flexibility=0.8)
            self.assertAlmostEqual(loss.λ_bal, 0.5)
            self.assertAlmostEqual(loss.λ_corr, 0.1)
            self.assertAlmostEqual(loss.λ_var, 2.0)

    class PhysicsConfigTest(unittest.TestCase):
        """Tests for PhysicsConfig dataclass."""

        def test_default_all_zero(self) -> None:
            cfg = PhysicsConfig()
            self.assertEqual(cfg.lambda_balance, 0.0)
            self.assertEqual(cfg.lambda_correlation, 0.0)
            self.assertEqual(cfg.lambda_variance, 0.0)
            self.assertEqual(cfg.lambda_kl, 0.0)

        def test_from_dict_with_unicode_aliases(self) -> None:
            raw = {"λ_bal": 0.5, "λ_corr": 0.1, "λ_var": 1.0, "λ_kl": 0.0}
            cfg = PhysicsConfig.from_dict(raw)
            self.assertAlmostEqual(cfg.lambda_balance, 0.5)
            self.assertAlmostEqual(cfg.lambda_correlation, 0.1)
            self.assertAlmostEqual(cfg.lambda_variance, 1.0)

        def test_roundtrip_serialization(self) -> None:
            cfg = PhysicsConfig(lambda_balance=2.0, lambda_correlation=0.5, lambda_variance=1.0)
            rebuilt = PhysicsConfig.from_dict(cfg.to_dict())
            self.assertEqual(cfg, rebuilt)

        def test_from_dict_drops_removed_keys(self) -> None:
            raw = {"lambda_balance": 1.0, "lambda_detection": 0.5, "λ_det": 0.3}
            cfg = PhysicsConfig.from_dict(raw)
            self.assertAlmostEqual(cfg.lambda_balance, 1.0)
            self.assertFalse(hasattr(cfg, 'lambda_detection'))

        def test_sweep_grid_coverage(self) -> None:
            """Verify that the physics sweep grid from the SLURM script is expressible."""
            bal_values = [0.0, 0.1, 0.5, 1.0, 2.0]
            corr_values = [0.0, 0.1, 0.5, 1.0]
            var_values = [0.0, 1.0, 2.0]
            configs = []
            for b in bal_values:
                for c in corr_values:
                    for v in var_values:
                        cfg = PhysicsConfig(
                            lambda_balance=b, lambda_correlation=c, lambda_variance=v,
                        )
                        configs.append(cfg)
            self.assertEqual(len(configs), 60)  # 5 × 4 × 3 = 60 as in submit_physics_sweep


if __name__ == '__main__':
    unittest.main()
