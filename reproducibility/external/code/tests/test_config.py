import unittest

from src.config import ExperimentConfig


class ExperimentConfigAliasTest(unittest.TestCase):
    def test_alias_mapping(self) -> None:
        raw = {
            "seed": 123,
            "physics": {
                "\u03bb_bal": 0.5,
                "\u03bb_corr": 0.1,
                "\u03bb_var": 0.2,
                "\u03bb_kl": 0.0,
            },
            "contrastive": {
                "enable_contrastive": True,
                "contrastive_weight": 0.3,
                "contrastive_temperature": 0.05,
            },
            "data": {
                "use_normalised": True,
                "batch_size": 4,
            },
        }

        config = ExperimentConfig.from_dict(raw)

        self.assertEqual(config.physics.lambda_balance, 0.5)
        self.assertEqual(config.physics.lambda_correlation, 0.1)
        self.assertEqual(config.physics.lambda_variance, 0.2)
        self.assertEqual(config.physics.lambda_kl, 0.0)
        self.assertTrue(config.contrastive.enabled)
        self.assertAlmostEqual(config.contrastive.weight, 0.3)
        self.assertAlmostEqual(config.contrastive.temperature, 0.05)
        self.assertTrue(config.data.normalized)
        self.assertEqual(config.data.batch_size, 4)

        serialised = config.to_dict()
        self.assertIn("lambda_balance", serialised["physics"])
        self.assertEqual(serialised["physics"]["lambda_balance"], 0.5)


if __name__ == "__main__":
    unittest.main()
