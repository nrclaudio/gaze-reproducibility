import os
import unittest


if os.environ.get('RUN_TORCH_TESTS') != '1':

    class ComponentBuilderTests(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable component builder tests.")
        def test_placeholder(self):
            pass

else:
    import torch

    from src.models.components import (
        build_encoders,
        build_flux_head,
        build_metabolite_head,
        build_gnn_layers,
        BaseConcatModel,
        MCRReaderConfig,
    )


    class ComponentBuilderTests(unittest.TestCase):
        def test_build_encoders(self):
            reaction_encoder, metabolite_encoder = build_encoders(embedding_dim=8, gene_dim=4, smiles_dim=6, hidden_dim=16)
            self.assertEqual(reaction_encoder.in_features, 12)
            self.assertEqual(reaction_encoder.out_features, 16)
            self.assertEqual(metabolite_encoder.in_features, 6)
            self.assertEqual(metabolite_encoder.out_features, 16)

        def test_flux_head_positive_output(self):
            head = build_flux_head(hidden_dim=16)
            sample = torch.randn(3, 16)
            out = head(sample)
            self.assertEqual(out.shape, (3, 1))
            self.assertTrue(torch.all(out >= 0))

        def test_metabolite_head_switch(self):
            reader_cfg = MCRReaderConfig(smiles_dim=6, hidden_dim=32, mlp_hidden=64, normalize_descriptors=False, dropout=0.1)
            reader, predictor = build_metabolite_head(
                use_mcr=True,
                smiles_dim=6,
                hidden_dim=16,
                dropout=0.1,
                reader_config=reader_cfg,
            )
            self.assertIsNotNone(reader)
            self.assertIsNone(predictor)

            reader, predictor = build_metabolite_head(
                use_mcr=False,
                smiles_dim=6,
                hidden_dim=16,
                dropout=0.1,
                reader_config=None,
            )
            self.assertIsNone(reader)
            self.assertIsNotNone(predictor)

        def test_gnn_layers_directional(self):
            layers = build_gnn_layers(
                num_layers=2,
                hidden_dim=32,
                dropout=0.1,
                use_directional=True,
                directional_concat=False,
            )
            self.assertEqual(len(layers), 2)


    class BaseModelSmokeTest(unittest.TestCase):
        def test_base_initialization(self):
            model = BaseConcatModel(
                embedding_dim=8,
                gene_dim=4,
                smiles_dim=6,
                hidden_dim=16,
                dropout=0.1,
                use_flux_messages=True,
                flux_message_depth=1,
                use_mass_balance_projection=True,
                use_mcr=False,
                reader_config=None,
            )
            sample = torch.randn(5, 12)
            encoded = model.reaction_encoder(sample)
            self.assertEqual(encoded.shape, (5, 16))
            self.assertIsNotNone(model.flux_message_layer)
            self.assertIsNone(model.metabolite_reader)


if __name__ == '__main__':
    unittest.main()
