import os
import unittest


SKIP_TORCH_TESTS = os.environ.get('RUN_TORCH_TESTS') != '1'


if SKIP_TORCH_TESTS:

    class FiLMDataModuleTest(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable FiLMDataModule tests.")
        def test_datamodule_skip(self) -> None:
            pass


else:
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    from src.config import DataConfig


    class _StubDataLoader:
        def __init__(self, dataset, *args, **kwargs):
            self.dataset = dataset
            self.args = args
            self.kwargs = kwargs


    class _StubSubset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]


    if 'torch' not in sys.modules:
        torch_stub = SimpleNamespace()
        data_pkg = SimpleNamespace(Subset=_StubSubset, DataLoader=_StubDataLoader)
        torch_stub.utils = SimpleNamespace(data=data_pkg)
        sys.modules['torch'] = torch_stub
        sys.modules['torch.utils'] = SimpleNamespace(data=data_pkg)
        sys.modules['torch.utils.data'] = data_pkg


    class _StubData:
        def __init__(self, num_nodes: int) -> None:
            self.num_nodes = num_nodes


    class _StubDataset:
        SIZE = 5

        def __init__(self, *args, **kwargs):
            self._size = self.SIZE
            self.embedding_dim = 8
            self.gene_dim = 4
            self.smiles_dim = 6
            self.node_types = ['metabolite', 'reaction']

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int):
            return _StubData(num_nodes=len(self.node_types))


    if 'src.data.dataset_smiles' not in sys.modules:
        dataset_module = ModuleType('src.data.dataset_smiles')
        dataset_module.FiLMDatasetWithSMILES = _StubDataset
        dataset_module.SpatialTranscriptomicsDataset = _StubDataset
        sys.modules['src.data.dataset_smiles'] = dataset_module

    from src.data.datamodule import FiLMDataModule


    class FiLMDataModuleTest(unittest.TestCase):
        @patch('src.data.datamodule.collate_metabolic_data', lambda batch: batch)
        @patch('src.data.datamodule.FiLMDatasetWithSMILES', new=_StubDataset)
        def test_split_proportions(self, *_args) -> None:
            _StubDataset.SIZE = 10
            config = DataConfig(train_fraction=0.6, val_fraction=0.2, batch_size=2)
            module = FiLMDataModule(config=config, seed=7, device=None)
            state = module.setup()

            self.assertEqual(state.embedding_dim, 8)
            splits = module.splits
            self.assertEqual(len(splits['train']), 6)
            self.assertEqual(len(splits['val']), 2)
            self.assertEqual(len(splits['test']), 2)


if __name__ == '__main__':
    unittest.main()
