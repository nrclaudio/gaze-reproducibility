import os
import unittest

import torch

from src.training.trainer_engine import Trainer
from src.training.callbacks import LoggingCallback


RUN_TORCH_TESTS = os.environ.get('RUN_TORCH_TESTS') == '1'


class DummyData:
    def __init__(self, num_nodes: int, device: torch.device) -> None:
        self.device = device
        self.x_embedding = torch.randn(num_nodes, 4, device=device)
        self.y = torch.randn(num_nodes, device=device)
        self.mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        self.num_nodes = num_nodes

    def to(self, device, non_blocking=False):
        self.x_embedding = self.x_embedding.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        self.device = device
        return self


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.contrastive_module = None

    def forward(self, data, return_embeddings=False, compute_physics=True):
        preds = self.linear(data.x_embedding).squeeze(-1)
        delta = torch.zeros_like(preds)
        flux_vec = torch.zeros_like(preds)
        embeddings = preds.unsqueeze(-1) if return_embeddings else None
        return preds, delta, flux_vec, embeddings


class DummyCriterion:
    def __init__(self):
        self.last_loss_components = {}

    def __call__(self, preds, targets, mask, delta, flux_vec, epoch, data):
        loss = torch.mean((preds - targets) ** 2)
        self.last_loss_components = {'MSE': float(loss.item())}
        return loss


@unittest.skipUnless(RUN_TORCH_TESTS, "Set RUN_TORCH_TESTS=1 to run trainer smoke test.")
class TrainerSmokeTest(unittest.TestCase):
    def test_trainer_runs_minimal_loop(self):
        device = torch.device('cpu')
        model = DummyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        criterion = DummyCriterion()

        train_loader = [DummyData(5, device) for _ in range(2)]
        val_loader = [DummyData(5, device)]
        test_loader = [DummyData(5, device)]

        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            gradient_accumulation_steps=1,
            contrastive_weight=0.0,
            enable_contrastive=False,
            callbacks=[LoggingCallback(verbose=False)],
        )

        history = trainer.fit(train_loader, val_loader, epochs=2, patience=3)
        self.assertTrue(len(history['train_loss']) >= 1)

        logs = trainer.test(test_loader, dataset=None, test_indices=None, has_physics=False)
        self.assertIn('test_loss', logs)
        self.assertIn('test_r2', logs)
        self.assertIn('test_rmse', logs)


if __name__ == '__main__':
    unittest.main()
