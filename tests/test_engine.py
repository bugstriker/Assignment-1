import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vgg6.engine import evaluate, train_model
from vgg6.model import VGG


class TrainingEngineTests(unittest.TestCase):
    def test_vgg_returns_one_score_per_class(self) -> None:
        model = VGG().eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(outputs.shape), (2, 10))

    def test_train_and_evaluate_small_dataset(self) -> None:
        device = torch.device("cpu")
        dataset = TensorDataset(
            torch.tensor([[0.0], [1.0], [0.1], [0.9]]),
            torch.tensor([0, 1, 0, 1]),
        )
        loader = DataLoader(dataset, batch_size=2)
        model = nn.Linear(1, 2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        history, best_accuracy, best_epoch = train_model(
            model,
            loader,
            loader,
            optimizer,
            device,
            epochs=1,
        )
        metrics = evaluate(model, loader, device)

        self.assertEqual(len(history.train_loss), 1)
        self.assertEqual(len(history.validation_accuracy), 1)
        self.assertGreaterEqual(best_accuracy, 0.0)
        self.assertEqual(best_epoch, 1)
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreaterEqual(metrics.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
