"""Command-line entry point for the VGG6 CIFAR-10 baseline."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from .data import create_data_loaders
from .engine import evaluate, select_device, set_seed, train_model
from .model import VGG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the VGG6 CIFAR-10 baseline."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", default="vgg6_baseline_best.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device()
    print(f"Device: {device}")

    loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = VGG().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    _, best_accuracy, best_epoch = train_model(
        model=model,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    test_metrics = evaluate(model, loaders.test, device)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Test loss: {test_metrics.loss:.4f}")
    print(f"Test top-1 accuracy: {test_metrics.accuracy:.2f}%")


if __name__ == "__main__":
    main()
