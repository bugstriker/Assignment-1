"""Training, evaluation, and deterministic seed utilities."""

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """Prefer Apple's MPS, then CUDA, and otherwise use the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Evaluation:
    """Evaluate average cross-entropy loss and top-1 accuracy."""
    loss_function = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return Evaluation(
        loss=total_loss / total,
        accuracy=100.0 * correct / total,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
) -> tuple[TrainingHistory, float, int]:
    """Train a model and restore its best validation-accuracy weights."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    best_weights: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        if total == 0:
            raise ValueError("Cannot train with an empty data loader")

        validation = evaluate(model, validation_loader, device, criterion)
        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.validation_loss.append(validation.loss)
        history.validation_accuracy.append(validation.accuracy)

        if validation.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation.accuracy
            best_epoch = epoch
            best_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Loss: {validation.loss:.4f} | "
            f"Validation Accuracy: {validation.accuracy:.2f}%"
        )

    if best_weights is None:
        raise RuntimeError("Training completed without producing model weights")
    model.load_state_dict(best_weights)
    print(
        f"Best validation accuracy: {best_validation_accuracy:.2f}% "
        f"(epoch {best_epoch})"
    )
    return history, best_validation_accuracy, best_epoch
