"""CIFAR-10 transforms, reproducible splits, and data loaders."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass(frozen=True)
class DataLoaders:
    """Training, validation, and test loaders for one experiment."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    validation_size: int = 5000,
    seed: int = 42,
) -> DataLoaders:
    """Download CIFAR-10 and create reproducible train/validation/test loaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    augmented_training_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    evaluation_training_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=evaluation_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=evaluation_transform,
    )

    if not 0 < validation_size < len(augmented_training_set):
        raise ValueError(
            "validation_size must be between zero and the training-set size"
        )

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(augmented_training_set),
        generator=split_generator,
    ).tolist()
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    train_generator = torch.Generator().manual_seed(seed)
    return DataLoaders(
        train=DataLoader(
            Subset(augmented_training_set, training_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=train_generator,
        ),
        validation=DataLoader(
            Subset(evaluation_training_set, validation_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        test=DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    )
