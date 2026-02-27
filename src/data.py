"""
Data loading and non-IID partitioning for federated learning.
Uses Dirichlet distribution for controlling data heterogeneity.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict


def get_dataset(dataset_name: str, data_dir: str = "./data") -> Tuple:
    """
    Load dataset with appropriate transforms.
    Returns (train_dataset, test_dataset).
    """
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset indices among clients using Dirichlet distribution.

    Args:
        dataset: Dataset with .targets attribute
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
                High (100.0) -> nearly IID
                Low (0.1) -> highly non-IID
        seed: Random seed

    Returns:
        List of index lists, one per client
    """
    rng = np.random.RandomState(seed)

    # Get labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([y for _, y in dataset])

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Ensure minimum 1 sample per client that gets any from this class
        # by adding a small floor
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()

        # Split indices according to proportions
        splits = (proportions * len(class_indices)).astype(int)

        # Handle rounding: give remaining samples to clients with fewest
        remainder = len(class_indices) - splits.sum()
        if remainder > 0:
            deficit_order = np.argsort(splits)
            for i in range(remainder):
                splits[deficit_order[i % num_clients]] += 1

        # Assign indices
        current = 0
        for i in range(num_clients):
            end = current + splits[i]
            client_indices[i].extend(class_indices[current:end].tolist())
            current = end

    # Shuffle each client's data
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def get_client_dataloaders(
    dataset,
    client_indices: List[List[int]],
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> List[DataLoader]:
    """Create a DataLoader for each client."""
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory if num_workers > 0 else False,
            persistent_workers=num_workers > 0,
        )
        loaders.append(loader)
    return loaders


def get_test_dataloader(
    dataset,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the global test set."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_class_distribution(
    dataset,
    client_indices: List[List[int]],
) -> np.ndarray:
    """
    Compute class distribution per client for analysis.
    Returns array of shape (num_clients, num_classes).
    """
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)
    distribution = np.zeros((num_clients, num_classes))

    for i, indices in enumerate(client_indices):
        for idx in indices:
            distribution[i, labels[idx]] += 1

    return distribution
