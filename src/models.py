"""
Neural network model definitions for federated learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.
    Architecture:
        Conv2d(3,32,3) -> ReLU -> Conv2d(32,64,3) -> ReLU -> MaxPool(2)
        Conv2d(64,64,3) -> ReLU -> MaxPool(2)
        FC(4096,128) -> ReLU -> FC(128,10)
    ~582K parameters - suitable for quick federated experiments.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size: after two pools on 32x32 -> 8x8 with 64 channels
        self._flat_size = 64 * 8 * 8

        self.fc1 = nn.Linear(self._flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN_MNIST(nn.Module):
    """Simple CNN for MNIST/Fashion-MNIST (1 input channel)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After two pools on 28x28 -> 7x7 with 64 channels
        self._flat_size = 64 * 7 * 7

        self.fc1 = nn.Linear(self._flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name: str, dataset: str) -> nn.Module:
    """Factory function to create models."""
    if dataset in ("mnist", "fashion_mnist"):
        return SimpleCNN_MNIST(num_classes=10)
    elif dataset == "cifar10":
        return SimpleCNN(num_classes=10, in_channels=3)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
