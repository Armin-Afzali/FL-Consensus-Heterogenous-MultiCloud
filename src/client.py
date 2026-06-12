"""
Federated learning cloud node implementation.
Each node represents a cloud instance in a multi-cloud deployment,
holding private regional data and performing local model training.
Supports simulation of honest, noisy, and compromised node behavior.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
from collections import OrderedDict


class FLClient:
    """
    Federated Learning cloud node.

    Represents a cloud instance in a multi-cloud deployment.
    Performs local training on private regional data and optionally
    simulates faulty or adversarial node behavior (noise, sign-flip, scaling).
    """

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device,
        local_epochs: int = 5,
        local_lr: float = 0.01,
        attack_type: str = "none",
        attack_params: Optional[Dict] = None,
        compute_speed: float = 1.0,
        use_amp: bool = True,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        self.compute_speed = compute_speed  # relative speed (0.3 = slow, 1.0 = fast)
        self.num_samples = len(dataloader.dataset)
        # Enable AMP on CUDA for ~2-3x speedup
        self.use_amp = use_amp and (device.type == "cuda")

    def set_local_model(self, model_template: nn.Module):
        """
        Store a persistent local model to avoid re-creating it each round.
        Call once at setup time. Much faster than copy.deepcopy every round.
        """
        self._local_model = copy.deepcopy(model_template)
        self._local_model.to(self.device)

    def train(
        self,
        global_model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local training and return the model update (delta).

        Args:
            global_model: Current global model

        Returns:
            Dictionary mapping parameter names to update tensors (delta = local - global)
        """
        # Fast path: reuse persistent local model, just reload weights
        if hasattr(self, '_local_model'):
            local_model = self._local_model
            local_model.load_state_dict(global_model.state_dict())
        else:
            local_model = copy.deepcopy(global_model)
            local_model.to(self.device)

        local_model.train()

        # Save initial (global) parameters — use state_dict (already detached)
        global_params = {
            name: param.detach().clone()
            for name, param in global_model.named_parameters()
        }

        # Local SGD
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        total_loss = 0.0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    outputs = local_model(batch_x)
                    loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

        avg_loss = total_loss / max(total_samples, 1)

        # Compute update delta = local_params - global_params
        delta = OrderedDict()
        for name, param in local_model.named_parameters():
            delta[name] = (param.detach() - global_params[name]).cpu()

        # Apply attack if this is a malicious client
        delta = self._apply_attack(delta)

        return delta, avg_loss

    def _apply_attack(
        self,
        delta: OrderedDict,
    ) -> OrderedDict:
        """Apply attack transformation to the update."""
        if self.attack_type == "none":
            return delta

        elif self.attack_type == "noise":
            # Add Gaussian noise to the update
            noise_std = self.attack_params.get("noise_std", 0.5)
            for name in delta:
                noise = torch.randn_like(delta[name]) * noise_std
                delta[name] = delta[name] + noise
            return delta

        elif self.attack_type == "sign_flip":
            # Flip the sign of the update (sends model in wrong direction)
            flip_scale = self.attack_params.get("flip_scale", 1.0)
            for name in delta:
                delta[name] = -flip_scale * delta[name]
            return delta

        elif self.attack_type == "scaling":
            # Scale the update by a large factor
            scaling_factor = self.attack_params.get("scaling_factor", 5.0)
            for name in delta:
                delta[name] = scaling_factor * delta[name]
            return delta

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def get_simulated_time(self, base_time: float = 1.0) -> float:
        """
        Get simulated compute time based on client's speed.
        Slower clients take longer to train.
        """
        return base_time / self.compute_speed


def create_clients(
    dataloaders: list,
    device: torch.device,
    local_epochs: int,
    local_lr: float,
    attack_config: Dict,
    compute_config: Dict,
    seed: int = 42,
) -> list:
    """
    Create all FL clients with appropriate configurations.

    Args:
        dataloaders: List of DataLoaders, one per client
        device: Torch device
        local_epochs: Number of local training epochs
        local_lr: Local learning rate
        attack_config: Attack configuration dict
        compute_config: Compute heterogeneity config
        seed: Random seed

    Returns:
        List of FLClient instances
    """
    rng = np.random.RandomState(seed)
    num_clients = len(dataloaders)

    # Determine which clients are malicious
    attack_type = attack_config.get("type", "none")
    attack_fraction = attack_config.get("fraction", 0.0)
    num_malicious = int(num_clients * attack_fraction)

    malicious_ids = set(rng.choice(num_clients, size=num_malicious, replace=False))

    # Assign compute speeds
    if compute_config.get("enabled", False):
        min_speed = compute_config.get("min_speed", 0.3)
        max_speed = compute_config.get("max_speed", 1.0)

        if compute_config.get("speed_distribution") == "bimodal":
            # Half fast, half slow
            speeds = np.zeros(num_clients)
            fast_ids = rng.choice(num_clients, size=num_clients // 2, replace=False)
            for i in range(num_clients):
                speeds[i] = max_speed if i in fast_ids else min_speed
        else:
            speeds = rng.uniform(min_speed, max_speed, size=num_clients)
    else:
        speeds = np.ones(num_clients)

    # Build attack params
    attack_params = {
        "noise_std": attack_config.get("noise_std", 0.5),
        "flip_scale": attack_config.get("flip_scale", 1.0),
        "scaling_factor": attack_config.get("scaling_factor", 5.0),
    }

    # Create clients
    clients = []
    for i in range(num_clients):
        client_attack = attack_type if i in malicious_ids else "none"
        client = FLClient(
            client_id=i,
            dataloader=dataloaders[i],
            device=device,
            local_epochs=local_epochs,
            local_lr=local_lr,
            attack_type=client_attack,
            attack_params=attack_params,
            compute_speed=speeds[i],
        )
        clients.append(client)

    # Log setup
    if num_malicious > 0:
        print(f"  Malicious clients ({attack_type}): {sorted(malicious_ids)}")
    print(f"  Compute speeds: min={speeds.min():.2f}, max={speeds.max():.2f}, "
          f"mean={speeds.mean():.2f}")

    return clients
