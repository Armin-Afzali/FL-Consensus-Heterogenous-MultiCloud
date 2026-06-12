"""
Federated Learning Aggregation Server.
Acts as the central hub in the multi-cloud deployment, orchestrating
rounds: node selection, update collection across regions,
trust-weighted aggregation, and global model update.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

from .aggregators import get_aggregator
from .network_sim import NetworkSimulator


class FLServer:
    """
    Federated Learning server with pluggable aggregation.
    """

    def __init__(
        self,
        global_model: nn.Module,
        aggregator_name: str,
        num_clients: int,
        clients_per_round: int,
        config: Dict,
        device: torch.device,
        seed: int = 42,
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.rng = np.random.RandomState(seed)

        # Create aggregator
        self.aggregator = get_aggregator(aggregator_name, num_clients, config)
        print(f"  Aggregator: {self.aggregator.name}")

        # Network simulator
        net_cfg = config.get("network", {})
        self.network_sim = NetworkSimulator(
            num_clients=num_clients,
            drop_rate=net_cfg.get("drop_rate", 0.0),
            straggler_fraction=net_cfg.get("straggler_fraction", 0.0),
            straggler_slowdown=net_cfg.get("straggler_slowdown", 3.0),
            timeout_multiplier=net_cfg.get("timeout_rounds", 1.5),
            seed=seed + 1,
        )

        # Minimum clients to proceed with a round
        self.min_clients = max(2, clients_per_round // 3)

    def select_clients(self) -> List[int]:
        """Randomly select clients for this round."""
        selected = self.rng.choice(
            self.num_clients,
            size=self.clients_per_round,
            replace=False,
        ).tolist()
        return selected

    def run_round(
        self,
        clients: list,
        selected_ids: List[int],
    ) -> Tuple[Optional[Dict], Dict]:
        """
        Execute one federated learning round.

        Args:
            clients: List of all FLClient instances
            selected_ids: IDs of selected clients

        Returns:
            (aggregation_info, round_metrics)
            aggregation_info is None if round was skipped
        """
        # Collect compute times for network simulation
        compute_times = {
            cid: clients[cid].get_simulated_time()
            for cid in selected_ids
        }

        # Simulate network (drops, timeouts, stragglers)
        responding_ids = self.network_sim.simulate_round(
            selected_ids, compute_times
        )

        net_info = self.network_sim.get_round_info(selected_ids, responding_ids)

        # Check minimum participation
        if len(responding_ids) < self.min_clients:
            return None, {
                "skipped": True,
                "reason": f"Only {len(responding_ids)} clients responded "
                          f"(need {self.min_clients})",
                "network": net_info,
            }

        # Collect updates from responding clients
        deltas = []
        num_samples = []
        client_losses = {}

        for cid in responding_ids:
            delta, loss = clients[cid].train(self.global_model)
            deltas.append(delta)
            num_samples.append(clients[cid].num_samples)
            client_losses[cid] = loss

        # Aggregate
        agg_delta, agg_info = self.aggregator.aggregate(
            deltas=deltas,
            num_samples=num_samples,
            client_ids=responding_ids,
            all_selected_ids=selected_ids,
        )

        # Apply aggregated update to global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.add_(agg_delta[name].to(self.device))

        round_metrics = {
            "skipped": False,
            "responding_clients": responding_ids,
            "num_responding": len(responding_ids),
            "client_losses": client_losses,
            "avg_client_loss": np.mean(list(client_losses.values())),
            "network": net_info,
        }

        return agg_info, round_metrics

    def evaluate(
        self,
        test_loader,
    ) -> Tuple[float, float]:
        """
        Evaluate global model on test set.

        Returns:
            (test_loss, test_accuracy)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        use_amp = self.device.type == "cuda"

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=use_amp
                ):
                    outputs = self.global_model(batch_x)
                    loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def get_model_state(self) -> OrderedDict:
        """Get a copy of the current global model state."""
        return copy.deepcopy(self.global_model.state_dict())
    