"""
Network heterogeneity simulation.
Simulates variable latency, bandwidth constraints, and client drop-outs.
"""

import numpy as np
from typing import List, Set, Dict


class NetworkSimulator:
    """
    Simulates network heterogeneity in federated learning.

    Models:
    - Variable latency per client
    - Random client drop-outs
    - Straggler clients (slow network + slow compute)
    - Timeout-based exclusion
    """

    def __init__(
        self,
        num_clients: int,
        drop_rate: float = 0.0,
        straggler_fraction: float = 0.0,
        straggler_slowdown: float = 3.0,
        timeout_multiplier: float = 1.5,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.drop_rate = drop_rate
        self.straggler_fraction = straggler_fraction
        self.straggler_slowdown = straggler_slowdown
        self.timeout_multiplier = timeout_multiplier
        self.rng = np.random.RandomState(seed)

        # Pre-assign straggler identities (persistent across rounds)
        num_stragglers = int(num_clients * straggler_fraction)
        self.straggler_ids = set(
            self.rng.choice(num_clients, size=num_stragglers, replace=False)
        )
        if num_stragglers > 0:
            print(f"  Network stragglers: {sorted(self.straggler_ids)}")

    def simulate_round(
        self,
        selected_ids: List[int],
        compute_times: Dict[int, float],
    ) -> List[int]:
        """
        Simulate network effects for a round.

        Args:
            selected_ids: IDs of clients selected for this round
            compute_times: Simulated compute time per client

        Returns:
            List of client IDs that successfully respond (after drops/timeouts)
        """
        # Compute effective times (compute + network overhead for stragglers)
        effective_times = {}
        for cid in selected_ids:
            base_time = compute_times.get(cid, 1.0)
            if cid in self.straggler_ids:
                base_time *= self.straggler_slowdown
            effective_times[cid] = base_time

        # Determine timeout threshold
        if effective_times:
            times_list = list(effective_times.values())
            median_time = np.median(times_list)
            timeout = median_time * self.timeout_multiplier
        else:
            timeout = float("inf")

        # Filter clients
        responding_ids = []
        for cid in selected_ids:
            # Random drop
            if self.rng.random() < self.drop_rate:
                continue

            # Timeout
            if effective_times[cid] > timeout:
                continue

            responding_ids.append(cid)

        return responding_ids

    def get_round_info(
        self,
        selected_ids: List[int],
        responding_ids: List[int],
    ) -> Dict:
        """Get info about network effects in a round."""
        dropped = set(selected_ids) - set(responding_ids)
        return {
            "selected": len(selected_ids),
            "responding": len(responding_ids),
            "dropped": sorted(dropped),
            "stragglers_selected": sorted(
                set(selected_ids) & self.straggler_ids
            ),
        }
