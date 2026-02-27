"""
Aggregation methods for federated learning.

Implements:
- FedAvg: Standard federated averaging
- TrimmedMean: Coordinate-wise trimmed mean
- Krum: Multi-Krum Byzantine-resilient aggregation
- TWAC: Trust-Weighted Adaptive Consensus (proposed method)
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional


def flatten_params(delta: OrderedDict) -> torch.Tensor:
    """Flatten an OrderedDict of tensors into a single 1D tensor."""
    return torch.cat([v.reshape(-1) for v in delta.values()])


def unflatten_params(flat: torch.Tensor, reference: OrderedDict) -> OrderedDict:
    """Unflatten a 1D tensor back into an OrderedDict matching the reference structure."""
    result = OrderedDict()
    offset = 0
    for name, ref_tensor in reference.items():
        numel = ref_tensor.numel()
        result[name] = flat[offset:offset + numel].reshape(ref_tensor.shape)
        offset += numel
    return result


# ===========================================================================
# FedAvg
# ===========================================================================

class FedAvg:
    """Standard Federated Averaging aggregation."""

    def __init__(self):
        self.name = "FedAvg"

    def aggregate(
        self,
        deltas: List[OrderedDict],
        num_samples: List[int],
        client_ids: List[int],
        **kwargs,
    ) -> Tuple[OrderedDict, Dict]:
        """
        Weighted average of client updates by number of samples.

        Returns:
            (aggregated_delta, info_dict)
        """
        total_samples = sum(num_samples)
        weights = [n / total_samples for n in num_samples]

        # Weighted average
        agg_delta = OrderedDict()
        for name in deltas[0]:
            agg_delta[name] = sum(
                w * d[name] for w, d in zip(weights, deltas)
            )

        info = {
            "weights": {cid: w for cid, w in zip(client_ids, weights)},
        }
        return agg_delta, info


# ===========================================================================
# Trimmed Mean
# ===========================================================================

class TrimmedMean:
    """Coordinate-wise trimmed mean aggregation."""

    def __init__(self, trim_ratio: float = 0.2):
        self.name = "TrimmedMean"
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        deltas: List[OrderedDict],
        num_samples: List[int],
        client_ids: List[int],
        **kwargs,
    ) -> Tuple[OrderedDict, Dict]:
        """
        Compute coordinate-wise trimmed mean.
        Trims trim_ratio fraction from each tail.
        """
        n = len(deltas)
        trim_count = max(1, int(n * self.trim_ratio))

        agg_delta = OrderedDict()
        for name in deltas[0]:
            # Stack all updates for this parameter: shape (n, ...)
            stacked = torch.stack([d[name] for d in deltas])
            original_shape = stacked.shape[1:]

            # Flatten to (n, d) for coordinate-wise operations
            flat = stacked.reshape(n, -1)

            # Sort along client dimension
            sorted_flat, _ = torch.sort(flat, dim=0)

            # Trim and average
            trimmed = sorted_flat[trim_count:n - trim_count]
            agg_delta[name] = trimmed.mean(dim=0).reshape(original_shape)

        info = {
            "trim_count": trim_count,
            "clients_used": n - 2 * trim_count,
        }
        return agg_delta, info


# ===========================================================================
# Krum (Multi-Krum)
# ===========================================================================

class Krum:
    """Multi-Krum Byzantine-resilient aggregation."""

    def __init__(self, num_select: int = 5):
        self.name = "Krum"
        self.num_select = num_select

    def aggregate(
        self,
        deltas: List[OrderedDict],
        num_samples: List[int],
        client_ids: List[int],
        **kwargs,
    ) -> Tuple[OrderedDict, Dict]:
        """
        Multi-Krum: select top-k updates with smallest sum of distances
        to their nearest neighbors, then average.
        """
        n = len(deltas)
        num_select = min(self.num_select, n)

        # Flatten all deltas
        flat_deltas = [flatten_params(d) for d in deltas]
        flat_stack = torch.stack(flat_deltas)  # (n, d)

        # Compute pairwise squared distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        norms_sq = (flat_stack ** 2).sum(dim=1)  # (n,)
        dot_products = flat_stack @ flat_stack.T  # (n, n)
        distances_sq = norms_sq.unsqueeze(1) + norms_sq.unsqueeze(0) - 2 * dot_products
        distances_sq = torch.clamp(distances_sq, min=0)  # numerical stability

        # For each client, compute sum of distances to n-2 nearest neighbors
        # (exclude self and the assumed Byzantine count)
        num_neighbors = max(1, n - 2)  # n - f - 2, with f estimated
        scores = torch.zeros(n)

        for i in range(n):
            dists = distances_sq[i].clone()
            dists[i] = float("inf")  # exclude self
            nearest, _ = torch.topk(dists, num_neighbors, largest=False)
            scores[i] = nearest.sum()

        # Select top-k with smallest scores
        _, selected_indices = torch.topk(scores, num_select, largest=False)
        selected_indices = selected_indices.tolist()

        # Average selected updates (weighted by sample count)
        selected_samples = [num_samples[i] for i in selected_indices]
        total = sum(selected_samples)
        weights = [s / total for s in selected_samples]

        agg_delta = OrderedDict()
        for name in deltas[0]:
            agg_delta[name] = sum(
                weights[j] * deltas[i][name]
                for j, i in enumerate(selected_indices)
            )

        info = {
            "selected_clients": [client_ids[i] for i in selected_indices],
            "krum_scores": {client_ids[i]: scores[i].item() for i in range(n)},
        }
        return agg_delta, info


# ===========================================================================
# TWAC: Trust-Weighted Adaptive Consensus
# ===========================================================================

class TWAC:
    """
    Trust-Weighted Adaptive Consensus (proposed method).

    Maintains per-client trust scores based on:
    1. Directional consistency (cosine similarity with reference direction)
    2. Magnitude reasonableness (penalize outlier norms)

    Trust scores are updated via EMA and used as aggregation weights.
    """

    def __init__(
        self,
        num_clients: int,
        trust_momentum: float = 0.7,
        magnitude_sensitivity: float = 2.0,
        magnitude_tolerance: float = 2.0,
        reference_momentum: float = 0.9,
        trust_floor: float = 0.1,
        straggler_decay: float = 0.98,
    ):
        self.name = "TWAC"
        self.num_clients = num_clients

        # Hyperparameters
        self.gamma = trust_momentum          # trust EMA momentum
        self.alpha = magnitude_sensitivity   # norm outlier sensitivity
        self.beta = magnitude_tolerance      # norm tolerance (multiples of median)
        self.mu = reference_momentum         # reference direction EMA
        self.tau_min = trust_floor           # minimum trust
        self.delta_decay = straggler_decay   # straggler trust decay

        # State
        self.trust_scores = {i: 1.0 for i in range(num_clients)}
        self.reference_direction: Optional[torch.Tensor] = None

        # History for analysis
        self.trust_history: List[Dict[int, float]] = []

    def aggregate(
        self,
        deltas: List[OrderedDict],
        num_samples: List[int],
        client_ids: List[int],
        all_selected_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[OrderedDict, Dict]:
        """
        TWAC aggregation with trust-weighted averaging.

        Args:
            deltas: List of client updates (from clients that responded)
            num_samples: Samples per responding client
            client_ids: IDs of responding clients
            all_selected_ids: IDs of all selected clients (including stragglers)
        """
        n = len(deltas)

        # Flatten all deltas for analysis
        flat_deltas = [flatten_params(d) for d in deltas]

        # --- Compute trust components ---

        # 1. Compute norms
        norms = torch.tensor([fd.norm().item() for fd in flat_deltas])
        median_norm = torch.median(norms).item()
        median_norm = max(median_norm, 1e-8)  # avoid division by zero

        # 2. Compute raw trust scores for each responding client
        raw_scores = {}
        for idx, cid in enumerate(client_ids):
            fd = flat_deltas[idx]

            # Component 1: Directional consistency
            if self.reference_direction is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    fd.unsqueeze(0),
                    self.reference_direction.unsqueeze(0),
                    eps=1e-8,
                ).item()
            else:
                # First round: no reference yet, everyone gets full trust
                cos_sim = 1.0

            # ReLU: zero out negatively-correlated updates
            direction_score = max(0.0, cos_sim)

            # Component 2: Magnitude reasonableness
            norm_ratio = norms[idx].item() / median_norm
            magnitude_score = np.exp(
                -self.alpha * max(0.0, norm_ratio - self.beta)
            )

            # Combined raw score
            raw_scores[cid] = direction_score * magnitude_score

        # 3. Update trust scores via EMA
        for cid in client_ids:
            old_trust = self.trust_scores.get(cid, 1.0)
            self.trust_scores[cid] = (
                self.gamma * old_trust + (1 - self.gamma) * raw_scores[cid]
            )

        # 4. Decay trust for stragglers (clients selected but didn't respond)
        if all_selected_ids is not None:
            stragglers = set(all_selected_ids) - set(client_ids)
            for cid in stragglers:
                self.trust_scores[cid] = self.delta_decay * self.trust_scores.get(cid, 1.0)

        # --- Aggregate with trust weights ---

        # Apply trust floor
        effective_trust = {
            cid: max(self.trust_scores[cid], self.tau_min)
            for cid in client_ids
        }

        # Compute weights: trust * num_samples
        raw_weights = [
            effective_trust[cid] * ns
            for cid, ns in zip(client_ids, num_samples)
        ]
        total_weight = sum(raw_weights)
        if total_weight < 1e-10:
            # Fallback to uniform if all weights are near zero
            weights = [1.0 / n] * n
        else:
            weights = [w / total_weight for w in raw_weights]

        # Weighted aggregation
        agg_delta = OrderedDict()
        for name in deltas[0]:
            agg_delta[name] = sum(
                w * d[name] for w, d in zip(weights, deltas)
            )

        # --- Update reference direction ---
        # Weighted sum of flat deltas using trust weights only (no sample size)
        trust_only_weights = [
            effective_trust[cid] for cid in client_ids
        ]
        trust_sum = sum(trust_only_weights)
        if trust_sum > 1e-10:
            ref_weights = [tw / trust_sum for tw in trust_only_weights]
        else:
            ref_weights = [1.0 / n] * n

        new_ref = sum(w * fd for w, fd in zip(ref_weights, flat_deltas))

        if self.reference_direction is not None:
            self.reference_direction = (
                self.mu * self.reference_direction + (1 - self.mu) * new_ref
            )
        else:
            self.reference_direction = new_ref

        # Save trust snapshot for analysis
        self.trust_history.append(dict(self.trust_scores))

        info = {
            "trust_scores": dict(self.trust_scores),
            "raw_scores": raw_scores,
            "weights": {cid: w for cid, w in zip(client_ids, weights)},
            "norms": {cid: norms[i].item() for i, cid in enumerate(client_ids)},
            "median_norm": median_norm,
        }
        return agg_delta, info

    def get_trust_history(self) -> List[Dict[int, float]]:
        """Return the full trust score history."""
        return self.trust_history


# ===========================================================================
# Factory
# ===========================================================================

def get_aggregator(
    name: str,
    num_clients: int,
    config: Dict,
) -> object:
    """Create an aggregator by name."""
    if name == "fedavg":
        return FedAvg()
    elif name == "trimmed_mean":
        trim_cfg = config.get("trimmed_mean", {})
        return TrimmedMean(
            trim_ratio=trim_cfg.get("trim_ratio", 0.2),
        )
    elif name == "krum":
        krum_cfg = config.get("krum", {})
        return Krum(
            num_select=krum_cfg.get("num_select", 5),
        )
    elif name == "twac":
        twac_cfg = config.get("twac", {})
        return TWAC(
            num_clients=num_clients,
            trust_momentum=twac_cfg.get("trust_momentum", 0.7),
            magnitude_sensitivity=twac_cfg.get("magnitude_sensitivity", 2.0),
            magnitude_tolerance=twac_cfg.get("magnitude_tolerance", 2.0),
            reference_momentum=twac_cfg.get("reference_momentum", 0.9),
            trust_floor=twac_cfg.get("trust_floor", 0.1),
            straggler_decay=twac_cfg.get("straggler_decay", 0.98),
        )
    else:
        raise ValueError(f"Unknown aggregator: {name}")
