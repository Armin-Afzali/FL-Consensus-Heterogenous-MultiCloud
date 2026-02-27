"""
Utility functions for logging, metrics, and configuration.
"""

import os
import json
import yaml
import torch
import numpy as np
import random
from typing import Dict, Any
from datetime import datetime


def set_seed(seed: int, fast_mode: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast_mode:
        # Faster but slightly non-deterministic on GPU
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ExperimentLogger:
    """Tracks and saves experiment metrics."""

    def __init__(self, experiment_name: str, save_dir: str = "results"):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Metric histories
        self.rounds: list = []
        self.test_accuracy: list = []
        self.test_loss: list = []
        self.train_loss: list = []
        self.round_info: list = []
        self.trust_histories: list = []

        self.start_time = datetime.now()

    def log_round(
        self,
        round_num: int,
        test_loss: float,
        test_acc: float,
        train_loss: float,
        agg_info: Dict = None,
        round_metrics: Dict = None,
    ):
        """Log metrics for a single round."""
        self.rounds.append(round_num)
        self.test_accuracy.append(test_acc)
        self.test_loss.append(test_loss)
        self.train_loss.append(train_loss)

        info = {
            "round": round_num,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_loss": train_loss,
        }

        if agg_info:
            # Save trust scores if available
            if "trust_scores" in agg_info:
                self.trust_histories.append({
                    "round": round_num,
                    "trust_scores": agg_info["trust_scores"],
                })
            # Save weights
            if "weights" in agg_info:
                info["weights"] = {
                    str(k): v for k, v in agg_info["weights"].items()
                }

        if round_metrics:
            info["num_responding"] = round_metrics.get("num_responding", 0)
            info["network"] = round_metrics.get("network", {})

        self.round_info.append(info)

    def save(self, config: Dict = None):
        """Save all metrics to JSON."""
        data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "config": config,
            "metrics": {
                "rounds": self.rounds,
                "test_accuracy": self.test_accuracy,
                "test_loss": self.test_loss,
                "train_loss": self.train_loss,
            },
            "trust_histories": self.trust_histories,
            "round_details": self.round_info,
        }

        filepath = os.path.join(self.save_dir, f"{self.experiment_name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"  Results saved to {filepath}")
        return filepath

    def get_summary(self) -> Dict:
        """Get experiment summary."""
        return {
            "final_accuracy": self.test_accuracy[-1] if self.test_accuracy else 0,
            "best_accuracy": max(self.test_accuracy) if self.test_accuracy else 0,
            "final_loss": self.test_loss[-1] if self.test_loss else float("inf"),
            "rounds_completed": len(self.rounds),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


def print_round_summary(
    round_num: int,
    total_rounds: int,
    test_loss: float,
    test_acc: float,
    train_loss: float,
    extra: str = "",
):
    """Print a formatted round summary."""
    print(
        f"  Round {round_num:3d}/{total_rounds} | "
        f"Test Acc: {test_acc:5.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Train Loss: {train_loss:.4f}"
        f"{' | ' + extra if extra else ''}"
    )
    