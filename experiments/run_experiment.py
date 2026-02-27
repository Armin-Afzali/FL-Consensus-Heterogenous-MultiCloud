"""
Main experiment runner for federated learning experiments.

Usage:
    # Run with default config:
    python experiments/run_experiment.py

    # Run with custom config:
    python experiments/run_experiment.py --config configs/default.yaml

    # Run a specific aggregator:
    python experiments/run_experiment.py --aggregator twac

    # Run comparison of all methods:
    python experiments/run_experiment.py --compare

    # Run with attack:
    python experiments/run_experiment.py --aggregator twac --attack sign_flip --attack_fraction 0.2
"""

import os
import sys
import argparse
import copy
import json
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, load_config, get_device, ExperimentLogger, print_round_summary
from src.models import get_model, count_parameters
from src.data import (
    get_dataset, dirichlet_partition, get_client_dataloaders,
    get_test_dataloader, compute_class_distribution,
)
from src.client import create_clients
from src.server import FLServer


def run_single_experiment(config: dict, experiment_name: str = None) -> dict:
    """
    Run a single federated learning experiment.

    Args:
        config: Experiment configuration dictionary
        experiment_name: Optional name for the experiment

    Returns:
        Dictionary with results summary
    """
    seed = config.get("seed", 42)
    fast_mode = config.get("fast_mode", False)
    set_seed(seed, fast_mode=fast_mode)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name or config['aggregator']}")
    print(f"Device: {device}" + (" (fast mode)" if fast_mode else ""))
    print(f"{'='*60}")

    # ---- Data ----
    print("\n[1/4] Loading data...")
    dataset_name = config.get("dataset", "cifar10")
    train_dataset, test_dataset = get_dataset(dataset_name)

    # Non-IID partition
    alpha = config.get("dirichlet_alpha", 0.5)
    num_clients = config["num_clients"]
    client_indices = dirichlet_partition(
        train_dataset, num_clients, alpha, seed=seed
    )
    print(f"  Dataset: {dataset_name}, Dirichlet α={alpha}")
    print(f"  Clients: {num_clients}, samples per client: "
          f"min={min(len(x) for x in client_indices)}, "
          f"max={max(len(x) for x in client_indices)}, "
          f"mean={sum(len(x) for x in client_indices) / num_clients:.0f}")

    # Create data loaders
    # IMPORTANT: Client loaders use num_workers=0 because clients train
    # sequentially — spawning workers per client wastes memory and CPU.
    # Only the test loader benefits from workers (single loader, reused).
    batch_size = config.get("batch_size", 32)
    use_cuda = device.type == "cuda"
    client_dataloaders = get_client_dataloaders(
        train_dataset, client_indices, batch_size,
        num_workers=0, pin_memory=use_cuda,
    )
    test_loader = get_test_dataloader(
        test_dataset, batch_size=256 if use_cuda else 128,
        num_workers=2, pin_memory=use_cuda,
    )

    # ---- Model ----
    print("\n[2/4] Creating model...")
    model = get_model(config.get("model", "simplecnn"), dataset_name)
    print(f"  Model: {model.__class__.__name__} "
          f"({count_parameters(model):,} parameters)")

    # torch.compile for PyTorch 2.0+ gives ~10-30% speedup on GPU
    if fast_mode and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception:
            print("  torch.compile: not available, skipping")

    # ---- Clients ----
    print("\n[3/4] Setting up clients...")
    clients = create_clients(
        dataloaders=client_dataloaders,
        device=device,
        local_epochs=config.get("local_epochs", 5),
        local_lr=config.get("local_lr", 0.01),
        attack_config=config.get("attack", {}),
        compute_config=config.get("compute", {}),
        seed=seed,
    )

    # ---- Server ----
    print("\n[4/4] Setting up server...")
    server = FLServer(
        global_model=model,
        aggregator_name=config["aggregator"],
        num_clients=num_clients,
        clients_per_round=config.get("clients_per_round", 10),
        config=config,
        device=device,
        seed=seed,
    )

    # ---- Logger ----
    if experiment_name is None:
        agg = config["aggregator"]
        atk = config.get("attack", {}).get("type", "none")
        atk_frac = config.get("attack", {}).get("fraction", 0)
        experiment_name = f"{agg}_a{alpha}_atk{atk}_{atk_frac}"

    logger = ExperimentLogger(experiment_name, save_dir="results")

    # ---- Training loop ----
    num_rounds = config.get("num_rounds", 100)
    log_interval = config.get("log_interval", 5)
    verbose = config.get("verbose", True)

    # Pre-allocate local models on each client (avoids deepcopy every round)
    print("  Pre-allocating client models (one-time cost)...")
    for c in clients:
        c.set_local_model(server.global_model)

    print(f"\n{'─'*60}")
    print(f"Starting training for {num_rounds} rounds...")
    print(f"{'─'*60}")

    # Initial evaluation
    test_loss, test_acc = server.evaluate(test_loader)
    print(f"  Initial | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    skipped_rounds = 0

    for round_num in range(1, num_rounds + 1):
        # Select clients
        selected_ids = server.select_clients()

        # Run round
        agg_info, round_metrics = server.run_round(clients, selected_ids)

        if round_metrics.get("skipped", False):
            skipped_rounds += 1
            if verbose:
                print(f"  Round {round_num:3d} SKIPPED: {round_metrics['reason']}")
            continue

        # Evaluate
        train_loss = round_metrics.get("avg_client_loss", 0)

        if round_num % log_interval == 0 or round_num == num_rounds:
            test_loss, test_acc = server.evaluate(test_loader)

            # Log
            logger.log_round(
                round_num=round_num,
                test_loss=test_loss,
                test_acc=test_acc,
                train_loss=train_loss,
                agg_info=agg_info,
                round_metrics=round_metrics,
            )

            if verbose:
                extra = ""
                if agg_info and "trust_scores" in agg_info:
                    trust_vals = list(agg_info["trust_scores"].values())
                    extra = (f"Trust: min={min(trust_vals):.2f}, "
                             f"max={max(trust_vals):.2f}")
                print_round_summary(
                    round_num, num_rounds,
                    test_loss, test_acc, train_loss, extra
                )
        # Only evaluate at log intervals — evaluation is expensive!
        # (Skip logging on non-interval rounds to save time)

    # ---- Save results ----
    print(f"\n{'─'*60}")
    summary = logger.get_summary()
    print(f"Experiment complete:")
    print(f"  Final Accuracy:  {summary['final_accuracy']:.2f}%")
    print(f"  Best Accuracy:   {summary['best_accuracy']:.2f}%")
    print(f"  Final Loss:      {summary['final_loss']:.4f}")
    print(f"  Rounds skipped:  {skipped_rounds}")
    print(f"  Duration:        {summary['duration_seconds']:.1f}s")

    results_path = logger.save(config=config)

    return {
        "experiment_name": experiment_name,
        "summary": summary,
        "results_path": results_path,
    }


def run_comparison(base_config: dict):
    """
    Run comparison experiments across all aggregation methods.
    """
    aggregators = ["fedavg", "trimmed_mean", "krum", "twac"]
    all_results = {}

    for agg in aggregators:
        config = copy.deepcopy(base_config)
        config["aggregator"] = agg

        atk = config.get("attack", {}).get("type", "none")
        atk_frac = config.get("attack", {}).get("fraction", 0)
        alpha = config.get("dirichlet_alpha", 0.5)
        name = f"{agg}_a{alpha}_atk{atk}_{atk_frac}"

        result = run_single_experiment(config, experiment_name=name)
        all_results[agg] = result

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Final Acc':>10} {'Best Acc':>10} "
          f"{'Final Loss':>10} {'Duration':>10}")
    print(f"{'─'*55}")
    for agg in aggregators:
        s = all_results[agg]["summary"]
        print(f"{agg:<15} {s['final_accuracy']:>9.2f}% "
              f"{s['best_accuracy']:>9.2f}% "
              f"{s['final_loss']:>10.4f} "
              f"{s['duration_seconds']:>9.1f}s")

    # Save comparison summary
    comparison_path = os.path.join("results", "comparison_summary.json")
    comp_data = {
        agg: {
            "final_accuracy": all_results[agg]["summary"]["final_accuracy"],
            "best_accuracy": all_results[agg]["summary"]["best_accuracy"],
            "final_loss": all_results[agg]["summary"]["final_loss"],
            "results_file": all_results[agg]["results_path"],
        }
        for agg in aggregators
    }
    with open(comparison_path, "w") as f:
        json.dump(comp_data, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Experiment Runner"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--aggregator", type=str, default=None,
        choices=["fedavg", "trimmed_mean", "krum", "twac"],
        help="Override aggregation method"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run comparison across all aggregation methods"
    )
    parser.add_argument(
        "--attack", type=str, default=None,
        choices=["none", "noise", "sign_flip", "scaling"],
        help="Override attack type"
    )
    parser.add_argument(
        "--attack_fraction", type=float, default=None,
        help="Override fraction of malicious clients"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override Dirichlet alpha (non-IID degree)"
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Override number of rounds"
    )
    parser.add_argument(
        "--straggler_fraction", type=float, default=None,
        help="Override straggler fraction"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick validation run (~5 min total for --compare): "
             "30 rounds, 2 local epochs, batch_size=128"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Thorough run for publication-quality results (~45 min for --compare): "
             "100 rounds, 5 local epochs, batch_size=32"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command-line overrides
    if args.aggregator:
        config["aggregator"] = args.aggregator
    if args.attack:
        config.setdefault("attack", {})["type"] = args.attack
    if args.attack_fraction is not None:
        config.setdefault("attack", {})["fraction"] = args.attack_fraction
    if args.alpha is not None:
        config["dirichlet_alpha"] = args.alpha
    if args.rounds is not None:
        config["num_rounds"] = args.rounds
    if args.straggler_fraction is not None:
        config.setdefault("network", {})["straggler_fraction"] = args.straggler_fraction
    if args.fast:
        config["fast_mode"] = True
        config["batch_size"] = max(config.get("batch_size", 32), 128)
        config["log_interval"] = max(config.get("log_interval", 5), 10)
        if args.rounds is None:
            config["num_rounds"] = min(config.get("num_rounds", 50), 30)
        config["local_epochs"] = min(config.get("local_epochs", 3), 2)
        print(f"⚡ Fast mode: batch_size={config['batch_size']}, "
              f"local_epochs={config['local_epochs']}, "
              f"rounds={config['num_rounds']}, log_interval={config['log_interval']}")
    elif args.full:
        config["num_rounds"] = 100
        config["local_epochs"] = 5
        config["batch_size"] = 32
        config["log_interval"] = 5
        print(f"🔬 Full mode: batch_size=32, local_epochs=5, "
              f"rounds=100, log_interval=5")

    # Run
    if args.compare:
        run_comparison(config)
    else:
        run_single_experiment(config)


if __name__ == "__main__":
    main()
    