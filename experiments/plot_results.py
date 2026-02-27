"""
Visualization script for federated learning experiments.

Generates all required plots:
1. Accuracy vs rounds (comparison)
2. Loss vs rounds (comparison)
3. Effect of heterogeneity
4. Impact of malicious clients
5. Trust score evolution (TWAC-specific)
6. Data distribution heatmap

Usage:
    python experiments/plot_results.py
    python experiments/plot_results.py --results_dir results
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
})

COLORS = {
    "fedavg": "#1f77b4",
    "trimmed_mean": "#ff7f0e",
    "krum": "#2ca02c",
    "twac": "#d62728",
}
LABELS = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Multi-Krum",
    "twac": "TWAC (Ours)",
}


def load_results(results_dir: str) -> dict:
    """Load all result JSON files from a directory."""
    results = {}
    for filepath in glob.glob(os.path.join(results_dir, "*.json")):
        if "comparison" in filepath or "summary" in filepath:
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        name = data.get("experiment_name", os.path.basename(filepath))
        results[name] = data
    return results


def identify_aggregator(name: str) -> str:
    """Extract aggregator name from experiment name."""
    for agg in ["fedavg", "trimmed_mean", "krum", "twac"]:
        if agg in name.lower():
            return agg
    return "unknown"


def plot_accuracy_comparison(results: dict, save_dir: str):
    """Plot accuracy vs rounds for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in sorted(results.items()):
        agg = identify_aggregator(name)
        metrics = data["metrics"]
        rounds = metrics["rounds"]
        accuracy = metrics["test_accuracy"]
        ax.plot(
            rounds, accuracy,
            color=COLORS.get(agg, "gray"),
            label=LABELS.get(agg, agg),
            linewidth=2.5,
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs. Communication Rounds")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_loss_comparison(results: dict, save_dir: str):
    """Plot loss vs rounds for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in sorted(results.items()):
        agg = identify_aggregator(name)
        metrics = data["metrics"]
        rounds = metrics["rounds"]
        loss = metrics["test_loss"]
        ax.plot(
            rounds, loss,
            color=COLORS.get(agg, "gray"),
            label=LABELS.get(agg, agg),
            linewidth=2.5,
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Loss")
    ax.set_title("Test Loss vs. Communication Rounds")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_trust_evolution(results: dict, save_dir: str):
    """Plot trust score evolution for TWAC experiments."""
    twac_results = {
        k: v for k, v in results.items()
        if "twac" in k.lower() and v.get("trust_histories")
    }

    if not twac_results:
        print("  No TWAC trust data found, skipping trust plot.")
        return

    for name, data in twac_results.items():
        trust_hist = data["trust_histories"]
        if not trust_hist:
            continue

        # Collect all client IDs
        all_client_ids = set()
        for entry in trust_hist:
            all_client_ids.update(entry["trust_scores"].keys())

        # Determine which clients are malicious (from config)
        config = data.get("config", {})
        attack_cfg = config.get("attack", {})
        attack_type = attack_cfg.get("type", "none")
        attack_fraction = attack_cfg.get("fraction", 0.0)

        fig, ax = plt.subplots(figsize=(12, 6))

        rounds = [entry["round"] for entry in trust_hist]

        for cid in sorted(all_client_ids, key=lambda x: int(x)):
            scores = [
                entry["trust_scores"].get(cid, 0.0)
                for entry in trust_hist
            ]
            # Use different style for visualization
            alpha_val = 0.4
            lw = 1.0
            ax.plot(rounds, scores, alpha=alpha_val, linewidth=lw, color="gray")

        # Overlay: compute mean trust for top/bottom
        if len(trust_hist) > 0:
            all_scores = np.array([
                [entry["trust_scores"].get(str(cid), entry["trust_scores"].get(cid, 0.0))
                 for entry in trust_hist]
                for cid in sorted(all_client_ids, key=lambda x: int(x))
            ])
            mean_trust = all_scores.mean(axis=0)
            min_trust = all_scores.min(axis=0)
            max_trust = all_scores.max(axis=0)

            ax.plot(rounds, mean_trust, color=COLORS["twac"],
                    linewidth=2.5, label="Mean trust")
            ax.fill_between(rounds, min_trust, max_trust,
                            alpha=0.15, color=COLORS["twac"])

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Trust Score")
        ax.set_title(f"TWAC Trust Score Evolution\n"
                     f"(Attack: {attack_type}, Fraction: {attack_fraction})")
        ax.legend(loc="lower right")
        ax.set_ylim(-0.05, 1.15)

        plt.tight_layout()
        path = os.path.join(save_dir, f"trust_evolution_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_combined_dashboard(results: dict, save_dir: str):
    """Create a combined dashboard with all key plots."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in sorted(results.items()):
        agg = identify_aggregator(name)
        m = data["metrics"]
        ax1.plot(m["rounds"], m["test_accuracy"],
                 color=COLORS.get(agg, "gray"),
                 label=LABELS.get(agg, agg), linewidth=2)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Test Accuracy")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(bottom=0)

    # 2. Loss comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for name, data in sorted(results.items()):
        agg = identify_aggregator(name)
        m = data["metrics"]
        ax2.plot(m["rounds"], m["test_loss"],
                 color=COLORS.get(agg, "gray"),
                 label=LABELS.get(agg, agg), linewidth=2)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title("Test Loss")
    ax2.legend(fontsize=9, loc="upper right")

    # 3. Final accuracy bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    agg_names = []
    final_accs = []
    colors = []
    for name, data in sorted(results.items()):
        agg = identify_aggregator(name)
        agg_names.append(LABELS.get(agg, agg))
        final_accs.append(data["metrics"]["test_accuracy"][-1])
        colors.append(COLORS.get(agg, "gray"))
    bars = ax3.bar(agg_names, final_accs, color=colors, alpha=0.8, edgecolor="black")
    ax3.set_ylabel("Final Accuracy (%)")
    ax3.set_title("Final Test Accuracy")
    for bar, acc in zip(bars, final_accs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    # 4. Trust evolution (if TWAC present)
    ax4 = fig.add_subplot(gs[1, 1])
    twac_data = None
    for name, data in results.items():
        if "twac" in name.lower() and data.get("trust_histories"):
            twac_data = data
            break

    if twac_data and twac_data["trust_histories"]:
        trust_hist = twac_data["trust_histories"]
        rounds = [e["round"] for e in trust_hist]

        all_cids = set()
        for entry in trust_hist:
            all_cids.update(entry["trust_scores"].keys())

        all_scores = []
        for cid in sorted(all_cids, key=lambda x: int(x)):
            scores = [
                entry["trust_scores"].get(cid, entry["trust_scores"].get(str(cid), 0))
                for entry in trust_hist
            ]
            all_scores.append(scores)
            ax4.plot(rounds, scores, alpha=0.3, linewidth=0.8, color="gray")

        if all_scores:
            arr = np.array(all_scores)
            ax4.plot(rounds, arr.mean(axis=0), color=COLORS["twac"],
                     linewidth=2.5, label="Mean")
        ax4.set_xlabel("Round")
        ax4.set_ylabel("Trust Score")
        ax4.set_title("TWAC Trust Scores")
        ax4.legend(fontsize=9)
        ax4.set_ylim(-0.05, 1.15)
    else:
        ax4.text(0.5, 0.5, "No TWAC trust data\navailable",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=14)
        ax4.set_title("TWAC Trust Scores")

    path = os.path.join(save_dir, "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_robustness_comparison(results_dir: str, save_dir: str):
    """
    If multiple attack experiments exist, plot robustness comparison.
    Groups results by attack fraction and shows accuracy for each method.
    """
    all_results = load_results(results_dir)
    if len(all_results) < 4:
        return

    # Try to group by attack scenario
    scenarios = {}
    for name, data in all_results.items():
        config = data.get("config", {})
        atk = config.get("attack", {}).get("type", "none")
        frac = config.get("attack", {}).get("fraction", 0.0)
        key = f"{atk}_{frac}"
        agg = identify_aggregator(name)
        if key not in scenarios:
            scenarios[key] = {}
        final_acc = data["metrics"]["test_accuracy"][-1]
        scenarios[key][agg] = final_acc

    if len(scenarios) <= 1:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios))
    width = 0.2
    aggregators = ["fedavg", "trimmed_mean", "krum", "twac"]

    for idx, agg in enumerate(aggregators):
        vals = [
            scenarios[key].get(agg, 0) for key in sorted(scenarios.keys())
        ]
        ax.bar(x + idx * width, vals, width,
               label=LABELS.get(agg, agg),
               color=COLORS.get(agg, "gray"),
               alpha=0.8, edgecolor="black")

    ax.set_xlabel("Attack Scenario")
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_title("Robustness Comparison Across Attack Scenarios")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(sorted(scenarios.keys()), rotation=15)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "robustness_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot FL experiment results")
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results/plots",
        help="Directory to save plots"
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading results...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found. Run experiments first:")
        print("  python experiments/run_experiment.py --compare")
        return

    print(f"Found {len(results)} experiment(s):")
    for name in results:
        acc = results[name]["metrics"]["test_accuracy"][-1]
        print(f"  - {name}: {acc:.2f}% final accuracy")

    print("\nGenerating plots...")
    plot_accuracy_comparison(results, args.save_dir)
    plot_loss_comparison(results, args.save_dir)
    plot_trust_evolution(results, args.save_dir)
    plot_combined_dashboard(results, args.save_dir)
    plot_robustness_comparison(args.results_dir, args.save_dir)

    print(f"\nAll plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
