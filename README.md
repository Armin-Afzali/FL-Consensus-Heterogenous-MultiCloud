# Lightweight Consensus for Fast and Accurate Federated Learning

## Trust-Weighted Adaptive Consensus (TWAC)

A federated learning framework that simulates a **multi-cloud environment** where distributed cloud nodes across different regions and providers collaboratively train a shared model. Implements a lightweight consensus mechanism (TWAC) that achieves robustness against compromised nodes and stragglers with O(nd) computational cost — matching FedAvg while significantly outperforming it under adversarial conditions.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick validation (~5 min for all 4 methods)
python experiments/run_experiment.py --compare --fast

# 3. Default comparison (~25 min for all 4 methods)
python experiments/run_experiment.py --compare

# 4. Full publication-quality run (~45+ min for all 4 methods)
python experiments/run_experiment.py --compare --full

# 5. Generate visualizations
python experiments/plot_results.py
```

### Speed tiers

| Flag     | Rounds | Epochs | Batch | ~Time (4 methods) |
|----------|--------|--------|-------|-------------------|
| `--fast` | 30     | 2      | 128   | ~5 min            |
| (default)| 50     | 3      | 64    | ~25 min           |
| `--full` | 100    | 5      | 32    | ~45+ min          |

## Project Structure

```
federated-learning/
├── configs/default.yaml          # Experiment configuration
├── src/
│   ├── models.py                 # CNN model definitions
│   ├── data.py                   # Non-IID data partitioning (Dirichlet) — simulates regional data
│   ├── client.py                 # Cloud node training + attack/fault simulation
│   ├── server.py                 # Aggregation server orchestration
│   ├── aggregators.py            # FedAvg, TrimmedMean, Krum, TWAC
│   ├── network_sim.py            # Multi-cloud network heterogeneity simulation
│   └── utils.py                  # Logging and helpers
├── experiments/
│   ├── run_experiment.py         # Main experiment runner
│   └── plot_results.py           # Visualization generation
├── results/                      # Experiment outputs
│   └── plots/                    # Generated figures
└── docs/
    └── BLUEPRINT.md              # Full technical blueprint
```

## Running Experiments

### Single Method
```bash
python experiments/run_experiment.py --aggregator twac
python experiments/run_experiment.py --aggregator fedavg
python experiments/run_experiment.py --aggregator krum
```

### With Attacks
```bash
# Sign-flip attack on 20% of clients
python experiments/run_experiment.py --aggregator twac --attack sign_flip --attack_fraction 0.2

# Gaussian noise attack on 30% of clients
python experiments/run_experiment.py --aggregator fedavg --attack noise --attack_fraction 0.3
```

### Varying Heterogeneity
```bash
# IID data
python experiments/run_experiment.py --aggregator twac --alpha 100.0

# Extreme non-IID
python experiments/run_experiment.py --aggregator twac --alpha 0.1
```

### With Stragglers
```bash
python experiments/run_experiment.py --aggregator twac --straggler_fraction 0.3
```

### Full Comparison
```bash
python experiments/run_experiment.py --compare --rounds 100
python experiments/run_experiment.py --compare --attack sign_flip --attack_fraction 0.2
python experiments/plot_results.py
```

## Method Overview

The system simulates a multi-cloud federated learning deployment where N cloud nodes (across different providers/regions) each hold private local data and collaboratively train a shared model by exchanging only model updates.

**TWAC** maintains per-node trust scores updated each round based on:

1. **Directional Consistency**: Cosine similarity between a node's update and the exponential moving average of past aggregated updates
2. **Magnitude Reasonableness**: Penalizes updates with norms far from the median (catches scaling attacks and faulty nodes)

Trust scores are updated via EMA and used as aggregation weights. Key advantages:
- **O(nd) complexity** — same as FedAvg, vs O(n²d) for Krum
- **Temporal memory** — transient cloud issues don't destroy trust; persistently compromised nodes get filtered out
- **Graceful straggler handling** — timeout-based (models WAN unreliability between regions) with mild trust decay

See `docs/BLUEPRINT.md` for complete mathematical formulation and algorithm pseudocode.

## Configuration

Edit `configs/default.yaml` to customize:
- Number of clients, rounds, local epochs
- Dirichlet alpha for data heterogeneity
- Attack type and fraction
- Network/straggler simulation
- TWAC hyperparameters (trust momentum, magnitude sensitivity, etc.)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list
- 