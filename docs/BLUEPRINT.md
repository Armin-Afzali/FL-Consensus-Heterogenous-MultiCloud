# Lightweight Consensus for Fast and Accurate Federated Learning Under Computational and Network Heterogeneity

## Complete Technical Blueprint

---

## 1. Executive Summary

This project proposes **Trust-Weighted Adaptive Consensus (TWAC)**, a lightweight consensus mechanism for federated learning that achieves robustness against noisy/malicious clients and stragglers while being computationally cheaper than traditional Byzantine-resilient methods. TWAC maintains per-client trust scores updated each round based on update consistency, magnitude reasonableness, and directional alignment, then uses these scores as aggregation weights. This replaces expensive pairwise distance computations (Krum: O(n²d)) with a linear-time mechanism (O(nd)).

---

## 2. Problem Statement

Standard Federated Averaging (FedAvg) assumes:
- All clients are honest and reliable
- Data is approximately IID across clients
- All clients complete training in similar time

In practice, federated systems face:
1. **Computational heterogeneity**: Clients have varying hardware, causing stragglers
2. **Network heterogeneity**: Variable latency, bandwidth, packet loss
3. **Data heterogeneity**: Non-IID data distributions across clients
4. **Byzantine behavior**: Noisy or malicious model updates

### Limitations of Existing Approaches

| Method | Limitation |
|--------|-----------|
| **FedAvg** | No robustness to any form of heterogeneity or attacks |
| **Krum** | O(n²d) complexity; selects only one update per round, slow convergence |
| **Trimmed Mean** | Requires coordinate-wise operation; weak against sophisticated attacks |
| **Median** | High variance; loses information from honest clients |
| **FedProx** | Addresses data heterogeneity but not Byzantine clients |
| **Bulyan** | Very expensive; requires n ≥ 4f + 3 clients |

---

## 3. Proposed Method: Trust-Weighted Adaptive Consensus (TWAC)

### 3.1 Core Idea

Instead of expensive Byzantine-resilient aggregation each round, TWAC builds **lightweight trust scores** that adapt over time. The intuition: honest clients produce consistent, reasonably-sized updates that align with the global learning direction. Malicious or noisy clients produce outliers that can be detected cheaply.

### 3.2 Mathematical Formulation

#### Notation
- $N$: total number of clients
- $K$: clients selected per round
- $t$: communication round
- $w^t$: global model at round $t$
- $\Delta_i^t = w_i^t - w^t$: client $i$'s update at round $t$
- $\tau_i^t$: trust score for client $i$ at round $t$
- $\bar{\Delta}^t$: reference update direction (exponential moving average)

#### Trust Score Components

**Component 1: Directional Consistency (cosine similarity)**

$$c_i^t = \frac{\Delta_i^t \cdot \bar{\Delta}^t}{\|\Delta_i^t\| \cdot \|\bar{\Delta}^t\| + \epsilon}$$

This measures whether client $i$'s update points in a similar direction to the running average of recent updates. Honest clients on reasonably distributed data will mostly agree on gradient direction.

**Component 2: Magnitude Reasonableness**

$$m_i^t = \exp\left(-\alpha \cdot \max\left(0, \frac{\|\Delta_i^t\|}{\text{median}_j(\|\Delta_j^t\|)} - \beta\right)\right)$$

Where $\alpha$ controls sensitivity and $\beta$ is the tolerance threshold. This penalizes updates with abnormally large norms (common in Byzantine attacks) while being robust via the median normalization.

**Component 3: Loss-based Performance Signal (optional, for labeled validation)**

$$p_i^t = \mathbb{1}[\mathcal{L}(w^t + \Delta_i^t; D_{val}) < \mathcal{L}(w^t; D_{val})]$$

Binary indicator: does applying this client's update improve validation loss?

#### Trust Score Update (Exponential Moving Average)

$$\tau_i^{t} = \gamma \cdot \tau_i^{t-1} + (1 - \gamma) \cdot s_i^t$$

Where the raw score is:

$$s_i^t = \text{ReLU}(c_i^t) \cdot m_i^t$$

And $\gamma \in [0.5, 0.9]$ is the momentum parameter. The ReLU ensures negatively-correlated updates get zero weight.

#### Aggregation

$$w^{t+1} = w^t + \sum_{i \in S_t} \frac{\hat{\tau}_i^t \cdot n_i}{\sum_{j \in S_t} \hat{\tau}_j^t \cdot n_j} \cdot \Delta_i^t$$

Where $\hat{\tau}_i^t = \max(\tau_i^t, \tau_{min})$ provides a floor to prevent complete exclusion (allowing recovery), and $n_i$ is client $i$'s dataset size.

#### Reference Direction Update

$$\bar{\Delta}^{t} = \mu \cdot \bar{\Delta}^{t-1} + (1-\mu) \cdot \sum_{i \in S_t} \frac{\hat{\tau}_i^t}{\sum_j \hat{\tau}_j^t} \cdot \Delta_i^t$$

### 3.3 Straggler Handling

- Set a **timeout** $T_{max}$ per round (simulated)
- Clients exceeding timeout are excluded from that round
- Their trust scores receive a mild decay: $\tau_i^t = \delta \cdot \tau_i^{t-1}$ where $\delta \in [0.95, 0.99]$
- Aggregation proceeds with available clients (minimum threshold $K_{min}$)

### 3.4 Complexity Analysis

| Method | Time Complexity | Space Complexity |
|--------|----------------|-----------------|
| FedAvg | O(nd) | O(d) |
| Krum | O(n²d) | O(n²) |
| Trimmed Mean | O(nd log n) | O(nd) |
| **TWAC (ours)** | **O(nd)** | **O(d + n)** |

Where $n$ = number of clients, $d$ = model dimension. TWAC adds only a constant factor over FedAvg.

### 3.5 Key Innovation

TWAC's novelty lies in the **temporal trust accumulation**:
- Unlike Krum/Trimmed Mean that treat each round independently, TWAC uses history
- A one-time malicious update from an otherwise honest client won't destroy its trust
- A persistent attacker will see trust decay to near-zero within a few rounds
- The combination of directional + magnitude signals catches both scaling attacks and direction-flip attacks
- Computational cost is essentially the same as FedAvg (one extra dot product per client)

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FL Server                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  Global   │  │    TWAC      │  │   Trust Score     │ │
│  │  Model    │  │  Aggregator  │  │   Manager         │ │
│  └──────────┘  └──────────────┘  └───────────────────┘ │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Straggler│  │  Round       │  │   Experiment      │ │
│  │ Handler  │  │  Coordinator │  │   Logger          │ │
│  └──────────┘  └──────────────┘  └───────────────────┘ │
└─────────────┬───────────────────────────┬───────────────┘
              │    Communication Layer     │
              │   (simulated latency +     │
              │    bandwidth + drops)      │
    ┌─────────┴──────┬──────────┬─────────┴────────┐
    ▼                ▼          ▼                   ▼
┌────────┐   ┌────────┐  ┌────────┐         ┌────────┐
│Client 1│   │Client 2│  │Client 3│  . . .  │Client N│
│(fast)  │   │(slow)  │  │(noisy) │         │(normal)│
│IID data│   │non-IID │  │corrupt │         │non-IID │
└────────┘   └────────┘  └────────┘         └────────┘
```

### 4.1 Component Descriptions

**Server Components:**
- **Global Model**: Maintains current model parameters
- **TWAC Aggregator**: Implements trust-weighted aggregation
- **Trust Score Manager**: Tracks and updates per-client trust scores
- **Straggler Handler**: Manages timeouts and partial aggregation
- **Round Coordinator**: Orchestrates client selection and round flow
- **Experiment Logger**: Records all metrics for analysis

**Client Components:**
- **Local Trainer**: Performs local SGD steps
- **Data Loader**: Manages local non-IID data partition
- **Network Simulator**: Adds simulated latency/drops
- **Behavior Module**: Configurable honest/noisy/malicious behavior

### 4.2 Communication Protocol (Simulated)

```
Round t:
  1. Server selects K clients (random or stratified)
  2. Server broadcasts global model w^t to selected clients
  3. Each client i:
     a. Receives w^t (with simulated network delay)
     b. Trains locally for E epochs on local data
     c. Computes update Δ_i^t = w_i^t - w^t
     d. Sends Δ_i^t back (with simulated delay + possible drop)
  4. Server collects updates (with timeout T_max)
  5. Server computes trust scores and TWAC aggregation
  6. Server updates global model: w^{t+1}
```

---

## 5. Algorithm Pseudocode

### Algorithm 1: TWAC-FL Server

```
Initialize: global model w⁰, trust scores τᵢ⁰ = 1.0 for all i, reference Δ̄⁰ = 0
For round t = 1, 2, ..., T:
    S_t ← select K clients (uniformly at random)
    Broadcast w^t to all clients in S_t
    
    # Collect updates with timeout
    received ← {}
    For each client i in S_t (with timeout T_max):
        Δ_i^t ← receive_update(i)  # includes simulated network delay
        if received within timeout:
            received ← received ∪ {i}
    
    if |received| < K_min:
        skip round, continue
    
    # Compute trust components
    norms ← {‖Δ_i^t‖ : i ∈ received}
    med_norm ← median(norms)
    
    For each i in received:
        c_i ← cosine_similarity(Δ_i^t, Δ̄^{t-1})
        m_i ← exp(-α · max(0, ‖Δ_i^t‖/med_norm - β))
        s_i ← ReLU(c_i) · m_i
        τ_i^t ← γ · τ_i^{t-1} + (1-γ) · s_i
    
    # Decay trust for stragglers
    For each i in (S_t \ received):
        τ_i^t ← δ · τ_i^{t-1}
    
    # Aggregate
    τ̂_i ← max(τ_i^t, τ_min) for all i in received
    weights ← normalize(τ̂_i · n_i for i in received)
    w^{t+1} ← w^t + Σ weights_i · Δ_i^t
    
    # Update reference direction
    Δ̄^t ← μ · Δ̄^{t-1} + (1-μ) · Σ (τ̂_i/Στ̂) · Δ_i^t
    
    # Log metrics
    evaluate(w^{t+1}) on test set
```

### Algorithm 2: Client Local Training

```
Input: global model w^t, local data D_i, epochs E, learning rate η
Output: update Δ_i^t

w_local ← w^t
For epoch = 1, ..., E:
    For batch (x, y) in D_i:
        w_local ← w_local - η · ∇L(w_local; x, y)

Δ_i^t ← w_local - w^t

# If malicious client:
if attack_type == "noise":
    Δ_i^t ← Δ_i^t + N(0, σ²)
elif attack_type == "sign_flip":
    Δ_i^t ← -λ · Δ_i^t
elif attack_type == "scaling":
    Δ_i^t ← κ · Δ_i^t

return Δ_i^t
```

---

## 6. Data Heterogeneity Modeling

### Non-IID Distribution via Dirichlet Allocation

We use a Dirichlet distribution to control the degree of non-IID-ness:

$$p_i \sim \text{Dir}(\alpha_{dir} \cdot \mathbf{1})$$

Where $\alpha_{dir}$ controls heterogeneity:
- $\alpha_{dir} \to \infty$: IID (uniform)
- $\alpha_{dir} = 1.0$: moderate heterogeneity
- $\alpha_{dir} = 0.1$: extreme heterogeneity (each client gets mostly 1-2 classes)

Each client receives data sampled according to their class probability vector $p_i$.

---

## 7. Experimental Design

### 7.1 Datasets
- **CIFAR-10**: 10 classes, 50K train / 10K test, 32×32 RGB images
- (Optional) **MNIST/Fashion-MNIST** for quick validation

### 7.2 Model
- **Simple CNN** (for reproducibility and speed):
  - Conv2d(3, 32, 3) → ReLU → Conv2d(32, 64, 3) → ReLU → MaxPool
  - Conv2d(64, 64, 3) → ReLU → MaxPool
  - FC(1024, 256) → ReLU → FC(256, 10)
  - ~300K parameters

### 7.3 Experimental Configurations

| Parameter | Value |
|-----------|-------|
| Number of clients (N) | 20 |
| Clients per round (K) | 10 |
| Local epochs (E) | 5 |
| Local learning rate (η) | 0.01 |
| Batch size | 32 |
| Communication rounds (T) | 100 |
| Non-IID α_dir | {0.1, 0.5, 1.0, 100.0} |
| Malicious fraction | {0%, 10%, 20%, 30%} |
| Straggler fraction | {0%, 20%, 40%} |

### 7.4 Attack Scenarios
1. **No attack** (baseline)
2. **Gaussian noise**: $\Delta + \mathcal{N}(0, \sigma^2 I)$, $\sigma = 0.5$
3. **Sign-flip**: $-\lambda \Delta$, $\lambda = 1.0$
4. **Scaling attack**: $\kappa \Delta$, $\kappa = 5.0$

### 7.5 Methods Compared
1. **FedAvg** — standard federated averaging (no defense)
2. **Trimmed Mean** — coordinate-wise trimmed mean (trim 20%)
3. **Krum** — multi-Krum selecting top-K/2 updates
4. **TWAC (ours)** — trust-weighted adaptive consensus

### 7.6 Evaluation Metrics
- **Test Accuracy** vs communication rounds
- **Test Loss** vs communication rounds
- **Final accuracy** at round T
- **Rounds to target accuracy** (e.g., 60%)
- **Trust score evolution** over rounds (TWAC-specific)
- **Robustness**: accuracy degradation under increasing attack fraction
- **Straggler impact**: accuracy under increasing straggler fraction

---

## 8. Folder Structure

```
federated-learning/
├── README.md                    # Project overview and usage instructions
├── requirements.txt             # Python dependencies
├── configs/
│   └── default.yaml             # Experiment configuration
├── src/
│   ├── __init__.py
│   ├── models.py                # Neural network definitions
│   ├── data.py                  # Data loading and non-IID partitioning
│   ├── client.py                # Client logic (local training + attacks)
│   ├── server.py                # Server logic (aggregation methods)
│   ├── aggregators.py           # FedAvg, TrimmedMean, Krum, TWAC
│   ├── network_sim.py           # Network heterogeneity simulation
│   └── utils.py                 # Logging, metrics, helpers
├── experiments/
│   ├── run_experiment.py        # Main experiment runner
│   └── plot_results.py          # Visualization script
├── results/
│   └── plots/                   # Generated figures
└── docs/
    └── BLUEPRINT.md             # This document
```

---

## 9. Implementation Plan

### Phase 1: Core Infrastructure (Days 1-3)
- [ ] Set up project structure and dependencies
- [ ] Implement CNN model
- [ ] Implement Dirichlet-based non-IID data partitioning
- [ ] Implement basic client training loop

### Phase 2: Aggregation Methods (Days 4-6)
- [ ] Implement FedAvg aggregator
- [ ] Implement Trimmed Mean aggregator
- [ ] Implement Krum aggregator
- [ ] Implement TWAC aggregator with trust score tracking

### Phase 3: Heterogeneity Simulation (Days 7-8)
- [ ] Implement computational heterogeneity (variable training time)
- [ ] Implement network heterogeneity (latency, bandwidth, drops)
- [ ] Implement straggler handling with timeouts
- [ ] Implement attack behaviors (noise, sign-flip, scaling)

### Phase 4: Experiments (Days 9-11)
- [ ] Run all baseline experiments
- [ ] Run robustness experiments (varying attack fraction)
- [ ] Run heterogeneity experiments (varying non-IID degree)
- [ ] Run straggler experiments

### Phase 5: Analysis & Documentation (Days 12-14)
- [ ] Generate all plots and visualizations
- [ ] Write results analysis
- [ ] Complete documentation
- [ ] Final testing and reproducibility check

---

## 10. Hyperparameter Guide for TWAC

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Trust momentum | γ | 0.7 | [0.5, 0.9] | Higher = more memory, slower adaptation |
| Magnitude sensitivity | α | 2.0 | [1.0, 5.0] | Higher = stricter norm filtering |
| Magnitude tolerance | β | 2.0 | [1.5, 3.0] | Multiples of median norm before penalty |
| Reference momentum | μ | 0.9 | [0.8, 0.95] | Smoothing of reference direction |
| Trust floor | τ_min | 0.1 | [0.01, 0.2] | Prevents permanent exclusion |
| Straggler decay | δ | 0.98 | [0.95, 0.99] | Trust decay for absent clients |

---

## 11. Discussion of Novelty vs. Related Work

### vs. Fang et al. (Robust FL with Noisy/Heterogeneous Clients)
- Fang et al. use norm-based clipping + noise-aware aggregation
- **Our improvement**: We add directional consistency checking and temporal trust, which catches sign-flip attacks that norm clipping alone misses

### vs. Rethinking Data Heterogeneity
- This work analyzes why non-IID hurts and proposes data-sharing solutions
- **Our approach**: Rather than requiring data sharing (privacy concern), TWAC implicitly handles non-IID by down-weighting outlier directions that may be due to extreme heterogeneity

### vs. Architecture-based approaches
- These modify the model architecture to handle heterogeneity
- **Our approach**: Architecture-agnostic; works with any model; focuses on the aggregation step

### vs. Robust FL in Heterogeneous Environment
- Proposes robust aggregation rules assuming known attack model
- **Our approach**: Attack-model-agnostic; the trust mechanism naturally adapts to whatever attack pattern emerges

---

## 12. Expected Results (Hypotheses)

1. **Under no attack**: TWAC ≈ FedAvg (minimal overhead, similar convergence)
2. **Under attacks**: TWAC > FedAvg, competitive with Krum/TrimmedMean
3. **Under stragglers**: TWAC > all baselines (graceful degradation via timeout + trust)
4. **Under non-IID**: TWAC handles direction divergence better than naive FedAvg
5. **Computation**: TWAC runs ~1.1x FedAvg wall-clock time vs ~2-3x for Krum

---

*End of Blueprint*
