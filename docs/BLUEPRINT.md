# Lightweight Consensus for Fast and Accurate Federated Learning Under Computational and Network Heterogeneity

## Complete Technical Blueprint

---

## 1. Executive Summary

This project proposes **Trust-Weighted Adaptive Consensus (TWAC)**, a lightweight consensus mechanism for federated learning in a **simulated multi-cloud environment**. The system models a realistic distributed setting in which cloud nodes across different regions and providers collaboratively train a shared model without exchanging raw data. Each node faces distinct conditions: heterogeneous compute capabilities (varying instance types), unreliable inter-region networking (variable latency, bandwidth, and packet loss), and non-uniform local data distributions. Some nodes may also exhibit faulty or adversarial behavior — submitting corrupted or malicious model updates.

TWAC achieves robustness against these challenges while being computationally cheaper than traditional Byzantine-resilient methods. It maintains per-node trust scores updated each round based on update consistency, magnitude reasonableness, and directional alignment, then uses these scores as aggregation weights. This replaces expensive pairwise distance computations (Krum: O(n²d)) with a linear-time mechanism (O(nd)).

---

## 2. Problem Statement

### 2.1 Multi-Cloud Federated Learning Context

In a multi-cloud federated learning deployment, participating cloud nodes — hosted across different providers (AWS, GCP, Azure) or different regions within a provider — collaboratively train a shared model. Each node retains its data locally (due to privacy, regulatory, or bandwidth constraints) and only shares model updates with a central aggregation server. This architecture is relevant for scenarios such as:

- **Cross-organizational ML**: Hospitals, banks, or enterprises in different cloud regions jointly training models without exposing private data
- **Geo-distributed training**: Regional data centers that each hold location-specific data (e.g., regional traffic patterns, local sensor data)
- **Edge-cloud hybrid**: A mix of powerful cloud instances and lightweight edge nodes contributing to a shared model

### 2.2 Challenges

Standard Federated Averaging (FedAvg) assumes:

- All nodes are honest and reliable
- Data is approximately IID across nodes
- All nodes complete training in similar time

In a real multi-cloud deployment, the system faces:

1. **Computational heterogeneity**: Nodes run on different instance types (e.g., a GPU-equipped VM vs. a small CPU instance), causing some to be significantly slower (stragglers)
2. **Network heterogeneity**: Inter-region and cross-provider links exhibit variable latency, bandwidth, and packet loss. Cross-continent communication is inherently slower and less reliable than intra-region
3. **Data heterogeneity**: Each region or organization collects data with a different distribution (non-IID) — e.g., a hospital in one region sees different patient demographics than another
4. **Byzantine behavior**: Compromised nodes, software bugs, or misconfigured training pipelines can produce noisy or adversarial model updates

### 2.3 Limitations of Existing Approaches

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

Instead of expensive Byzantine-resilient aggregation each round, TWAC builds **lightweight trust scores** that adapt over time. The intuition: honest cloud nodes produce consistent, reasonably-sized updates that align with the global learning direction, regardless of their region or compute capability. Compromised or faulty nodes produce outliers that can be detected cheaply.

### 3.2 Mathematical Formulation

#### Notation

- $N$: total number of cloud nodes
- $K$: nodes selected per round
- $t$: communication round
- $w^t$: global model at round $t$
- $\Delta_i^t = w_i^t - w^t$: node $i$'s update at round $t$
- $\tau_i^t$: trust score for node $i$ at round $t$
- $\bar{\Delta}^t$: reference update direction (exponential moving average)

#### Trust Score Components

**Component 1: Directional Consistency (cosine similarity)**

$$c_i^t = \frac{\Delta_i^t \cdot \bar{\Delta}^t}{\|\Delta_i^t\| \cdot \|\bar{\Delta}^t\| + \epsilon}$$

This measures whether node $i$'s update points in a similar direction to the running average of recent updates. Honest nodes on reasonably distributed data will mostly agree on gradient direction.

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

In a multi-cloud setting, stragglers arise naturally from slower instance types, cross-continent network delays, or temporary cloud provider throttling.

- Set a **timeout** $T_{max}$ per round (simulated based on median expected response time)
- Nodes exceeding timeout are excluded from that round (models WAN timeout behavior)
- Their trust scores receive a mild decay: $\tau_i^t = \delta \cdot \tau_i^{t-1}$ where $\delta \in [0.95, 0.99]$
- Aggregation proceeds with available nodes (minimum threshold $K_{min}$)
- Persistently slow but honest nodes will recover trust once they respond

### 3.4 Complexity Analysis

| Method | Time Complexity | Space Complexity |
|--------|----------------|-----------------|
| FedAvg | O(nd) | O(d) |
| Krum | O(n²d) | O(n²) |
| Trimmed Mean | O(nd log n) | O(nd) |
| **TWAC (ours)** | **O(nd)** | **O(d + n)** |

Where $n$ = number of nodes, $d$ = model dimension. TWAC adds only a constant factor over FedAvg — critical for cost-sensitive multi-cloud deployments where aggregation server compute is billed per-use.

### 3.5 Key Innovation

TWAC's novelty lies in the **temporal trust accumulation**:

- Unlike Krum/Trimmed Mean that treat each round independently, TWAC uses history
- A one-time faulty update from an otherwise reliable node won't destroy its trust (important for transient cloud issues)
- A persistently compromised node will see trust decay to near-zero within a few rounds
- The combination of directional + magnitude signals catches both scaling attacks and direction-flip attacks
- Computational cost is essentially the same as FedAvg (one extra dot product per node)

---

## 4. System Architecture

The system simulates a multi-cloud federated learning deployment on a single machine. Each cloud node is modeled as an independent client with its own data, compute profile, and network characteristics.

```
                        ┌──────────────────────┐
                        │   Cloud Region A     │
                        │   (Aggregation Hub)  │
┌───────────────────────┤   FL Server          ├───────────────────────┐
│                       │                      │                       │
│  ┌──────────┐  ┌──────┴──────┐  ┌───────────┴───────┐              │
│  │  Global   │  │    TWAC      │  │   Trust Score     │              │
│  │  Model    │  │  Aggregator  │  │   Manager         │              │
│  └──────────┘  └──────────────┘  └───────────────────┘              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐              │
│  │ Straggler│  │  Round       │  │   Experiment      │              │
│  │ Handler  │  │  Coordinator │  │   Logger          │              │
│  └──────────┘  └──────────────┘  └───────────────────┘              │
└─────────────┬───────────────────────────┬───────────────────────────┘
              │  Simulated Network Layer   │
              │  (inter-region latency,    │
              │   cross-provider drops,    │
              │   bandwidth limits)        │
    ┌─────────┴──────┬──────────┬─────────┴────────┐
    ▼                ▼          ▼                   ▼
┌────────┐   ┌────────┐  ┌────────┐         ┌────────┐
│Node 1  │   │Node 2  │  │Node 3  │  . . .  │Node N  │
│Region B│   │Region C│  │Region D│         │Region E│
│GPU inst.│  │CPU inst.│ │Comprom.│         │GPU inst│
│IID data│   │non-IID │  │corrupt │         │non-IID │
└────────┘   └────────┘  └────────┘         └────────┘
  AWS           GCP        Azure              AWS
  eu-west     us-east    ap-south          us-west
```

### 4.1 Component Descriptions

**Aggregation Server (Central Hub):**

- **Global Model**: Maintains current model parameters, hosted in a designated region
- **TWAC Aggregator**: Implements trust-weighted aggregation of node updates
- **Trust Score Manager**: Tracks and updates per-node trust scores across rounds
- **Straggler Handler**: Manages timeouts for slow or unreachable nodes, enables partial aggregation
- **Round Coordinator**: Orchestrates node selection and round flow
- **Experiment Logger**: Records all metrics for analysis

**Cloud Nodes (Distributed Participants):**

- **Local Trainer**: Performs local SGD steps on the node's private data
- **Data Loader**: Manages the node's local non-IID data partition
- **Network Profile**: Simulated inter-region latency, bandwidth, and drop characteristics
- **Behavior Module**: Configurable honest/noisy/malicious behavior (models compromised nodes)

### 4.2 Simulated Multi-Cloud Communication Protocol

```
Round t:
  1. Server selects K nodes for participation (random or stratified)
  2. Server broadcasts global model w^t to selected nodes
     (simulated inter-region transfer delay per node)
  3. Each node i:
     a. Receives w^t (with simulated network latency based on region)
     b. Trains locally for E epochs on its private regional data
        (training time varies by instance type / compute speed)
     c. Computes update Δ_i^t = w_i^t - w^t
     d. Sends Δ_i^t back to server (with simulated delay + possible drop)
  4. Server collects updates (with timeout T_max — models WAN unreliability)
  5. Server computes trust scores and TWAC aggregation
  6. Server updates global model: w^{t+1}
```

### 4.3 Simulation Fidelity

The simulation captures the essential characteristics of a real multi-cloud deployment without requiring actual distributed infrastructure:

| Real-World Factor | Simulation Mechanism |
|-------------------|---------------------|
| Different cloud providers/regions | Per-node network latency and drop profiles |
| Different instance types (GPU vs CPU) | Per-node `compute_speed` parameter (0.3 = weak CPU, 1.0 = fast GPU) |
| Inter-region WAN unreliability | `NetworkSimulator` with configurable drop rate and straggler behavior |
| Regional data distribution differences | Dirichlet-based non-IID data partitioning per node |
| Compromised or buggy nodes | Attack module: noise injection, sign-flip, scaling attacks |
| Round timeout for unresponsive nodes | `timeout_multiplier` on median expected response time |

---

## 5. Algorithm Pseudocode

### Algorithm 1: TWAC-FL Server

```
Initialize: global model w⁰, trust scores τᵢ⁰ = 1.0 for all nodes i, reference Δ̄⁰ = 0
For round t = 1, 2, ..., T:
    S_t ← select K nodes (uniformly at random)
    Broadcast w^t to all nodes in S_t
    
    # Collect updates with timeout (models WAN unreliability)
    received ← {}
    For each node i in S_t (with timeout T_max):
        Δ_i^t ← receive_update(i)  # includes simulated inter-region delay
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
    
    # Decay trust for straggler nodes (slow instances or high-latency regions)
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

### Algorithm 2: Cloud Node Local Training

```
Input: global model w^t, local regional data D_i, epochs E, learning rate η
Output: update Δ_i^t

w_local ← w^t
For epoch = 1, ..., E:
    For batch (x, y) in D_i:
        w_local ← w_local - η · ∇L(w_local; x, y)

Δ_i^t ← w_local - w^t

# If compromised/faulty node:
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

### Regional Data Distribution via Dirichlet Allocation

In a multi-cloud setting, each node collects data locally from its region or organization, leading to naturally non-IID distributions. For example, a hospital in Asia might see different disease patterns than one in Europe; a regional data center might collect traffic data with local characteristics.

We simulate this using a Dirichlet distribution to control the degree of non-IID-ness across nodes:

$$p_i \sim \text{Dir}(\alpha_{dir} \cdot \mathbf{1})$$

Where $\alpha_{dir}$ controls heterogeneity:

- $\alpha_{dir} \to \infty$: IID (all nodes see identical data distribution — unrealistic but useful baseline)
- $\alpha_{dir} = 1.0$: moderate heterogeneity (regional differences in data mix)
- $\alpha_{dir} = 0.1$: extreme heterogeneity (each node sees mostly 1-2 classes — models highly specialized regional data)

Each node receives data sampled according to its class probability vector $p_i$, simulating regional or organizational data bias.

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
| Number of cloud nodes (N) | 20 |
| Nodes per round (K) | 10 |
| Local epochs (E) | 5 |
| Local learning rate (η) | 0.01 |
| Batch size | 32 |
| Communication rounds (T) | 100 |
| Non-IID α_dir | {0.1, 0.5, 1.0, 100.0} |
| Compromised node fraction | {0%, 10%, 20%, 30%} |
| Straggler node fraction | {0%, 20%, 40%} |

### 7.4 Failure and Attack Scenarios

1. **No attack** (baseline — all nodes honest)
2. **Gaussian noise** (models noisy hardware or software bugs): $\Delta + \mathcal{N}(0, \sigma^2 I)$, $\sigma = 0.5$
3. **Sign-flip** (models a compromised node actively sabotaging training): $-\lambda \Delta$, $\lambda = 1.0$
4. **Scaling attack** (models a node trying to dominate aggregation): $\kappa \Delta$, $\kappa = 5.0$

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
- **Our improvement**: We add directional consistency checking and temporal trust, which catches sign-flip attacks that norm clipping alone misses. In a multi-cloud setting, temporal trust is particularly valuable because transient cloud issues (brief network spikes, instance throttling) should not permanently penalize a node

### vs. Rethinking Data Heterogeneity

- This work analyzes why non-IID hurts and proposes data-sharing solutions
- **Our approach**: Rather than requiring cross-region data sharing (which violates data sovereignty and privacy regulations in multi-cloud deployments), TWAC implicitly handles non-IID by down-weighting outlier directions that may be due to extreme regional data heterogeneity

### vs. Architecture-based approaches

- These modify the model architecture to handle heterogeneity
- **Our approach**: Architecture-agnostic; works with any model; focuses on the aggregation step. This is important in multi-cloud settings where different organizations may need to use standardized model architectures

### vs. Robust FL in Heterogeneous Environment

- Proposes robust aggregation rules assuming known attack model
- **Our approach**: Attack-model-agnostic; the trust mechanism naturally adapts to whatever failure pattern emerges — whether that's a compromised node, a buggy training pipeline, or a misconfigured cloud instance

---

## 12. Expected Results (Hypotheses)

1. **Under no attack**: TWAC ≈ FedAvg (minimal overhead when all nodes are honest)
2. **Under attacks**: TWAC > FedAvg, competitive with Krum/TrimmedMean (compromised nodes are effectively isolated)
3. **Under stragglers**: TWAC > all baselines (graceful degradation via timeout + trust — models WAN unreliability between cloud regions)
4. **Under non-IID**: TWAC handles regional data divergence better than naive FedAvg by down-weighting extreme outlier directions
5. **Computation**: TWAC runs ~1.1x FedAvg wall-clock time vs ~2-3x for Krum (crucial for cost-sensitive multi-cloud deployments)

---

*End of Blueprint*
