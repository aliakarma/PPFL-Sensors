# Methodology

## 1. Federated Learning Setup

### Objective

We consider a standard cross-device federated learning scenario with $N$ clients and one central server. Each client $i$ holds a private local dataset $\mathcal{D}_i$ drawn from distribution $\mathcal{P}_i$. The distributions may differ across clients (non-IID). The global objective is to minimise:

$$F(w) = \sum_{i=1}^{N} \frac{|\mathcal{D}_i|}{|\mathcal{D}|} F_i(w)$$

where $F_i(w) = \frac{1}{|\mathcal{D}_i|} \sum_{(x,y) \in \mathcal{D}_i} \ell(f_w(x), y)$ is the local empirical risk.

### FedAvg Algorithm

Each communication round $t$ proceeds as follows:

1. **Broadcast**: Server sends global weights $w^{(t)}$ to all clients.
2. **Local update**: Each client $i$ runs $E$ epochs of SGD on $\mathcal{D}_i$ starting from $w^{(t)}$, producing $w_i^{(t+1)}$.
3. **Upload**: Client sends weight delta $\Delta w_i^{(t)} = w_i^{(t+1)} - w^{(t)}$ to server.
4. **Aggregate**: Server computes $w^{(t+1)} = w^{(t)} + \sum_i \frac{|\mathcal{D}_i|}{|\mathcal{D}|} \Delta w_i^{(t)}$.

### Data Partitioning Strategies

Five strategies are implemented to model different degrees of statistical heterogeneity:

| Strategy | Description | Parameter |
|---|---|---|
| IID | Uniform random split | — |
| Dirichlet | Label proportions from $\text{Dir}(\alpha)$ | $\alpha \in (0, \infty)$ |
| Pathological | Each client receives $K$ of $C$ classes | $K \in [1, C]$ |
| Quantity skew | Sample counts from $\text{Dir}(\beta)$ | $\beta \in (0, \infty)$ |
| Feature skew | Per-client Gaussian feature noise | $\sigma_i$ per client |

---

## 2. Threat Model

### Attacker Capabilities

We consider an **honest-but-curious server** (semi-honest adversary):

- The server executes the FL protocol faithfully.
- The server also attempts to infer sensitive attributes from client uploads.
- The attacker has access to all gradient uploads from all clients in all rounds.
- The attacker has no access to client raw data.
- The attacker knows the FL model architecture and the number of clients.

### Attack Surface

The attack surface is the set of weight delta vectors $\{\Delta w_i^{(t)}\}$ uploaded by client $i$ at round $t$. The server observes all of these across all rounds.

### Attack Objective

**Identity inference**: given a gradient upload $\Delta w$, predict which client $i$ produced it.

Formally, the attacker trains a classifier $A: \mathbb{R}^D \to [N]$ on a set of labelled gradient-client pairs, then applies it to new uploads.

---

## 3. Attack Formulation

### Feature Extraction

Raw gradient uploads are flattened to 1-D vectors of dimension $D = \sum_l |w_l|$ (total parameter count). Before classification, vectors are normalised using one of:

- **L2**: $g \leftarrow g / \|g\|_2$
- **L∞**: $g \leftarrow g / \|g\|_\infty$
- **Z-score**: $g \leftarrow (g - \mu) / \sigma$

### Attack Dataset Construction

- **Training set**: gradient uploads from rounds $1 \ldots N_\text{collect}$.
- **Test set**: gradient uploads from rounds $N_\text{collect}+1 \ldots T$.
- **Labels**: client IDs (0-indexed).
- The attack model is trained exactly **once** after round $N_\text{collect}$.

### Attack Models

Five classifiers of increasing expressive power:

| Model | Type | Notes |
|---|---|---|
| Random baseline | Uniform random | Hard lower bound: $1/N$ |
| Majority baseline | Always predict majority class | Guards against imbalance |
| Logistic regression | Linear | Tests linear separability |
| Random Forest | Non-linear ensemble | Strong non-parametric baseline |
| MLP | Neural network | Most expressive; strongest attack |

---

## 4. Defense Mechanisms

### 4.1 Gradient Clipping

Clip each client's update to have L2 norm at most $C$:

$$\Delta \tilde{w}_i = \Delta w_i \cdot \min\left(1,\ \frac{C}{\|\Delta w_i\|_2}\right)$$

This bounds the **sensitivity** of each upload to $C$, limiting how much a single client can influence the global model or reveal about itself.

### 4.2 Gaussian Noise Injection

After clipping, add isotropic Gaussian noise:

$$\hat{w}_i = \Delta \tilde{w}_i + \mathcal{N}(0,\ \sigma^2 C^2 \mathbf{I})$$

This is the Gaussian mechanism from differential privacy. The noise scale $\sigma C$ is calibrated relative to the sensitivity $C$.

**Standard DP application order**: clip first → then add noise.

### 4.3 Ensemble FL

Instead of training a single global model:

1. Partition clients into $K$ groups.
2. Train $K$ independent sub-models, one per group.
3. At inference, combine sub-model predictions via majority vote or average logits.

**Privacy benefit**: the server never sees all clients' gradients flowing into a single model. An attacker observing sub-model $k$ can only infer identities among the $N/K$ clients in group $k$.

---

## 5. Privacy–Utility Tradeoff

### Metrics

| Metric | Definition | Direction |
|---|---|---|
| FL task accuracy | Top-1 accuracy of global model on test set | Higher is better |
| Attack accuracy | Top-1 accuracy of best attack model on gradient test set | Lower is better |
| Privacy score | $1 - \text{attack\_accuracy}$ | Higher is better |

### Tradeoff

Defenses improve privacy (reduce attack accuracy) at the cost of utility (reduce FL accuracy). The optimal operating point depends on the application's privacy requirement.

---

## 6. Limitations

- **Simulation**: FL is simulated in-process; no real network or encryption.
- **No formal DP accounting**: $(\varepsilon, \delta)$ bounds are not computed. For formal DP, use Opacus or TensorFlow Privacy.
- **Single dataset family**: experiments use UCI HAR or synthetic tabular data; results may not generalise to image or text modalities.
- **Passive attacker only**: the honest-but-curious model does not cover active Byzantine adversaries.
- **Fixed attack window**: the collect/eval split is fixed; adaptive attacks that update continuously are not modelled.
