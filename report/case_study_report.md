# Case Study Report – Self-Pruning Neural Network

## Candidate

V Krishnakumar
RA2311026020074
[kv4553@srmist.edu.in](mailto:kv4553@srmist.edu.in)

---

# Problem Statement

Design a neural network that learns to prune its own weights during training instead of relying on post-training pruning.

The objective is to jointly optimize:

* classification accuracy
* sparse connectivity
* efficient architecture

---

# Proposed Solution

A custom `PrunableLinear` layer was implemented where each weight has an associated learnable gate parameter.

Effective weight:

```text id="m5w4uq"
W_pruned = W × sigmoid(score / temperature)
```

Connections with low gate values become inactive.

---

# Model Architecture

CNN feature extractor:

* Conv(32)
* Conv(64)
* Conv(128)

Sparse classifier:

* PrunableLinear(2048, 256)
* PrunableLinear(256, 10)

---

# Training Objective

Total loss:

```text id="txw3xp"
CrossEntropy + λ × L1(Gates)
```

Where:

* CrossEntropy optimizes classification
* L1 regularization encourages sparsity

---

# Experimental Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1.5e-5 | 82.35%   | 26.63%   |
| 2e-5   | 81.99%   | 36.77%   |
| 3e-5   | 81.31%   | 43.88%   |
| 5e-5   | 81.05%   | 43.98%   |
| 7e-5   | 80.28%   | 38.82%   |

---

# Best Balanced Model

```text id="m19m9l"
Lambda = 5e-5
Accuracy = 81.05%
Sparsity = 43.98%
```

This model removed nearly half of effective connections while preserving strong accuracy.

---

# Key Insights

1. CNN feature extraction significantly improved performance over MLP baseline.
2. Higher lambda increases pruning pressure.
3. Moderate lambda provides best accuracy-sparsity tradeoff.

---

# Conclusion

The project successfully demonstrates train-time differentiable pruning using learnable gates.

It balances predictive performance with efficiency and reflects practical AI engineering focused on deployable systems.
