# 🧠 Self-Pruning Neural Network

<p align="center">
  <b>Dynamic Sparse Learning with Differentiable Gates in PyTorch</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&labelColor=111827" />
  <img src="https://img.shields.io/badge/CIFAR--10-Training-2563EB?style=for-the-badge&labelColor=111827" />
  <img src="https://img.shields.io/badge/Sparsity-Learned-16A34A?style=for-the-badge&labelColor=111827" />
  <img src="https://img.shields.io/badge/Status-Research_Prototype-9333EA?style=for-the-badge&labelColor=111827" />
</p>

---

## 🌍 Vision

Train neural networks that automatically remove unnecessary connections during learning.

---

## 📌 Problem Statement

Traditional pruning requires:

Train Dense Model → Prune → Fine-tune Again

This increases compute cost and training complexity.

---

## 🚀 Solution

A self-pruning neural network that learns sparse connectivity during training using differentiable gate parameters.

Effective weight:

```text
W_pruned = W × sigmoid(score / temperature)
```

---

## 🏗️ Architecture

```text
Input (32x32x3)
↓
CNN Feature Extractor
↓
PrunableLinear(2048 → 256)
↓
ReLU + Dropout
↓
PrunableLinear(256 → 10)
```

---

## ⚙️ Core Features

### 🧠 Learnable Gates

Every weight learns whether it should stay active.

### ✂️ Dynamic Sparsity

Connections collapse toward zero during training.

### 📉 Compression vs Accuracy Tradeoff

Controlled using λ regularization.

### 🔥 Temperature Annealing

Soft gates early, sharper pruning later.

### 📊 Real Experiments

Validated on CIFAR-10.

---

## 📈 Best Result

```text
81.05% Accuracy
43.98% Learned Sparsity
```

Nearly half the network effectively removed while maintaining strong performance.

---

## 🛠️ Tech Stack

| Layer            | Technology          |
| ---------------- | ------------------- |
| Language         | Python              |
| Framework        | PyTorch             |
| Dataset          | CIFAR-10            |
| Analysis         | Pandas + Matplotlib |
| Deployment Ready | FastAPI (planned)   |

---

## 📂 Repository Structure

```text
train.py
model.py
evaluate.py
outputs/
checkpoints/
report/
```

---

## ⚙️ Run Locally

```bash
pip install -r requirements.txt
python train.py
```

---

## 🎯 Why It Matters

Useful for:

* Edge AI
* Low-latency inference
* Efficient deployment
* Sparse expert systems
* Cost-efficient ML serving

---

## 🔮 Future Work

* Transformer pruning
* Quantization hybrid
* MoE sparse routing
* FastAPI serving layer

---

## 👨‍💻 Author

**V Krishnakumar**
