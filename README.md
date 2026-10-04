# Epsilon: An Adaptive Optimisation Algorithm Based on Directional Gradient Similarity for Image Classification

Official implementation and benchmark reproduction scripts for the paper:
> **Epsilon: An adaptive optimisation algorithm based on directional gradient similarity for image classification**  
> *Marcela de los Ángeles Yanes-Pérez, José Adán Hernández-Nolasco, and Miguel A. Wister*  
> Juarez Autonomous University of Tabasco (UJAT).

---

## Overview

**Epsilon** is an adaptive optimization algorithm that regulates momentum decay via the **cosine directional similarity between consecutive gradients**. Rather than accumulating squared gradients, it assesses update coherence and normalises step sizes against accumulated momentum magnitude, mitigating erratic trajectories under high spatial gradient variance.

### Key Highlights
- **FashionMNIST**: **89.25%** test accuracy (+5.05 percentage points vs Adam 84.20%).
- **CIFAR-10**: **85.80%** test accuracy (+1.79 percentage points vs Adam 84.01%).
- **CIFAR-100 (ResNet18)**: **66.21%** test accuracy (+1.57 percentage points vs Adam 64.64%).
- Consistently higher predictive accuracy than **Adam**, **AMSGrad**, and **Adamax** across all three benchmarks.

---

## Repository Structure

```
epsilon-optimizer/
├── epsilon.py               # Standalone Epsilon PyTorch Optimizer
├── train_fashionmnist.py    # FashionMNIST benchmark (Epsilon vs Adam vs AMSGrad vs Adamax)
├── train_cifar10.py         # CIFAR-10 benchmark (Epsilon vs Adam vs AMSGrad vs Adamax)
├── train_cifar100.py        # CIFAR-100 benchmark on ResNet18 (Epsilon vs Adam vs AMSGrad vs Adamax)
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---

## Installation & Usage

### 1. Requirements
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Using Epsilon in your own project
You can directly import and use `Epsilon` in any standard PyTorch workflow:

```python
import torch
from epsilon import Epsilon

model = YourModel()

# Standard paper configuration
optimizer = Epsilon(
    model.parameters(),
    lr=1e-4,
    beta_max=0.9,
    alpha=0.85,
    beta_min=0.5,
    gamma=0.5,
    epsilon=1e-8
)

# Training step
optimizer.zero_grad()
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step()
```

### 3. Reproducing Paper Experiments

To run the exact benchmarks reported in the manuscript:

- **FashionMNIST**:
  ```bash
  python train_fashionmnist.py
  ```
- **CIFAR-10**:
  ```bash
  python train_cifar10.py
  ```
- **CIFAR-100 (ResNet18)**:
  ```bash
  python train_cifar100.py
  ```

---

## Summary of Empirical Results

| Dataset | Optimiser | Test Accuracy (%) | Test Loss | Macro AUC |
|---|---|---|---|---|
| **FashionMNIST** | **Epsilon** | **89.25%** | **0.3043** | **0.9914** |
| | Adam | 84.20% | 0.4114 | 0.9849 |
| | AMSGrad | 81.07% | 0.4812 | 0.9791 |
| | Adamax | 78.86% | 0.5302 | 0.9752 |
| **CIFAR-10** | **Epsilon** | **85.80%** | **0.4631** | **0.9873** |
| | Adam | 84.01% | 0.4701 | 0.9855 |
| | AMSGrad | 83.25% | 0.4853 | 0.9846 |
| | Adamax | 78.22% | 0.6271 | 0.9755 |
| **CIFAR-100** | **Epsilon** | **66.21%** | **1.2115** | **0.9889** |
| | Adam | 64.64% | 1.2525 | 0.9866 |
| | AMSGrad | 63.94% | 1.2739 | 0.9865 |
| | Adamax | 57.39% | 1.5703 | 0.9781 |

---

## Citation

```bibtex
@article{yanes2026epsilon,
  title={Epsilon: An adaptive optimisation algorithm based on directional gradient similarity for image classification},
  author={Yanes-P{'e}rez, Marcela de los {'A}ngeles and Hern{'a}ndez-Nolasco, Jos{'e} Ad{'a}n and Wister, Miguel A.},
  journal={International Journal of Combinatorial Optimization Problems and Informatics (IJCOPI)},
  year={2026}
}
```
