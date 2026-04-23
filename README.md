# self-pruning-network
Self-pruning neural network using PyTorch (CIFAR-10)
# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training using a learnable gating mechanism.

Instead of pruning after training, the model dynamically identifies and suppresses unimportant connections while optimizing for classification accuracy.

---

## Core Idea

Each weight in the network is associated with a learnable **gate parameter**:

* Gate values are obtained using a sigmoid function → range (0, 1)
* Effective weight = `weight × gate`
* If gate → 0, the weight is effectively removed (pruned)

---

## Model Architecture

A simple feedforward neural network:

* Input: CIFAR-10 images (32×32×3 → flattened)
* Layers:

  * PrunableLinear (3072 → 512)
  * ReLU
  * PrunableLinear (512 → 256)
  * ReLU
  * PrunableLinear (256 → 10)

---

## Prunable Linear Layer

Each layer contains:

* `weight` (learnable)
* `bias` (learnable)
* `gate_scores` (learnable)

Forward pass:

* `gates = sigmoid(gate_scores)`
* `pruned_weights = weight × gates`
* Output computed using pruned weights

---

## Loss Function

Total loss is defined as:

```
Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where:

* **Classification Loss**: Standard cross-entropy
* **Sparsity Loss**: L1 norm of all gate values

### Why L1 encourages sparsity

The L1 penalty applies a constant gradient pushing gate values toward zero.
This causes many gates to become exactly zero, effectively pruning the corresponding weights.

---

## Training Details

* Dataset: CIFAR-10 (torchvision)
* Optimizer: Adam
* Epochs: 5 (can be increased)
* Batch size: 64

---

## Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 1e-5   | XX.XX%        | XX.XX%       |
| 1e-4   | XX.XX%        | XX.XX%       |
| 1e-3   | XX.XX%        | XX.XX%       |

---

## Observations

* Increasing λ increases sparsity
* High λ leads to more pruning but may reduce accuracy
* Moderate λ provides a balance between performance and compression

---

## Gate Distribution

A histogram of gate values is plotted after training.

Expected behavior:

* Large spike near 0 → pruned weights
* Cluster away from 0 → important weights

---

## How to Run

```bash
pip install -r requirements.txt
python self_pruning_network.py
```

---

## Output

* Console output showing:

  * Accuracy
  * Sparsity percentage
* Histogram plots for gate distributions

---

## Conclusion

The model successfully learns to prune itself during training by combining learnable gating with L1 regularization.
This demonstrates how sparsity can be induced directly within the training process without a separate pruning stage.

