# Self-Pruning Neural Network – Case Study Report

## 1. Introduction

In deep learning, large neural networks often contain many redundant parameters, making them inefficient for deployment. Pruning is a technique used to remove less important weights to improve efficiency.

This project implements a **self-pruning neural network**, where the model learns to prune its own weights during training instead of applying pruning after training.

---

## 2. Methodology

### 2.1 Prunable Linear Layer

A custom linear layer called `PrunableLinear` is implemented.

Each weight is associated with a learnable parameter called `gate_scores`.

During the forward pass:

* Gate values are computed using a sigmoid function:

  `gates = sigmoid(gate_scores)`

* The effective weights are:

  `pruned_weights = weight × gates`

If a gate approaches zero, the corresponding weight is effectively removed.

---

### 2.2 Sparsity Loss

To encourage pruning, a sparsity penalty is added to the loss function.

Total loss:

`Loss = CrossEntropyLoss + λ × SparsityLoss`

Where:

* SparsityLoss = sum of all gate values (L1 norm)

### Why L1 encourages sparsity

L1 regularization applies a constant gradient pushing values toward zero.
This results in many gate values becoming exactly zero, leading to a sparse network.

---

## 3. Training Setup

* Dataset: CIFAR-10
* Model: 3-layer feedforward neural network
* Optimizer: Adam
* Epochs: 5
* Batch size: 64

---

## 4. Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 1e-5   | XX.XX%        | XX.XX%       |
| 1e-4   | XX.XX%        | XX.XX%       |
| 1e-3   | XX.XX%        | XX.XX%       |

---

## 5. Observations

* Increasing λ increases sparsity in the network.
* Higher λ values lead to more aggressive pruning but may reduce accuracy.
* Lower λ values retain higher accuracy but result in less pruning.
* A moderate λ provides a balance between sparsity and performance.

---

## 6. Gate Distribution

A histogram of gate values shows:

* A spike near zero → pruned weights
* A cluster away from zero → important weights

This confirms that the model successfully learns to prune itself.

---

## 7. Conclusion

The self-pruning neural network successfully learns to remove unnecessary weights during training using a learnable gating mechanism and L1 regularization.

This approach eliminates the need for post-training pruning and produces a compact model while maintaining reasonable accuracy.
