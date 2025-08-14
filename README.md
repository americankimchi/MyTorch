# MyTorch

*A tiny, NumPy‑only deep learning library that mirrors the core ideas behind PyTorch.*

> **What is this?** MyTorch is a learning/teaching project that builds a minimal deep learning stack from scratch: a `Tensor` object with automatic differentiation, a small set of neural network layers, basic loss functions, and simple optimizers—implemented with **NumPy only**.

---

## Highlights

* **NumPy‑backed `Tensor`** with scalar and tensor ops, gradients, and reverse‑mode autodiff (a tape/graph engine).
* **Layers (nn)**: building blocks like Linear, activations (ReLU/Tanh/Sigmoid), (optionally) Conv1D/2D, RNN/GRU cells, Dropout, BatchNorm. *(Some modules may still be WIP.)*
* **Losses**: MSE, Cross‑Entropy, etc.
* **Optimizers**: SGD (with momentum), Adam, etc.
* **Clean, readable code** focused on clarity over speed—ideal for study, debugging, and experimentation.

> Project status: **experimental / WIP**. Expect frequent API changes.

---

## Repository structure

```
MyTorch/
├── tensor.py            # Core Tensor object (NumPy wrapper) and utilities
├── autograd_engine.py   # Reverse‑mode autodiff (computational graph/tape)
├── nn/                  # Layers & modules (e.g., Linear, activations, conv, rnn)
├── optim/               # Optimizers (e.g., SGD, Adam)
└── README.md            # You are here
```

---

## Installation

This repo is intentionally lightweight and not packaged to PyPI.

```bash
# clone
git clone https://github.com/americankimchi/MyTorch.git
cd MyTorch

# (optional) create a virtualenv
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\\Scripts\\activate

# use in-place
python -c "import sys; sys.path.append('.') ; import tensor ; print('OK')"
```

If you prefer editable installs, you can create a minimal `pyproject.toml` later and run `pip install -e .`.

---

## Quickstart (minimal example)

Below is a tiny example that demonstrates forward, loss, backward, and a manual optimizer step using the core `Tensor`. **Adjust names if your local API differs.**

```python
# Example only — adapt to the exact API in tensor.py / autograd_engine.py
from tensor import Tensor  # or: from MyTorch.tensor import Tensor

# toy data: X ∈ R^{N×D}, y ∈ R^{N×1}
N, D, H, C = 32, 10, 16, 1
X = Tensor.randn(N, D, requires_grad=True)
y = Tensor.randn(N, C)

# parameters
W1 = Tensor.randn(D, H, requires_grad=True)
b1 = Tensor.zeros(H, requires_grad=True)
W2 = Tensor.randn(H, C, requires_grad=True)
b2 = Tensor.zeros(C, requires_grad=True)

# forward
h = (X @ W1 + b1).relu()
y_hat = X @ W1 @ W2 + b2  # or: (h @ W2 + b2)

# loss (MSE)
loss = ((y_hat - y) ** 2).mean()

# backward
loss.backward()

# gradient step (SGD)
lr = 1e-2
for p in (W1, b1, W2, b2):
    p.data -= lr * p.grad
    p.grad = 0  # or p.zero_grad() if available
```

If you prefer using higher‑level layers and optimizers, the pattern will look familiar:

```python
# Example only — adapt to actual modules available under nn/ and optim/
from nn import Linear, ReLU, Sequential
from optim import SGD
from tensor import Tensor

model = Sequential(
    Linear(10, 16),
    ReLU(),
    Linear(16, 1),
)

opt = SGD(model.parameters(), lr=1e-2)

for step in range(500):
    y_pred = model(X)
    loss = ((y_pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## Design notes

* **Autograd**: operators create a graph of `Function` nodes; `Tensor.backward()` walks this graph in reverse topological order to accumulate gradients.
* **Modules**: layers track parameters and expose `.parameters()` for optimizers.
* **Numerical stability**: log‑sum‑exp and softmax should be implemented with stability in mind; batch norm tracks running stats; dropout is active only in training mode.
* **No GPUs**: MyTorch intentionally uses **NumPy** for transparency; vectorization is preferred over Python loops.

---

## Roadmap (suggested)

* [ ] Finish a consistent `Module` base class (training/eval mode, `.parameters()`).
* [ ] Expand ops coverage in `Tensor` (broadcasting semantics, slicing/reshape, etc.).
* [ ] Add `Conv2D`, `MaxPool2D`, `Flatten`, and `Sequential` utilities.
* [ ] Add `CrossEntropyLoss` (with numerically stable softmax + log likelihood).
* [ ] Add common optimizers beyond SGD (Adam, RMSProp) with weight decay.
* [ ] Unit tests and simple examples (MNIST MLP/CNN, character‑level RNN).

---

## Contributing

Contributions, bug reports, and feature requests are very welcome! If you submit PRs, please keep the code:

* **Readable** (teaching‑oriented docs & comments)
* **Deterministic** (seeded randomness where applicable)
* **Tested** (add small unit tests where possible)

---

## License

If you haven’t added a license yet, consider MIT or Apache‑2.0. Create a `LICENSE` file in the repo root so others know how they can use the code.

---

## Acknowledgments

* Inspired by the design of **PyTorch** (tensors, autograd, modules, and optimizers).
* Thanks to the many minimal‑DL projects that make learning internals fun.
