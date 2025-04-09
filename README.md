# Neural Network Library

A tiny neural network framework built from scratch using only Python — no external libraries.

---

## Features
- Auto-diff with a custom `Value` class
- Neurons, Layers, and full Networks
- Forward + Backward pass (manual training)
- Tanh activation
- Supports training simple tasks like XOR

---

## Quick Start

### 1. Define a Network
```python
from CalcGrad import Network, Value

net = Network([2, 4, 1])  # 2 inputs → 4 hidden → 1 output
```

### 2. Training Data (XOR)
```python
data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]
```

### 3. Training Loop
```python
lr = 0.05
for epoch in range(1000):
    total_loss = Value(0.0)
    for x, y in data:
        inputs = [Value(i) for i in x]
        targets = [Value(t) for t in y]
        outputs = net(inputs)
        loss = net.calc_loss(targets, outputs)

        for p in net.parameters(): p.zero_grad()
        loss.backward()
        for p in net.parameters(): p.data -= lr * p.grad

        total_loss += loss

    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {total_loss.data:.4f}")
```

### Predict
```python
for x, _ in data:
    out = net([Value(i) for i in x])
    print(f"{x} → {[o.data for o in out]}")
```

### License
MIT — Use it, learn from it, hack it! 🔧


