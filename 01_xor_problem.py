"""
examples/01_xor_problem.py
===========================
🔵 EXAMPLE 1 — The "Hello World" of Neural Networks: XOR Gate

XOR cannot be solved by a single straight line (it's not linearly separable).
A multi-layer network learns to solve it using hidden representations.

Run it:
    python examples/01_xor_problem.py

Expected output:
    Training...
    Epoch   500 | Loss: 0.249123
    Epoch  1000 | Loss: 0.112456
    ...
    Epoch  5000 | Loss: 0.001234

    XOR Results:
    [0, 0] → 0.02  (expected 0) ✅
    [0, 1] → 0.97  (expected 1) ✅
    [1, 0] → 0.96  (expected 1) ✅
    [1, 1] → 0.03  (expected 0) ✅
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from examples.pure_python_fallback import load_engine

# ─── Load the best available backend ─────────────────────────────────────────
ai = load_engine("pybind11")

# ─── Dataset: 4 XOR samples ───────────────────────────────────────────────────
#   Input:   [x1, x2]
#   Output:  x1 XOR x2
X = np.array([
    [0, 0],   # 0 XOR 0 = 0
    [0, 1],   # 0 XOR 1 = 1
    [1, 0],   # 1 XOR 0 = 1
    [1, 1],   # 1 XOR 1 = 0
], dtype=np.float64)

y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# ─── Build Network ────────────────────────────────────────────────────────────
#   Architecture:  2 inputs → 8 hidden (ReLU) → 8 hidden (ReLU) → 1 output (Sigmoid)
#   Why ReLU?      Fast to compute, avoids vanishing gradients in hidden layers
#   Why Sigmoid?   Output layer — squashes to [0, 1] for binary classification

np.random.seed(42)

model = ai.NeuralNetwork([
    ai.LayerConfig(2, 8, ai.Activation.RELU),
    ai.LayerConfig(8, 8, ai.Activation.RELU),
    ai.LayerConfig(8, 1, ai.Activation.SIGMOID),
], learning_rate=0.1)

print("=" * 50)
print("  EXAMPLE 1: XOR Gate Classifier")
print("=" * 50)
print(f"\nModel: {model}")
print(f"\nDataset:\n  X = {X.tolist()}\n  y = {y.flatten().tolist()}\n")

# ─── Training Loop ────────────────────────────────────────────────────────────
print("Training...\n")
EPOCHS = 5000
losses = []

for epoch in range(1, EPOCHS + 1):
    loss = model.train_step(X, y)
    losses.append(loss)

    if epoch % 500 == 0:
        print(f"  Epoch {epoch:5d}/{EPOCHS} | Loss: {loss:.6f}")

# ─── Predictions ─────────────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("XOR Predictions (threshold = 0.5):")
print("-" * 50)

preds = model.predict(X)
correct = 0

for i in range(len(X)):
    raw    = float(preds[i, 0])
    binary = 1 if raw >= 0.5 else 0
    expected = int(y[i, 0])
    ok = "✅" if binary == expected else "❌"
    print(f"  [{int(X[i,0])}, {int(X[i,1])}]  →  {raw:.4f}  →  {binary}  (expected {expected}) {ok}")
    correct += (binary == expected)

print(f"\nAccuracy: {correct}/{len(X)} = {100*correct/len(X):.0f}%")

# ─── Plot learning curve (optional — requires matplotlib) ─────────────────────
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), losses, color='royalblue', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("XOR Training Curve")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("xor_training_curve.png", dpi=150)
    print("\n📊 Learning curve saved to xor_training_curve.png")
    plt.show()
except ImportError:
    print("\n(Install matplotlib to see the learning curve: pip install matplotlib)")
