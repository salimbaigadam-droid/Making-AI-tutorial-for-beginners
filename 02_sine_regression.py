"""
examples/02_sine_regression.py
================================
🟠 EXAMPLE 2 — Sine Wave Regression

We train the network to approximate f(x) = sin(x) using only raw inputs.
This shows the network learning a smooth, continuous non-linear function
with no prior knowledge about trigonometry.

Run it:
    python examples/02_sine_regression.py

What you'll see:
  - Network starts with random noise predictions
  - Gradually learns the sine shape
  - Final MSE < 0.001 if trained long enough
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from examples.pure_python_fallback import load_engine

ai = load_engine("pybind11")

# ─── Dataset: sin(x) for x in [-π, π] ────────────────────────────────────────
np.random.seed(0)

N = 500  # Number of training samples
X_raw = np.linspace(-np.pi, np.pi, N)
y_raw = np.sin(X_raw)

# Reshape to (N, 1) for the network
X = X_raw.reshape(-1, 1).astype(np.float64)
y = y_raw.reshape(-1, 1).astype(np.float64)

# Normalize X to [-1, 1] — always normalize inputs!
X_norm = X / np.pi

# Train / test split
split = int(0.8 * N)
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y[:split], y[split:]

print("=" * 55)
print("  EXAMPLE 2: Sine Wave Regression")
print("=" * 55)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"Input range:      [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Output range:     [{y_train.min():.2f}, {y_train.max():.2f}]")

# ─── Build Network ────────────────────────────────────────────────────────────
#   Architecture: 1 → 64 → 64 → 32 → 1
#   Tanh is great for regression because output is in [-1, 1], matching sin(x)
#   Final layer uses LINEAR activation — no squashing, raw output

model = ai.NeuralNetwork([
    ai.LayerConfig(1,  64, ai.Activation.TANH),
    ai.LayerConfig(64, 64, ai.Activation.TANH),
    ai.LayerConfig(64, 32, ai.Activation.TANH),
    ai.LayerConfig(32,  1, ai.Activation.LINEAR),
], learning_rate=0.005)

print(f"\nModel: {model}")

# ─── Training Loop with mini-batches ─────────────────────────────────────────
print("\nTraining...\n")
EPOCHS     = 3000
BATCH_SIZE = 64
train_losses = []
test_losses  = []

for epoch in range(1, EPOCHS + 1):
    # Shuffle data each epoch
    idx = np.random.permutation(len(X_train))
    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, len(X_train), BATCH_SIZE):
        batch_idx = idx[start:start + BATCH_SIZE]
        Xb = X_train[batch_idx]
        yb = y_train[batch_idx]
        epoch_loss += model.train_step(Xb, yb)
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    # Compute test loss (no gradient update)
    test_preds = model.predict(X_test)
    test_loss  = float(np.mean((test_preds - y_test) ** 2))
    test_losses.append(test_loss)

    if epoch % 300 == 0:
        print(f"  Epoch {epoch:5d}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("Evaluation on test set:")
final_preds = model.predict(X_test)
final_mse   = float(np.mean((final_preds - y_test) ** 2))
final_mae   = float(np.mean(np.abs(final_preds - y_test)))
max_err     = float(np.max(np.abs(final_preds - y_test)))

print(f"  MSE (Mean Squared Error):     {final_mse:.6f}")
print(f"  MAE (Mean Absolute Error):    {final_mae:.6f}")
print(f"  Max Error:                    {max_err:.6f}")

# Show some sample predictions
print("\nSample predictions (input in radians):")
print(f"  {'x (rad)':>10}  {'sin(x)':>10}  {'predicted':>10}  {'error':>10}")
print(f"  {'-'*44}")
sample_idx = np.linspace(0, len(X_test)-1, 8, dtype=int)
for i in sample_idx:
    x_orig  = float(X_test[i, 0]) * np.pi
    actual  = float(y_test[i, 0])
    pred    = float(final_preds[i, 0])
    err     = abs(pred - actual)
    print(f"  {x_orig:>10.3f}  {actual:>10.4f}  {pred:>10.4f}  {err:>10.6f}")

# ─── Visualize ────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    X_plot     = np.linspace(-1, 1, 300).reshape(-1, 1)
    y_true     = np.sin(X_plot * np.pi)
    y_pred_all = model.predict(X_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Sine approximation ──────────────────────────────────────────────
    ax1.plot(X_plot * np.pi, y_true, 'k-', linewidth=2, label='True sin(x)', zorder=2)
    ax1.plot(X_plot * np.pi, y_pred_all, 'r--', linewidth=2,
             label='Network prediction', zorder=3)
    ax1.scatter(X_test * np.pi, y_test, s=5, alpha=0.4, color='gray', label='Test data')
    ax1.set_xlabel("x (radians)")
    ax1.set_ylabel("sin(x)")
    ax1.set_title("Sine Wave Approximation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right: Training curves ────────────────────────────────────────────────
    ax2.plot(train_losses, label='Train Loss', color='royalblue')
    ax2.plot(test_losses,  label='Test Loss',  color='tomato')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sine_regression.png", dpi=150)
    print("\n📊 Plot saved to sine_regression.png")
    plt.show()

except ImportError:
    print("\n(Install matplotlib for plots: pip install matplotlib)")
