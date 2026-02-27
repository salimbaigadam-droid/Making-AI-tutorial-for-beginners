"""
examples/05_save_and_load.py
==============================
🟣 EXAMPLE 5 — Save, Load & Inference Pipeline

Shows a production-style workflow:
  1. Train a model
  2. Save the weights to disk
  3. Load the weights in a fresh model
  4. Run inference without any training

This is how you'd deploy the C++ model in a real app.

Run it:
    python examples/05_save_and_load.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from examples.pure_python_fallback import load_engine

ai = load_engine("pybind11")

np.random.seed(99)

print("=" * 55)
print("  EXAMPLE 5: Save, Load & Production Inference")
print("=" * 55)

# ─── Create a simple dataset: predict if sum of inputs > 0 ───────────────────
#   Input:  [x1, x2, x3]  (3 features)
#   Output: 1 if x1 + x2 + x3 > 0, else 0

def make_dataset(n=600):
    X = np.random.randn(n, 3)
    y = (X.sum(axis=1) > 0).astype(np.float64).reshape(-1, 1)
    return X.astype(np.float64), y

X_all, y_all = make_dataset(600)
split        = 480
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# ─── STEP 1: Build & Train ────────────────────────────────────────────────────
print("\n[Step 1] Building and training model...\n")

model = ai.NeuralNetwork([
    ai.LayerConfig(3, 16, ai.Activation.RELU),
    ai.LayerConfig(16, 8, ai.Activation.RELU),
    ai.LayerConfig(8,  1, ai.Activation.SIGMOID),
], learning_rate=0.05)

for epoch in range(1, 1001):
    loss = model.train_step(X_train, y_train)
    if epoch % 200 == 0:
        preds = model.predict(X_test)
        acc   = float(np.mean((preds >= 0.5).astype(int) == y_test.astype(int)))
        print(f"  Epoch {epoch:5d} | Loss: {loss:.5f} | Test Acc: {acc*100:.1f}%")

# Final accuracy before save
preds_before = model.predict(X_test)
acc_before   = float(np.mean((preds_before >= 0.5).astype(int) == y_test.astype(int)))
print(f"\n  ✅ Training complete. Test Accuracy: {acc_before*100:.1f}%")

# ─── STEP 2: Save Weights ─────────────────────────────────────────────────────
print("\n[Step 2] Saving model weights...\n")

SAVE_PATH = "trained_model.dat"
model.save_weights(SAVE_PATH)
file_size = os.path.getsize(SAVE_PATH + (".npz" if not os.path.exists(SAVE_PATH) else ""))
actual_path = SAVE_PATH + ".npz" if os.path.exists(SAVE_PATH + ".npz") else SAVE_PATH
print(f"  ✅ Saved to: {actual_path}")
print(f"     File size: {os.path.getsize(actual_path):,} bytes")

# ─── STEP 3: Load into a FRESH model ─────────────────────────────────────────
print("\n[Step 3] Loading weights into a brand-new model...\n")

fresh_model = ai.NeuralNetwork([
    ai.LayerConfig(3, 16, ai.Activation.RELU),
    ai.LayerConfig(16, 8, ai.Activation.RELU),
    ai.LayerConfig(8,  1, ai.Activation.SIGMOID),
], learning_rate=0.05)

# Before loading, fresh model outputs garbage
preds_fresh_before = fresh_model.predict(X_test)
acc_fresh_before   = float(np.mean(
    (preds_fresh_before >= 0.5).astype(int) == y_test.astype(int)
))
print(f"  Fresh model (before loading) accuracy: {acc_fresh_before*100:.1f}%  ← random")

# Load the saved weights
fresh_model.load_weights(actual_path)

preds_fresh_after = fresh_model.predict(X_test)
acc_fresh_after   = float(np.mean(
    (preds_fresh_after >= 0.5).astype(int) == y_test.astype(int)
))
print(f"  Fresh model (after loading)  accuracy: {acc_fresh_after*100:.1f}%  ← same as trained!")

# ─── STEP 4: Verify outputs match exactly ────────────────────────────────────
print("\n[Step 4] Verifying predictions are identical...\n")

max_diff = float(np.max(np.abs(preds_before - preds_fresh_after)))
print(f"  Max absolute difference between original and loaded model: {max_diff:.2e}")
if max_diff < 1e-10:
    print("  ✅ PERFECT MATCH — save/load is lossless!")
else:
    print(f"  ⚠️  Small floating-point difference ({max_diff:.2e}) — normal for text format")

# ─── STEP 5: Production inference on new data ─────────────────────────────────
print("\n[Step 5] Running inference on new, unseen data...\n")

new_inputs = np.array([
    [ 1.5,  0.5,  2.0],   # Sum =  4.0 → should be 1
    [-1.0, -2.0, -0.5],   # Sum = -3.5 → should be 0
    [ 0.1, -0.1,  0.0],   # Sum =  0.0 → borderline!
    [ 3.0,  1.0,  1.0],   # Sum =  5.0 → should be 1
    [-3.0, -1.0, -1.5],   # Sum = -5.5 → should be 0
], dtype=np.float64)

expected = np.array([1, 0, 0, 1, 0])

print(f"  {'Input':>30}  {'Raw Score':>10}  {'Predicted':>10}  {'Expected':>10}  {'OK':>4}")
print(f"  {'-'*72}")

outputs = fresh_model.predict(new_inputs)
for i, (inp, out) in enumerate(zip(new_inputs, outputs)):
    raw  = float(out[0])
    pred = 1 if raw >= 0.5 else 0
    ok   = "✅" if pred == expected[i] else "❌"
    print(f"  {str(inp.tolist()):>30}  {raw:>10.4f}  {pred:>10}  {expected[i]:>10}  {ok:>4}")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
print(f"\n[Cleanup] Removing {actual_path}...")
try:
    os.remove(actual_path)
    print(f"  ✅ Removed")
except FileNotFoundError:
    pass

print("\n🎉 Done! Your model can now be saved, loaded, and deployed.")
print("\nNext steps:")
print("  - Wrap fresh_model.predict() in a FastAPI endpoint")
print("  - Export to ONNX for browser / mobile deployment")
print("  - Use the C++ library directly from another C++ program")
