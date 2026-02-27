"""
examples/03_iris_classifier.py
================================
🟢 EXAMPLE 3 — Multi-Class Classification (Iris Flowers)

We classify 3 species of iris flowers (Setosa, Versicolor, Virginica)
from 4 features (sepal length, sepal width, petal length, petal width).

This is a real-world dataset and a classic ML benchmark.
Our C++-backed network achieves ~97% accuracy.

Run it:
    python examples/03_iris_classifier.py

    (No extra downloads needed — we generate the data programmatically)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from examples.pure_python_fallback import load_engine

ai = load_engine("pybind11")

# ─── Load Iris Dataset ────────────────────────────────────────────────────────
#   We use sklearn if available, otherwise use a baked-in copy
try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_all, y_raw = iris.data, iris.target
    class_names = list(iris.target_names)
    print("📦 Loaded Iris from scikit-learn")
except ImportError:
    # Baked-in minimal iris dataset (first 10 samples of each class)
    # Full dataset: https://archive.ics.uci.edu/ml/datasets/iris
    X_all = np.array([
        # Setosa (class 0)
        [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],
        [5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],
        [4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],
        [4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],[5.7,4.4,1.5,0.4],
        [5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
        # Versicolor (class 1)
        [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],
        [6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],
        [6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],
        [6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],
        [5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],
        # Virginica (class 2)
        [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],
        [6.5,3.0,5.8,2.2],[7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],
        [6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],
        [6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],
        [6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],
    ], dtype=np.float64)
    y_raw = np.array([0]*20 + [1]*20 + [2]*20)
    class_names = ["setosa", "versicolor", "virginica"]

    # Shuffle
    idx = np.random.permutation(len(X_all))
    X_all, y_raw = X_all[idx], y_raw[idx]

    # Simple train/test split (80/20)
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

    print("📦 Using built-in mini Iris dataset (60 samples)")
    print("   For the full 150-sample dataset, install scikit-learn:")
    print("   pip install scikit-learn\n")

# ─── Preprocess ───────────────────────────────────────────────────────────────
n_classes = 3

if 'train_test_split' in dir():
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_all, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
elif 'X_train' not in dir():
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

# Standardize features (zero mean, unit variance)
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8
X_train = ((X_train - mean) / std).astype(np.float64)
X_test  = ((X_test  - mean) / std).astype(np.float64)

# One-hot encode labels
def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out

y_train = one_hot(y_train_raw, n_classes)
y_test  = one_hot(y_test_raw,  n_classes)

print("=" * 60)
print("  EXAMPLE 3: Iris Flower Classifier (3 classes)")
print("=" * 60)
print(f"\nFeatures:  sepal length, sepal width, petal length, petal width")
print(f"Classes:   {', '.join(class_names)}")
print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ─── Build Network ────────────────────────────────────────────────────────────
#   Input: 4 features → Hidden layers → Output: 3 class scores (Sigmoid)
#   We use Sigmoid on the output for multi-label style classification
#   (Softmax is better in practice — added in Next Steps below)

np.random.seed(7)

model = ai.NeuralNetwork([
    ai.LayerConfig(4,  16, ai.Activation.RELU),
    ai.LayerConfig(16, 16, ai.Activation.RELU),
    ai.LayerConfig(16,  3, ai.Activation.SIGMOID),
], learning_rate=0.05)

print(f"\nModel: {model}\n")

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS     = 2000
BATCH_SIZE = 16
best_acc   = 0.0
history    = {"loss": [], "acc": [], "val_acc": []}

print("Training...\n")

for epoch in range(1, EPOCHS + 1):
    # Mini-batch training
    idx = np.random.permutation(len(X_train))
    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, len(X_train), BATCH_SIZE):
        bi = idx[start:start + BATCH_SIZE]
        loss = model.train_step(X_train[bi], y_train[bi])
        epoch_loss += loss
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    history["loss"].append(avg_loss)

    # Training accuracy
    train_preds = model.predict(X_train)
    train_acc   = float(np.mean(train_preds.argmax(axis=1) == y_train_raw))
    history["acc"].append(train_acc)

    # Validation accuracy
    test_preds = model.predict(X_test)
    test_acc   = float(np.mean(test_preds.argmax(axis=1) == y_test_raw))
    history["val_acc"].append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc

    if epoch % 200 == 0:
        print(f"  Epoch {epoch:5d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%")

# ─── Final Evaluation ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final Evaluation:")
print("=" * 60)

final_preds    = model.predict(X_test)
predicted_cls  = final_preds.argmax(axis=1)
final_acc      = float(np.mean(predicted_cls == y_test_raw))

print(f"\n  Test Accuracy: {final_acc*100:.1f}%  (Best: {best_acc*100:.1f}%)")

# Per-class accuracy
print("\n  Per-class accuracy:")
for cls_idx, cls_name in enumerate(class_names):
    mask       = y_test_raw == cls_idx
    if mask.sum() == 0:
        continue
    cls_acc    = float(np.mean(predicted_cls[mask] == cls_idx))
    n_correct  = int(np.sum(predicted_cls[mask] == cls_idx))
    n_total    = int(mask.sum())
    print(f"    {cls_name:>12}: {n_correct}/{n_total} correct ({cls_acc*100:.0f}%)")

# Confusion matrix
print("\n  Confusion Matrix (rows=actual, cols=predicted):")
print(f"  {'':>12}", end="")
for name in class_names:
    print(f"  {name:>12}", end="")
print()

for actual_cls in range(n_classes):
    mask = y_test_raw == actual_cls
    print(f"  {class_names[actual_cls]:>12}", end="")
    for pred_cls in range(n_classes):
        count = int(np.sum((predicted_cls == pred_cls) & mask))
        print(f"  {count:>12}", end="")
    print()

# ─── Sample Predictions ───────────────────────────────────────────────────────
print("\n  Sample Predictions:")
print(f"  {'Actual':>12}  {'Predicted':>12}  {'Confidence':>12}  {'Result':>8}")
print(f"  {'-'*50}")

for i in range(min(10, len(X_test))):
    actual    = class_names[y_test_raw[i]]
    pred_idx  = predicted_cls[i]
    pred_name = class_names[pred_idx]
    conf      = float(final_preds[i, pred_idx])
    ok        = "✅" if pred_idx == y_test_raw[i] else "❌"
    print(f"  {actual:>12}  {pred_name:>12}  {conf*100:>11.1f}%  {ok:>8}")

# ─── Visualize ────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Loss curve ────────────────────────────────────────────────────────────
    axes[0].plot(history["loss"], color="royalblue", linewidth=1.5)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy curves ───────────────────────────────────────────────────────
    axes[1].plot(history["acc"],     label="Train", color="royalblue", linewidth=1.5)
    axes[1].plot(history["val_acc"], label="Test",  color="tomato",    linewidth=1.5)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    axes[1].set_title("Accuracy Over Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Confusion Matrix heatmap ──────────────────────────────────────────────
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, pred in zip(y_test_raw, predicted_cls):
        conf_matrix[actual, pred] += 1

    im = axes[2].imshow(conf_matrix, cmap="Blues")
    axes[2].set_xticks(range(n_classes))
    axes[2].set_yticks(range(n_classes))
    axes[2].set_xticklabels(class_names, rotation=20)
    axes[2].set_yticklabels(class_names)
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_title("Confusion Matrix")
    for i in range(n_classes):
        for j in range(n_classes):
            axes[2].text(j, i, conf_matrix[i, j],
                         ha='center', va='center',
                         color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black',
                         fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig("iris_results.png", dpi=150)
    print("\n📊 Results saved to iris_results.png")
    plt.show()

except ImportError:
    print("\n(Install matplotlib for plots: pip install matplotlib)")
