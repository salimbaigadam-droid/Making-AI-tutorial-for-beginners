# 🧪 Examples AI Engine with Pybind11 & Cython

Each example is self-contained and runs **without needing to compile C++** first.
The `pure_python_fallback.py` module automatically kicks in if the C++ extension
isn't built yet, using the **exact same API** so your code is identical either way.

---

## ▶️ Quick Start

```bash
# Install Python dependencies
pip install numpy matplotlib scikit-learn

# Run any example directly
python examples/01_xor_problem.py
python examples/02_sine_regression.py
python examples/03_iris_classifier.py
python examples/04_speed_benchmark.py
python examples/05_save_and_load.py
```

---

## 📁 Example Overview

| File | Topic | Difficulty |
|------|-------|------------|
| `pure_python_fallback.py` | NumPy fallback, same API as C++ | — |
| `01_xor_problem.py` | XOR gate (hello world of NNs) | ⭐ Beginner |
| `02_sine_regression.py` | Sine wave approximation | ⭐⭐ Intermediate |
| `03_iris_classifier.py` | 3-class flower classification | ⭐⭐ Intermediate |
| `04_speed_benchmark.py` | Python vs C++ speed comparison | ⭐⭐⭐ Advanced |
| `05_save_and_load.py` | Save weights, load & deploy | ⭐⭐ Intermediate |

---

## 🔄 Switching from Python → C++ backend

Every example uses `load_engine()` to pick the best available backend:

```python
from examples.pure_python_fallback import load_engine

# Tries pybind11 first, then cython, then falls back to pure Python
ai = load_engine("pybind11")

model = ai.NeuralNetwork([
    ai.LayerConfig(2, 8, ai.Activation.RELU),
    ai.LayerConfig(8, 1, ai.Activation.SIGMOID),
], learning_rate=0.1)
```

Once you build the C++ extension (`pip install -e .`), the **same code**
automatically runs at full C++ speed — no changes needed.

---

## 🔬 What Each Example Teaches

### 01 — XOR Problem
- Why simple networks can't solve XOR (linear separability)
- How hidden layers create non-linear decision boundaries
- Binary classification with Sigmoid output
- Plotting a learning curve

### 02 — Sine Regression  
- Regression (predicting continuous values)
- Why Tanh is better than ReLU for smooth functions
- Mini-batch training to avoid overfitting
- Train/validation split and generalization

### 03 — Iris Classifier
- Multi-class classification with one-hot labels
- Feature normalization (zero mean, unit variance)
- Confusion matrix and per-class accuracy
- Real-world dataset workflow

### 04 — Speed Benchmark
- Measuring throughput (steps/second)
- Proper benchmarking: warmup rounds, perf_counter
- Visualizing speedup with bar charts
- Why C++ is 20–100x faster than Python loops

### 05 — Save & Load
- Saving model weights to disk
- Loading weights into a fresh model
- Verifying bit-exact reproduction
- Production inference on new data

---

## 📊 Expected Output (Example 01)

```
⚡ Using C++ Pybind11 backend
==================================================
  EXAMPLE 1: XOR Gate Classifier
==================================================

Model: NeuralNetwork([2→8, 8→8, 8→1], lr=0.1)

Training...

  Epoch   500/5000 | Loss: 0.243871
  Epoch  1000/5000 | Loss: 0.187342
  Epoch  1500/5000 | Loss: 0.089234
  Epoch  2000/5000 | Loss: 0.024512
  Epoch  2500/5000 | Loss: 0.006123
  Epoch  3000/5000 | Loss: 0.002341
  Epoch  3500/5000 | Loss: 0.001245
  Epoch  4000/5000 | Loss: 0.000834
  Epoch  4500/5000 | Loss: 0.000612
  Epoch  5000/5000 | Loss: 0.000489

XOR Predictions (threshold = 0.5):
  [0, 0]  →  0.0231  →  0  (expected 0) ✅
  [0, 1]  →  0.9734  →  1  (expected 1) ✅
  [1, 0]  →  0.9712  →  1  (expected 1) ✅
  [1, 1]  →  0.0198  →  0  (expected 0) ✅

Accuracy: 4/4 = 100%
📊 Learning curve saved to xor_training_curve.png
```

---

## 🛠 Troubleshooting

**`ModuleNotFoundError: pure_python_fallback`**  
→ Run examples from the project root: `python examples/01_xor_problem.py`

**`MemoryError` or slow on large batches**  
→ Reduce `BATCH_SIZE` in the example file

**Plots not showing**  
→ Install matplotlib: `pip install matplotlib`

**C++ extension not loading**  
→ Build it first: `pip install -e .` (see main README for C++ setup)
