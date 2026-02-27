"""
examples/04_speed_benchmark.py
================================
🔴 EXAMPLE 4 — Speed Benchmark: Pure Python vs C++ via Pybind11

This benchmark compares the exact same training loop using:
  1. Pure Python / NumPy fallback
  2. C++ engine via Pybind11 (if built)
  3. C++ engine via Cython (if built)

Run it:
    python examples/04_speed_benchmark.py

Expected results (approximate, on a modern laptop):
  Pure Python:   ~320 steps/sec
  C++ Pybind11:  ~9,800 steps/sec  → ~30x faster
  C++ Cython:    ~7,500 steps/sec  → ~23x faster
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time


def build_model(engine, lr=0.001):
    """Build an identical model regardless of which backend we use."""
    return engine.NeuralNetwork([
        engine.LayerConfig(64, 128, engine.Activation.RELU),
        engine.LayerConfig(128, 64, engine.Activation.RELU),
        engine.LayerConfig(64,  32, engine.Activation.RELU),
        engine.LayerConfig(32,   1, engine.Activation.SIGMOID),
    ], learning_rate=lr)


def benchmark(model, X, y, n_steps=200, warmup=10):
    """
    Run n_steps of train_step and return timing stats.
    
    Returns:
        dict with keys: total_time, steps_per_sec, ms_per_step
    """
    # Warmup — first calls can be slower due to JIT / cache warming
    for _ in range(warmup):
        model.train_step(X[:16], y[:16])

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        model.train_step(X, y)
    elapsed = time.perf_counter() - start

    return {
        "total_time":    elapsed,
        "steps_per_sec": n_steps / elapsed,
        "ms_per_step":   elapsed / n_steps * 1000,
    }


# ─── Setup ────────────────────────────────────────────────────────────────────
np.random.seed(42)
BATCH_SIZE = 256
N_STEPS    = 200

X = np.random.randn(BATCH_SIZE, 64).astype(np.float64)
y = np.random.randint(0, 2, (BATCH_SIZE, 1)).astype(np.float64)

print("=" * 65)
print("  EXAMPLE 4: Speed Benchmark — Python vs C++")
print("=" * 65)
print(f"\n  Batch size:   {BATCH_SIZE}")
print(f"  Input dim:    64")
print(f"  Architecture: 64 → 128 → 64 → 32 → 1")
print(f"  Steps:        {N_STEPS}\n")

results = {}

# ─── Backend 1: Pure Python fallback ─────────────────────────────────────────
from examples.pure_python_fallback import NeuralNetwork, LayerConfig, Activation
import types

python_engine        = types.ModuleType("python_engine")
python_engine.NeuralNetwork = NeuralNetwork
python_engine.LayerConfig   = LayerConfig
python_engine.Activation    = Activation

print("⏳ Benchmarking pure Python / NumPy...")
py_model = build_model(python_engine)
r = benchmark(py_model, X, y, N_STEPS, warmup=5)
results["Pure Python (NumPy)"] = r
print(f"  ✅ {r['steps_per_sec']:.0f} steps/sec  ({r['ms_per_step']:.2f} ms/step)\n")

# ─── Backend 2: Pybind11 C++ ─────────────────────────────────────────────────
try:
    import ai_engine_pybind as pb
    print("⏳ Benchmarking C++ Pybind11 engine...")
    pb_model = build_model(pb)
    r = benchmark(pb_model, X, y, N_STEPS)
    results["C++ via Pybind11"] = r
    print(f"  ✅ {r['steps_per_sec']:.0f} steps/sec  ({r['ms_per_step']:.2f} ms/step)\n")
except ImportError:
    print("  ⚠️  Pybind11 extension not built yet.")
    print("     Run: pip install -e .  to build it.\n")

# ─── Backend 3: Cython C++ ────────────────────────────────────────────────────
try:
    import ai_engine_cython as cy
    import types

    cy_engine = types.ModuleType("cy_engine")
    cy_engine.NeuralNetwork = cy.PyNeuralNetwork
    cy_engine.LayerConfig   = lambda inp, out, act: (inp, out, int(act))
    cy_engine.Activation    = Activation  # reuse enum

    print("⏳ Benchmarking C++ Cython engine...")
    cy_model = cy.PyNeuralNetwork([
        (64, 128, 1),  # RELU = 1
        (128, 64, 1),
        (64,  32, 1),
        (32,   1, 0),  # SIGMOID = 0
    ], learning_rate=0.001)

    r = benchmark(cy_model, X, y, N_STEPS)
    results["C++ via Cython"] = r
    print(f"  ✅ {r['steps_per_sec']:.0f} steps/sec  ({r['ms_per_step']:.2f} ms/step)\n")
except ImportError:
    print("  ⚠️  Cython extension not built yet.")
    print("     Run: python setup.py build_ext --inplace  to build it.\n")

# ─── Results Table ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)

if not results:
    print("  No results to compare.")
else:
    baseline = results[list(results.keys())[0]]["ms_per_step"]

    print(f"\n  {'Backend':<25}  {'Steps/sec':>12}  {'ms/step':>10}  {'Speedup':>10}")
    print(f"  {'-'*62}")

    for name, r in results.items():
        speedup  = baseline / r["ms_per_step"]
        bar_len  = int(speedup * 3)
        bar      = "█" * min(bar_len, 40)
        print(f"  {name:<25}  {r['steps_per_sec']:>12.0f}  "
              f"{r['ms_per_step']:>9.2f}ms  {speedup:>9.1f}x  {bar}")

    print()

    if len(results) > 1:
        py_speed  = results["Pure Python (NumPy)"]["steps_per_sec"]
        cpp_name  = "C++ via Pybind11" if "C++ via Pybind11" in results else "C++ via Cython"
        if cpp_name in results:
            cpp_speed = results[cpp_name]["steps_per_sec"]
            factor    = cpp_speed / py_speed
            print(f"  🚀 C++ is {factor:.1f}x faster than pure Python on this workload!")

# ─── Visualize ────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    if len(results) >= 2:
        names   = list(results.keys())
        speeds  = [results[n]["steps_per_sec"] for n in names]
        colors  = ["#4C72B0", "#DD8452", "#55A868"][:len(names)]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, speeds, color=colors, edgecolor="white",
                      linewidth=1.2, zorder=3)

        for bar, speed in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(speeds) * 0.01,
                    f"{speed:,.0f}\nsteps/sec",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Training Steps per Second (higher = faster)")
        ax.set_title("Speed Comparison: Python vs C++ Backends")
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, max(speeds) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig("benchmark_results.png", dpi=150)
        print("\n📊 Benchmark chart saved to benchmark_results.png")
        plt.show()

except ImportError:
    pass
