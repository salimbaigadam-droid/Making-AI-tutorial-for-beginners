"""
pure_python_fallback.py
========================
A pure NumPy implementation of the C++ AI engine.
It mirrors the EXACT same API as ai_engine_pybind / ai_engine_cython,
so ALL the example scripts work even before you compile the C++ extension.

This is also a great way to understand what the C++ code is doing.
"""

import numpy as np
from enum import IntEnum


# ─── Activation Enum (matches C++ enum) ───────────────────────────────────────
class Activation(IntEnum):
    SIGMOID = 0
    RELU    = 1
    TANH    = 2
    LINEAR  = 3


# ─── Activation functions ─────────────────────────────────────────────────────
def _sigmoid(x):      return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def _sigmoid_d(x):    s = _sigmoid(x); return s * (1 - s)
def _relu(x):         return np.maximum(0, x)
def _relu_d(x):       return (x > 0).astype(np.float64)
def _tanh(x):         return np.tanh(x)
def _tanh_d(x):       return 1 - np.tanh(x) ** 2
def _linear(x):       return x
def _linear_d(x):     return np.ones_like(x)

_ACT  = {Activation.SIGMOID: _sigmoid, Activation.RELU: _relu,
          Activation.TANH: _tanh,       Activation.LINEAR: _linear}
_ACT_D = {Activation.SIGMOID: _sigmoid_d, Activation.RELU: _relu_d,
           Activation.TANH: _tanh_d,       Activation.LINEAR: _linear_d}


# ─── LayerConfig (matches C++ struct) ─────────────────────────────────────────
class LayerConfig:
    def __init__(self, input_size: int, output_size: int,
                 activation: Activation = Activation.RELU):
        self.input_size  = input_size
        self.output_size = output_size
        self.activation  = activation

    def __repr__(self):
        return (f"LayerConfig({self.input_size} → {self.output_size}, "
                f"act={self.activation.name})")


# ─── NeuralNetwork (matches C++ class + Pybind11 API) ─────────────────────────
class NeuralNetwork:
    """
    Multi-Layer Perceptron with the same API as the C++ Pybind11 module.
    Swap this class out for `ai_engine_pybind.NeuralNetwork` for a speed boost.
    """

    def __init__(self, layers: list, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self._activations  = [l.activation for l in layers]
        self._weights = []
        self._biases  = []

        for layer in layers:
            # Xavier / He initialization
            scale = np.sqrt(2.0 / layer.input_size)
            W = np.random.randn(layer.input_size, layer.output_size) * scale
            b = np.zeros((1, layer.output_size))
            self._weights.append(W)
            self._biases.append(b)

        self._z_cache = []   # Pre-activation values
        self._a_cache = []   # Post-activation values

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run input through the network, return output array."""
        self._z_cache = []
        self._a_cache = [X]
        current = X

        for W, b, act in zip(self._weights, self._biases, self._activations):
            z = current @ W + b
            a = _ACT[act](z)
            self._z_cache.append(z)
            self._a_cache.append(a)
            current = a

        return current

    # ── Backward Pass ─────────────────────────────────────────────────────────
    def backward(self, X: np.ndarray, y: np.ndarray):
        """Backpropagation — updates weights and biases."""
        n = X.shape[0]
        output = self._a_cache[-1]
        delta  = output - y   # dL/da for MSE

        for i in reversed(range(len(self._weights))):
            act_d  = _ACT_D[self._activations[i]](self._z_cache[i])
            delta_z = delta * act_d

            dW = self._a_cache[i].T @ delta_z / n
            db = delta_z.mean(axis=0, keepdims=True)

            if i > 0:
                delta = delta_z @ self._weights[i].T

            self._weights[i] -= self.learning_rate * dW
            self._biases[i]  -= self.learning_rate * db

    # ── Compute Loss ──────────────────────────────────────────────────────────
    def compute_loss(self, output: np.ndarray, target: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((output - target) ** 2))

    # ── Train Step ────────────────────────────────────────────────────────────
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One forward + backward pass. Returns the loss."""
        output = self.forward(X)
        loss   = self.compute_loss(output, y)
        self.backward(X, y)
        return loss

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference (no gradient tracking)."""
        return self.forward(X)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save_weights(self, path: str):
        """Save weights to a .npz file."""
        data = {}
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            data[f"W{i}"] = W
            data[f"b{i}"] = b
        np.savez(path, **data)
        print(f"✅ Weights saved to {path}.npz")

    def load_weights(self, path: str):
        """Load weights from a .npz file."""
        data = np.load(path)
        for i in range(len(self._weights)):
            self._weights[i] = data[f"W{i}"]
            self._biases[i]  = data[f"b{i}"]
        print(f"✅ Weights loaded from {path}")

    def __repr__(self):
        shapes = [f"{W.shape[0]}→{W.shape[1]}" for W in self._weights]
        return f"NeuralNetwork([{', '.join(shapes)}], lr={self.learning_rate})"


# ─── Smart importer ───────────────────────────────────────────────────────────
def load_engine(prefer: str = "pybind11"):
    """
    Try to import the fast C++ engine, fall back to pure Python.
    
    Args:
        prefer: "pybind11" | "cython"
    Returns:
        module with NeuralNetwork, LayerConfig, Activation
    """
    import types, sys

    if prefer == "pybind11":
        try:
            import ai_engine_pybind as m
            print("⚡ Using C++ Pybind11 backend")
            return m
        except ImportError:
            pass

    if prefer == "cython":
        try:
            import ai_engine_cython as m
            print("⚡ Using C++ Cython backend")
            return m
        except ImportError:
            pass

    print("🐍 Using pure Python fallback (build C++ extension for 100x speedup!)")
    # Return this module as a fake "engine module"
    m = types.ModuleType("ai_engine_fallback")
    m.NeuralNetwork = NeuralNetwork
    m.LayerConfig   = LayerConfig
    m.Activation    = Activation
    return m
