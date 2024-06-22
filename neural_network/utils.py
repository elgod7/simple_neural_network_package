import os
import numpy as np

def normalize_data(X, min_val=0, max_val=1):
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_norm * (max_val - min_val) + min_val

def apply_threshold(output, threshold=0.5):
    return (output >= threshold).astype(int)

def save_parameters(weights, biases, prefix):
    for i, (w, b) in enumerate(zip(weights, biases)):
        np.save(f"{prefix}_weights_{i}.npy", w)
        np.save(f"{prefix}_biases_{i}.npy", b)

def load_parameters(layers, prefix):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.load(f"{prefix}_weights_{i}.npy"))
        biases.append(np.load(f"{prefix}_biases_{i}.npy"))
    return weights, biases

def reset_parameters(layers, prefix):
    for i in range(len(layers) - 1):
        weight_file = f"{prefix}_weights_{i}.npy"
        bias_file = f"{prefix}_biases_{i}.npy"
        if os.path.exists(weight_file):
            os.remove(weight_file)
        if os.path.exists(bias_file):
            os.remove(bias_file)