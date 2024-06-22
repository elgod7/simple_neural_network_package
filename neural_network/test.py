from .model import NeuralNetwork
from .utils import load_parameters, apply_threshold, normalize_data

def test_network(X, layers, load_prefix, threshold=0.5, normalize=True):
    if normalize:
        X = normalize_data(X)
        
    weights, biases = load_parameters(layers, load_prefix)
    nn = NeuralNetwork(layers)
    nn.weights = weights
    nn.biases = biases

    output = nn.predict(X)
    predicted_output = apply_threshold(output, threshold)
    return output, predicted_output
