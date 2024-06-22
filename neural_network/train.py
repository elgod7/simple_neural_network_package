import numpy as np
from .model import NeuralNetwork
from .utils import save_parameters, normalize_data

def train_network(X, y, layers, epochs=20000, learning_rate=0.5, save_prefix=None, normalize=True):
    if normalize:
        X = normalize_data(X)
        y = normalize_data(y)
        
    nn = NeuralNetwork(layers)
    for epoch in range(epochs):
        activations = nn.forward(X)
        nn.backward(X, y, activations, learning_rate)
        
        if epoch % 1000 == 0:
            loss = np.mean(np.square(y - activations[-1]))
            print(f'Epoch {epoch}, Loss: {loss}')

    if save_prefix:
        save_parameters(nn.weights, nn.biases, save_prefix)
    return nn
