import numpy as np

#sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

#define the NeuralNetwork Class

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self._initialize_weights()

# Xavier initialization for weights
    def _initialize_weights(self):
        np.random.seed(42) # For reproducibility
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1])*np.sqrt(1/self.layers[i]))
            self.biases.append(np.zeros((1, self.layers[i+1])))
    
    def forward(self, X):
        activations = [X]
        inputs = X
        for i in range(len(self.weights)):
            inputs = sigmoid(np.dot(inputs, self.weights[i]) + self.biases[i])
            activations.append(inputs)
        return activations
    
    def backward(self, X, y, activations, learning_rate):
        deltas = [y - activations[-1]]
        for i in reversed(range(len(self.weights)-1)):
            delta = deltas[-1].dot(self.weights[i+1].T)*sigmoid_derivative(activations[i+1])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i])*learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True)*learning_rate
    
    def predict(self, X):
        return self.forward(X)[-1]