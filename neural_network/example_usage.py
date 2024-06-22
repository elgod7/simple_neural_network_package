from neural_network.train import train_network
from neural_network.test import test_network
import numpy as np

from neural_network.utils import reset_parameters

# Example usage with XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the neural network layers: input layer, hidden layers, and output layer
layers = [2, 4, 1]  # 2 inputs, 1 hidden layer with 4 neurons, 1 output

# Train the network
nn = train_network(X, y, layers, epochs=20000, learning_rate=0.5, save_prefix='xor_model')

# Test the network
predicted_output, predicted_output_thresholded = test_network(X, layers, load_prefix='xor_model')

print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", predicted_output)
print("Predicted Output:\n", predicted_output_thresholded)

# Reset the saved model parameters
#reset_parameters(layers, 'xor_model')

# Confirm reset by attempting to test again (this should fail if reset was successful)
try:
    predicted_output_after_reset = test_network(X, layers, load_prefix='xor_model')
    print("Predicted Output After Reset:\n", predicted_output_after_reset)
except FileNotFoundError as e:
    print("Model parameters have been reset. Load failed as expected:", e)