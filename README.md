# Neural Network Package

This package provides a simple neural network implementation for training and testing models.

## Installation

To install the package locally, navigate to the project directory and run:

```bash
pip install -e .
```

## Usage

Here's an example of how to use the package:

```python
Copy code
from neural_network import train_network, test_network, reset_parameters
import numpy as np

# Example data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the neural network layers
layers = [2, 4, 1]

# Train the network
nn = train_network(X, y, layers, epochs=20000, learning_rate=0.5, save_prefix='model')

# Test the network
predicted_output = test_network(X, layers, load_prefix='model')

print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", predicted_output)

# Reset the saved model parameters
reset_parameters(layers, 'model')

```

## License

This project is licensed under the MIT License.
