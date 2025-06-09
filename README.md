# Neural Network from Scratch

A complete implementation of a neural network built from scratch using only NumPy, with save/load functionality for model persistence.

## Features

- **Pure NumPy Implementation**: No external deep learning frameworks required
- **Modular Architecture**: Clean separation of concerns with individual modules for layers, activations, and losses
- **Save/Load Functionality**: Persist trained models and load them for inference
- **Batch Normalization**: Includes batch normalization for improved training stability
- **Multiple Activation Functions**: ReLU and Sigmoid activations
- **Binary Classification**: Optimized for binary classification tasks with binary cross-entropy loss
- **Mini-batch Training**: Efficient training with configurable batch sizes
- **Comprehensive Testing**: Unit tests for all major components

## Project Structure

```
neural_network_project/
├── src/                    # Source code
│   ├── __init__.py
│   ├── activations.py      # Activation functions (ReLU, Sigmoid)
│   ├── layers.py           # Neural network layers (Dense, BatchNorm)
│   ├── losses.py           # Loss functions (Binary Cross-Entropy)
│   ├── model.py            # Main neural network model
│   └── utils.py            # Utility functions for data handling
├── models/                 # Directory for saved models
├── examples/               # Example scripts
│   ├── train_model.py      # Training script
│   └── inference.py        # Inference script
├── tests/                  # Unit tests
│   └── test_model.py
├── requirements.txt        # Dependencies
└── README.md
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

```bash
cd neural_network_project
python examples/train_model.py
```

This will:
- Generate synthetic binary classification data
- Build a neural network with dense layers, batch normalization, and ReLU activations
- Train the model for 100 epochs
- Save the trained model to `models/trained_model.pkl`
- Save normalization parameters to `models/normalization_params.pkl`

### 2. Use the Trained Model for Inference

```bash
python examples/inference.py
```

This will:
- Load the saved model and normalization parameters
- Demonstrate predictions on sample data
- Provide an interactive mode for making predictions on custom inputs

## Usage Examples

### Building a Custom Model

```python
from src import NeuralNetwork, DenseLayer, BatchNormLayer, ReLU, Sigmoid

# Create model
model = NeuralNetwork()

# Add layers
model.add_layer(DenseLayer(input_size=2, output_size=16))
model.add_layer(BatchNormLayer(input_shape=16))
model.add_layer(ReLU())
model.add_layer(DenseLayer(16, 8))
model.add_layer(BatchNormLayer(input_shape=8))
model.add_layer(ReLU())
model.add_layer(DenseLayer(8, 1))
model.add_layer(Sigmoid())

# Train model
model.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)

# Save model
model.save_model("my_model.pkl")
```

### Loading and Using a Saved Model

```python
from src import NeuralNetwork
import numpy as np

# Load model
model = NeuralNetwork()
model.load_model("my_model.pkl")

# Make predictions
predictions = model.predict(X_new)
binary_predictions = (predictions > 0.5).astype(int)
```

### Data Generation and Preprocessing

```python
from src import generate_data, train_test_split, normalize_data

# Generate synthetic data
X, y = generate_data(num_samples=1000, num_features=2, random_seed=42)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize data
X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
```

## Model Architecture

The default model architecture includes:

1. **Input Layer**: 2 features
2. **Hidden Layer 1**: 16 neurons with batch normalization and ReLU activation
3. **Hidden Layer 2**: 8 neurons with batch normalization and ReLU activation
4. **Output Layer**: 1 neuron with sigmoid activation (for binary classification)

## Components

### Layers
- **DenseLayer**: Fully connected layer with weights and biases
- **BatchNormLayer**: Batch normalization for training stability

### Activations
- **ReLU**: Rectified Linear Unit activation
- **Sigmoid**: Sigmoid activation for binary classification output

### Loss Functions
- **BinaryCrossEntropy**: For binary classification tasks

### Optimizers
- Simple gradient descent with configurable learning rate

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python tests/test_model.py
```

## Key Features Explained

### Save/Load Functionality

The model can save and load its complete state, including:
- All layer weights and biases
- Batch normalization running statistics (mean and variance)
- Model architecture information

### Batch Normalization

Includes proper handling of training vs. inference modes:
- **Training mode**: Uses batch statistics and updates running averages
- **Inference mode**: Uses saved running statistics for consistent predictions

### Data Preprocessing

Includes utilities for:
- Synthetic data generation for testing
- Train/test splitting with reproducible random seeds
- Data normalization with proper handling of test data

## Customization

### Adding New Activation Functions

```python
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, d_output):
        return d_output * (1 - self.output**2)
```

### Adding New Layer Types

```python
class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, x):
        if self.training:
            self.mask = np.random.binomial(1, 1-self.dropout_rate, x.shape) / (1-self.dropout_rate)
            return x * self.mask
        return x
    
    def backward(self, d_output):
        return d_output * self.mask if self.training else d_output
```

## Performance Notes

- The implementation prioritizes clarity and educational value over performance
- For production use, consider frameworks like TensorFlow or PyTorch
- Training time depends on data size, model complexity, and number of epochs

## Dependencies

- NumPy >= 1.20.0
- Python >= 3.6

Optional:
- Matplotlib >= 3.3.0 (for visualization)
- Pandas >= 1.3.0 (for data handling)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Model Inspection

### `read_model_weights.py`

This script allows you to inspect the weights, biases, and architecture details of a saved model (`.pkl` file). It can print the model structure and parameters to the console in a human-readable format or save them as a JSON file.

**Usage:**

```bash
python read_model_weights.py <path_to_model.pkl> [options]
```

**Arguments & Options:**

*   `<path_to_model.pkl>`: (Required) Path to the saved model file (e.g., `models/trained_model.pkl`).
*   `--output <filename.json>`: (Optional) If provided, saves the model details to the specified JSON file.
*   `--pretty`: (Optional) If specified, pretty-prints the JSON output to the console or file, making it more readable.

**Examples:**

1.  Print model details to the console (pretty-printed):
    ```bash
    python read_model_weights.py models/trained_model.pkl --pretty
    ```

2.  Save model details to a JSON file (pretty-printed):
    ```bash
    python read_model_weights.py models/trained_model.pkl --output model_details.json --pretty
    ```

## Future Enhancements

Potential improvements could include:
- Additional layer types (Convolutional, LSTM)
- More optimization algorithms (Adam, RMSprop)
- Learning rate scheduling
- Data augmentation utilities
- Visualization tools for training progress
- Export to other formats (ONNX, etc.)