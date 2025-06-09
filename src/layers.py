"""
Neural network layers with save/load functionality.
"""
import numpy as np


class DenseLayer:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_output):
        self.d_weights = np.dot(self.input.T, d_output)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.weights.T)
        return d_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
    
    def get_params(self):
        """Get layer parameters for saving."""
        return {
            'type': 'DenseLayer',
            'input_size': self.input_size,
            'output_size': self.output_size,
            'weights': self.weights,
            'biases': self.biases
        }
    
    def set_params(self, params):
        """Set layer parameters from loaded data."""
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.weights = params['weights']
        self.biases = params['biases']


class BatchNormLayer:
    """Batch normalization layer."""
    
    def __init__(self, input_shape, epsilon=1e-5, momentum=0.9):
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones(input_shape)  # Learnable scale parameter
        self.beta = np.zeros(input_shape)  # Learnable shift parameter

        # For training: running mean and variance
        self.running_mean = np.zeros(input_shape)
        self.running_var = np.ones(input_shape)

        # Cache for backward pass
        self.x_norm = None
        self.x_minus_mean = None
        self.std_inv = None
        self.input_shape_cache = None
        self.batch_size = None
        self.mean = None
        self.var = None
        self.is_training = True # Flag to switch between training and inference mode

    def forward(self, x):
        # x shape: (N, D) where N is batch size, D is number of features
        self.input_shape_cache = x.shape
        self.batch_size = x.shape[0]

        if self.is_training:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0) # Using variance directly, np.std is sqrt(var)

            # Update running mean and variance for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            # Normalize
            self.x_minus_mean = x - self.mean
            self.std_inv = 1. / np.sqrt(self.var + self.epsilon)
            self.x_norm = self.x_minus_mean * self.std_inv
        else:
            # Inference mode: use running mean and variance
            x_minus_running_mean = x - self.running_mean
            std_inv_running = 1. / np.sqrt(self.running_var + self.epsilon)
            self.x_norm = x_minus_running_mean * std_inv_running
            # Store values needed for potential (though less common) backward pass in eval mode
            self.x_minus_mean = x_minus_running_mean
            self.std_inv = std_inv_running

        # Scale and shift
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, d_output):
        # d_output shape: (N, D)

        # Gradients for gamma and beta
        self.d_gamma = np.sum(d_output * self.x_norm, axis=0)
        self.d_beta = np.sum(d_output, axis=0)

        # Gradient for x_norm
        d_x_norm = d_output * self.gamma

        # Gradient for variance (intermediate step)
        d_std_inv = np.sum(d_x_norm * self.x_minus_mean, axis=0)
        d_var = d_std_inv * (-0.5) * (self.var + self.epsilon)**(-1.5)

        # Gradient for mean (intermediate step)
        d_x_minus_mean_term1 = d_x_norm * self.std_inv
        d_x_minus_mean_term2 = (2.0 / self.batch_size) * self.x_minus_mean * d_var

        d_x_minus_mean = d_x_minus_mean_term1 + d_x_minus_mean_term2
        d_mean = -1 * np.sum(d_x_minus_mean, axis=0)

        # Gradient for input x
        d_input = d_x_minus_mean + (1.0/self.batch_size) * d_mean

        return d_input

    def update(self, learning_rate):
        self.gamma -= learning_rate * self.d_gamma
        self.beta -= learning_rate * self.d_beta
    
    def get_params(self):
        """Get layer parameters for saving."""
        return {
            'type': 'BatchNormLayer',
            'input_shape': self.input_shape,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'gamma': self.gamma,
            'beta': self.beta,
            'running_mean': self.running_mean,
            'running_var': self.running_var
        }
    
    def set_params(self, params):
        """Set layer parameters from loaded data."""
        self.input_shape = params['input_shape']
        self.epsilon = params['epsilon']
        self.momentum = params['momentum']
        self.gamma = params['gamma']
        self.beta = params['beta']
        self.running_mean = params['running_mean']
        self.running_var = params['running_var']