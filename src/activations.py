"""
Activation functions for the neural network.
"""
import numpy as np


class ReLU:
    """Rectified Linear Unit activation function."""
    
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_output):
        return d_output * (self.input > 0)


class Sigmoid:
    """Sigmoid activation function."""
    
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x_clipped))
        return self.output

    def backward(self, d_output):
        return d_output * self.output * (1 - self.output)