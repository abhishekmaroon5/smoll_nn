# src/__init__.py
"""
Neural Network Package
"""
from .model import NeuralNetwork
from .layers import DenseLayer, BatchNormLayer
from .activations import ReLU, Sigmoid
from .losses import BinaryCrossEntropy
from .utils import generate_data, train_test_split, normalize_data

__all__ = [
    'NeuralNetwork',
    'DenseLayer', 
    'BatchNormLayer',
    'ReLU', 
    'Sigmoid',
    'BinaryCrossEntropy',
    'generate_data',
    'train_test_split',
    'normalize_data'
]

# examples/__init__.py
"""
Example scripts for training and inference
"""

# tests/__init__.py
"""
Test suite for the neural network package
"""