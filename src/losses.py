"""
Loss functions for the neural network.
"""
import numpy as np


class BinaryCrossEntropy:
    """Binary cross-entropy loss function."""
    
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        # Clip predictions to avoid log(0)
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss

    def backward(self):
        epsilon = 1e-12
        y_pred_clipped = np.clip(self.y_pred, epsilon, 1. - epsilon)
        d_loss = - (self.y_true / y_pred_clipped - (1 - self.y_true) / (1 - y_pred_clipped)) / len(self.y_true)
        return d_loss