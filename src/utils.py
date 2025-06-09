"""
Utility functions for data generation and preprocessing.
"""
import numpy as np


def generate_data(num_samples=200, num_features=2, noise_level=0.5, random_seed=None):
    """
    Generate synthetic binary classification data.
    
    Args:
        num_samples (int): Number of samples to generate
        num_features (int): Number of input features
        noise_level (float): Amount of noise to add
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Simple linearly separable data for binary classification
    X = np.random.rand(num_samples, num_features) * 10 - 5
    # Create a decision boundary (e.g., x1 - x2 > 0)
    y = (X[:, 0] - X[:, 1] > 0).astype(int).reshape(-1, 1)
    
    # Add some noise to make it a bit more challenging
    X += np.random.randn(num_samples, num_features) * noise_level
    
    return X, y


def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split data into training and testing sets.
    
    Args:
        X (np.array): Features
        y (np.array): Labels
        test_size (float): Proportion of data to use for testing
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_samples = X.shape[0]
    num_test = int(num_samples * test_size)
    
    # Random permutation
    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test=None):
    """
    Normalize data using mean and standard deviation from training set.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array, optional): Test features
    
    Returns:
        tuple: Normalized training and test sets (and normalization parameters)
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    X_train_norm = (X_train - mean) / std
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        return X_train_norm, X_test_norm, mean, std
    
    return X_train_norm, mean, std