"""
Unit tests for the neural network implementation.
"""
import sys
import os
import tempfile
import unittest
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import NeuralNetwork, DenseLayer, BatchNormLayer, ReLU, Sigmoid
from src import generate_data, train_test_split, normalize_data


class TestNeuralNetwork(unittest.TestCase):
    """Test cases for the neural network implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = NeuralNetwork()
        self.X, self.y = generate_data(num_samples=100, random_seed=42)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertEqual(len(self.model.layers), 0)
        self.assertIsNotNone(self.model.loss_function)
    
    def test_add_layers(self):
        """Test adding layers to the model."""
        self.model.add_layer(DenseLayer(2, 4))
        self.model.add_layer(ReLU())
        self.model.add_layer(DenseLayer(4, 1))
        self.model.add_layer(Sigmoid())
        
        self.assertEqual(len(self.model.layers), 4)
        self.assertIsInstance(self.model.layers[0], DenseLayer)
        self.assertIsInstance(self.model.layers[1], ReLU)
        self.assertIsInstance(self.model.layers[2], DenseLayer)
        self.assertIsInstance(self.model.layers[3], Sigmoid)
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        self.model.add_layer(DenseLayer(2, 4))
        self.model.add_layer(ReLU())
        self.model.add_layer(DenseLayer(4, 1))
        self.model.add_layer(Sigmoid())
        
        output = self.model.predict(self.X[:5])
        
        self.assertEqual(output.shape, (5, 1))
        # Sigmoid output should be between 0 and 1
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
    
    def test_training(self):
        """Test basic training functionality."""
        self.model.add_layer(DenseLayer(2, 4))
        self.model.add_layer(ReLU())
        self.model.add_layer(DenseLayer(4, 1))
        self.model.add_layer(Sigmoid())
        
        # Train for a few epochs
        initial_weights = self.model.layers[0].weights.copy()
        self.model.train(self.X, self.y, epochs=5, learning_rate=0.01, batch_size=32, verbose=False)
        final_weights = self.model.layers[0].weights
        
        # Weights should have changed
        self.assertFalse(np.array_equal(initial_weights, final_weights))
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Build and train a simple model
        self.model.add_layer(DenseLayer(2, 4))
        self.model.add_layer(ReLU())
        self.model.add_layer(DenseLayer(4, 1))
        self.model.add_layer(Sigmoid())
        
        self.model.train(self.X, self.y, epochs=5, learning_rate=0.01, batch_size=32, verbose=False)
        
        # Get predictions before saving
        predictions_before = self.model.predict(self.X[:10])
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            self.model.save_model(temp_path)
            
            # Create new model and load
            new_model = NeuralNetwork()
            new_model.load_model(temp_path)
            
            # Get predictions after loading
            predictions_after = new_model.predict(self.X[:10])
            
            # Predictions should be identical
            np.testing.assert_array_almost_equal(predictions_before, predictions_after, decimal=6)
            
            # Model structure should be the same
            self.assertEqual(len(self.model.layers), len(new_model.layers))
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_batch_normalization(self):
        """Test batch normalization layer."""
        self.model.add_layer(DenseLayer(2, 4))
        self.model.add_layer(BatchNormLayer(4))
        self.model.add_layer(ReLU())
        self.model.add_layer(DenseLayer(4, 1))
        self.model.add_layer(Sigmoid())
        
        # Test training mode
        output_train = self.model.predict(self.X[:10])
        
        # Test inference mode
        for layer in self.model.layers:
            if isinstance(layer, BatchNormLayer):
                layer.is_training = False
        
        output_inference = self.model.predict(self.X[:10])
        
        # Outputs should be different (training vs inference mode)
        self.assertFalse(np.array_equal(output_train, output_inference))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_generate_data(self):
        """Test data generation."""
        X, y = generate_data(num_samples=100, num_features=2, random_seed=42)
        
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100, 1))
        self.assertTrue(np.all(np.isin(y, [0, 1])))
    
    def test_train_test_split(self):
        """Test train-test split."""
        X, y = generate_data(num_samples=100, random_seed=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
        
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
    
    def test_normalize_data(self):
        """Test data normalization."""
        X, _ = generate_data(num_samples=100, random_seed=42)
        X_train, X_test, _, _ = train_test_split(X, np.zeros((100, 1)), test_size=0.2, random_seed=42)
        
        X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
        
        # Training data should have mean~0 and std~1
        np.testing.assert_array_almost_equal(np.mean(X_train_norm, axis=0), [0, 0], decimal=10)
        np.testing.assert_array_almost_equal(np.std(X_train_norm, axis=0), [1, 1], decimal=10)
        
        # Test data should be normalized using training statistics
        self.assertEqual(X_test_norm.shape, X_test.shape)


class TestLayers(unittest.TestCase):
    """Test cases for individual layers."""
    
    def test_dense_layer(self):
        """Test dense layer functionality."""
        layer = DenseLayer(3, 5)
        X = np.random.randn(10, 3)
        
        # Test forward pass
        output = layer.forward(X)
        self.assertEqual(output.shape, (10, 5))
        
        # Test backward pass
        d_output = np.random.randn(10, 5)
        d_input = layer.backward(d_output)
        self.assertEqual(d_input.shape, (10, 3))
        
        # Test parameter updates
        old_weights = layer.weights.copy()
        layer.update(0.01)
        self.assertFalse(np.array_equal(old_weights, layer.weights))
    
    def test_activation_functions(self):
        """Test activation functions."""
        X = np.array([[-1, 0, 1, 2]])
        
        # Test ReLU
        relu = ReLU()
        relu_output = relu.forward(X)
        expected_relu = np.array([[0, 0, 1, 2]])
        np.testing.assert_array_equal(relu_output, expected_relu)
        
        # Test Sigmoid
        sigmoid = Sigmoid()
        sigmoid_output = sigmoid.forward(X)
        self.assertTrue(np.all(sigmoid_output >= 0) and np.all(sigmoid_output <= 1))


if __name__ == '__main__':
    unittest.main()