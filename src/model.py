"""
Main neural network model with save/load functionality.
"""
import pickle
import os
import numpy as np
from .activations import ReLU, Sigmoid
from .layers import DenseLayer, BatchNormLayer
from .losses import BinaryCrossEntropy


class NeuralNetwork:
    """Neural network model with save/load capabilities."""
    
    def __init__(self):
        self.layers = []
        self.loss_function = BinaryCrossEntropy()
        self.model_config = {
            'version': '1.0',
            'layers': [],
            'loss_function': 'BinaryCrossEntropy'
        }

    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def predict(self, X):
        """Make predictions on input data."""
        output = X
        for layer in self.layers:
            # Set BatchNorm to inference mode for predictions
            if isinstance(layer, BatchNormLayer):
                layer.is_training = False
            output = layer.forward(output)
        return output

    def train(self, X_train, y_train, epochs, learning_rate, batch_size, verbose=True):
        """Train the neural network."""
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0
            # Mini-batch training
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                output = X_batch
                for layer in self.layers:
                    if isinstance(layer, BatchNormLayer): # Ensure BN is in training mode
                        layer.is_training = True
                    output = layer.forward(output)

                # Calculate loss
                loss = self.loss_function.forward(output, y_batch)
                epoch_loss += loss * X_batch.shape[0] # Weighted by batch size

                # Backward pass
                d_loss = self.loss_function.backward()
                for layer in reversed(self.layers):
                    d_loss = layer.backward(d_loss)

                # Update weights
                for layer in self.layers:
                    if hasattr(layer, 'update'):
                        layer.update(learning_rate)
            
            avg_epoch_loss = epoch_loss / num_samples
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

    def save_model(self, filepath):
        """Save the trained model to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'config': self.model_config,
            'layers': []
        }
        
        # Save layer parameters and types
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                # Layers with parameters (Dense, BatchNorm)
                model_data['layers'].append(layer.get_params())
            else:
                # Activation functions
                layer_info = {
                    'type': type(layer).__name__
                }
                model_data['layers'].append(layer_info)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Clear existing layers
        self.layers = []
        self.model_config = model_data['config']
        
        # Reconstruct layers
        for layer_data in model_data['layers']:
            layer_type = layer_data['type']
            
            if layer_type == 'DenseLayer':
                layer = DenseLayer(layer_data['input_size'], layer_data['output_size'])
                layer.set_params(layer_data)
                self.layers.append(layer)
            
            elif layer_type == 'BatchNormLayer':
                layer = BatchNormLayer(layer_data['input_shape'], 
                                     layer_data['epsilon'], 
                                     layer_data['momentum'])
                layer.set_params(layer_data)
                self.layers.append(layer)
            
            elif layer_type == 'ReLU':
                self.layers.append(ReLU())
            
            elif layer_type == 'Sigmoid':
                self.layers.append(Sigmoid())
            
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        print(f"Model loaded from {filepath}")

    def evaluate(self, X, y, threshold=0.5):
        """Evaluate the model on given data."""
        # Set all BatchNorm layers to inference mode
        for layer in self.layers:
            if isinstance(layer, BatchNormLayer):
                layer.is_training = False
        
        predictions = self.predict(X)
        binary_predictions = (predictions > threshold).astype(int)
        accuracy = np.mean(binary_predictions == y)
        return accuracy, predictions

    def summary(self):
        """Print a summary of the model architecture."""
        print("Model Summary:")
        print("=" * 50)
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            
            if isinstance(layer, DenseLayer):
                params = layer.weights.size + layer.biases.size
                total_params += params
                print(f"Layer {i+1}: {layer_name} ({layer.input_size} -> {layer.output_size}) - {params} parameters")
            
            elif isinstance(layer, BatchNormLayer):
                params = layer.gamma.size + layer.beta.size
                total_params += params
                print(f"Layer {i+1}: {layer_name} ({layer.input_shape} features) - {params} parameters")
            
            else:
                print(f"Layer {i+1}: {layer_name} - 0 parameters")
        
        print("=" * 50)
        print(f"Total parameters: {total_params}")
        print("=" * 50)