"""
Training script for the neural network model.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import NeuralNetwork, DenseLayer, BatchNormLayer, ReLU, Sigmoid
from src import generate_data, train_test_split, normalize_data
import numpy as np


def main():
    """Main training function."""
    print("=" * 60)
    print("Neural Network Training")
    print("=" * 60)
    
    # Hyperparameters
    input_features = 2
    hidden_units1 = 16
    hidden_units2 = 8
    output_units = 1
    learning_rate = 0.01
    epochs = 100
    batch_size = 32
    
    # Data generation parameters
    num_samples = 600
    test_size = 0.2
    random_seed = 42
    
    print(f"Generating {num_samples} samples with {input_features} features...")
    
    # Generate data
    X, y = generate_data(num_samples=num_samples, 
                        num_features=input_features, 
                        random_seed=random_seed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                       test_size=test_size, 
                                                       random_seed=random_seed)
    
    # Normalize data (optional - usually helps with training)
    X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print()
    
    # Build the model
    print("Building model architecture...")
    model = NeuralNetwork()
    model.add_layer(DenseLayer(input_features, hidden_units1))
    model.add_layer(BatchNormLayer(input_shape=hidden_units1))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(hidden_units1, hidden_units2))
    model.add_layer(BatchNormLayer(input_shape=hidden_units2))
    model.add_layer(ReLU())
    model.add_layer(DenseLayer(hidden_units2, output_units))
    model.add_layer(Sigmoid()) # Sigmoid for binary classification output
    
    # Print model summary
    model.summary()
    print()
    
    # Train the model
    print("Training the model...")
    print("-" * 40)
    model.train(X_train_norm, y_train, epochs, learning_rate, batch_size)
    print("-" * 40)
    print()
    
    # Evaluate the model
    print("Evaluating the model...")
    
    # Training accuracy
    train_accuracy, train_predictions = model.evaluate(X_train_norm, y_train)
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Test accuracy
    test_accuracy, test_predictions = model.evaluate(X_test_norm, y_test)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print()
    
    # Save the model
    model_path = "models/trained_model.pkl"
    print(f"Saving model to {model_path}...")
    model.save_model(model_path)
    
    # Save normalization parameters for inference
    norm_params_path = "models/normalization_params.pkl"
    import pickle
    with open(norm_params_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    print(f"Normalization parameters saved to {norm_params_path}")
    print()
    
    # Example of a single prediction
    print("Example prediction:")
    sample_input = np.array([[1.0, -2.0]]) # Example input
    sample_input_norm = (sample_input - mean) / std # Normalize
    single_prediction_raw = model.predict(sample_input_norm)
    single_prediction_class = (single_prediction_raw > 0.5).astype(int)
    print(f"Input: {sample_input[0]}")
    print(f"Raw output: {single_prediction_raw[0,0]:.4f}")
    print(f"Predicted class: {single_prediction_class[0,0]}")
    print()
    
    print("Training completed successfully!")
    print("You can now run 'python examples/inference.py' to test the saved model.")


if __name__ == "__main__":
    main()