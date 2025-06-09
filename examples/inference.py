"""
Inference script for the saved neural network model.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import NeuralNetwork, generate_data
import numpy as np
import pickle


def load_normalization_params(filepath):
    """Load normalization parameters."""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    return params['mean'], params['std']


def normalize_input(X, mean, std):
    """Normalize input using saved parameters."""
    return (X - mean) / std


def main():
    """Main inference function."""
    print("=" * 60)
    print("Neural Network Inference")
    print("=" * 60)
    
    # Model paths
    model_path = "models/trained_model.pkl"
    norm_params_path = "models/normalization_params.pkl"
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'python examples/train_model.py' first to train and save a model.")
        return
    
    if not os.path.exists(norm_params_path):
        print(f"Error: Normalization parameters not found at {norm_params_path}")
        print("Please run 'python examples/train_model.py' first.")
        return
    
    # Load the trained model
    print("Loading trained model...")
    model = NeuralNetwork()
    model.load_model(model_path)
    
    # Load normalization parameters
    print("Loading normalization parameters...")
    mean, std = load_normalization_params(norm_params_path)
    print(f"Normalization - Mean: {mean}, Std: {std}")
    print()
    
    # Print model summary
    model.summary()
    print()
    
    # Generate some test data for demonstration
    print("Generating test data for demonstration...")
    X_demo, y_demo = generate_data(num_samples=10, num_features=2, random_seed=123)
    
    # Normalize the demo data
    X_demo_norm = normalize_input(X_demo, mean, std)
    
    # Make predictions
    print("Making predictions on demo data...")
    print("-" * 40)
    predictions = model.predict(X_demo_norm)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Display results
    print(f"{'Input':<20} {'True Label':<12} {'Predicted':<12} {'Confidence':<12} {'Correct':<8}")
    print("-" * 70)
    
    for i in range(len(X_demo)):
        input_str = f"[{X_demo[i,0]:.2f}, {X_demo[i,1]:.2f}]"
        true_label = y_demo[i,0]
        pred_label = binary_predictions[i,0]
        confidence = predictions[i,0]
        is_correct = "✓" if pred_label == true_label else "✗"
        
        print(f"{input_str:<20} {true_label:<12} {pred_label:<12} {confidence:.4f}{'':8} {is_correct:<8}")
    
    print("-" * 70)
    
    # Calculate accuracy
    accuracy = np.mean(binary_predictions.flatten() == y_demo.flatten())
    print(f"Demo Data Accuracy: {accuracy*100:.2f}%")
    print()
    
    # Interactive prediction mode
    print("Interactive Prediction Mode")
    print("Enter two numbers separated by space (or 'quit' to exit):")
    print("Example: 1.5 -2.3")
    print()
    
    while True:
        try:
            user_input = input("Enter input values: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse input
            values = list(map(float, user_input.split()))
            
            if len(values) != 2:
                print("Please enter exactly 2 numbers separated by space.")
                continue
            
            # Create input array and normalize
            user_input_array = np.array([values])
            user_input_norm = normalize_input(user_input_array, mean, std)
            
            # Make prediction
            prediction = model.predict(user_input_norm)
            binary_pred = (prediction > 0.5).astype(int)
            
            print(f"Input: [{values[0]:.2f}, {values[1]:.2f}]")
            print(f"Normalized: [{user_input_norm[0,0]:.4f}, {user_input_norm[0,1]:.4f}]")
            print(f"Raw prediction: {prediction[0,0]:.4f}")
            print(f"Binary prediction: {binary_pred[0,0]}")
            print(f"Confidence: {prediction[0,0] if binary_pred[0,0] == 1 else 1-prediction[0,0]:.4f}")
            print()
            
        except ValueError:
            print("Invalid input. Please enter two numbers separated by space.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Inference completed!")


if __name__ == "__main__":
    main()