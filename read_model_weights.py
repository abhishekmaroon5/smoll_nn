#!/usr/bin/env python3
"""
Script to read model weights from saved neural network and print them in JSON format.
"""

import pickle
import json
import numpy as np
import argparse
import os
import sys


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def load_model_weights(model_path):
    """Load model weights from saved pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_weights_info(model_data):
    """Extract weight information from model data."""
    weights_info = {
        "model_config": model_data.get("config", {}),
        "layers": []
    }
    
    layer_count = 0
    
    for i, layer_data in enumerate(model_data.get("layers", [])):
        layer_type = layer_data.get("type", "Unknown")
        layer_info = {
            "layer_index": i,
            "layer_type": layer_type
        }
        
        if layer_type == "DenseLayer":
            layer_count += 1
            weights = layer_data.get("weights")
            biases = layer_data.get("biases")
            
            layer_info.update({
                "layer_name": f"dense_{layer_count}",
                "input_size": layer_data.get("input_size"),
                "output_size": layer_data.get("output_size"),
                "weights_shape": list(weights.shape) if weights is not None else None,
                "biases_shape": list(biases.shape) if biases is not None else None,
                "weights": weights,
                "biases": biases,
                "num_parameters": (weights.size + biases.size) if (weights is not None and biases is not None) else 0
            })
            
        elif layer_type == "BatchNormLayer":
            gamma = layer_data.get("gamma")
            beta = layer_data.get("beta")
            running_mean = layer_data.get("running_mean")
            running_var = layer_data.get("running_var")
            
            layer_info.update({
                "layer_name": f"batch_norm_{i}",
                "input_shape": layer_data.get("input_shape"),
                "epsilon": layer_data.get("epsilon"),
                "momentum": layer_data.get("momentum"),
                "gamma_shape": list(gamma.shape) if gamma is not None else None,
                "beta_shape": list(beta.shape) if beta is not None else None,
                "gamma": gamma,
                "beta": beta,
                "running_mean": running_mean,
                "running_var": running_var,
                "num_parameters": (gamma.size + beta.size) if (gamma is not None and beta is not None) else 0
            })
            
        else:
            # Activation layers (ReLU, Sigmoid, etc.)
            layer_info.update({
                "layer_name": f"{layer_type.lower()}_{i}",
                "num_parameters": 0
            })
        
        weights_info["layers"].append(layer_info)
    
    return weights_info


def print_model_summary(weights_info):
    """Print a summary of the model structure."""
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = 0
    print(f"{'Layer':<15} {'Type':<15} {'Shape':<20} {'Parameters':<12}")
    print("-" * 60)
    
    for layer in weights_info["layers"]:
        layer_name = layer.get("layer_name", "unknown")
        layer_type = layer.get("layer_type", "unknown")
        num_params = layer.get("num_parameters", 0)
        total_params += num_params
        
        if layer_type == "DenseLayer":
            shape_str = f"{layer['input_size']} -> {layer['output_size']}"
        elif layer_type == "BatchNormLayer":
            shape_str = f"({layer['input_shape']} features)"
        else:
            shape_str = "-"
        
        print(f"{layer_name:<15} {layer_type:<15} {shape_str:<20} {num_params:<12}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {'':15} {'':20} {total_params:<12}")
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Read and display neural network model weights")
    parser.add_argument("model_path", help="Path to the saved model file (.pkl)")
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")
    parser.add_argument("--summary", "-s", action="store_true", help="Show model summary only")
    parser.add_argument("--layer", "-l", type=int, help="Show specific layer weights (layer index)")
    parser.add_argument("--no-weights", action="store_true", help="Exclude actual weight values from JSON")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print JSON with indentation")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist")
        sys.exit(1)
    
    # Load model data
    print(f"Loading model from: {args.model_path}")
    model_data = load_model_weights(args.model_path)
    
    if model_data is None:
        sys.exit(1)
    
    # Extract weights information
    weights_info = extract_weights_info(model_data)
    
    # Show summary if requested
    if args.summary:
        print_model_summary(weights_info)
        return
    
    # Show specific layer if requested
    if args.layer is not None:
        if 0 <= args.layer < len(weights_info["layers"]):
            layer_data = weights_info["layers"][args.layer]
            print(f"Layer {args.layer} ({layer_data['layer_type']}):")
            print(json.dumps(layer_data, cls=NumpyEncoder, indent=2))
        else:
            print(f"Error: Layer index {args.layer} out of range (0-{len(weights_info['layers'])-1})")
        return
    
    # Remove actual weight values if requested
    if args.no_weights:
        for layer in weights_info["layers"]:
            # Keep shapes and metadata, but remove actual weight values
            keys_to_remove = ["weights", "biases", "gamma", "beta", "running_mean", "running_var"]
            for key in keys_to_remove:
                if key in layer:
                    del layer[key]
    
    # Convert to JSON
    indent = 2 if args.pretty else None
    json_output = json.dumps(weights_info, cls=NumpyEncoder, indent=indent)
    
    # Output to file or stdout
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"Weights saved to: {args.output}")
        except Exception as e:
            print(f"Error saving to file: {e}")
    else:
        print(json_output)


if __name__ == "__main__":
    # Example usage when run without arguments
    if len(sys.argv) == 1:
        print("Neural Network Model Weights Reader")
        print("=" * 40)
        print("Usage examples:")
        print("  python read_model_weights.py models/trained_model.pkl")
        print("  python read_model_weights.py models/trained_model.pkl --summary")
        print("  python read_model_weights.py models/trained_model.pkl --layer 0")
        print("  python read_model_weights.py models/trained_model.pkl --output weights.json --pretty")
        print("  python read_model_weights.py models/trained_model.pkl --no-weights --pretty")
        print("\nOptions:")
        print("  --summary, -s      Show model summary only")
        print("  --layer N, -l N    Show specific layer weights")
        print("  --output FILE, -o  Save JSON to file")
        print("  --no-weights       Exclude weight values from JSON")
        print("  --pretty, -p       Pretty print with indentation")
        sys.exit(0)
    
    main()