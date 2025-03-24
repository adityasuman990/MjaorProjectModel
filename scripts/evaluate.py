import numpy as np
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_gen=None, test_steps=None):
    """Evaluate model performance with proper error handling"""
    try:
        model = load_model(model_path)
        
        if test_gen is not None and test_steps is not None:
            results = model.evaluate(test_gen, steps=test_steps, verbose=1)
            print("\n=== EVALUATION RESULTS ===")
            print(f"Loss: {results[0]:.4f}")
            print(f"Accuracy: {results[1]:.4f}")
            print(f"AUC: {results[2]:.4f}")
            return results
        else:
            print("Warning: No test data provided - loaded model only")
            return model
            
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise