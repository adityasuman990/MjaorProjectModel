import os
from scripts.preprocess import load_and_preprocess_odir
from scripts.train import build_model, train_model
from scripts.evaluate import evaluate_model

def main():
    # Configure paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "ODIR-5K", "Training Images")
    CSV_PATH = os.path.join(BASE_DIR, "data", "ODIR-5K", "ODIR-5K_Training_Annotations.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "odir_model.keras")  # Changed to .keras format
    
    # Verify paths
    print("\n=== SYSTEM CHECK ===")
    print(f"Project Root: {BASE_DIR}")
    print(f"Images Found: {os.path.exists(DATA_DIR)}")
    print(f"CSV Found: {os.path.exists(CSV_PATH)}")
    
    if not all([os.path.exists(DATA_DIR), os.path.exists(CSV_PATH)]):
        raise FileNotFoundError("Missing required data files")
    
    # Load and preprocess data
    print("\n=== LOADING DATA ===")
    train_gen, train_steps, val_gen, val_steps = load_and_preprocess_odir(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        batch_size=32
    )
    
    # Build and train model
    print("\n=== TRAINING MODEL ===")
    model = build_model()
    history = train_model(
        model,
        train_gen,
        train_steps,
        val_gen,
        val_steps,
        epochs=15,
        save_path=MODEL_PATH
    )
    
    # Evaluate with validation data
    print("\n=== EVALUATION ===")
    evaluate_model(
        model_path=MODEL_PATH,
        test_gen=val_gen,
        test_steps=val_steps
    )

if __name__ == "__main__":
    main()