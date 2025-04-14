from scripts.train import build_improved_model, calculate_class_weights
from scripts.preprocess import create_generators
import tensorflow as tf

def retrain_model():
    # Data
    train_gen, val_df = create_generators(
        "data/ODIR-5K/Training Images",
        "data/ODIR-5K/ODIR-5K_Training_Annotations.csv"
    )
    
    # Class weights
    class_weights = calculate_class_weights(
        "data/ODIR-5K/ODIR-5K_Training_Annotations.csv")
    
    # Model
    model = build_improved_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/improved_model.keras",
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        train_gen,
        epochs=50,
        class_weight=class_weights,
        callbacks=callbacks,
        validation_data=val_df  # Adapt this to your validation generator
    )
    
    return history