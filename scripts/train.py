# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras import layers, models
# import os
# import numpy as np
# from typing import Tuple, Any
# import pandas as pd

# def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3), 
#                 num_classes: int = 8) -> tf.keras.Model:
#     """
#     Builds and compiles an EfficientNetB0-based model for multi-label classification
    
#     Args:
#         input_shape: Shape of input images (height, width, channels)
#         num_classes: Number of output classes (8 for ODIR-5K)
    
#     Returns:
#         Compiled Keras model
#     """
#     # Load pre-trained base model
#     base_model = EfficientNetB0(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
#     base_model.trainable = False  # Freeze base layers

#     # Custom classification head
#     inputs = tf.keras.Input(shape=input_shape)
#     x = base_model(inputs, training=False)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(num_classes, activation='sigmoid')(x)

#     model = tf.keras.Model(inputs, outputs)
    
#     # Compile with multiple metrics
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.AUC(name='auc'),
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall')
#         ]
#     )
    
#     return model

# def train_model(
#     model: tf.keras.Model,
#     train_gen: tf.keras.utils.Sequence,
#     train_steps: int,
#     val_gen: tf.keras.utils.Sequence,
#     val_steps: int,
#     epochs: int = 15,
#     save_path: str = "models/odir_model.keras"
# ) -> tf.keras.callbacks.History:
#     """
#     Trains the model with callbacks and saves the best version
    
#     Args:
#         model: Compiled Keras model
#         train_gen: Training data generator
#         train_steps: Steps per training epoch
#         val_gen: Validation data generator
#         val_steps: Validation steps
#         epochs: Maximum training epochs
#         save_path: Path to save the best model
    
#     Returns:
#         Training history object
#     """
#     # Create model directory if needed
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Configure callbacks
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(
#             save_path,
#             monitor='val_auc',
#             mode='max',
#             save_best_only=True,
#             save_weights_only=False,
#             verbose=1
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_auc',
#             patience=5,
#             mode='max',
#             restore_best_weights=True,
#             verbose=1
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.2,
#             patience=3,
#             min_lr=1e-6,
#             verbose=1
#         ),
#         tf.keras.callbacks.TensorBoard(
#             log_dir='logs',
#             histogram_freq=1
#         )
#     ]

#     # Train the model
#     try:
#         history = model.fit(
#             train_gen,
#             steps_per_epoch=train_steps,
#             validation_data=val_gen,
#             validation_steps=val_steps,
#             epochs=epochs,
#             callbacks=callbacks,
#             verbose=2
#         )
#         return history
#     except Exception as e:
#         print(f"\nError during training: {str(e)}")
#         raise

# def get_class_weights(csv_path: str) -> dict:
#     """
#     Calculates class weights for imbalanced data
    
#     Args:
#         csv_path: Path to training annotations CSV
    
#     Returns:
#         Dictionary of class weights
#     """
#     df = pd.read_csv(csv_path)
#     targets = np.array([eval(x) for x in df['target']])
#     class_counts = targets.sum(axis=0)
#     total = len(df)
#     return {i: total/(len(class_counts)*count) for i, count in enumerate(class_counts)}






import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from typing import Dict

def build_improved_model(input_shape=(224, 224, 3), num_classes=8):
    """Build model with better architecture for eye disease detection"""
    # Load base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Unfreeze top layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-15]:  # Freeze all but last 15 layers
        layer.trainable = False
    
    # Custom head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Use focal loss for class imbalance
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def calculate_class_weights(csv_path: str) -> Dict[int, float]:
    """Calculate effective class weights"""
    df = pd.read_csv(csv_path)
    targets = np.array([eval(x) for x in df['target']])
    class_counts = targets.sum(axis=0)
    
    # Smooth weights to prevent extreme values
    total = len(df)
    smoothing_factor = 0.1
    return {
        i: (total - class_counts[i] + smoothing_factor) / (class_counts[i] + smoothing_factor)
        for i in range(len(class_counts))
    }