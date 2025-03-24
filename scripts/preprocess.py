import pandas as pd
import numpy as np
import os
import tensorflow as tf  # Added this import
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_odir(data_dir, csv_path, img_size=(224, 224), batch_size=32, test_size=0.2):
    """Load and preprocess ODIR-5K dataset with guaranteed float32 output"""
    # Load CSV and convert targets to proper float32 arrays
    df = pd.read_csv(csv_path)
    
    # Convert string targets to numpy arrays
    def parse_target(target_str):
        arr = np.fromstring(target_str.strip('[]'), sep=',')
        return arr.astype(np.float32)  # Force float32
    
    df['target'] = df['target'].apply(parse_target)
    
    # Create absolute file paths
    df['filepath'] = df['filename'].apply(
        lambda x: os.path.join(data_dir, x)
    )
    
    # Verify all targets have 8 values
    assert all(len(t) == 8 for t in df['target']), "All targets must have 8 values"
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Custom generator to ensure proper dtypes
    def create_generator(df, datagen, batch_size):
        while True:
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                images = []
                labels = []
                for _, row in batch_df.iterrows():
                    img = tf.keras.preprocessing.image.load_img(
                        row['filepath'], target_size=img_size)
                    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    images.append(img.astype(np.float32))
                    labels.append(row['target'].astype(np.float32))
                yield np.array(images), np.array(labels)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Create generators with guaranteed float32 output
    train_steps = len(train_df) // batch_size
    val_steps = len(val_df) // batch_size
    
    train_gen = create_generator(train_df, train_datagen, batch_size)
    val_gen = create_generator(val_df, ImageDataGenerator(), batch_size)
    
    return train_gen, train_steps, val_gen, val_steps