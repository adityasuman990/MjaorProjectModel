import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_image(model_path, img_path, img_size=(224, 224)):
    """Make prediction on single image"""
    model = load_model(model_path)
    
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    diseases = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
               'AMD', 'Hypertension', 'Myopia', 'Others']
    
    print("Predicted Probabilities:")
    for disease, prob in zip(diseases, preds):
        print(f"{disease}: {prob:.4f}")
    
    return preds