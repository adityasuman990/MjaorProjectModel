# # import numpy as np
# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.models import load_model

# # def predict_image(model_path, img_path, img_size=(224, 224)):
# #     """Make prediction on single image"""
# #     model = load_model(model_path)
    
# #     img = image.load_img(img_path, target_size=img_size)
# #     img_array = image.img_to_array(img) / 255.0
# #     img_array = np.expand_dims(img_array, axis=0)
    
# #     preds = model.predict(img_array)[0]
# #     diseases = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
# #                'AMD', 'Hypertension', 'Myopia', 'Others']
    
# #     print("Predicted Probabilities:")
# #     for disease, prob in zip(diseases, preds):
# #         print(f"{disease}: {prob:.4f}")
    
# #     return preds

# import numpy as np
# import tensorflow as tf 

# class Predictor:
#     def __init__(self, model_path: str):
#         self.model = tf.keras.models.load_model(model_path)
#         self.class_names = [
#             'Normal', 'Diabetic Retinopathy', 'Glaucoma', 
#             'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other'
#         ]
    
#     def get_interpreted_prediction(self, pred_prob: float, class_id: int) -> dict:
#         """Convert raw prediction to human-readable format"""
#         diagnosis = "Normal" if class_id == 0 else self.class_names[class_id]
        
#         # Confidence interpretation
#         if pred_prob < 0.4:
#             confidence_level = "Low"
#         elif 0.4 <= pred_prob < 0.7:
#             confidence_level = "Medium"
#         else:
#             confidence_level = "High"
        
#         return {
#             "disease": diagnosis,
#             "class_id": int(class_id),
#             "confidence_score": float(pred_prob),
#             "confidence_level": confidence_level,
#             "recommendation": "No action needed" if class_id == 0 else "Consult an ophthalmologist"
#         }

#     def predict_image(self, image_path: str) -> dict:
#         """Enhanced prediction with interpretation"""
#         try:
#             # Preprocess image
#             img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#             img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)
            
#             # Get prediction
#             preds = self.model.predict(img_array, verbose=0)[0]
#             primary_class = np.argmax(preds)
            
#             # Return interpreted results
#             return self.get_interpreted_prediction(preds[primary_class], primary_class)
            
#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "message": "Prediction failed"
#             }



import tensorflow as tf
import numpy as np
from typing import Dict
import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)

class Predictor:  # Changed from EnhancedPredictor to match import
    def __init__(self, model_path: str):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.class_names = [
                'Normal', 'Diabetic Retinopathy', 'Glaucoma',
                'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other'
            ]
            self.abnormal_threshold = 0.35
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, image_path: str) -> Dict[str, any]:
        """Make prediction with proper error handling"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            img = self._preprocess(image_path)
            preds = self.model.predict(img, verbose=0)[0]
            
            return {
                'diagnosis': self._interpret_prediction(preds),
                'confidence_scores': {
                    name: float(score) 
                    for name, score in zip(self.class_names, preds)
                }
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'error': str(e),
                'message': 'Prediction failed - check server logs'
            }

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def _interpret_prediction(self, preds: np.ndarray) -> str:
        """Determine if any abnormality exceeds threshold"""
        if any(preds[1:] > self.abnormal_threshold):  # Check non-Normal classes
            abnormal_indices = np.where(preds[1:] > self.abnormal_threshold)[0]
            return f"Abnormal findings: {', '.join([self.class_names[i+1] for i in abnormal_indices])}"
        return "Normal"