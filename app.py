# from flask import Flask, request, jsonify, render_template
# import os
# from scripts.predict import Predictor

# app = Flask(__name__)
# predictor = Predictor("models/odir_model.keras")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     try:
#         # Save temporarily
#         temp_path = os.path.join("uploads", file.filename)
#         os.makedirs(os.path.dirname(temp_path), exist_ok=True)
#         file.save(temp_path)
        
#         # Get prediction
#         result = predictor.predict_image(temp_path)
        
#         # Clean up
#         os.remove(temp_path)
        
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify, render_template, redirect, url_for
from scripts.predict import Predictor
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize predictor
try:
    predictor = Predictor("models/odir_model.keras")
except Exception as e:
    print(f"Failed to initialize predictor: {str(e)}")
    predictor = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            
            result = predictor.predict(temp_path)
            os.remove(temp_path)
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)