from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')

# Load the trained model
model = load_model('ai_image_detector.h5')

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img_resized, axis=0)

@app.route('/')
def index():
    """Render the upload page."""
    return render_template('update.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and classification."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        result = "AI-generated" if prediction > 0.5 else "Real"
        
        return render_template(
            'update.html',
            result=result,
            image_url=url_for('static', filename=f'uploads/{file.filename}')
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
