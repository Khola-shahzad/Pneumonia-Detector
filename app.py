from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------------------------
# Step 1: Download model from Google Drive
# --------------------------------------------
def download_model_from_drive():
    model_dir = "saved_models"
    model_path = os.path.join(model_dir, "pneumonia_cnn_model.keras")
    drive_file_id = "1bRBYG7qsKTaBHJOwCYeS2CT20CrwNz4S"  # ← Replace this with your actual file ID
    url = f"https://drive.google.com/file/d/1bRBYG7qsKTaBHJOwCYeS2CT20CrwNz4S/view?usp=sharing"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(model_path):
        print("[INFO] Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    else:
        print("[INFO] Model already exists.")

# Download and load model
download_model_from_drive()
model = load_model('saved_models/pneumonia_cnn_model.keras')

# --------------------------------------------
# Routes
# --------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = 'Pneumonia' if prediction > 0.5 else 'Normal'
    confidence = round(100 * prediction if prediction > 0.5 else 100 * (1 - prediction), 2)

    if label == 'Pneumonia':
        return render_template('selfcare.html', confidence=confidence, filename=file.filename)

    return render_template('result.html', label=label, confidence=confidence, filename=file.filename)

@app.route('/selfcare')
def selfcare():
    return render_template('selfcare.html')

if __name__ == '__main__':
    app.run(debug=True)
