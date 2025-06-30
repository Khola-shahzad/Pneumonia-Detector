from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('saved_models/pneumonia_cnn_model.keras')

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
