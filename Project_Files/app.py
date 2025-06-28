from flask import Flask, render_template, request
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import zipfile
import os
from tensorflow.keras.models import load_model

# Step 1: Unzip the file
zip_path = 'healthy_vs_rotten.zip'
extracted_model_path = 'Healthy_vs_Rotten.h5'

if not os.path.exists(extracted_model_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()


app = Flask(__name__)
model = load_model(extracted_model_path)

# Load class index
with open("index_to_class.pkl", "rb") as f:
    index_to_class = pickle.load(f)

def predict_image_class(model, img_path, index_to_class, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = index_to_class[predicted_index]

    return predicted_class, confidence

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            img_path = os.path.join("static", img_file.filename)
            img_file.save(img_path)

            predicted_class, confidence = predict_image_class(model, img_path, index_to_class)
            prediction = f"The image is predicted to be: {predicted_class}"
            confidence = f"Confidence: {confidence:.2%}"
            image_path = img_path

    return render_template("predict.html", prediction=prediction, confidence=confidence, image=image_path)



@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            img_path = os.path.join("static", img_file.filename)
            img_file.save(img_path)

            predicted_class, confidence = predict_image_class(model, img_path, index_to_class)
            prediction = f"The image is predicted to be: {predicted_class}"
            confidence = f"Confidence: {confidence:.2%}"

            return render_template("index.html", prediction=prediction, confidence=confidence, image=img_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
