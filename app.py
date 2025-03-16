from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "models/healthscan_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(image_path):
    """Preprocess image for CNN prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image / 255.0  # Normalize

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        # Preprocess & Predict
        image = prepare_image(file_path)
        prediction = model.predict(image)[0][0]
        result = "Anomaly Detected" if prediction > 0.5 else "Normal"

        return render_template("index.html", result=result, image_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
