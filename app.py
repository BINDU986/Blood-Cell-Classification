import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class labels
model = load_model("Bloodcell.h5")
class_labels = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']


def predict_image_class(image_path, model):
    """Predict the class of a blood cell image"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    confidence = float(np.max(predictions))
    return predicted_class_label, confidence, img_rgb


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Get prediction
            predicted_class_label, confidence, img_rgb = predict_image_class(file_path, model)

            # Convert image to base64 for display
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            return render_template("result.html",
                                   class_label=predicted_class_label,
                                   confidence=f"{confidence * 100:.1f}",
                                   img_data=img_str,
                                   filename=file.filename)
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)