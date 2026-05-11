from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback
import os

# =========================
# FLASK APP
# =========================

app = Flask(
    __name__,
    template_folder="../"
)

# Enable CORS
CORS(app)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "face_model.h5",
    compile=False
)

# =========================
# LOAD EMBEDDING DATA
# =========================

with open("face_model.pkl", "rb") as f:
    model_data = pickle.load(f)

mean_emb = np.array(model_data["mean_emb"])
threshold = float(model_data["threshold"])

print("✅ Model Loaded Successfully")

# =========================
# HOME ROUTE
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("📥 Request received")

        # Check file uploaded
        if "file" not in request.files:
            return jsonify({
                "result": "No file uploaded"
            }), 400

        file = request.files["file"]

        print("📸 File:", file.filename)

        # Open image
        image = Image.open(file).convert("RGB")

        # Resize image
        image = image.resize((128, 128))

        # Convert to numpy
        img = np.array(image).astype("float32") / 255.0

        # Expand dimensions
        img = np.expand_dims(img, axis=0)

        print("🤖 Predicting...")

        # Predict embedding
        embedding = model.predict(
            img,
            verbose=0
        )[0]

        embedding = np.array(embedding)

        # Calculate distance
        distance = np.linalg.norm(
            embedding - mean_emb
        )

        print("📏 Distance:", distance)

        # Classification
        if distance < threshold:
            result = "Human Face ✅"
        else:
            result = "Non Human Face ❌"

        print("✅ Result:", result)

        return jsonify({
            "result": result
        })

    except Exception as e:

        print(traceback.format_exc())

        return jsonify({
            "result": str(e)
        }), 500

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
