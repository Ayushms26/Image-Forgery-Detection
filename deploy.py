# app.py (Flask server)
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
model = load_model("image_forgery_cnn.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).resize((256, 256))
    image_array = np.array(image)/255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)[0][0]
    result = "Tampered" if prediction > 0.5 else "Authentic"

    return jsonify({"prediction": result, "confidence": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
