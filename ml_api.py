from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict_sign_language():
    try:
        # Get the image file from the request
        file = request.files['file']
        # Read image file
        img_bytes = file.read()
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Preprocess the image
        img = preprocess_image(img)
        # Make prediction
        prediction = model.predict([img])[0]
        return jsonify({'predicted_class': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
