from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import numpy as np
import torch
import joblib
import albumentations
import cnn_models

app = FastAPI()

# Load the label binarizer
lb = joblib.load('lb.pkl')

# Load the model
model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the augmentation pipeline
aug = albumentations.Compose([
    albumentations.Resize(224, 224, always_apply=True),
])

# Define function for sign language prediction on a single frame
def predict_sign_language_single_frame(frame):
    # Apply the augmentation
    frame = aug(image=frame)['image']
    frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)
    frame = torch.tensor(frame, dtype=torch.float)
    frame = frame.unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(frame)
    _, preds = torch.max(outputs, 1)
    predicted_class = lb.classes_[preds]
    
    return predicted_class

@app.post("/predict")
async def predict_sign_language(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Make prediction on the frame
    prediction = predict_sign_language_single_frame(frame)
    prediction_text = str(prediction)  # Convert prediction to string
    
    # Return prediction as JSON response
    return {"prediction": prediction_text}

@app.get("/realtime")
async def predict_real_time():
    # Open camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Could not open camera."}
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            return {"error": "Could not read frame."}
        
        # Make prediction on the frame
        prediction = predict_sign_language_single_frame(frame)
        prediction_text = str(prediction)  # Convert prediction to string

        # Display the result on the image
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the frame to JPEG format and stream it as response
        frame_jpeg = cv2.imencode('.jpg', frame)[1]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg.tobytes() + b'\r\n')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
