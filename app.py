from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import os

app = FastAPI(title="Emotion Detection API")

# Load model and labels at startup
model_path = os.path.join("saved_model", "emotion_model.keras")
label_path = os.path.join("saved_model", "label_mapping.pkl")

model = None
index_to_label = {}

@app.on_event("startup")
def load_model():
    global model, index_to_label
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            index_to_label = pickle.load(f)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_emotion(input_data: TextInput):
    if model is None:
        return {"error": "Model not loaded. Please run train.py first."}
    
    # Prediction
    prediction = model.predict(np.array([input_data.text]))
    predicted_index = int(np.argmax(prediction[0]))
    predicted_emotion = index_to_label.get(predicted_index, "Unknown")
    confidence = float(prediction[0][predicted_index])
    
    return {
        "text": input_data.text,
        "emotion": predicted_emotion,
        "confidence": confidence
    }

@app.get("/", response_class=HTMLResponse)
def root():
    # Return the beautiful HTML frontend
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
