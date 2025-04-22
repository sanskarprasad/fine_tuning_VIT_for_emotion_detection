from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import TFViTForImageClassification, AutoImageProcessor
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Load model and processor
model_path = "/Users/harshshivhare/Emotion-Detection-By-Finetuning-VIT/vit-unfreeze-top3"
model = TFViTForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="tf")
    
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    top_3 = probs.argsort()[-3:][::-1]
    
    results = [
        {"label": class_names[i], "probability": float(probs[i])}
        for i in top_3
    ]
    
    return JSONResponse(content={"top_3_predictions": results})
