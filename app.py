from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# 1. Model ko globally initialize karein
model = None

try:
    # Model load karein
    model = tf.keras.models.load_model('rps_cnn_model.h5')
    print("✓ Rock-Paper-Scissors Model successfully loaded!")
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Class Names
class_names = ['paper', 'rock', 'scissors']

@app.get("/")
async def root():
    return {"message": "Asad, RPS Classifier is Ready!"}

def preprocess_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((180, 180))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check karein ke kya model load hua hai
    if model is None:
        return {"error": "Model is not loaded on the server. Check terminal for errors."}
    
    try:
        bytes_data = await file.read()
        processed_image = preprocess_image(bytes_data)
        
        # Prediction
        predictions = model.predict(processed_image)
        
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = float(np.max(predictions))

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
    except Exception as e:
        return {"error": f"Internal Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
