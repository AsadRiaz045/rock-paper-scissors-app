from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import gc

app = FastAPI()

# Model load karne ka rasta
MODEL_PATH = 'rps_cnn_model.h5'
model = None

def load_model_safely():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # compile=False aur memory management ke sath model load karna
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Model loading error: {e}")
    else:
        print(f"✗ File not found at {MODEL_PATH}")

@app.on_event("startup")
async def startup_event():
    load_model_safely()

@app.get("/")
async def root():
    # Asad, ye check karne ke liye ke API zinda hai
    return {
        "message": "Asad, RPS Classifier is Ready!",
        "model_status": "Loaded" if model else "Not Loaded"
    }

def preprocess_image(data) -> np.ndarray:
    # Image resize aur normalization (Aapke project ke mutabiq 180x180)
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((180, 180)) 
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model is not loaded on the server. Check logs."}
    
    try:
        content = await file.read()
        processed_img = preprocess_image(content)
        
        # Prediction logic
        predictions = model.predict(processed_img)
        classes = ['Paper', 'Rock', 'Scissors']
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # Memory saaf karna taake aglay request ke liye jagah banay
        del processed_img
        gc.collect()

        return {
            "prediction": classes[predicted_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "filename": file.filename
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    # Render ke liye port management
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
