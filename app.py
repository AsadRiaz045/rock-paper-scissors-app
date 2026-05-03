from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import gc

app = FastAPI()

# Model configuration
MODEL_PATH = 'rps_cnn_model.h5'
model = None

def load_model_safely():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Keras 3/TF 2.16+ mein h5 files ke liye compile=False lazmi hai
            # Agar version mismatch ho to ye 'batch_shape' error ko handle kar leta hai
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✓ SUCCESS: Model loaded successfully!")
        except Exception as e:
            print(f"✗ ERROR: Model loading failed. Details: {e}")
    else:
        print(f"✗ ERROR: {MODEL_PATH} not found in the directory.")

@app.on_event("startup")
async def startup_event():
    load_model_safely()

@app.get("/")
async def root():
    return {
        "message": "Asad, RPS Classifier is Ready!",
        "model_status": "Active" if model else "Inactive",
        "tensorflow_version": tf.__version__
    }

def preprocess_image(data) -> np.ndarray:
    # Image resize (180, 180) as per your training
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((180, 180)) 
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model is not loaded. Please check server logs."}
    
    try:
        content = await file.read()
        processed_img = preprocess_image(content)
        
        # Inference
        predictions = model.predict(processed_img)
        classes = ['Paper', 'Rock', 'Scissors']
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # Cleanup memory
        del processed_img
        gc.collect()

        return {
            "prediction": classes[predicted_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "filename": file.filename
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

if __name__ == "__main__":
    # Render dynamic port binding
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
