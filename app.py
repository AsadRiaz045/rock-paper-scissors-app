import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('rps_cnn_model.h5')

# Define the class names
class_names = ['paper', 'rock', 'scissors']

# Set the title of the Streamlit app
st.title("Rock Paper Scissors Image Classifier")

# Add a file uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Include a conditional statement to proceed only if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image
    # Resize the image to the target size (180x180)
    image = image.resize((180, 180))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Expand the dimensions of the array to match the model's input shape (add a batch dimension)
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize the pixel values by dividing by 255.0
    image_array = image_array / 255.0

    # Make a prediction using the loaded model
    predictions = model.predict(image_array)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    # Display the predicted class
    st.write(f"Prediction: {predicted_class}")
