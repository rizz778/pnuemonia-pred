import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Define the class names for binary classification
class_names = ['NORMAL', 'PNEUMONIA']

# Use st.cache_resource to cache the model loading
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("notebooks/xray_model.hdf5")
    return model

# Load the model with spinner
with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """
         )

file = st.file_uploader("Please upload a chest scan file", type=["jpg","jpeg", "png"])

# Preprocess function as per your request
def preprocess_image(image_path, target_size=(180, 180)):
    img = image_path.convert("RGB")  # Ensure the image is in RGB format
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0  # Scale pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function
def import_and_predict(image_data, model):
    # Preprocess the uploaded image
    img_reshape = preprocess_image(image_data)
    # Make prediction
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    
    st.write(predictions)  # Raw prediction output for debugging
    st.write(score)  # Softmax score for debugging

    # Display the result
    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
