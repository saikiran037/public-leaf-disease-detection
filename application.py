import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("model.h5")

# Dictionary for disease solutions
disease_solutions = {
    "Healthy": "No treatment needed",
    "Powdery": "Apply fungicide",
    "Rust": "Needs water"
}

# Define class indices manually
class_indices = {
    0: "Healthy",
    1: "Powdery",
    2: "Rust"
}

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(225, 225)):
    img = image.resize(target_size)  # Resize the image to target size
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x

# Streamlit app layout
st.title("Leaf Disease Detection")

# File uploader widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    x = preprocess_image(image)

    # Make prediction
    try:
        predictions = model.predict(x)
        predicted_index = np.argmax(predictions)
        predicted_label = class_indices[predicted_index]  # Use the defined class indices
        solution = disease_solutions.get(predicted_label, "Solution not available")
        
        # Display the result
        st.write(f"Predicted Disease: **{predicted_label}**")
        st.write(f"Suggested Solution: **{solution}**")
    except Exception as e:
        st.write(f"An error occurred during prediction: {e}")
