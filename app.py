import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define the path to the saved model
MODEL_PATH = './CNN_Model.h5'

@st.cache(allow_output_mutation=True)
def load_trained_model():
    # Make sure to include any custom objects here if you had any in your model
    my_model = load_model(MODEL_PATH)
    return my_model

model = load_trained_model()

# Add a title and description
st.title('Facial Expression Recognition App')
st.write('This app uses a CNN to predict facial expressions.')

# Function to preprocess the image to fit the model's expected input format
def preprocess_image(uploaded_image, target_size=(128, 128)):
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image if the model expects pixel values in [0,1]
    return image

# Upload file interface
uploaded_file = st.file_uploader("Upload an image with a face...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(uploaded_file)
    
    # Use the model to make a prediction
    prediction = model.predict(preprocessed_image)
    
    # Assuming the model returns an array with the predicted probabilities for each class
    # And you have a list of class names in the order that the model outputs
    class_names = ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral']  # Update this list as necessary
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display the prediction
    st.write(f'Predicted expression: {predicted_class}')
