import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Function to load the trained saved model
def load_model():
    return tf.saved_model.load('saved_model')  # Load the SavedModel format

vae_model = load_model()

def generate_output(input_image):
    # Resize and normalize the input image
    input_image = np.array(input_image.resize((128, 128))) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Ensure the input is in the correct dtype, TensorFlow typically expects float32
    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
    
    # Create a dictionary for input as expected by the serving signature
    # 'inputs' is the key that we found needs to be used from the error message and model signature inspection
    input_dict = {'inputs': input_tensor}

    # Use the serving signature with the correct input format
    output = vae_model.signatures['serving_default'](**input_dict)

    # Extracting the output assuming 'output_0' is the key for the desired model output
    output_image = output['output_0']

    return output_image[0].numpy()  # Convert to numpy array if needed

# Streamlit UI
st.title('VAE Model Deployment')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image', use_column_width=True)
    if st.button('Generate Output'):
        with st.spinner('Generating Output...'):
            output_image = generate_output(input_image)
            st.image(output_image, caption='Generated Output', use_column_width=True)
            st.success('Output generated successfully!')
