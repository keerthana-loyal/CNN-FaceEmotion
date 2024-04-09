import streamlit as st
import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

# Function to generate images using the specified model
def generate_images(generator, num_images, latent_size):
    # Generate latent vectors
    latent_vectors = torch.randn(num_images, latent_size, 1, 1)
    # Generate images
    with torch.no_grad():
        generated_images = generator(latent_vectors)
    return generated_images

# Dictionary to map expressions to corresponding model paths
expression_to_model = {
    'Angry': './models/angry_generator_model.pth',
    'Happy': './models/Happy_generator_model.pth',
    'Neutral': './models/Neutral_generator_model.pth',
    'Sad': './models/Sad_generator_model.pth',
    'Surprise': './models/Surprise_generator_model.pth'
}

# Streamlit app
st.title("Image Generation")

# Dropdown to select expression
expression = st.selectbox("Select Expression", options=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'])

# Load the selected model
@st.cache_resource
def load_model(expression):
    model_path = expression_to_model[expression]
    return torch.load(model_path, map_location=torch.device('cpu'))  # Load the model

# Load model
model = load_model(expression)

# Specify number of images to generate
num_images = st.number_input("Number of Images to Generate", value=1)

# Generate and display images
if st.button("Generate Images"):
    latent_size = 128  # Assuming latent size is 100
    generated_images = generate_images(model, num_images, latent_size)
    # Create a grid of generated images
    grid = make_grid(generated_images, nrow=4, normalize=True)
    # Convert tensor to PIL Image
    image = Image.fromarray(grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
    st.image(image, caption=f"Generated {num_images} images", use_column_width=True)
