import streamlit as st
import tensorflow as tf
from tensorflow import keras
#import cv2
from tempfile import NamedTemporaryFile
from PIL import Image, ImageOps
import numpy as np

categories = {0:'Angry',1:'Happy',2:'Neutral',3:'Sad',4:'Surprise'}

emojis = {0:':angry:',1:':smile:',2:':neutral_face:',3:':cry:',4:':frowning:'}

# Prediction function which take image as input and return string

def predict_image(filename):
    # Path of the model where model is stored
    path_to_model = r'model.hdf5'

    with st.spinner('Model is being loaded..'):
        # Load model using load_model function of Keras
        model = keras.models.load_model(path_to_model, compile=False)
        model.compile()
        print("Done!")


    #Image loading
    img_ = tf.keras.utils.load_img(filename, target_size=(224, 224))

    #converting image to array
    img_array = tf.keras.utils.img_to_array(img_)

    # blur = cv2.GaussianBlur(img_array, (5, 5), 0)

    # img = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    x = np.asarray(img_array) / 255.0
    img_processed = np.expand_dims(x, axis=0)
    img_processed /= 255.

    # prediction using already loaded model
    prediction = model.predict(img_processed)

    #finding the maximum value which indicates the highest probablity of the detected object
    index = np.argmax(prediction)

    return index



st.title("""
         # Face Expressions Classification
         """
         )




#option has been set for the file uploader
st.set_option('deprecation.showfileUploaderEncoding', False)
# Take file as input using file_uploader function of streamlit
buffer = st.file_uploader("Upload a JPG File", type=['jpg'])
temp_file = NamedTemporaryFile(delete=False)

if buffer is None:
    st.text("Please upload an image file")
else:

    image = Image.open(buffer)
    temp_file.write(buffer.getvalue())
    st.image(image, use_column_width=True)
    predict = predict_image(temp_file.name)
    face = categories[predict]
    emo = emojis[predict]
    st.write("This image most likely belongs to {} - {}".format(face,emo),font_size=30)
