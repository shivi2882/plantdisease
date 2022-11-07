import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Plant Disease Identification')
st.markdown("<h2 style='font-family:georgia; text-align: center; color:#4d7902' > Identifying Plant Diseases & Deficiencies using AI </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'> Upload image of a leaf to see if it has any disease </h2>", unsafe_allow_html=True)

class_dict = {'Scab': 0, 'Black rot': 1, 'Cedar apple rust': 2, 'Healthy!!': 3, 'Healthy!': 4,
              'Powdery mildew': 5, 'Looks Healthy!': 6, 'Cercospora leaf spot (Gray leaf spot)': 7, 'Common rust': 8,
              'Northern Leaf Blight': 9, 'Looks Healthy!!': 10, 'Black rot detected': 11, 'Esca(Black Measles)': 12,
              'Leaf blight(Isariopsis Leaf Spot)': 13, 'Looks healthy!': 14, 'Huanglongbing (Citrus greening)': 15, 'Bacterial spot': 16,
              'Looks healthy!!': 17, 'Bacterial spot detected': 18, 'healthy!': 19, 'Early blight': 20,
              'Late blight': 21, 'It\'s healthy!': 22, 'Leaf looks Healthy!': 23, 'Leaf looks healthy!': 24, 'Powdery mildew detected': 25,
              'Leaf scorch': 26, 'Leaf looks healthy.': 27, 'Leaf seems to have Bacterial spot': 28, 'Early blight detected': 29,
              'Late blight detected': 30, 'Leaf Mold': 31, 'Septoria leaf spot': 32, 'Spider mites (Two-spotted spider mite)': 33,
              'Target Spot': 34, 'Yellow Leaf Curl Virus': 35, 'Mosaic virus': 36, 'Leaf looks Healthy.': 37}

li = list(class_dict.keys())

# @st.cache(allow_output_mutation=True)

model_path = 'plant_model.h5'
model = load_model(model_path)

file = st.file_uploader("",type=["jpg", "png"])

if file is None:
    st.text("Please upload an image of a leaf")
else:
    loaded_img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(loaded_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    # predicting the image -
    prediction = model.predict(img)
    fl = prediction.flatten()
    flm = fl.max()
    for index, item in enumerate(fl):
        if item == flm:
            class_name = li[index]
    if 'healthy' not in class_name.lower():
        st.write(f"Prediction: [{class_name}](https://www.google.com/search?q={(class_name).replace(' ', '+')}+Leaf+Disease)")
    else:
        st.write(class_name)
    #ploting image with predicted class name
    image_demo = Image.open(file)
    st.image(image_demo, caption='Uploaded image')

st.header('About')
about = """
            This application uses Deep Learning to detect plant diseases using images of leaves.
            The images can be uploaded from device storage or clicked using a phone camera.
"""
st.write(about)