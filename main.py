import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
from tensorflow.keras.models import load_model


st.set_page_config(page_title='Alexander Kovalevskiy')

'''
# Hello!

This toy website allows you to **interact** with ML models.

---
'''

'''
## DCGANime
Deep Convolutional Generative Adversarial Net trained on anime faces dataset to generate not existing animes.
'''

model_path = 'model/generator.h5'
generator = load_model(model_path)
latent_dims = 100


grid_size = st.slider('Select grid size: ', 2, 6)
# ganime_URL = 'http://127.0.0.1:8000'

# image = (np.array(requests.get(ganime_URL).json()['Image'][0]) + 1) / 2
# print(image.shape, type(image))
# print(image)

def test(grid_size):
    with st.container():
        ind = 0
        for row in range(grid_size):
            for col in st.columns(grid_size):
                with col:
                    noise = np.random.normal(0, 1, (1, latent_dims))
                    image = (generator.predict(noise) + 1) / 2
                    st.image(
                        image,
                        use_column_width="always",
                    )
                    ind += 1

if st.button('Generate') or grid_size:
    test(grid_size)


'''
---

## Cats VS Dogs

Cat/Dog image classification based on transfer learning MobileNetV2 model.
'''

file = st.file_uploader('Upload your cat or dog image file', accept_multiple_files=False)

if file:
    st.image(file)
    with Image.open(file) as im:
        image_array = np.asarray(im)
    # print(image_array.shape)
    # response = 

'''
---

## FaceID

Simple face identification based on dlib and resnet_v1. This model extracts faces' descriptors \
     and calculates euclidean distance. If distance < threshold returns **True**.
'''
    
st.file_uploader(
    'Select two files with single faces. And we will answer you: is it the same person?',
    accept_multiple_files=True
    )
