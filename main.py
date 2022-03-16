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

model_dcganime_path = 'model/generator.h5'
generator = load_model(model_dcganime_path)
latent_dims = 100


model_cats_path = 'model/cats_vs_dogs.h5'
model = load_model(model_cats_path)


grid_size = st.slider('Select grid size: ', 2, 6)


def plot_images(grid_size):
    with st.container():
        for row in range(grid_size):
            for col in st.columns(grid_size):
                with col:
                    noise = np.random.normal(0, 1, (1, latent_dims))
                    image = (generator.predict(noise) + 1) / 2
                    st.image(
                        image,
                        use_column_width="always",
                    )

if st.button('Generate') or grid_size:
    plot_images(grid_size)


'''
---

## Cats VS Dogs

Cat/Dog image classification based on transfer learning MobileNetV2 model.
'''

file = st.file_uploader('Upload your cat or dog image file', accept_multiple_files=False)

if file:
    st.image(file)
    with Image.open(file) as im:
        image_array = np.asarray(im, dtype=np.uint8)
        prediction = model.predict(np.expand_dims(image_array, axis=0))[0]
        f'''
        ##### Result
        Cat: {np.round(prediction[0] * 100, 4)}%
        Dog: {np.round(prediction[1] * 100, 4)}%
        '''

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
