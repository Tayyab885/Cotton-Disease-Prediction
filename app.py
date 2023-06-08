import streamlit as st
import numpy as np
from keras.models import load_model
import cv2

# Load the model
model = load_model('cotton_disease_prediction.h5')


## Header of the webpage
st.set_page_config( page_title="Cotton Disease Prediction",page_icon=":memo:", layout="centered")
st.header("Cotton Disease Prediction")
st.text("Upload a cotton leaf image for image classification as diseased or fresh")
file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"], key="image")
st.set_option('deprecation.showfileUploaderEncoding', False) 
st.markdown("---")
button = st.button("Predict")
if button:
    if file is None:
        st.text("Please upload an image file")
    else:
        st.image(file, use_column_width=True)
        image = file.read()
        image = cv2.imdecode(np.fromstring(image, np.uint8), 1)
        image = cv2.resize(image, (150,150))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        prediction = np.argmax(prediction)
        if prediction == 0:
            st.write("<h3>The leaf is diseased cotton leaf</h3>", unsafe_allow_html=True)
        elif prediction == 1:
            st.write("<h3>The leaf is diseased cotton plant</h3>", unsafe_allow_html=True)
        elif prediction == 2:
            st.write("<h3>The leaf is fresh cotton leaf</h3>", unsafe_allow_html=True)
        else:
            st.write("<h3>The leaf is fresh cotton plant</h3>", unsafe_allow_html=True)