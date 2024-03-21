import streamlit as st
import numpy as np
from PIL import Image
import cv2
import abc

st.title("TSC AQC DETECTION")
st.info("This job was completed by New Ocean Company")
st.header("Input Data")

def hex_to_rgb(hex):
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return tuple(rgb)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity
# Visualize input, template images
uploaded_img_file = st.file_uploader("Please choose an image",accept_multiple_files=False, type=["jpg", "jpeg", "png"])
if uploaded_img_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_img_file)
    st.image(uploaded_img_file, caption="Input Image", width=256)

uploaded_tmp_file = st.file_uploader("Please choose a template image",accept_multiple_files=False, type=["jpg", "jpeg", "png"])
if uploaded_tmp_file is not None:
    # Display the uploaded image
    image_tmp = Image.open(uploaded_img_file)
    st.image(uploaded_tmp_file, caption="Template Image", width=256)

color_garment = st.color_picker('Pick a color of Garment', '#00f900')
st.write('The current color is', color_garment)
RGB_color_garment = hex_to_rgb(color_garment[1:])
st.write(RGB_color_garment)
HSV_garment = cv2.cvtColor(np.array(RGB_color_garment, 'uint8').reshape(1,1,3), cv2.COLOR_RGB2LAB)


color_actual = st.color_picker('Pick a color of Actual Garment', '#00f901')
st.write('The current color is', color_actual)
RGB_color_actual = hex_to_rgb(color_actual[1:])
st.write(RGB_color_actual)
HSV_actual = cv2.cvtColor(np.array(RGB_color_actual, 'uint8').reshape(1,1,3), cv2.COLOR_RGB2LAB)

st.write('Similarity of two colors (RGB): ', cosine_similarity(np.array(RGB_color_garment, 'float32'), np.array(RGB_color_actual, 'float32')))
st.write('Similarity of two colors (HSV): ', cosine_similarity(HSV_actual.astype('float32').reshape(-1), HSV_garment.astype('float32').reshape(-1)))
#Show result
result0, result1, result2 = st.columns(3)
st.header("Predict")
with result0:
    st.image("download.jpg",caption="Bird")
with result1:
    st.image("download.jpg",caption="Bird")
with result2:
    st.image("download.jpg",caption="Bird")

garment_size = 'L'

style = f"""
        border: 1px solid white; 
        padding:10px;
"""
with st.container():
        html = f"""
        <div style="{style}">
            <p>Collar front: {True}</p>
            <p>Design match {True}</p>
            <p>Print area rotation: {7} degree</p>
            <p>Print area offset: {True}</p>
            <p>Garment size: {garment_size}</p>
        </div>
        """
        results = st.markdown(html, unsafe_allow_html=True)

