import streamlit as st
import tensorflow as tf
import cv2

import time


from util import *

st.set_page_config(page_title='DogPal', page_icon = 'logo.jpeg')
st.title("DogPal single shot inference")

detector = load_model()

method = st.selectbox('Capture or Upload an Image', ('Upload Image', 'Capture Image'))
#for image upload
if method == 'Upload Image':
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
else:
    image_file = st.camera_input("Capture Image")
#for group upload
if image_file:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.001)
            progress_bar.progress(i + 1)
        st.info('Image Uploaded successfully!')
        st.image(image_file.getvalue())
        image = tf.io.decode_image(image_file.getvalue())

        image = cv2.resize(image.numpy(), (512, 512))
        image_np = np.asarray(image)
        detections = detector.detect(image_np)
        image_np = visualize(image_np, detections)

        st.image(image_np)
