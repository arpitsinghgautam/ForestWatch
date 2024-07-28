import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
loaded_model = load_model('deforestation_model.h5')

st.title("Deforestation and Pollution Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = loaded_model.predict(x)
    deforestation_prob = preds[0][0]
    pollution_prob = preds[0][1]

    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Deforestation Probability: {deforestation_prob:.4f}")
    st.write(f"Pollution Probability: {pollution_prob:.4f}")
