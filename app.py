import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("catdog_mobilenetv2.h5")

model = load_model()
st.title("🐱 vs 🐶 Classifier with MobileNetV2")

uploaded = st.file_uploader("Unggah gambar kucing atau anjing", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((128,128))
    arr = np.array(img_resized)/255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    pred = model.predict(arr)[0][0]
    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    confidence = pred if pred>0.5 else 1-pred
    st.write(f"**Prediksi:** {label} (confidence: {confidence:.2%})")
