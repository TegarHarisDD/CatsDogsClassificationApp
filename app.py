import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache model agar tidak reload setiap rerun
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("catdog_mobilenetv2_saved")

model = load_model()

# UI
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload gambar dan model MobileNetV2 akan mengklasifikasikan sebagai **Kucing** atau **Anjing**.")

# Upload gambar
file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediksi
    pred = model.predict(img_array)[0][0]
    label = "🐶 Anjing" if pred > 0.5 else "🐱 Kucing"
    confidence = pred if pred > 0.5 else 1 - pred

    st.markdown(f"### Prediksi: **{label}**")
    st.write(f"Tingkat keyakinan: `{confidence:.2%}`")
