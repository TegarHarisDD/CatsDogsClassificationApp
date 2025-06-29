# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("🐱 Cat vs Dog Classifier")

# Debug: tampilkan struktur file di root agar kita tahu di mana kita berada
st.sidebar.header("Debug Info")
st.sidebar.write("Current working dir:", os.getcwd())
st.sidebar.write("Root files:", os.listdir())

MODEL_DIR = "catdog_mobilenetv2_saved"

# Pastikan folder model ada
if not os.path.isdir(MODEL_DIR):
    st.error(f"❌ Folder model `{MODEL_DIR}` tidak ditemukan di root!\n\n" +
             "Silakan pastikan:\n"
             "1. Anda telah push folder `catdog_mobilenetv2_saved/` ke GitHub,\n"
             "2. Tidak ada di `.gitignore`,\n"
             "3. Path sesuai (cek daftar file di sidebar).")
    st.stop()

@st.cache_resource
def load_model():
    # load SavedModel dari folder
    return tf.keras.models.load_model(MODEL_DIR)

# Muat model
model = load_model()

st.write("✅ Model berhasil dimuat!")

st.write("Upload gambar dan model akan mengklasifikasikan sebagai Kucing atau Anjing.")

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
