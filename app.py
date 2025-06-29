import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Set judul halaman
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾")

# Judul aplikasi
st.title("🐱 Cats vs Dogs Classifier 🐶")
st.markdown("## Meow or Woof? Upload an image to find out!")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/cats_dogs_mobilenetv2.h5')

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader(
    "Choose an image of a cat or dog...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', width=300)
    
    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Prediksi
    prediction = model.predict(img_array)
    result = "DOG 🐶" if prediction[0][0] > 0.5 else "CAT 😸"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    # Tampilkan hasil
    st.subheader("Prediction Result:")
    st.success(f"This is a **{result}**!")
    st.info(f"Confidence: **{confidence:.2%}**")
    
    # Tampilkan penjelasan
    if result == "DOG 🐶":
        st.markdown("### Woof! Woof! 🐕")
        st.write("Characteristics detected:")
        st.write("- Canine facial features")
        st.write("- Dog ear structure")
        st.write("- Typical dog posture")
    else:
        st.markdown("### Meow! 😻")
        st.write("Characteristics detected:")
        st.write("- Feline eye shape")
        st.write("- Pointed cat ears")
        st.write("- Typical cat body posture")

# Footer
st.markdown("---")
