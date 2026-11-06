
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# KONFIGURASI APLIKASI
# =====================================

st.set_page_config(
    page_title="Malaria Cell Classification",
    page_icon="🔬",
    layout="centered"
)

IMAGE_SIZE = (128, 128)

CLASS_NAMES = {
    0: "Uninfected",
    1: "Parasitized"
}

MODEL_PATH = "final_model.keras"


# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# =====================================
# PREPROCESSING GAMBAR
# =====================================

def preprocess_image(image):
    # Pastikan gambar memiliki 3 channel RGB
    image = image.convert("RGB")

    # Ubah gambar menjadi array
    image = np.array(image)

    # Ubah ke tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Resize sesuai preprocessing training
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=False
    )

    # Normalisasi nilai piksel 0-1
    image = image / 255.0

    # Tambahkan dimensi batch
    image = tf.expand_dims(image, axis=0)

    return image


# =====================================
# TAMPILAN APLIKASI
# =====================================

st.title("🔬 Malaria Cell Classification")

st.write(
    "Aplikasi klasifikasi citra sel darah merah "
    "untuk mendeteksi kemungkinan infeksi malaria "
    "menggunakan model CNN."
)

st.divider()

uploaded_file = st.file_uploader(
    "Unggah gambar sel darah merah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Gambar yang Diunggah")
    st.image(
        image,
        caption="Citra sel darah merah",
        use_container_width=True
    )

    if st.button("Mulai Prediksi", type="primary"):

        try:
            model = load_model()

            # Preprocessing
            processed_image = preprocess_image(image)

            # Prediksi
            prediction = model.predict(
                processed_image,
                verbose=0
            )

            # Asumsi output model berupa sigmoid
            probability = float(np.squeeze(prediction))

            if not 0 <= probability <= 1:
                st.error(
                    "Output model bukan probabilitas sigmoid. "
                    "Periksa kembali konfigurasi output model."
                )
                st.stop()

            # Klasifikasi berdasarkan threshold 0.5
            predicted_class = (
                1 if probability >= 0.5 else 0
            )

            confidence = (
                probability
                if predicted_class == 1
                else 1 - probability
            )

            # Menampilkan hasil
            st.divider()
            st.subheader("Hasil Prediksi")

            if predicted_class == 1:
                st.error("Hasil: Parasitized")
            else:
                st.success("Hasil: Uninfected")

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )

            st.write("**Rincian probabilitas:**")

            st.write(
                f"Uninfected: {(1 - probability) * 100:.2f}%"
            )

            st.write(
                f"Parasitized: {probability * 100:.2f}%"
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

st.divider()

st.caption(
    "Model CNN untuk klasifikasi citra sel darah merah. "
    "Hasil prediksi merupakan keluaran model dan bukan diagnosis medis."
)
