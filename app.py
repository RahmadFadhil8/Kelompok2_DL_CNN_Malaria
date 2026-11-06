
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
import hashlib

# =========================================
# 1. KONFIGURASI APLIKASI
# =========================================

st.set_page_config(
    page_title="Malaria Cell Classification",
    page_icon="🔬",
    layout="wide"
)

IMAGE_SIZE = (128, 128)
MODEL_PATH = "final_model.keras"

CLASS_NAMES = {
    0: "Uninfected",
    1: "Parasitized"
}

# Parameter hasil kalibrasi validation set
TEMPERATURE = 1.2147856356111453
TAU = 0.755

# Tingkat masking untuk deletion test
MASK_FRACTIONS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
]

# Jumlah pengulangan random masking
RANDOM_REPEATS = 10


# =========================================
# 2. LOAD MODEL
# =========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Mencari Conv2D terakhir pada CNN
    last_conv = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

    if last_conv is None:
        raise ValueError(
            "Layer Conv2D tidak ditemukan."
        )

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv.output,
            model.output
        ]
    )

    return model, grad_model


# =========================================
# 3. PREPROCESSING
# =========================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = np.array(image)

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=False
    )

    # Menyamakan preprocessing dengan notebook
    image = tf.cast(
        tf.clip_by_value(
            tf.round(image),
            0,
            255
        ),
        tf.uint8
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return tf.expand_dims(image, axis=0)


# =========================================
# 4. TEMPERATURE SCALING
# =========================================

def apply_temperature(probability):

    eps = 1e-7

    probability = np.clip(
        probability,
        eps,
        1 - eps
    )

    logit = np.log(
        probability / (1 - probability)
    )

    calibrated = 1 / (
        1 + np.exp(-logit / TEMPERATURE)
    )

    return float(calibrated)


# =========================================
# 5. PREDIKSI DAN ABSTENTION
# =========================================

def predict_with_confidence(probability):

    predicted_class = int(
        probability >= 0.5
    )

    confidence = (
        probability
        if predicted_class == 1
        else 1 - probability
    )

    if confidence < TAU:
        label = "Tidak yakin"
    else:
        label = CLASS_NAMES[predicted_class]

    return label, confidence, predicted_class


# =========================================
# 6. GENERATE GRAD-CAM
# =========================================

def generate_gradcam(
    grad_model,
    image_tensor,
    class_index
):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        if class_index == 1:
            score = predictions[:, 0]
        else:
            score = 1 - predictions[:, 0]

    gradients = tape.gradient(
        score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE
    )

    return tf.squeeze(
        heatmap
    ).numpy()


# =========================================
# 7. GRAD-CAM OVERLAY
# =========================================

def create_gradcam_overlay(
    image_array,
    heatmap
):

    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap)[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    overlay = (
        0.6 * image_array +
        0.4 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return overlay


# =========================================
# 8. MASKING AREA
# =========================================

def create_mask(
    heatmap,
    fraction,
    random_mask=False,
    seed=42
):

    height, width = heatmap.shape

    total_pixels = height * width

    n_pixels = max(
        1,
        int(round(total_pixels * fraction))
    )

    rng = np.random.default_rng(seed)

    if random_mask:

        selected_indices = rng.choice(
            total_pixels,
            size=n_pixels,
            replace=False
        )

    else:

        selected_indices = np.argsort(
            heatmap.reshape(-1)
        )[-n_pixels:]

    mask = np.zeros(
        total_pixels,
        dtype=bool
    )

    mask[selected_indices] = True

    return mask.reshape(
        height,
        width
    )


def apply_mask(
    image_array,
    mask
):

    masked_image = image_array.copy()

    # Menggunakan warna rata-rata gambar
    mean_color = image_array.mean(
        axis=(0, 1)
    )

    masked_image[mask] = mean_color

    return masked_image


# =========================================
# 9. DELETION TEST UNTUK SATU GAMBAR
# =========================================

def get_class_confidence(
    model,
    image_array,
    class_index
):

    image_tensor = (
        image_array.astype(np.float32) / 255.0
    )

    image_tensor = np.expand_dims(
        image_tensor,
        axis=0
    )

    prediction = model.predict(
        image_tensor,
        verbose=0
    )

    probability = float(
        np.squeeze(prediction)
    )

    # Mengukur confidence terhadap kelas awal
    if class_index == 1:
        return probability

    return 1 - probability


def run_deletion_test(
    model,
    image_array,
    heatmap,
    class_index
):

    # Confidence awal gambar
    original_confidence = get_class_confidence(
        model,
        image_array,
        class_index
    )

    results = []

    for i, fraction in enumerate(
        MASK_FRACTIONS
    ):

        # -----------------------------
        # Grad-CAM masking
        # -----------------------------

        gradcam_mask = create_mask(
            heatmap,
            fraction,
            random_mask=False
        )

        gradcam_image = apply_mask(
            image_array,
            gradcam_mask
        )

        gradcam_confidence = get_class_confidence(
            model,
            gradcam_image,
            class_index
        )

        gradcam_drop = (
            original_confidence -
            gradcam_confidence
        )

        # -----------------------------
        # Random masking
        # -----------------------------

        random_drops = []

        for repeat in range(RANDOM_REPEATS):

            random_mask = create_mask(
                heatmap,
                fraction,
                random_mask=True,
                seed=1000 + i * 100 + repeat
            )

            random_image = apply_mask(
                image_array,
                random_mask
            )

            random_confidence = get_class_confidence(
                model,
                random_image,
                class_index
            )

            random_drop = (
                original_confidence -
                random_confidence
            )

            random_drops.append(
                random_drop
            )

        results.append({
            "Masking (%)": fraction * 100,
            "Grad-CAM Drop": gradcam_drop,
            "Random Drop": np.mean(random_drops),
            "Grad-CAM Confidence": gradcam_confidence,
            "Random Confidence": original_confidence - np.mean(random_drops)
        })

    return pd.DataFrame(results)


# =========================================
# 10. TAMPILAN UTAMA
# =========================================

st.title("🔬 Malaria Cell Classification")

st.write(
    "Aplikasi klasifikasi citra sel darah merah "
    "menggunakan CNN dengan Grad-CAM, Deletion Test, "
    "dan kalibrasi keyakinan."
)

st.info(
    "Unggah gambar sel darah merah untuk melihat "
    "hasil klasifikasi, visualisasi Grad-CAM, "
    "dan evaluasi interpretabilitas."
)

st.divider()

uploaded_file = st.file_uploader(
    "Unggah gambar sel darah merah",
    type=["jpg", "jpeg", "png"]
)


# =========================================
# 11. PROSES GAMBAR
# =========================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image_hash = hashlib.md5(
        image_bytes
    ).hexdigest()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    st.subheader("Gambar yang Diunggah")

    st.image(
        image,
        caption="Citra sel darah merah",
        width=300
    )

    # Hapus hasil lama jika gambar berubah
    if st.session_state.get("image_hash") != image_hash:

        st.session_state["image_hash"] = image_hash

        st.session_state.pop(
            "analysis",
            None
        )

    if st.button(
        "Mulai Analisis",
        type="primary"
    ):

        try:

            with st.spinner(
                "Model sedang menganalisis gambar..."
            ):

                model, grad_model = load_model()

                processed_image = preprocess_image(
                    image
                )

                image_array = (
                    processed_image[0].numpy() * 255
                ).round().clip(
                    0, 255
                ).astype(np.uint8)

                # -------------------------
                # Prediksi
                # -------------------------

                raw_prediction = model.predict(
                    processed_image,
                    verbose=0
                )

                raw_probability = float(
                    np.squeeze(raw_prediction)
                )

                calibrated_probability = apply_temperature(
                    raw_probability
                )

                label, confidence, predicted_class = (
                    predict_with_confidence(
                        calibrated_probability
                    )
                )

                # -------------------------
                # Grad-CAM
                # -------------------------

                heatmap = generate_gradcam(
                    grad_model,
                    processed_image,
                    predicted_class
                )

                overlay = create_gradcam_overlay(
                    image_array,
                    heatmap
                )

                # -------------------------
                # Deletion Test
                # -------------------------

                deletion_results = run_deletion_test(
                    model,
                    image_array,
                    heatmap,
                    predicted_class
                )

                # Simpan hasil agar tidak dihitung
                # ulang ketika slider berubah
                st.session_state["analysis"] = {
                    "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                    "label": label,
                    "confidence": confidence,
                    "predicted_class": predicted_class,
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "image_array": image_array,
                    "deletion_results": deletion_results
                }

        except Exception as e:

            st.error(
                f"Terjadi kesalahan: {e}"
            )


# =========================================
# 12. MENAMPILKAN HASIL ANALISIS
# =========================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    st.divider()

    st.header("Hasil Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Klasifikasi",
            result["label"]
        )

    with col2:
        st.metric(
            "Confidence",
            f'{result["confidence"] * 100:.2f}%'
        )

    with col3:
        st.metric(
            "Threshold",
            f"{TAU * 100:.1f}%"
        )

    if result["label"] == "Parasitized":

        st.error(
            "Hasil: Parasitized"
        )

    elif result["label"] == "Uninfected":

        st.success(
            "Hasil: Uninfected"
        )

    else:

        st.warning(
            "Hasil: Tidak yakin. Model belum cukup "
            "yakin untuk memberikan klasifikasi."
        )

    # =====================================
    # 13. INFORMASI KALIBRASI MODEL
    # =====================================

    st.divider()

    st.header("Informasi Kalibrasi Model")

    st.write(
        "Bagian ini memperlihatkan perubahan probabilitas "
        "model sebelum dan setelah temperature scaling "
        "pada gambar yang sedang dianalisis."
    )

    cal_col1, cal_col2 = st.columns(2)

    with cal_col1:

        st.metric(
            "Temperature",
            f"{TEMPERATURE:.4f}"
        )

    with cal_col2:

        st.metric(
            "Threshold Tidak Yakin",
            f"{TAU * 100:.1f}%"
        )

    st.subheader(
        "Perbandingan Probabilitas"
    )

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:

        st.metric(
            "Probabilitas Mentah",
            f'{result["raw_probability"] * 100:.2f}%'
        )

        st.progress(
            result["raw_probability"]
        )

    with prob_col2:

        st.metric(
            "Probabilitas Setelah Kalibrasi",
            f'{result["calibrated_probability"] * 100:.2f}%'
        )

        st.progress(
            result["calibrated_probability"]
        )

    st.write(
        f"**Confidence kelas prediksi:** "
        f'{result["confidence"] * 100:.2f}%'
    )

    if result["confidence"] < TAU:

        st.warning(
            "Confidence berada di bawah threshold. "
            "Karena itu, model memberikan keputusan "
            "'Tidak yakin'."
        )

    else:

        st.success(
            "Confidence memenuhi threshold sehingga "
            "model memberikan hasil klasifikasi."
        )

    st.caption(
        "Temperature dan threshold ditentukan menggunakan "
        "data validasi, sedangkan probabilitas dan confidence "
        "di atas dihitung dari gambar yang diunggah."
    )

    # =====================================
    # 14. VISUALISASI GRAD-CAM
    # =====================================

    st.divider()

    st.header("Visualisasi Grad-CAM")

    st.write(
        "Grad-CAM memperlihatkan area pada gambar yang "
        "berkontribusi terhadap prediksi kelas model."
    )

    cam_col1, cam_col2 = st.columns(2)

    with cam_col1:

        st.image(
            result["image_array"],
            caption="Gambar Asli",
            use_container_width=True
        )

    with cam_col2:

        st.image(
            result["overlay"],
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    st.caption(
        "Heatmap dibuat berdasarkan gambar yang sedang "
        "diunggah dan kelas prediksi model."
    )

    # =====================================
    # 15. EVALUASI INTERPRETABILITAS MODEL
    # =====================================

    st.divider()

    st.header("Evaluasi Interpretabilitas Model")

    st.write(
        "Deletion Test digunakan untuk melihat perubahan "
        "confidence ketika area penting berdasarkan Grad-CAM "
        "dihilangkan. Hasilnya dibandingkan dengan penghapusan "
        "area secara acak pada gambar yang sama."
    )

    deletion_results = result["deletion_results"]

    st.subheader(
        "Pengaturan Area Masking"
    )

    selected_fraction = st.slider(
        "Pilih persentase area yang ingin dihapus",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    fraction = selected_fraction / 100

    # Masking untuk visualisasi sesuai slider
    gradcam_mask = create_mask(
        result["heatmap"],
        fraction,
        random_mask=False
    )

    random_mask = create_mask(
        result["heatmap"],
        fraction,
        random_mask=True,
        seed=42
    )

    gradcam_masked = apply_mask(
        result["image_array"],
        gradcam_mask
    )

    random_masked = apply_mask(
        result["image_array"],
        random_mask
    )

    st.write(
        f"**Perbandingan gambar setelah "
        f"{selected_fraction}% area dihapus**"
    )

    mask_col1, mask_col2 = st.columns(2)

    with mask_col1:

        st.image(
            gradcam_masked,
            caption="Grad-CAM Masking",
            use_container_width=True
        )

    with mask_col2:

        st.image(
            random_masked,
            caption="Random Masking",
            use_container_width=True
        )

    # Menampilkan hasil untuk tingkat masking terpilih
    selected_row = deletion_results[
        deletion_results["Masking (%)"] == selected_fraction
    ].iloc[0]

    st.subheader(
        "Perubahan Confidence"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            "Confidence Drop Grad-CAM",
            f'{selected_row["Grad-CAM Drop"] * 100:.2f}%'
        )

    with metric_col2:

        st.metric(
            "Confidence Drop Random",
            f'{selected_row["Random Drop"] * 100:.2f}%'
        )

    st.caption(
        "Confidence drop dihitung terhadap kelas prediksi awal. "
        "Random masking menggunakan rata-rata dari "
        f"{RANDOM_REPEATS} pengulangan."
    )

    # =====================================
    # 16. GRAFIK DELETION TEST
    # =====================================

    st.subheader(
        "Grafik Deletion Test"
    )

    st.write(
        "Grafik berikut dihitung berdasarkan gambar yang "
        "sedang diunggah, bukan hasil agregat dari dataset uji."
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        deletion_results["Masking (%)"],
        deletion_results["Grad-CAM Drop"],
        marker="o",
        linewidth=2,
        label="Grad-CAM"
    )

    ax.plot(
        deletion_results["Masking (%)"],
        deletion_results["Random Drop"],
        marker="o",
        linewidth=2,
        label="Random"
    )

    ax.set_xlabel(
        "Area yang Dihapus (%)"
    )

    ax.set_ylabel(
        "Confidence Drop"
    )

    ax.set_title(
        "Deletion Test pada Gambar yang Diunggah"
    )

    ax.set_xticks(
        [5, 10, 20, 30, 50]
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # Tabel hasil
    st.subheader(
        "Tabel Hasil Deletion Test"
    )

    display_results = deletion_results.copy()

    st.dataframe(
        display_results.style.format({
            "Masking (%)": "{:.0f}%",
            "Grad-CAM Drop": "{:.4f}",
            "Random Drop": "{:.4f}",
            "Grad-CAM Confidence": "{:.4f}",
            "Random Confidence": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Nilai confidence drop yang lebih besar menunjukkan "
        "bahwa penghapusan area tersebut menyebabkan penurunan "
        "confidence yang lebih besar. Hasil ini merupakan "
        "evaluasi satu gambar, bukan metrik akurasi model."
    )


# =========================================
# 17. FOOTER
# =========================================

st.divider()

st.caption(
    "Model CNN untuk klasifikasi citra sel darah merah. "
    "Hasil prediksi merupakan keluaran model dan bukan "
    "diagnosis medis."
)
