# 🔬 Malaria Cell Classification

Aplikasi web klasifikasi citra sel darah merah (**Uninfected** vs **Parasitized**) berbasis CNN, dilengkapi **Grad-CAM**, **Deletion Test**, dan **kalibrasi keyakinan** (temperature scaling + abstention).

> ⚠️ Hasil aplikasi ini adalah keluaran model dan **bukan diagnosis medis**.

🔗 **Demo:** https://kelompok2-deep-learning.streamlit.app/ 
🎥 **Video demo:** 

## Fitur

- Klasifikasi gambar sel darah merah (JPG, JPEG, PNG).
- **Kalibrasi keyakinan**: temperature scaling (T = 1.2148) dan threshold τ = 0.755. Jika confidence di bawah τ, hasilnya "Tidak yakin".
- **Grad-CAM**: visualisasi area gambar yang memengaruhi prediksi.
- **Deletion Test**: membandingkan penurunan confidence antara penghapusan area berdasarkan Grad-CAM dan penghapusan acak (5%–50%, 10 pengulangan), lengkap dengan slider, grafik, dan tabel.

## Struktur Repository

```
.
├── app.py               # Aplikasi Streamlit + skrip inferensi
├── final_model.keras    # Model CNN terlatih
├── requirements.txt     # Dependensi
└── README.md
```

## Alur Kerja Aplikasi

1. Gambar diunggah → diubah ke RGB, di-resize ke 128×128, dinormalisasi (0–1).
2. Model memprediksi probabilitas kelas *Parasitized* atau *Uninfected*.
3. Probabilitas dikalibrasi dengan temperature scaling.
4. Confidence dibandingkan dengan threshold τ untuk menentukan label atau "Tidak yakin".
5. Grad-CAM dihitung dari lapisan Conv2D terakhir.
6. Deletion Test dijalankan dan hasilnya ditampilkan.

## Instalasi dan Menjalankan Secara Lokal

```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## requirements.txt (contoh, sesuaikan dengan versi saat pelatihan)

```
streamlit
tensorflow
numpy
pandas
matplotlib
pillow
```

## Deployment (Streamlit Community Cloud)

1. Unggah seluruh berkas ke repository GitHub (gunakan Git LFS jika model > 100 MB).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan GitHub.
3. Klik **New app**, pilih repository, branch `main`, dan file `app.py`.
4. Klik **Deploy**.

## Cara Penggunaan

1. Unggah gambar sel darah merah.
2. Klik **Mulai Analisis**.
3. Lihat hasil prediksi, informasi kalibrasi, Grad-CAM, dan evaluasi Deletion Test.
4. Geser slider untuk mengubah persentase area yang dihapus.

## Parameter

| Parameter | Nilai | Keterangan |
|---|---|---|
| IMAGE_SIZE | 128×128 | Ukuran input model |
| TEMPERATURE | 1.2148 | Hasil kalibrasi pada validation set |
| TAU | 0.755 | Threshold confidence untuk "Tidak yakin" |
| MASK_FRACTIONS | 5%–50% | Tingkat masking Deletion Test |
| RANDOM_REPEATS | 10 | Pengulangan random masking |

## Catatan

- Deletion Test dihitung pada **satu gambar yang diunggah**, bukan agregat dataset uji.
- Preprocessing disamakan dengan notebook pelatihan agar hasil konsisten.

## Lisensi

[Tuliskan lisensi, misalnya MIT]