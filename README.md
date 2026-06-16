# Klasifikasi Penyakit Tanaman Padi
### Sehat | Bacterial Leaf Blight | Blast

> Proyek UAS Mata Kuliah Machine Learning — Program Studi Teknologi Informasi, Universitas Lambung Mangkurat

---

## Deskripsi Proyek

Penelitian ini membangun dan membandingkan **5 model Machine Learning dan Deep Learning** untuk mengklasifikasikan tiga kondisi tanaman padi berdasarkan citra daun:

| Kelas | Keterangan |
|-------|-----------|
| **Sehat** | Daun padi tanpa gejala penyakit |
| **Bacterial Leaf Blight (Blight)** | Infeksi bakteri *Xanthomonas oryzae* pv. *oryzae* |
| **Blast** | Infeksi cendawan *Magnaporthe oryzae* |

Dataset dikumpulkan secara **mandiri dari lahan sawah Kalimantan Selatan** sebagai bentuk pemanfaatan kearifan lokal daerah.

---

## Struktur Repository

```
klasifikasi-penyakit-padi-ml/
│
├── Klasifikasi_Penyakit_Padi.ipynb   # Notebook utama (semua model)
├── README.md                                      # File ini
│
├── hasil/                                         # Output visualisasi
│   ├── 00_sample_dataset.png
│   ├── 01_bar_chart_semua_model.png
│   ├── 02_confusion_matrix_semua_model.png
│   ├── 03_heatmap_metrik.png
│   ├── 04_radar_chart.png
│   ├── history_MobileNetV2.png
│   ├── history_ResNet50.png
│   └── history_CNN(Feature Extractor).png
│
└── dataset/
    └── README.md     # Berisi link Google Drive dataset
```

---

## Dataset

- **Sumber:** Dataset primer — dikumpulkan mandiri dari lahan padi Kalimantan Selatan
- **Total:** 810 gambar (270 per kelas)
- **Komposisi per kelas:** 135 gambar *field background* (sawah) + 135 gambar *white background* (HVS)
- **Ukuran preprocessing:** 224 × 224 piksel
- **Pembagian:** 80% training (648) | 10% validation (81) | 10% testing (81)

 **Link Dataset Google Drive:** `https://drive.google.com/drive/folders/1v0GgpWXapnsErbLkRas0U6h9D321QgTM?usp=drive_link`

---

## Model yang Digunakan

| No | Model | Tipe | Keterangan |
|----|-------|------|-----------|
| 1 | **SVM** | Machine Learning | Baseline dengan PCA 200 komponen (variansi 91.38%) |
| 2 | **SVM + HOG** | Machine Learning | Fitur HOG 6.084 dimensi + SVM kernel RBF |
| 3 | **MobileNetV2** | Deep Learning | Transfer learning ImageNet, ~2.24M parameter |
| 4 | **ResNet-50** | Deep Learning | Transfer learning ImageNet, ~23M parameter |
| 5 | **CNN + SVM** | Deep Learning Hybrid | CNN custom feature extractor 256-dim + SVM classifier |

---

## Hasil Utama

| Model | Accuracy | Precision | Recall | F1-Score | Rank |
|-------|:--------:|:---------:|:------:|:--------:|:----:|
| **CNN + SVM** | **91.36%** | **92.30%** | **91.36%** | **91.42%** | 🥇 1 |
| MobileNetV2 | 82.72% | 82.70% | 82.72% | 82.57% | 🥈 2 |
| SVM | 74.07% | 75.39% | 74.07% | 74.37% | 🥉 3 |
| SVM + HOG | 74.07% | 75.53% | 74.07% | 74.06% | 4 |
| ResNet-50 | 70.37% | 70.73% | 70.37% | 70.48% | 5 |

>  **Model terbaik: CNN + SVM** dengan accuracy **91.36%**

### Hasil per Kelas - CNN + SVM (Model Terbaik)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Sehat | 0.96 | 0.93 | 0.94 | 27 |
| Blight | 0.88 | 0.96 | 0.92 | 27 |
| Blast | 0.96 | 0.85 | 0.90 | 27 |
| **Weighted Avg** | **0.923** | **0.914** | **0.914** | **81** |

---

## Cara Menjalankan Program

### Prasyarat
- Akun Google (untuk Google Colaboratory)
- Dataset tersimpan di Google Drive dengan struktur folder yang benar

### Langkah-langkah

**1. Buka notebook di Google Colab**

Upload file `Klasifikasi_Penyakit_Padi.ipynb` ke [Google Colab](https://colab.research.google.com/)

**2. Aktifkan GPU**
```
Runtime → Change runtime type → Hardware accelerator: GPU (T4)
```

**3. Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**4. Siapkan struktur folder dataset di Google Drive**
```
MyDrive/
└── dataset_penyakit_daun_padi/
    ├── Sehat/        ← 270 gambar (.jpg/.png)
    ├── Blight/       ← 270 gambar (.jpg/.png)
    └── Blast/        ← 270 gambar (.jpg/.png)
```

**5. Sesuaikan path di cell konfigurasi (cell ke-2)**
```python
DATASET_DIR = '/content/drive/MyDrive/dataset_penyakit_daun_padi'
```

**6. Jalankan semua cell secara berurutan**
```
Runtime → Run all
```

>  **Estimasi waktu total:** ~60–90 menit (tergantung ketersediaan GPU Colab)
> - SVM & SVM+HOG: ~5-10 menit
> - MobileNetV2: ~15-20 menit
> - ResNet-50: ~25-35 menit
> - CNN+SVM: ~20-25 menit

---

##  Library yang Digunakan

```
tensorflow==2.20.0
keras==3.13.2
scikit-learn==1.6.1
scikit-image
opencv-python-headless
numpy
pandas
matplotlib
seaborn
```

Install otomatis di cell pertama notebook:
```python
!pip install -q scikit-image opencv-python-headless
```

---

##  Visualisasi Hasil

Semua file visualisasi tersimpan otomatis ke folder output:

| File | Keterangan |
|------|-----------|
| `00_sample_dataset.png` | Contoh gambar per kelas setelah preprocessing |
| `01_bar_chart_semua_model.png` | Perbandingan 4 metrik semua model |
| `02_confusion_matrix_semua_model.png` | Confusion matrix 5 model berdampingan |
| `03_heatmap_metrik.png` | Heatmap accuracy, precision, recall, F1 |
| `04_radar_chart.png` | Radar chart profil performa multidimensi |
| `history_MobileNetV2.png` | Training history MobileNetV2 |
| `history_ResNet50.png` | Training history ResNet-50 |
| `history_CNN (Feature Extractor).png` | Training history CNN feature extractor |


---

## Kesimpulan

1. **CNN+SVM adalah model terbaik** (accuracy 91.36%) - sinergi ekstraksi fitur CNN dengan presisi klasifikasi SVM menghasilkan performa tertinggi di semua metrik dan semua kelas
2. **MobileNetV2 pilihan terbaik untuk deployment mobile** (82.72%, hanya ~2.24M parameter) - trade-off terbaik antara akurasi dan efisiensi
3. **Deep Learning secara umum mengungguli ML konvensional** - selisih 8–17 poin persentase
4. **ResNet-50 underperform** (70.37%) - frozen backbone terlalu rigid untuk dataset lokal Kalimantan Selatan yang kecil, memerlukan fine-tuning
5. **SVM dan SVM+HOG** menghasilkan accuracy sama (74.07%) namun distribusi error berbeda - HOG lebih baik mendeteksi Blast

---

## Informasi Mahasiswa

| | |
|--|--|
| **Nama** | Muhammad Azwin Hakim |
| **NIM** | 2310817310012 |
| **Mata Kuliah** | Machine Learning |
| **Program Studi** | Teknologi Informasi |
| **Universitas** | Universitas Lambung Mangkurat |
| **Tahun Akademik** | 2025/2026 |

---

## Referensi Utama

- Li et al. (2024). Deep learning-based methods for multi-class rice disease detection. *Agronomy*, 14(9), 1879.
- Yang et al. (2023). Stacking-based and improved CNN for rice leaf disease identification. *Frontiers in Plant Science*, 14, 1165940.
- Oktaviana et al. (2021). Klasifikasi penyakit padi menggunakan ResNet101. *Jurnal RESTI*, 5(6), 1216–1222.
- Asharaf et al. (2023). Rice leaf diagnosis using efficient CNN. *IJERT*, Special Issue ICCIDT-2023.
- Setiawan et al. (2023). Rice leaf disease classification using Nu-SVM. *Indonesian Journal of Data and Science*, 4(3).
