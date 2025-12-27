# Klasifikasi Gempa Indonesia

Proyek Data Mining untuk mengklasifikasikan gempa bumi di Indonesia berdasarkan magnitudo menggunakan algoritma Naïve Bayes, sebagai bagian dari artikel yang sedang ditulis.

## Deskripsi Project

Project ini menganalisis dan mengklasifikasikan data gempa bumi di Indonesia periode 2023-2025 berdasarkan kategori magnitudo (Ringan, Sedang, Kuat) menggunakan metode Gaussian Naïve Bayes.

## Struktur Project

```
KlasifikasiGempa/
├── klasifikasi-gempa.ipynb    # Notebook Jupyter lengkap
├── klasifikasi-gempa.py       # Script Python
├── README.md
└── csv/
    ├── katalog_gempa.csv      # Dataset asli
    ├── hasil_sort.csv         # Dataset setelah sorting
    └── gempa_final.csv        # Dataset final hasil training
```

## Tahapan Project

### 1. **Pemuatan dan Inspeksi Data**
- Load dataset katalog gempa dari CSV
- Melihat informasi dan struktur data awal
- Dataset source: `csv/katalog_gempa.csv`

### 2. **Sorting Data**
- Sorting data berdasarkan `datetime` (descending)
- Seleksi 30,000 baris data (dari indeks 2600-32600)
- Memilih 10 kolom utama
- Output: `csv/hasil_sort.csv`

### 3. **Menangani Missing Value**
- **Kategorikal**: Mengisi dengan modus (nilai terbanyak)
- **Numerikal**: Mengisi dengan median
- Verifikasi tidak ada missing value tersisa

### 4. **Deteksi dan Penanganan Outlier**
- Menggunakan metode **IQR (Interquartile Range)**
- Kolom yang dianalisis: `phasecount`, `depth`
- Visualisasi menggunakan boxplot
- Filter data: 25,981 data bersih
- Output: `gempa_clean.csv`

### 5. **Klasifikasi Magnitudo**

Kategori gempa berdasarkan magnitudo:
```python
def kategori_magnitude(m):
    if m < 3.0:
        return 'Ringan'
    elif 3.0 <= m < 5.0:
        return 'Sedang'
    else:
        return 'Kuat'
```

### 6. **Preprocessing**

**Encoding Kategorikal:**
- `mag_type` → `mag_type_encoded` (LabelEncoder)
- `location` → `location_encoded` (LabelEncoder)

**Normalisasi Fitur Numerik (Min-Max Scaler):**
- `latitude`
- `longitude`
- `depth`
- `phasecount`
- `azimuth_gap`

### 7. **Training Model**

**Fitur yang digunakan:**
```python
features = ['latitude', 'longitude', 'depth', 'phasecount', 'azimuth_gap', 'mag_type_encoded']
```

**Target:**
- `kategori_magnitude` (Ringan, Sedang, Kuat)

**Model:**
- Algoritma: **Gaussian Naïve Bayes**
- Split data: 80% training, 20% testing
- Stratified split untuk menjaga proporsi kelas

### 8. **Evaluasi Model**

Metrik evaluasi:
- **Accuracy Score**
- **Classification Report** (precision, recall, f1-score)
- **Confusion Matrix**
- Visualisasi Confusion Matrix menggunakan heatmap

## Visualisasi

Project ini menghasilkan beberapa visualisasi:

1. **Boxplot** untuk deteksi outlier (`phasecount`, `depth`)
2. **Histogram** distribusi magnitudo gempa
3. **Scatter plot** sebaran lokasi gempa (longitude vs latitude)
4. **Heatmap** Confusion Matrix hasil klasifikasi

## Library yang Digunakan

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

## Output Dataset

- **gempa_clean.csv**: Dataset bersih setelah handling missing value dan outlier
- **gempa_final.csv**: Dataset final dengan fitur encoding dan normalisasi

## Cara Menjalankan

### Menggunakan Jupyter Notebook:
```bash
jupyter notebook klasifikasi-gempa.ipynb
```

### Menggunakan Python Script:
```bash
python klasifikasi-gempa.py
```

## Catatan

- Dataset mencakup periode **2023-2025**
- Total data setelah cleaning: **25,981 records**
- Range magnitudo: **0.19 - 7.60**
- Model menggunakan stratified split untuk menjaga proporsi kelas target

## Author

Project ini dibuat untuk keperluan klasifikasi data gempa Indonesia menggunakan machine learning.

---

**Note**: Pastikan file CSV berada di folder `csv` atau sesuaikan path di code jika menggunakan lokasi berbeda.
