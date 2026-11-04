# Document Image Classification SD234504

Klasifikasi otomatis citra dokumen ke dalam **16 kategori** menggunakan **PyTorch**. Proyek ini merupakan solusi untuk kompetisi Kaggle *Document Image Classification SD234504*.

**Hasil utama:** ResNet18 pretrained yang di-*fine-tune* mencapai **akurasi validasi 87,18%** (macro F1 0,8703).

---

## Daftar Isi

1. [Latar Belakang dan Tujuan](#1-latar-belakang-dan-tujuan)
2. [Dataset](#2-dataset)
3. [Struktur Repository](#3-struktur-repository)
4. [Instalasi](#4-instalasi)
5. [Cara Unduh Data](#5-cara-unduh-data)
6. [Cara Menjalankan](#6-cara-menjalankan)
7. [Workflow Analisis](#7-workflow-analisis)
8. [Hasil Evaluasi](#8-hasil-evaluasi)
9. [Laporan Interpretasi](#9-laporan-interpretasi)
10. [Keterbatasan dan Pengembangan Lanjutan](#10-keterbatasan-dan-pengembangan-lanjutan)
11. [Referensi](#11-referensi)

---

## 1. Latar Belakang dan Tujuan

Pengelompokan dokumen secara manual memerlukan waktu yang besar dan rentan terhadap kesalahan. Proyek ini membangun model klasifikasi citra yang menentukan jenis dokumen secara otomatis.

- **Tujuan:** membangun model dengan akurasi setinggi mungkin pada data uji.
- **Metrik:** *categorical accuracy*, yaitu proporsi prediksi yang sesuai dengan label sebenarnya.

## 2. Dataset

| Item | Keterangan |
|---|---|
| Sumber | Kaggle, kompetisi `document-image-classification-sd234504` |
| Format citra | TIFF, grayscale (mode `L`), satu frame per berkas |
| Data training | 32.003 citra berlabel (32.002 terbaca, lihat [Bagian 9](#9-laporan-interpretasi)) |
| Data test | 7.994 citra tanpa label |
| Jumlah kelas | 16 |
| Berkas tabel | `train.csv` (id, filename, class_name, label), `test.csv` (id, filename), `sample_submission.csv` (id, label) |

**Daftar kelas (label 0–15):** advertisement, budget, email, file_folder, form, handwritten, invoice, letter, memo, news_article, presentation, questionnaire, resume, scientific_publication, scientific_report, specification.

## 3. Struktur Repository

```
document-image-classification-sd234504/
├── .gitignore                  # mengabaikan data/, checkpoint/, cache/, outputs/
├── README.md                   # dokumentasi proyek (file ini)
├── requirements.txt            # daftar dependensi Python
├── notebooks/
│   └── document_image_classification.ipynb   # alur analisis: EDA → preprocessing → modelling → evaluasi → prediksi
└── src/
    ├── __init__.py             # penanda paket
    ├── config.py               # path proyek dan seluruh hyperparameter (CONFIG)
    ├── utils.py                # seed, pemilihan device, hashing MD5, cek ruang disk
    ├── data.py                 # pemuatan CSV, pembersihan duplikat, split, augmentasi, cache, Dataset
    ├── model.py                # DocumentCNN (buatan sendiri) dan DocumentResNet (ResNet18 pretrained)
    ├── engine.py               # loop training, validasi, early stopping, checkpoint, prediksi batch
    └── predict.py              # pembuatan dan verifikasi submission.csv
```

Folder yang **dibuat saat menjalankan kode** dan **tidak** masuk repository:

| Folder | Isi | Ukuran perkiraan |
|---|---|---|
| `data/` | dataset hasil unduhan Kaggle | besar (> 100 MB) |
| `checkpoint/` | bobot model `.pt` (`best_*.pt`, `last_*.pt`) | puluhan MB |
| `cache/` | citra ter-resize dalam format `.npy` | ±13 GB |
| `outputs/` | `submission.csv` | kecil |

## 4. Instalasi

Persyaratan: Python 3.10 atau lebih baru. GPU dengan CUDA sangat disarankan, karena training CPU memakan waktu sangat lama.

```bash
# 1. Clone repository
git clone https://github.com/<username>/document-image-classification-sd234504.git
cd document-image-classification-sd234504

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Pasang dependensi
pip install -r requirements.txt
```

> **Catatan GPU:** jika ingin memakai CUDA, pasang PyTorch sesuai versi CUDA kamu dari [pytorch.org](https://pytorch.org) sebelum menjalankan `pip install -r requirements.txt`.

## 5. Cara Unduh Data

Data **tidak** disertakan dalam repository karena ukurannya besar dan merupakan milik kompetisi. Ikuti langkah berikut.

**Prasyarat:** punya akun Kaggle dan sudah menerima aturan kompetisi di halaman kompetisi *Document Image Classification SD234504*.

**Langkah 1 — Pasang Kaggle CLI** (sudah termasuk dalam `requirements.txt`):

```bash
pip install kaggle
```

**Langkah 2 — Simpan API token:**

1. Buka Kaggle → *Settings* → *API* → **Create New Token**. Berkas `kaggle.json` akan terunduh.
2. Pindahkan berkas tersebut ke folder konfigurasi Kaggle:

```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

```powershell
# Windows (PowerShell)
mkdir $HOME\.kaggle -Force
move $HOME\Downloads\kaggle.json $HOME\.kaggle\
```

**Langkah 3 — Unduh dan ekstrak data** dari root repository:

```bash
mkdir -p data
kaggle competitions download -c document-image-classification-sd234504 -p data
unzip -q data/document-image-classification-sd234504.zip -d data
rm data/document-image-classification-sd234504.zip     # opsional, menghemat ruang
```

Pada Windows, ekstrak zip dengan klik kanan → *Extract All* ke folder `data/`.

**Struktur hasil ekstraksi** yang diharapkan:

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   └── train/<class_name>/*.tif      # contoh: data/train/train/resume/50497291-7292.tif
└── test/
    └── test/*.tif                    # contoh: data/test/test/test_00001.tif
```

Jika struktur folder hasil ekstraksi berbeda, path dapat disesuaikan pada fungsi `load_competition_tables` di `src/data.py`.

## 6. Cara Menjalankan

### Opsi A — Notebook (disarankan untuk analisis)

```bash
jupyter lab notebooks/document_image_classification.ipynb
```

Pilih **Run → Run All Cells**. Notebook dijalankan dari atas ke bawah dan memanggil fungsi dari folder `src/`. Notebook disimpan **tanpa output** agar repository tetap ringan, sehingga hasil akan muncul kembali setelah dijalankan.

### Opsi B — Mengubah konfigurasi

Semua hyperparameter berada di `src/config.py`. Contoh perubahan:

| Ingin | Ubah di `CONFIG` |
|---|---|
| Memakai CNN buatan sendiri (tanpa pretrained) | `"model_name": "docnet"` |
| Mengubah jumlah epoch | `"epochs"` |
| Mematikan cache jika disk terbatas | `"use_cache": False` |
| Mengubah ukuran input | `"img_h"`, `"img_w"` |

### Hasil yang dihasilkan

| Berkas | Lokasi | Keterangan |
|---|---|---|
| Bobot terbaik | `checkpoint/best_resnet18.pt` | checkpoint dengan validation loss terendah |
| Bobot terakhir | `checkpoint/last_resnet18.pt` | dipakai untuk melanjutkan training (`RESUME = True`) |
| Submission | `outputs/submission.csv` | siap diunggah ke Kaggle |

**Mengirim submission ke Kaggle:**

```bash
kaggle competitions submit -c document-image-classification-sd234504 \
    -f outputs/submission.csv -m "resnet18 fine-tune, 384x288"
```

## 7. Workflow Analisis

Alur kerja mengikuti struktur praktikum: **import → baca data → EDA → preprocessing → modelling → evaluasi → prediksi**.

| Tahap | Isi | Lokasi kode |
|---|---|---|
| 1. Import & read data | Membaca `train.csv`, `test.csv`, `sample_submission.csv`; memeriksa keberadaan file | `load_competition_tables` (`src/data.py`) |
| 2. EDA | Missing value, konsistensi label, distribusi kelas, metadata TIFF, resolusi, ukuran file, contoh gambar, duplikasi MD5 | notebook bagian 2; `read_image_metadata`, `add_md5` |
| 3. Preprocessing | Pembersihan duplikat, split *group-aware*, cache citra, augmentasi, DataLoader | notebook bagian 3; `clean_duplicates`, `split_train_val`, `build_all_caches`, `build_transforms`, `build_dataloaders` |
| 4. Modelling | ResNet18 fine-tune atau DocumentCNN, AdamW, cosine annealing, mixed precision, early stopping | notebook bagian 4; `build_model`, `train_model` |
| 5. Evaluation | Classification report, confusion matrix, akurasi per kelas | notebook bagian 5; `predict_batches` |
| 6. Prediksi | Menghasilkan dan memverifikasi `submission.csv` | notebook bagian 6; `write_submission` |

## 8. Hasil Evaluasi

Model terbaik dipilih berdasarkan **validation loss terendah**, yaitu checkpoint **epoch 12**.

| Metrik (validasi, n = 6.390) | Nilai |
|---|---|
| **Accuracy** | **0,8718** (5.571 benar dari 6.390) |
| Macro F1 | 0,8703 |
| Weighted F1 | 0,872 |
| Validation loss terbaik | 0,9550 |

**Performa per kelas:**

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| advertisement | 0,900 | 0,902 | 0,901 | 410 |
| budget | 0,861 | 0,835 | 0,848 | 400 |
| email | 0,982 | 0,984 | **0,983** | 387 |
| file_folder | 0,874 | 0,969 | 0,919 | 415 |
| form | 0,738 | 0,762 | **0,750** | 374 |
| handwritten | 0,893 | 0,928 | 0,910 | 404 |
| invoice | 0,907 | 0,848 | 0,877 | 402 |
| letter | 0,879 | 0,870 | 0,874 | 416 |
| memo | 0,928 | 0,919 | 0,923 | 418 |
| news_article | 0,886 | 0,881 | 0,883 | 378 |
| presentation | 0,752 | 0,797 | **0,774** | 365 |
| questionnaire | 0,801 | 0,823 | 0,812 | 351 |
| resume | 0,959 | 0,927 | 0,943 | 425 |
| scientific_publication | 0,910 | 0,895 | 0,903 | 409 |
| scientific_report | 0,731 | 0,722 | **0,726** | 410 |
| specification | 0,936 | 0,864 | 0,899 | 426 |

## 9. Laporan Interpretasi

### 9.1 Temuan dari EDA

**Kualitas data baik.** Tidak ada nilai kosong di ketiga tabel. Setiap label memetakan tepat ke satu `class_name` (16 pasangan), label berada pada rentang 0–15, dan folder kelas pada path sesuai dengan `class_name`, baik di CSV maupun di disk.

**Kelas hampir seimbang.** Setiap kelas memiliki 6,09% sampai 6,42% dari total data, dengan rasio maksimum/minimum 1,055 dan koefisien variasi 0,0134. Ketidakseimbangan ini tidak signifikan, sehingga *class weighting* tidak diperlukan.

**Satu berkas rusak.** Satu berkas (`train_08795`) tidak dapat dibuka oleh PIL dan dibuang sebelum tahap berikutnya, sehingga tersisa 32.002 citra.

**Duplikasi perlu ditangani.** Terdapat 11 grup citra yang isinya identik (59 baris) di data training. Tiga grup di antaranya memiliki label berbeda. Label yang benar pada grup tersebut tidak dapat ditentukan, sehingga seluruh grup konflik dibuang. Grup identik dengan label sama disisakan satu salinan. Total 51 baris dibuang, dan data training tersisa **31.951 citra**. Di data test terdapat 16 duplikat internal. Setelah pembersihan, **6 citra test masih identik dengan citra training**.

**Resolusi relatif seragam.** Tinggi seluruh citra adalah 1.000 piksel. Lebar paling umum adalah 754 piksel (47% training dan 48% test). Distribusi lebar dan ukuran berkas di training serta test hampir sama, sehingga data test diasumsikan berasal dari distribusi yang sama. Seluruh citra juga memiliki sisi terpendek di atas 224 piksel, sehingga semuanya diperkecil saat preprocessing.

### 9.2 Keputusan Preprocessing dan Alasannya

**Split yang mencegah kebocoran.** Validasi dibuat dengan `StratifiedGroupKFold` berdasarkan MD5. Hasilnya, **0 citra validasi identik dengan citra training**, sehingga estimasi akurasi tidak bias ke atas akibat duplikasi. Distribusi kelas pada training dan validasi juga selisihnya maksimal 0,74 poin persentase.

**Augmentasi yang konservatif.** Augmentasi hanya berupa rotasi kecil (maks. 3°), translasi, skala 95–105%, dan jitter kecerahan atau kontras. Flip tidak dipakai karena orientasi dan tata letak dokumen merupakan ciri kelas (misalnya letter, memo, dan form). Random crop agresif juga dihindari karena dapat memotong bagian yang menjadi pembeda antar kelas.

**Normalisasi ImageNet.** Mean dan std ImageNet dipakai karena bobot ResNet18 pretrained dilatih dengan normalisasi tersebut.

### 9.3 Interpretasi Proses Training

**Kurva loss dan akurasi.** Validation loss turun secara monoton dari 1,2929 (epoch 1) menjadi 0,9550 (epoch 12). Validation accuracy naik dari 74,26% menjadi 87,17%. Checkpoint terbaik jatuh pada epoch terakhir dan early stopping tidak aktif. Akurasi validasi mulai mendatar sejak epoch 10 (87,03%, 86,98%, 87,17%), sehingga model hampir konvergen meski loss masih menurun sedikit.

**Gap antara training dan validasi.** Pada epoch terakhir, train accuracy mencapai 99,11%, sedangkan validation accuracy 87,17%. Sebagian selisih ini wajar, karena akurasi training dihitung saat augmentasi dan dropout aktif, sehingga angkanya tidak sepenuhnya sebanding. Namun selisih sekitar 12 poin tetap menunjukkan bahwa model cukup banyak menghafal data training. Regularisasi yang dipakai (weight decay 0,02, dropout 0,5, label smoothing 0,1) membantu, tetapi belum sepenuhnya menutup gap tersebut.

**Selisih kecil 0,8717 dan 0,8718.** Angka validasi pada log training (0,8717) dihitung dengan mixed precision FP16, sedangkan evaluasi ulang setelah training berjalan dalam FP32 dan menghasilkan 0,8718. Selisih 0,0001 ini berasal dari presisi numerik dan tidak bermakna secara statistik.

### 9.4 Interpretasi Hasil Evaluasi

**Akurasi dan keseimbangan metrik.** Akurasi validasi sebesar 87,18% dengan macro F1 0,8703 dan weighted F1 0,872. Ketiganya hampir sama karena data memang seimbang, sehingga performa model tidak timpang antar kelas.

**Ketidakpastian estimasi.** Dengan 6.390 sampel validasi, standar error akurasi sekitar 0,42 poin persentase. Interval kepercayaan 95% berada pada kisaran **86,4% sampai 88,0%**. Angka ini hanya menggambarkan ketidakpastian pada data validasi, bukan pada data test.

**Kelas yang paling kuat.** Email (F1 0,983), resume (0,943), dan memo (0,923) memiliki tata letak yang khas sehingga mudah dikenali. File_folder memiliki recall tinggi (0,969).

**Kelas yang paling lemah.** Scientific_report (F1 0,726), form (0,750), dan presentation (0,774) menjadi kelas dengan performa terendah.

**Pola kesalahan.** Kesalahan terbesar terjadi antar pasangan yang visualnya mirip:

| Label sebenarnya | Diprediksi sebagai | Jumlah |
|---|---|---:|
| scientific_report | presentation | 30 |
| presentation | scientific_report | 25 |
| invoice | form | 18 |
| scientific_report | questionnaire | 16 |
| scientific_report | form | 16 |
| form | questionnaire | 15 |
| questionnaire | form | 14 |
| invoice | budget | 14 |

Kesalahan terkonsentrasi pada dokumen dengan teks padat dan tata letak tabular atau formulir. Kesalahan scientific_report ↔ presentation terjadi dua arah dengan jumlah yang hampir sama, sehingga kedua kelas ini sulit dibedakan oleh model saat ini.

**Asimetri presisi pada file_folder.** Recall file_folder (0,969) jauh lebih tinggi daripada presisinya (0,874). Artinya model cenderung menebak file_folder untuk citra yang sebenarnya termasuk kelas lain.

**Hipotesis penyebab.** Citra diperkecil dari sekitar 754×1000 piksel menjadi 288×384 piksel. Pengecilan ini bisa menurunkan keterbacaan teks kecil, yang sebenarnya penting untuk membedakan scientific_report dan presentation. Hipotesis ini belum diuji. Pengujian dengan resolusi lebih tinggi merupakan langkah lanjutan yang paling jelas.

### 9.5 Implikasi

Untuk pengelompokan otomatis, model ini cukup andal pada sebagian besar kelas, terutama email, resume, dan memo. Untuk tiga kelas dengan F1 di bawah 0,78 (scientific_report, form, presentation), prediksi sebaiknya ditinjau manual atau diberi ambang keyakinan sebelum dipakai secara otomatis.

## 10. Keterbatasan dan Pengembangan Lanjutan

**Keterbatasan:**
- Evaluasi hanya memakai satu fold validasi (sekitar 20% data), belum cross-validation penuh.
- Belum ada test-time augmentation, ensemble, atau tuning hyperparameter yang sistematis.
- Enam citra test yang identik dengan training tidak dimanfaatkan secara eksplisit.
- Resolusi input dibatasi 384×288 karena keterbatasan memori dan waktu training.
- Skor public leaderboard tidak dicatat di repository.

**Pengembangan lanjutan yang disarankan:**
- Menguji resolusi input yang lebih tinggi, terutama untuk kelas teks padat.
- Mencoba arsitektur lain yang sudah pretrained, misalnya EfficientNet atau ConvNeXt.
- Menambahkan test-time augmentation dan ensemble beberapa model.
- Menjalankan lebih banyak epoch dengan kesabaran (`patience`) yang lebih besar.

## 11. Referensi

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. (AdamW)
- Kaggle. *Document Image Classification SD234504* (kompetisi).
- PyTorch Documentation: https://pytorch.org/docs
