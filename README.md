# SentimenLab

SentimenLab adalah aplikasi analisis sentimen berbahasa Indonesia untuk komentar marketplace dan media sosial. Aplikasi menyediakan analisis teks tunggal, pemrosesan CSV, segmentasi aspek (ABSA), dashboard web, CLI, REST API lokal, dan ekstensi Chromium.

## Fitur

- Klasifikasi **positif**, **netral**, dan **negatif**.
- Analisis satu komentar atau banyak baris CSV.
- Segmentasi aspek berdasarkan kata kunci, seperti pengiriman, produk, pelayanan, dan harga.
- Tiga pilihan engine:

  | Engine | Cara kerja | Kebutuhan |
  | --- | --- | --- |
  | Linear SVM | TF-IDF dan classifier yang dilatih dari `dataset_ulasan.csv` | Dependensi dasar |
  | IndoBERT | Model IndoBERT yang di-fine-tune dengan dataset lokal | Dependensi Transformer dan bobot hasil fine-tuning |
  | Pre-trained | Model sentimen dari Hugging Face, tanpa fine-tuning lokal | Dependensi Transformer dan internet saat model pertama kali diunduh |

- Pengelolaan sampel data serta fine-tuning IndoBERT melalui dashboard.
- Ekstensi browser untuk memindai komentar pada halaman atau menganalisis teks yang dipilih.

Model Pre-trained yang tersedia:

- `w11wo/indonesian-roberta-base-sentiment-classifier` (pilihan awal)
- `cardiffnlp/twitter-xlm-roberta-base-sentiment`

Engine yang belum siap akan menampilkan pesan yang jelas, bukan otomatis berganti ke model lain. Jika model IndoBERT belum di-fine-tune, pilih Linear SVM atau latih IndoBERT terlebih dahulu.

## Persyaratan

- Python 3.9 atau lebih baru dengan dependensi yang kompatibel. Snapshot yang disertakan dibuat dan suite regresinya dijalankan pada Windows dengan Python 3.14.2.
- Browser berbasis Chromium untuk ekstensi.
- Koneksi internet saat pertama kali mengunduh model Hugging Face.

## Instalasi

Klon repositori:

```bash
git clone https://github.com/josapton/sentimen-lab.git
cd sentimen-lab
```

Buat dan aktifkan virtual environment.

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux atau macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Pasang dependensi dasar untuk dashboard, API, CLI, dan Linear SVM:

```bash
python -m pip install -r requirements.txt
```

Untuk memakai IndoBERT atau model Pre-trained, pasang juga dependensi Transformer:

```bash
python -m pip install -r requirements-transformers.txt
```

PyTorch dapat memerlukan perintah instalasi berbeda untuk perangkat GPU. Ikuti pemilih instalasi resmi PyTorch jika akan menggunakan GPU.

`requirements-lock.txt` berisi snapshot lengkap termasuk dependensi Transformer untuk lingkungan Windows Python 3.14.2. Gunakan berkas itu pada lingkungan Python yang sesuai jika ingin mereproduksi versi dependensi yang diuji, bukan bersamaan dengan dua berkas requirements lainnya.

## Dataset dan pelatihan

Dataset awal `dataset_ulasan.csv` menggunakan dua kolom:

```csv
komentar,sentimen
"Pengiriman cepat dan barang sesuai deskripsi",positif
"Barang belum saya gunakan",netral
"Paket datang terlambat dan kemasan rusak",negatif
```

Label yang diterima adalah `positif`, `netral`, dan `negatif`. Pelatihan memerlukan setidaknya lima contoh untuk setiap kelas. Dataset yang sudah ada tidak diganti otomatis, termasuk dataset kustom berukuran kecil. Fungsi pembuat dataset juga menolak menimpa file yang sudah ada kecuali dipanggil secara eksplisit dengan `overwrite=True`.

Dataset yang disertakan dibuat dengan bantuan Google Gemini dan **tidak termasuk dalam lisensi MIT proyek ini**. Repositori ini tidak memberikan izin penggunaan ulang dataset; periksa ketentuan layanan yang berlaku sebelum menggunakan atau mendistribusikannya.

Model SVM dan bobot IndoBERT tidak disertakan di Git. Dashboard dan API dapat membuat model SVM jika artefaknya belum ada, tetapi menjalankan training secara eksplisit di bawah ini lebih jelas saat menyiapkan clone baru. Untuk memilih IndoBERT di extension, jalankan fine-tuning terlebih dahulu.

Latih dan evaluasi model klasik:

```bash
python train_model.py
```

Pemilihan model dilakukan dengan cross-validation hanya pada partisi latih. Partisi holdout dipakai untuk evaluasi akhir, lalu model final dilatih ulang menggunakan seluruh dataset dan disimpan sebagai `model_sentimen.joblib`.

Untuk fine-tuning IndoBERT:

```bash
python train_model.py --bert --epochs 3
```

Perintah ini membutuhkan dependensi Transformer dan mengunduh bobot awal `indobenchmark/indobert-base-p1`. Bobot hasil fine-tuning disimpan ke `model_indobert/`.

## Menjalankan aplikasi

### Dashboard web

```bash
streamlit run app_web.py
```

Buka [http://localhost:8501](http://localhost:8501). Dashboard menyediakan analisis teks, pemrosesan CSV, pengelolaan dataset, dan fine-tuning IndoBERT.

### CLI

```bash
# Mode interaktif
python cli.py

# Analisis satu komentar dengan Linear SVM
python cli.py -t "Pengiriman cepat, tetapi kemasan penyok" -e svm

# Analisis dengan model Pre-trained
python cli.py -t "Kontennya bermanfaat banget" -e pretrained --json

# Proses CSV yang memiliki kolom komentar
python cli.py -f ulasan.csv -o hasil_analisis.csv
```

Pilihan engine: `svm`, `indobert`, dan `pretrained`. Model yang belum siap akan menghasilkan pesan yang menjelaskan cara menyiapkannya.

### REST API dan ekstensi

Jalankan API lokal:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

API menyediakan:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`

Contoh body untuk `POST /predict`:

```json
{
  "text": "Pengiriman cepat, tetapi kemasan penyok",
  "engine": "svm",
  "include_absa": true
}
```

Nilai `engine` adalah `svm`, `indobert`, atau `pretrained`. Untuk model Pre-trained, `pretrained_model` hanya menerima ID dalam katalog di atas. Batas API adalah 5.000 karakter per teks, 100 teks per batch, dan 100 klausa ABSA per permintaan. CORS dibatasi ke origin dashboard lokal. Jalankan API pada `127.0.0.1`; jangan mengeksposnya ke jaringan publik tanpa menambahkan autentikasi dan kontrol akses.

Muat ekstensi di Chrome, Edge, atau Brave:

1. Pastikan API berjalan di `127.0.0.1:8000`.
2. Buka halaman ekstensi browser (`chrome://extensions` atau `edge://extensions`) dan aktifkan **Developer mode**.
3. Pilih **Load unpacked**, lalu pilih folder `extension/`.
4. Buka halaman yang berisi komentar dan pilih **Pindai Komentar Halaman Ini**. Untuk pemindaian saat scroll, aktifkan **Auto-Scan saat Scroll**.
5. Untuk analisis satu komentar, blok teks pada halaman lalu gunakan menu konteks SentimenLab.

Ekstensi memakai content script pada semua situs agar teks yang dipilih dapat dianalisis di halaman mana pun. Teks halaman dikirim ke API lokal hanya saat pengguna meminta pemindaian atau mengaktifkan Auto-Scan.

## Batasan dan privasi

- ABSA saat ini mendeteksi aspek dengan kata kunci, memecah teks menjadi klausa, lalu mengklasifikasikan sentimen per klausa. Jika satu klausa menyebut beberapa aspek, aspek-aspek itu dapat memperoleh sentimen yang sama. Ini adalah perkiraan heuristik, bukan evaluasi aspek tersupervisi; validasi dengan data aspek dan sentimen yang dianotasi manusia sebelum mengandalkannya untuk keputusan penting.
- Hasil model bergantung pada mutu dan cakupan dataset. Dataset contoh bukan jaminan akurasi untuk semua domain atau gaya bahasa.
- API berjalan lokal dan ekstensi mengirim teks ke API lokal. Bobot model Pre-trained diunduh dari Hugging Face saat diperlukan; komentar tidak dikirim ke Hugging Face oleh kode inferensi.
- Model SVM, bobot IndoBERT, cache, dan hasil ekspor diabaikan oleh Git karena merupakan artefak lokal. Dataset sumber dan kode tetap disertakan.

## Pengujian

Jalankan suite regresi:

```bash
python -m unittest discover -s tests -v
```

## Struktur repositori

```text
.
├── .streamlit/config.toml
├── .gitignore
├── PRODUCT.md
├── api_server.py
├── app_web.py
├── cli.py
├── dataset_ulasan.csv
├── extension/
├── requirements.txt
├── requirements-transformers.txt
├── requirements-lock.txt
├── tests/
├── train_model.py
├── transformer_model.py
└── utils.py
```

## Lisensi

Kode sumber dan dokumentasi proyek dilisensikan di bawah [MIT License](./LICENSE). Lisensi tersebut tidak mencakup `dataset_ulasan.csv`, model Hugging Face, dependensi, atau aset pihak ketiga; masing-masing mengikuti ketentuan yang berlaku.
