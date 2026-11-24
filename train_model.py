import os
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_ulasan.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_sentimen.joblib")

def buat_dataset_300(overwrite: bool = False):
    if os.path.exists(DATASET_PATH) and not overwrite:
        raise FileExistsError(
            f"Dataset '{DATASET_PATH}' sudah ada. Gunakan overwrite=True setelah membuat cadangan."
        )

    # ==========================================================
    # 1. KELAS POSITIF (50 Marketplace + 50 Media Sosial = 100)
    # ==========================================================
    positif_marketplace = [
        "Produk ini sangat bagus dan kualitasnya memuaskan",
        "Pelayanan ramah, pengiriman cepat, mantap sekali!",
        "Suka banget sama barangnya, sesuai ekspektasi",
        "Harga terjangkau tapi kualitas luar biasa bagus",
        "Recommended seller, barang ori dan berfungsi baik",
        "Luar biasa, aplikasi berjalan lancar tanpa kendala",
        "Oke banget barangnya, puas belanja di sini",
        "Keren, desainnya rapi dan sangat nyaman dipakai",
        "Terima kasih, pesanan mendarat dengan selamat dan aman",
        "Mantap jiwa, respon penjual cepat dan solutif",
        "Top markotop, kualitas bahan tebal dan awet",
        "Sangat terbantu dengan fitur ini, kerja jadi mudah",
        "Bintang lima, bakal langganan terus di toko ini",
        "Packing rapi, barang mulus tidak ada cacat sedikitpun",
        "Bagus, oke, sesuai gambar dan deskripsi",
        "Worth it banget dengan harga segini, top!",
        "Pengemasan super aman pakai bubble wrap tebal",
        "Admin sangat membantu saat saya bingung pilih ukuran",
        "Hasilnya memuaskan, tidak menyesal beli produk ini",
        "Kualitas juara, pengiriman kilat sehari langsung sampai",
        "Kurir sangat sopan dan mengantar paket tepat waktu",
        "Baterai sangat awet dipakai seharian penuh tanpa dicas",
        "Antarmuka aplikasi bersih, ringan, dan mudah dipahami",
        "Harga paling murah dibanding toko sebelah tapi original",
        "Warna dan modelnya elegan, istri saya sangat suka",
        "Proses klaim garansi mudah dan ditangani dengan cepat",
        "Fitur barunya sangat berguna untuk produktivitas harian",
        "Mantul gan, barang berfungsi normal seratus persen",
        "Segel masih utuh, produk resmi dan bergaransi",
        "Pelayanan terbaik yang pernah saya rasakan saat belanja online",
        "Suara jernih, bass mantap, sangat puas dengan performanya",
        "Bahan adem dan jahitan sangat rapi, kualitas premium",
        "Pengiriman ke luar pulau tetap cepat dan aman terkendali",
        "Sangat direkomendasikan untuk pemula maupun profesional",
        "Tidak ada bug sama sekali setelah pembaruan sistem",
        "Seller amanah, dikasih bonus aksesoris tambahan juga",
        "Ukuran pas di badan, nyaman dipakai beraktivitas",
        "Respon chat admin hitungan detik langsung dibalas ramah",
        "Kinerja laptop kencang, render video jadi jauh lebih singkat",
        "Kemasan eksklusif dan sangat aman dari benturan",
        "Puas sekali, pesanan datang lebih cepat dari estimasi",
        "Kualitas gambar tajam dan jernih, luar biasa bagus",
        "Sistem pembayaran praktis dan langsung terverifikasi otomatis",
        "Barang kokoh dan tidak ringkih, pemakaian jangka panjang oke",
        "Senang berbelanja di sini, selalu konsisten menjaga kualitas",
        "Tampilan modern dan fitur lengkap sesuai kebutuhan saya",
        "Hemat daya dan tidak cepat panas saat digunakan lama",
        "Penjelasan deskripsi produk sangat jujur dan akurat",
        "Komplit semua kelengkapannya ada di dalam kotak",
        "Sempurna, tidak ada kekurangan sedikitpun pada produk ini",
    ]

    positif_medsos = [
        "Keren banget editannya, transisinya mulus parah",
        "Makasih min infonya sangat bermanfaat buat pemula",
        "Valid no debat, setuju banget sama thread ini",
        "Ngakak banget liat kelakuan kucingnya, menghibur di kala penat",
        "Menyala abangku, terus berkarya jangan kasih kendor",
        "Bagus banget puisinya, jujur tersentuh bacanya",
        "Konten edukatif kayak gini yang harusnya rame dan viral",
        "Salut sama warga sekitar yang kompak gotong royong",
        "Penjelasannya daging semua, mudah dipahami orang awam",
        "Gokil sih inovasinya, bangga sama karya anak bangsa",
        "Alhamdulillah akhirnya kabar baik yang ditunggu datang juga",
        "Suka banget sama pembawaannya yang santai tapi berbobot",
        "Respect buat kreatornya yang mau riset mendalam sebelum posting",
        "Visualnya memanjakan mata, pemilihan musiknya juga pas banget",
        "Akhirnya ada yang bahas topik ini secara objektif dan jernih",
        "Terharu bacanya, sehat selalu orang-orang baik di luar sana",
        "Inspiratif sekali ceritanya, bikin semangat kerja lagi hari ini",
        "Kocak abis komentar netizen di sini, sukses bikin ketawa seharian",
        "Mantap min, selalu update berita terkini dengan akurat",
        "Lagu ini gak pernah bosen didengerin, liriknya dalem banget",
        "Setuju seratus persen, semoga kebijakan ini berjalan lancar",
        "Hasil fotonya tajam dan estetik banget, keren parah",
        "Terima kasih sudah berbagi ilmu gratis, sangat membantu tugas saya",
        "Adem banget liat interaksi di kolom komentar postingan ini",
        "Definisi konten berkualitas tanpa perlu drama dan sensasi",
        "Juara banget aktingnya, dapet banget emosinya ke penonton",
        "Sangat menginspirasi anak muda untuk mulai peduli lingkungan",
        "Layout dan desain grafisnya rapi banget, enak dibaca",
        "Bangga banget liat prestasi tim Indonesia di kancah dunia",
        "Tutorialnya langsung berhasil dipraktikkan, makasih banyak bang",
        "Humor yang cerdas dan segar, gak nyakitin pihak mana pun",
        "Salute, langkah nyata yang patut dicontoh daerah lainnya",
        "Makin sukses terus channelnya, selalu ditunggu video barunya",
        "Kualitas audio jernih banget, enak didengerin sambil kerja",
        "Info penting banget ini, izin bagikan ke grup keluarga ya min",
        "Pemikiran yang sangat dewasa dan solutif menghadapi masalah",
        "Gak salah follow akun ini, tiap hari selalu dapet insight baru",
        "Seneng banget liat perkembangannya sekarang jauh lebih baik",
        "Kreatif tanpa batas, ide kontennya selalu out of the box",
        "Bener banget kata abangnya, relate sama kehidupan sehari-hari",
        "Dukungan penuh buat gerakan positif seperti ini",
        "Singkat, padat, jelas, dan langsung ke inti permasalahan",
        "Keren abis, detail kecilnya aja diperhatiin dengan teliti",
        "Semangat terus kakak, karyamu selalu dinanti banyak orang",
        "Jujur ini salah satu utas terbaik yang pernah lewat di beranda",
        "Ramah banget adminnya balesin satu-satu pertanyaan pengikut",
        "Pilihan warna dan tipografinya sangat profesional",
        "Bikin mood langsung naik setelah nonton video lucu ini",
        "Sangat layak diapresiasi kerja keras seluruh tim di baliknya",
        "Mantap betul, semoga makin banyak konten positif berseliweran",
    ]

    # ==========================================================
    # 2. KELAS NEGATIF (50 Marketplace + 50 Media Sosial = 100)
    # ==========================================================
    negatif_marketplace = [
        "Produk sangat buruk dan mengecewakan, tidak sesuai foto",
        "Pelayanan lambat, admin tidak responsif, parah banget",
        "Barang rusak saat diterima, rugi beli di sini",
        "Kualitas jelek, baru dipakai sebentar sudah error",
        "Tidak puas sama sekali, pengiriman sangat lama",
        "Sangat buruk, fitur sering gagal dan bikin kesal",
        "Penipu, barang yang dikirim beda jauh dan cacat",
        "Sampah, produk murahan gampang patah dan hancur",
        "Bikin emosi, aplikasi lemot dan sering keluar sendiri",
        "Kapok beli di sini, komplain tidak pernah ditanggapi",
        "Anjing, barang rusak malah disuruh bayar ongkir lagi",
        "Goblok banget sistemnya, sudah bayar tapi pesanan batal",
        "Tolol, admin diajak ngomong tidak nyambung sama sekali",
        "Bangsat, paket hilang tidak ada tanggung jawabnya",
        "Kontol, pelayanannya sampah dan bikin rugi pembeli",
        "Warna yang dikirim salah, penjual tidak mau tukar",
        "Baru sehari dipakai sudah mati total, buang uang saja",
        "Kecewa berat, deskripsi menipu tidak sesuai kenyataan",
        "Kurir kasar dan paket penyok parah saat tiba",
        "Aplikasi sering bug dan force close, sangat mengganggu",
        "Harga mahal tapi kualitas murahan seperti barang bekas",
        "Packing asal-asalan tanpa pelindung, barang pecah di jalan",
        "Baterai bocor dan cepat habis padahal baru dibuka",
        "Pengiriman molor berminggu-minggu tanpa kejelasan status",
        "Admin ketus dan tidak sopan saat ditanya soal pesanan",
        "Zonk banget, ukuran kekecilan dan bahannya panas kasar",
        "Sistem sering down dan gagal login terus menerus",
        "Barang palsu alias kw, sangat tidak sesuai iklan",
        "Suara kresek-kresek dan mati sebelah, produk cacat",
        "Pengajuan pengembalian dana dipersulit oleh penjual",
        "Lemot parah, buka satu halaman saja butuh waktu lama",
        "Bau menyengat dan kotor seperti stok lama di gudang",
        "Sangat menyesal beli di toko ini, pelayanan mengecewakan",
        "Komponen tidak lengkap, kabel charger tidak ada di dus",
        "Update terbaru malah bikin aplikasi berat dan boros baterai",
        "Kurir malas mengantar ke alamat dan asal menandai selesai",
        "Kualitas jahitan berantakan dan banyak benang lepas",
        "Tidak berfungsi sama sekali, tombol macet sejak awal",
        "Pemborosan uang, performa jauh di bawah ekspektasi",
        "Respon lambat, chat baru dibalas setelah tiga hari",
        "Layar bergaris dan berkedip setelah pemakaian dua hari",
        "Barang yang datang kurang satu unit dari jumlah pesanan",
        "Sangat tidak profesional, pesanan dibatalkan sepihak",
        "Koneksi sering putus nyambung, tidak stabil dipakai",
        "Penipuan berkedok diskon, harga dinaikkan dulu baru dipotong",
        "Bikin darah tinggi, fitur pembayaran selalu gagal diproses",
        "Bahan tipis dan mudah robek, kualitas sangat rendah",
        "Tidak ada garansi resmi seperti yang dijanjikan penjual",
        "Buruk sekali, pengalaman belanja terparah tahun ini",
        "Gagal total, produk tidak bisa menyala walau sudah dicas",
    ]

    negatif_medsos = [
        "Gaje banget kontennya, buang-buang kuota aja",
        "Adminnya nyebelin, tiap ditanya gak pernah kasih solusi",
        "Hoaks ini mah, jangan gampang percaya berita tanpa sumber",
        "Katanya bebas berpendapat tapi kritik langsung dihapus",
        "Bikin emosi baca beritanya, makin hari makin gak masuk akal",
        "Cringe banget liatnya, maksa banget biar viral",
        "Toxic banget kolom komentarnya, isinya saling hujat semua",
        "Makin ke sini makin aneh aturannya, nyusahin rakyat kecil",
        "Judul clickbait doang, isinya kosong melompong gak ada faedahnya",
        "Bosen liat drama settingan kayak gini tiap hari di beranda",
        "Sarkas dikit: sistemnya canggih banget sampai tiap hari gangguan",
        "Gak jelas maksud postingannya apa, bikin orang salah paham",
        "Parah banget kelakuannya, gak punya empati sama sekali",
        "Konten sampah, bisanya cuma nyebar kebencian dan adu domba",
        "Kualitas videonya burik banget, audionya juga kresek-kresek",
        "Capek banget liat kebijakan yang berubah-ubah tanpa sosialisasi",
        "Sombong amat jawab komentarnya, padahal netizen nanya baik-baik",
        "Plagiat karya orang lain tanpa izin dan tanpa cantumin sumber",
        "Rugi waktu nonton sampai habis ternyata cuma promosi terselubung",
        "Beli gas aja sekarang susahnya minta ampun kayak nyari harta karun",
        "Lebay banget reaksinya, masalah sepele dibesar-besarin",
        "Fomo doang tapi gak paham substansi masalahnya apa",
        "Makin gak mutu tontonan zaman sekarang, isinya pamer doang",
        "Komentar julid semua, bukannya mendukung malah menjatuhkan",
        "Servernya down mulu pas lagi dibutuhin, bikin kerjaan terhambat",
        "Informasi menyesatkan, tolong jangan sebar fitnah di media sosial",
        "Gak mutu blas, argumennya muter-muter gak ada poinnya",
        "Kecewa banget sama klarifikasinya yang terkesan cuci tangan",
        "Bikin darah tinggi liat antrean panjang tanpa kejelasan petugas",
        "Spam komentar judi di mana-mana, mengganggu kenyamanan pengguna",
        "Kacau balau manajemen acaranya, penonton dibiarkan kepanasan",
        "Hancur sudah kepercayaan publik kalau penanganannya lamban begini",
        "Bacot doang gede tapi realisasinya nol besar di lapangan",
        "Nyesel buka kolom balasan, isinya debat kusir gak berujung",
        "Tampilan aplikasinya makin ribet setelah diperbarui, bikin bingung",
        "Gak profesional sama sekali, jadwal mundur berjam-jam tanpa info",
        "Mirip penipuan berkedok investasi, hati-hati jangan ketipu",
        "Basi banget topiknya, udah dibahas berkali-kali masih diulang",
        "Nyusahin orang banyak aja kalau jalanan ditutup tanpa pemberitahuan",
        "Kualitas layanan publiknya makin menurun drastis tahun ini",
        "Goblok banget yang bikin aturan tanpa mikirin dampak ke bawah",
        "Anjir kesel banget liat orang buang sampah sembarangan di sungai",
        "Bangsat emang pelaku penipuan online yang nipu orang tua",
        "Tolol dipelihara, dikasih fakta malah ngeyel gak mau kalah",
        "Miris liat kelakuan oknum yang menyalahgunakan wewenang",
        "Buruk banget cara komunikasinya ke masyarakat, terkesan meremehkan",
        "Gagal paham sama jalan pikiran orang-orang yang ngebela kesalahan",
        "Suaranya putus-putus dan gambarnya patah-patah sepanjang live",
        "Sangat tidak mendidik, bahasanya kasar dan penuh umpatan",
        "Makin parah macetnya sejak proyek itu mangkrak di tengah jalan",
    ]

    # ==========================================================
    # 3. KELAS NETRAL (50 Marketplace + 50 Media Sosial = 100)
    # ==========================================================
    netral_marketplace = [
        "Paket sudah sampai tadi siang, belum sempat dicoba",
        "Barang diterima dalam kondisi standar sesuai deskripsi",
        "Pengiriman menggunakan kurir reguler ke alamat kantor",
        "Produk ini tersedia dalam pilihan warna hitam dan putih",
        "Saya membeli produk ini pada hari Senin kemarin",
        "Spesifikasi produk sesuai dengan yang tertulis di katalog",
        "Apakah stok untuk ukuran besar masih tersedia?",
        "Pesanan sedang dalam proses pengemasan oleh penjual",
        "Berat paket sekitar satu kilogram termasuk dus",
        "Pembayaran dilakukan menggunakan transfer bank",
        "Toko buka dari hari Senin sampai Sabtu pukul sembilan",
        "Nomor resi pengiriman sudah diperbarui di sistem",
        "Produk ini buatan dalam negeri dengan bahan katun",
        "Saya memesan dua unit untuk keperluan kantor",
        "Informasi garansi tertera di dalam kartu petunjuk",
        "Bagaimana cara mengajukan klaim garansi resmi?",
        "Paket dititipkan ke satpam komplek karena rumah kosong",
        "Ukuran panjang produk dua puluh sentimeter",
        "Varian rasa cokelat dan vanila memiliki harga sama",
        "Transaksi tercatat pada tanggal sepuluh bulan ini",
        "Kapan jadwal pengiriman untuk pesanan luar kota?",
        "Produk menggunakan daya listrik sebesar lima puluh watt",
        "Saya sedang menunggu konfirmasi ketersediaan warna biru",
        "Buku panduan tersedia dalam bahasa Indonesia dan Inggris",
        "Sistem sedang melakukan pemeliharaan rutin malam ini",
        "Alamat pengiriman sudah sesuai dengan titik koordinat peta",
        "Paket dikirim dari gudang cabang kota Surabaya",
        "Isi dalam kotak terdiri dari unit utama dan kabel daya",
        "Apakah produk ini kompatibel dengan sistem operasi terbaru?",
        "Pesanan baru saja diserahkan kepada pihak ekspedisi",
        "Saya memilih metode pembayaran bayar di tempat",
        "Dimensi kemasan cukup ringkas untuk disimpan di tas",
        "Garansi berlaku selama dua belas bulan sejak pembelian",
        "Mohon informasi detail mengenai bahan dasar produk ini",
        "Status pelacakan menunjukkan paket berada di gudang sortir",
        "Terdapat tiga pilihan kapasitas penyimpanan pada seri ini",
        "Saya membeli barang ini sebagai hadiah ulang tahun teman",
        "Waktu operasional layanan pelanggan mengikuti jam kerja",
        "Invoice pembelian sudah dikirimkan ke alamat email terdaftar",
        "Produk ini merupakan versi pembaruan dari seri sebelumnya",
        "Apakah ada perbedaan spesifikasi antara varian standar dan pro?",
        "Kurir menghubungi nomor telepon sebelum mengantar paket",
        "Bahan kemasan terbuat dari kardus daur ulang berwarna cokelat",
        "Pengaturan awal perangkat memerlukan koneksi internet",
        "Jumlah stok yang tertera di halaman produk adalah sepuluh unit",
        "Saya membandingkan beberapa tipe sebelum memutuskan membeli",
        "Pengiriman dijadwalkan ulang pada hari kerja berikutnya",
        "Suhu ruangan memengaruhi daya tahan penyimpanan makanan",
        "Label pengiriman tertempel pada bagian atas kotak paket",
        "Silakan periksa kembali rincian pesanan sebelum checkout",
    ]

    netral_medsos = [
        "Ada yang tau judul lagu di video ini apa ya?",
        "Info lokasi kejadiannya di daerah mana min?",
        "Izin save videonya buat referensi tugas kuliah",
        "Ternyata baru rilis kemarin sore beritanya",
        "Cek DM min, mau tanya soal jadwal acaranya",
        "Kalau dari arah stasiun naik transportasi apa ya ke sana?",
        "Oh jadi gitu kronologi awalnya, baru paham sekarang",
        "Sudah tayang di bioskop mulai hari ini tanggal dua puluh",
        "Spill merek kamera yang dipakai buat rekam video ini dong",
        "Jam berapa siaran langsung pertandingannya dimulai nanti malam?",
        "Titip jejak dulu biar nanti gampang nyari postingannya lagi",
        "Apakah acara tersebut terbuka untuk umum atau khusus mahasiswa?",
        "Baru tahu kalau ternyata ada fitur tersembunyi di pengaturan",
        "Link pendaftaran ada di bio akun resmi penyelenggara ya",
        "Ini lokasi syutingnya di sekitar kawasan kota lama bukan sih?",
        "Saya sudah mengisi formulir registrasi kemarin siang",
        "Perbedaan versi reguler dan versi pro ada di kapasitas memori",
        "Berapa harga tiket masuk untuk akhir pekan ini?",
        "Postingan ini diunggah sekitar dua jam yang lalu",
        "Kira-kira kapan pendaftaran gelombang kedua dibuka kembali?",
        "Tadi pagi sempat lewat jalan tersebut masih normal lalu lintasnya",
        "Ada yang punya kontak panitia yang bisa dihubungi?",
        "Menunggu pengumuman resmi dari pihak terkait besok pagi",
        "Catatan: kegiatan dilaksanakan secara daring melalui tautan zoom",
        "Nama font yang dipakai di thumbnail ini jenis sans serif",
        "Siapa nama narasumber yang berbicara di menit ketiga tadi?",
        "Persyaratan dokumen bisa diunduh melalui laman utama website",
        "Sekadar mengingatkan bahwa batas pengumpulan berkas hari Jumat",
        "Lokasi parkir pengunjung berada di sisi selatan gedung utama",
        "Apakah tersedia siaran ulangnya di kanal YouTube resmi?",
        "Foto diambil menggunakan lensa sudut lebar pada sore hari",
        "Jumlah peserta yang hadir dalam diskusi sekitar lima puluh orang",
        "Menu sarapan hari ini nasi goreng dan teh hangat",
        "Saya sedang membaca utas ini dari bagian pertama sampai akhir",
        "Kantor layanan buka kembali setelah jam istirahat pukul satu",
        "Rute perjalanan memakan waktu kurang lebih empat puluh menit",
        "Format file yang diterima sistem hanya dokumen pdf dan docx",
        "Hari ini cuaca di wilayah Boyolali dan sekitarnya cerah berawan",
        "Silakan tuliskan pertanyaan di kolom komentar di bawah ini",
        "Data statistik tersebut diambil dari laporan tahunan lembaga",
        "Akun resmi sudah memberikan tanggapan di kolom komentar teratas",
        "Pembagian jadwal sesi pagi dan sesi siang sudah ditempel di papan",
        "Apakah aplikasi ini bisa dijalankan di sistem operasi Windows?",
        "Warna latar belakang postingan menggunakan kombinasi abu dan hitam",
        "Saya baru saja menonton episode pertamanya tadi malam",
        "Informasi selengkapnya dapat dilihat pada slide terakhir unggahan",
        "Nomor antrean diambil melalui mesin tiket di pintu masuk",
        "Pertemuan rutin dijadwalkan setiap hari Rabu pukul tiga sore",
        "Buku tersebut terdiri dari sebelas bab pembahasan materi",
        "Terima kasih atas informasinya, saya catat dulu jadwalnya",
    ]

    data = (
        [(t, "positif") for t in (positif_marketplace + positif_medsos)]
        + [(t, "negatif") for t in (negatif_marketplace + negatif_medsos)]
        + [(t, "netral") for t in (netral_marketplace + netral_medsos)]
    )
    df = pd.DataFrame(data, columns=["komentar", "sentimen"])
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    print(f"[INFO] File '{DATASET_PATH}' berhasil diperbarui dengan {len(df)} data (Marketplace + Media Sosial).")

def load_training_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        buat_dataset_300()
    elif os.path.getsize(DATASET_PATH) == 0:
        raise ValueError(f"Dataset '{DATASET_PATH}' kosong; dataset tidak akan ditimpa otomatis.")

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    kolom_wajib = {"komentar", "sentimen"}
    if not kolom_wajib.issubset(df.columns):
        raise ValueError("Dataset wajib memiliki kolom 'komentar' dan 'sentimen'.")

    df = df.dropna(subset=["komentar", "sentimen"]).copy()
    df["komentar"] = df["komentar"].astype(str).str.strip()
    df["sentimen"] = df["sentimen"].astype(str).str.strip().str.lower()
    df = df[df["komentar"] != ""]
    label_valid = {"positif", "netral", "negatif"}
    label_tidak_valid = set(df["sentimen"]) - label_valid
    if label_tidak_valid:
        raise ValueError(f"Label sentimen tidak valid: {', '.join(sorted(label_tidak_valid))}.")
    if df.empty:
        raise ValueError("Dataset tidak memiliki komentar berlabel yang dapat dilatih.")

    df["clean_text"] = df["komentar"].apply(preprocess_text)
    return df


def jalankan_training(train_bert: bool = False, epochs: int = 3):
    df = load_training_dataset()
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["sentimen"], random_state=42
    )
    X_train = df_train["clean_text"]
    y_train = df_train["sentimen"]
    X_test = df_test["clean_text"]
    y_test = df_test["sentimen"]
    min_train_examples = int(y_train.value_counts().min())
    if min_train_examples < 4:
        raise ValueError("Diperlukan setidaknya lima contoh per kelas untuk evaluasi silang dan kalibrasi.")

    # Tuning TF-IDF (Unigram + Bigram, sublinear_tf=True) & Model Klasifikasi[cite: 38, 39, 44, 48]
    pipeline_nb = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5))
    ])

    pipeline_svm = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=3))
    ])

    # Model selection only sees the training partition; the test set remains untouched.
    cv = StratifiedKFold(
        n_splits=min(5, min_train_examples),
        shuffle=True,
        random_state=42,
    )
    f1_nb = cross_val_score(pipeline_nb, X_train, y_train, cv=cv, scoring="f1_macro").mean()
    f1_svm = cross_val_score(pipeline_svm, X_train, y_train, cv=cv, scoring="f1_macro").mean()

    print(f"\n=== PERBANDINGAN {cv.n_splits}-FOLD CROSS-VALIDATION (MACRO F1) ===")
    print(f"Multinomial Naive Bayes : {f1_nb:.4f}")
    print(f"Linear SVM (Tuned)      : {f1_svm:.4f}")

    model_terbaik = pipeline_svm if f1_svm >= f1_nb else pipeline_nb
    nama_model = "Linear SVM (Tuned)" if f1_svm >= f1_nb else "Multinomial Naive Bayes"

    model_terbaik.fit(X_train, y_train)
    y_pred = model_terbaik.predict(X_test)

    print(f"\n=== CLASSIFICATION REPORT ({nama_model}) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== CONFUSION MATRIX ===")
    labels_urut = ["positif", "netral", "negatif"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_urut)
    print(pd.DataFrame(cm, index=[f"Aktual_{l}" for l in labels_urut],
                       columns=[f"Pred_{l}" for l in labels_urut]))

    # Latih pada seluruh dataset & simpan artefak model klasik
    model_terbaik.fit(df["clean_text"], df["sentimen"])
    joblib.dump({"model": model_terbaik, "nama_model": nama_model}, MODEL_PATH)
    print(f"\n[SUKSES] Model '{nama_model}' disimpan ke '{MODEL_PATH}'.")

    # Opsional: Evaluasi & Fine-tuning IndoBERT dengan 300 data
    if train_bert:
        from transformer_model import TRANSFORMER_AVAILABLE, latih_indobert, prediksi_indobert
        if not TRANSFORMER_AVAILABLE:
            print("[PERINGATAN] Dependensi torch/transformers belum terpasang, melewati pelatihan IndoBERT.")
        else:
            print(f"\n[INFO] Mengevaluasi IndoBERT pada 20% data uji ({len(y_test)} sampel)...")
            latih_indobert(df_train, epochs=epochs)
            y_pred_bert = [prediksi_indobert(t)[0].lower() for t in df_test["komentar"]]

            print("\n=== CLASSIFICATION REPORT (IndoBERT Fine-Tuned) ===")
            print(classification_report(df_test["sentimen"], y_pred_bert, zero_division=0))

            print("=== CONFUSION MATRIX (IndoBERT) ===")
            cm_bert = confusion_matrix(df_test["sentimen"], y_pred_bert, labels=labels_urut)
            print(pd.DataFrame(cm_bert, index=[f"Aktual_{l}" for l in labels_urut],
                               columns=[f"Pred_{l}" for l in labels_urut]))

            print(f"\n[INFO] Melatih IndoBERT final ({epochs} epoch) pada seluruh {len(df)} data...")
            latih_indobert(df, epochs=epochs)
            print("[SUKSES] Model IndoBERT terbaru berhasil disimpan ke 'model_indobert/'.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Model Analisis Sentimen")
    parser.add_argument("--bert", action="store_true", help="Sekaligus lakukan fine-tuning ulang IndoBERT")
    parser.add_argument("--epochs", type=int, default=3, help="Jumlah epoch IndoBERT (default: 3)")
    args = parser.parse_args()

    jalankan_training(train_bert=args.bert, epochs=args.epochs)
