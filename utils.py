import re

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    USE_STEMMER = True
except ImportError:
    USE_STEMMER = False

# Kamus normalisasi bahasa informal / slang marketplace & media sosial[cite: 22, 23]
SLANG_DICT = {
    "gk": "tidak", "gak": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak", "kagak": "tidak",
    "bgt": "banget", "bgtt": "banget", "bgs": "bagus", "jlk": "jelek",
    "mantul": "mantap", "good": "bagus", "rekomen": "recommended", "worth": "layak",
    "ok": "oke", "sip": "oke", "syip": "oke", "thx": "terima kasih", "makasih": "terima kasih",
    "krg": "kurang", "jg": "juga", "yg": "yang", "dg": "dengan", "dgn": "dengan",
    "tpi": "tapi", "tp": "tapi", "klo": "kalau", "kalo": "kalau", "utk": "untuk", "krn": "karena",
    "zonk": "buruk", "lemot": "lambat", "lelet": "lambat", "fast": "cepat",
    # Slang media sosial
    "kocak": "lucu", "ngakak": "lucu", "wkwk": "lucu", "wkwkwk": "lucu",
    "gaje": "tidak jelas", "gj": "tidak jelas", "baper": "tersinggung",
    "valid": "benar", "menyala": "keren", "gokil": "keren", "anjay": "keren",
    "cringe": "memalukan", "hoax": "bohong", "hoaks": "bohong",
    "lebay": "berlebihan", "julid": "iri", "toxic": "kasar", "parah": "buruk",
    "bgst": "bangsat", "ajg": "anjing", "kntl": "kontol", "gblk": "goblok",
    "min": "admin", "mimin": "admin", "info": "informasi", "infonya": "informasi"
}

# Stopwords selektif (kata negasi 'tidak', 'bukan', 'kurang', 'jangan' dipertahankan)[cite: 53]
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "adalah",
    "ini", "itu", "saya", "aku", "dia", "mereka", "nya", "juga", "pada", "pun",
    "gw", "gue", "lu", "lo", "nih", "sih", "deh", "dong", "tuh", "mah", "kan", "ya"
}

# Kamus kata kunci aspek: Marketplace + Media Sosial[cite: 21, 22]
ASPECT_KEYWORDS = {
    "Pengiriman & Kurir": ["kirim", "pengiriman", "kurir", "paket", "ongkir", "ekspedisi", "sampai", "tiba", "resi", "antar"],
    "Kualitas & Performa": ["produk", "barang", "kualitas", "bahan", "baterai", "layar", "warna", "ukuran", "awet", "rusak", "cacat", "ori", "patah", "mouse", "mousepad", "kabel", "lampu", "rgb", "sensitivitas", "responsif", "fungsi"],
    "Kenyamanan": ["nyaman", "kenyamanan", "halus", "enak", "pas", "ergonomis", "adem", "kasar"],
    "Pelayanan & Admin": ["pelayanan", "layanan", "admin", "penjual", "seller", "respon", "cs", "ramah", "komplain", "tanggung jawab"],
    "Harga & Biaya": ["harga", "murah", "mahal", "terjangkau", "biaya", "diskon", "promo", "uang", "rugi", "bayar", "modal", "ribuan", "worth", "kuota"],
    "Aplikasi & Sistem": ["aplikasi", "sistem", "fitur", "bug", "error", "force close", "lancar", "lemot", "lambat", "login", "server"],
    "Kemasan & Packing": ["packing", "kemasan", "bungkus", "bubble", "dus", "penyok", "rapi", "segel"],
    "Konten & Topik": ["konten", "postingan", "post", "thread", "tweet", "utas", "berita", "informasi", "cerita", "puisi", "tulisan", "caption", "topik", "isu", "aturan", "kebijakan"],
    "Visual & Audio": ["video", "foto", "gambar", "edit", "editannya", "musik", "lagu", "suara", "audio", "kamera", "grafik", "visual", "desain", "wallpaper"],
    "Komunitas & Netizen": ["netizen", "warga", "komentar", "komen", "reply", "balas", "followers", "kreator", "creator"]
}

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Case Folding (Lowercasing)[cite: 52, 59]
    text = text.lower()
    # 2. Regex Cleaning: hapus URL, mention, tanda baca, dan angka[cite: 58, 59]
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Normalisasi Slang & Stopwords Removal[cite: 23, 53]
    tokens = []
    for word in text.split():
        word = SLANG_DICT.get(word, word)
        if word not in STOPWORDS_ID:
            tokens.append(word)

    clean_text = " ".join(tokens)

    # 4. Stemming Bahasa Indonesia[cite: 55]
    if USE_STEMMER and clean_text:
        clean_text = stemmer.stem(clean_text)

    return clean_text

def deteksi_aspek(klausa: str) -> list[str]:
    klausa_lower = klausa.lower()
    aspek_terdeteksi = []
    for nama_aspek, keywords in ASPECT_KEYWORDS.items():
        if any(re.search(rf"\b{kw}\w*\b", klausa_lower) for kw in keywords):
            aspek_terdeteksi.append(nama_aspek)
    return aspek_terdeteksi if aspek_terdeteksi else ["Umum"]

def analisis_aspek(teks: str, fungsi_prediksi) -> list[dict]:
    """
    Memecah kalimat menjadi klausa (Sentence/Clause Segmentation), mendeteksi aspek,
    dan memprediksi sentimen tiap aspek secara terpisah[cite: 21, 22].
    """
    pola_pemisah = r"[.!?;,]+|\b(?:tetapi|tapi|tpi|namun|sedangkan|meskipun|walau|walaupun|hanya saja)\b"
    klausa_list = [k.strip() for k in re.split(pola_pemisah, teks, flags=re.IGNORECASE) if len(k.strip()) >= 3]

    if not klausa_list:
        klausa_list = [teks.strip()]

    hasil_absa = []
    for klausa in klausa_list:
        daftar_aspek = deteksi_aspek(klausa)
        sentimen, keyakinan, _, _ = fungsi_prediksi(klausa)
        for aspek in daftar_aspek:
            hasil_absa.append({
                "Klausa": klausa,
                "Aspek": aspek,
                "Sentimen": sentimen,
                "Keyakinan (%)": round(keyakinan, 1)
            })
    return hasil_absa