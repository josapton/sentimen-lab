import io
import os
import html
import joblib
import pandas as pd
import streamlit as st
from utils import preprocess_text, analisis_aspek
from train_model import jalankan_training, DATASET_PATH, MODEL_PATH
from transformer_model import (
    TRANSFORMER_AVAILABLE,
    PRETRAINED_SENTIMENT_CATALOG,
    DEFAULT_SENTIMENT_PRETRAINED,
    alasan_model_belum_siap,
    model_indobert_siap,
    latih_indobert,
    prediksi_indobert_detail,
    prediksi_pretrained_detail,
)

st.set_page_config(
    page_title="SentimenLab - Analisis Sentimen & ABSA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS: TYPOGRAPHY, CARDS & ADAPTIVE THEME
# ==========================================
if "mode_tema" not in st.session_state:
    st.session_state["mode_tema"] = "Otomatis (Sistem)"

tema_aktif = st.session_state["mode_tema"]

DARK_VARS = """
    color-scheme: dark;
    --sl-bg-app: #09090B;
    --sl-bg-sidebar: #121216;
    --sl-bg-card: #121216;
    --sl-bg-surface: #18181B;
    --sl-bg-chip: #1E1E24;
    --sl-border: #27272A;
    --sl-border-strong: #3F3F46;
    --sl-text-primary: #FAFAFA;
    --sl-text-secondary: #D4D4D8;
    --sl-text-muted: #A1A1AA;
"""

LIGHT_VARS = """
    color-scheme: light;
    --sl-bg-app: #FFFFFF;
    --sl-bg-sidebar: #F4F4F5;
    --sl-bg-card: #F8FAFC;
    --sl-bg-surface: #F1F5F9;
    --sl-bg-chip: #E2E8F0;
    --sl-border: #E4E4E7;
    --sl-border-strong: #CBD5E1;
    --sl-text-primary: #09090B;
    --sl-text-secondary: #27272A;
    --sl-text-muted: #52525B;
"""

if tema_aktif == "Gelap (Dark)":
    blok_variabel_tema = f":root, html, body {{{DARK_VARS}}}"
elif tema_aktif == "Terang (Light)":
    blok_variabel_tema = f":root, html, body {{{LIGHT_VARS}}}"
else:
    blok_variabel_tema = f"""
    :root, html, body {{{LIGHT_VARS}}}
    @media (prefers-color-scheme: dark) {{
        :root, html, body {{{DARK_VARS}}}
    }}
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

{blok_variabel_tema}

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}

* {{
    scrollbar-color: var(--sl-border-strong) var(--sl-bg-app);
    scrollbar-width: thin;
}}

*::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}

*::-webkit-scrollbar-track {{
    background: var(--sl-bg-app);
}}

*::-webkit-scrollbar-thumb {{
    background-color: var(--sl-border-strong);
    border: 2px solid var(--sl-bg-app);
    border-radius: 999px;
}}

code, pre, .mono-text {{
    font-family: 'JetBrains Mono', monospace !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{
    background-color: var(--sl-bg-app) !important;
}}

button[aria-controls="stSidebar"] {{
    background-color: var(--sl-bg-surface) !important;
    color: var(--sl-text-primary) !important;
    border: 1px solid var(--sl-border-strong) !important;
    border-radius: 8px !important;
    opacity: 1 !important;
}}

button[aria-controls="stSidebar"]:hover {{
    background-color: var(--sl-bg-chip) !important;
    border-color: #3B82F6 !important;
}}

button[aria-controls="stSidebar"] [data-testid="stIconMaterial"] {{
    color: var(--sl-text-primary) !important;
}}

/* Latar Utama & Sidebar */
.stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--sl-bg-app) !important;
    color: var(--sl-text-primary) !important;
}}

section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
    background-color: var(--sl-bg-sidebar) !important;
    border-right: 1px solid var(--sl-border);
}}

/* Tipografi Global & Komponen Streamlit */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp label,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stThumbValue"],
button[data-baseweb="tab"] p {{
    color: var(--sl-text-primary) !important;
}}

.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMetricLabel"] p {{
    color: var(--sl-text-muted) !important;
}}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
    background-color: var(--sl-bg-card) !important;
    color: var(--sl-text-primary) !important;
    border: 1px solid var(--sl-border-strong) !important;
}}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"])
    [data-testid="stMarkdownContainer"] p,
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"])
    [data-testid="stMarkdownContainer"] strong {{
    color: var(--sl-text-primary) !important;
}}

.stApp hr {{
    border-color: var(--sl-border) !important;
}}

/* Kode Inline */
.stApp code {{
    background-color: var(--sl-bg-chip) !important;
    color: var(--sl-text-primary) !important;
    border: 1px solid var(--sl-border-strong) !important;
    padding: 0.15rem 0.45rem !important;
    border-radius: 5px !important;
}}

/* Input Teks, Textarea & Placeholder */
.stTextInput [data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"],
.stTextInput div[data-baseweb="input"],
.stTextInput div[data-baseweb="base-input"],
.stTextArea div[data-baseweb="textarea"],
.stTextArea div[data-baseweb="base-input"] {{
    background-color: var(--sl-bg-surface) !important;
    border: 1px solid var(--sl-border-strong) !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}}

.stTextInput [data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within {{
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}}

.stTextInput input,
.stTextArea textarea {{
    background-color: transparent !important;
    color: var(--sl-text-primary) !important;
    caret-color: var(--sl-text-primary) !important;
    border: 0 !important;
    outline: none !important;
    box-shadow: none !important;
}}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: var(--sl-text-muted) !important;
    opacity: 0.85 !important;
}}

/* Selectbox & Menu Dropdown Popover */
div[data-baseweb="select"],
div[data-baseweb="select"] > div {{
    background-color: var(--sl-bg-surface) !important;
    border-color: var(--sl-border-strong) !important;
    box-shadow: none !important;
}}

div[data-baseweb="select"] > div {{
    border: 1px solid var(--sl-border-strong) !important;
}}

div[data-baseweb="select"] > div div {{
    background-color: transparent !important;
    color: var(--sl-text-primary) !important;
}}

[data-testid="stSelectbox"] [role="group"] {{
    background-color: var(--sl-bg-surface) !important;
    color: var(--sl-text-primary) !important;
    border: 1px solid var(--sl-border-strong) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}}

[data-testid="stSelectbox"] [role="group"]:focus-within {{
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}}

[data-testid="stSelectbox"] [role="combobox"] {{
    background-color: transparent !important;
    color: var(--sl-text-primary) !important;
    border: 0 !important;
    outline: none !important;
    box-shadow: none !important;
}}

[data-testid="stSelectbox"] button {{
    background-color: transparent !important;
    color: var(--sl-text-primary) !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] svg {{
    color: var(--sl-text-primary) !important;
    fill: var(--sl-text-primary) !important;
}}

div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
[role="listbox"] {{
    background-color: var(--sl-bg-card) !important;
    color: var(--sl-text-primary) !important;
    border: 1px solid var(--sl-border-strong) !important;
}}

[role="option"] {{
    background-color: transparent !important;
    color: var(--sl-text-primary) !important;
}}

[role="option"]:hover,
[role="option"][aria-selected="true"] {{
    background-color: var(--sl-bg-chip) !important;
}}

/* Radio Button Unchecked & Checked */
[data-testid="stRadioOption"] > div > div:first-child {{
    background-color: var(--sl-bg-surface) !important;
    border: 1.5px solid var(--sl-border-strong) !important;
    box-shadow: none !important;
}}

[data-testid="stRadioOption"] > div > div:first-child > div {{
    background-color: transparent !important;
}}

[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {{
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
}}

[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div {{
    background-color: #FFFFFF !important;
}}

label[data-baseweb="radio"] input[type="radio"] {{
    accent-color: #3B82F6;
    color-scheme: inherit;
}}

label[data-baseweb="radio"] input:not(:checked) + div {{
    background-color: var(--sl-bg-surface) !important;
    border: 1.5px solid var(--sl-border-strong) !important;
    box-shadow: none !important;
}}

label[data-baseweb="radio"] input:not(:checked) + div > div {{
    background-color: transparent !important;
}}

label[data-baseweb="radio"] input:checked + div {{
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
}}

label[data-baseweb="radio"] input:checked + div > div {{
    background-color: #FFFFFF !important;
}}

/* Expander & File Uploader */
[data-testid="stExpander"] {{
    background-color: var(--sl-bg-card) !important;
    border: 1px solid var(--sl-border) !important;
    border-radius: 8px !important;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary svg {{
    color: var(--sl-text-primary) !important;
    fill: var(--sl-text-primary) !important;
}}

[data-testid="stFileUploader"] section {{
    background-color: var(--sl-bg-surface) !important;
    border: 1px dashed var(--sl-border-strong) !important;
}}

[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span {{
    color: var(--sl-text-muted) !important;
}}

/* Tombol Sekunder, Tombol Upload & Primary */
.stButton button[kind="secondary"],
.stButton button[data-testid="stBaseButton-secondary"],
.stDownloadButton button,
[data-testid="stFileUploader"] button {{
    background-color: var(--sl-bg-surface) !important;
    color: var(--sl-text-primary) !important;
    border: 1px solid var(--sl-border-strong) !important;
}}

.stButton button[kind="secondary"]:hover,
.stButton button[data-testid="stBaseButton-secondary"]:hover,
.stDownloadButton button:hover,
[data-testid="stFileUploader"] button:hover {{
    border-color: #3B82F6 !important;
}}

.stButton button[kind="primary"],
.stButton button[data-testid="stBaseButton-primary"] {{
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
    border: none !important;
}}

.stButton button p,
.stDownloadButton button p,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p {{
    color: inherit !important;
}}

[data-testid="stFileUploader"] button svg {{
    fill: var(--sl-text-primary) !important;
}}

/* Layout & Komponen Kustom SentimenLab */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1120px;
}}

.app-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1.25rem;
    border-bottom: 1px solid var(--sl-border);
    padding-bottom: 1.1rem;
    margin-bottom: 1.5rem;
}}
.app-header > div:first-child {{
    min-width: 0;
}}
.app-title {{
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--sl-text-primary);
    margin: 0;
}}
.app-subtitle {{
    font-size: 0.9rem;
    color: var(--sl-text-muted);
    margin: 0.25rem 0 0 0;
    line-height: 1.5;
}}
.engine-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    background: var(--sl-bg-surface);
    border: 1px solid var(--sl-border);
    color: var(--sl-text-secondary);
    white-space: nowrap;
    flex-shrink: 0;
    max-width: 42%;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.engine-dot {{
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}}

.verdict-card {{
    background: var(--sl-bg-card);
    border: 1px solid var(--sl-border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.5rem;
}}
.verdict-label-caption {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--sl-text-muted);
    margin-bottom: 0.35rem;
}}
.verdict-value {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}}
.verdict-conf {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--sl-text-muted);
    margin-top: 0.2rem;
}}

.prob-row {{
    margin-bottom: 0.65rem;
}}
.prob-header {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--sl-text-secondary);
    margin-bottom: 0.25rem;
}}
.prob-track {{
    width: 100%;
    height: 6px;
    background: var(--sl-border);
    border-radius: 999px;
    overflow: hidden;
}}
.prob-fill {{
    height: 100%;
    border-radius: 999px;
}}

.token-box {{
    background: var(--sl-bg-card);
    border: 1px solid var(--sl-border);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-top: 0.5rem;
}}
.token-chip {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: var(--sl-bg-chip);
    border: 1px solid var(--sl-border-strong);
    color: var(--sl-text-primary);
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
    margin: 0.2rem 0.25rem 0.2rem 0;
}}

.kpi-card {{
    background: var(--sl-bg-card);
    border: 1px solid var(--sl-border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}}
.kpi-title {{
    font-size: 0.75rem;
    color: var(--sl-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.kpi-num {{
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--sl-text-primary);
    margin-top: 0.15rem;
}}

@media (max-width: 900px) {{
    .app-header {{
        align-items: flex-start;
        flex-direction: column;
        gap: 0.75rem;
    }}
    .engine-badge {{
        max-width: 100%;
    }}
}}

@media (max-width: 1100px) {{
    [data-testid="stColumn"]:not(:first-child) {{
        min-width: 0;
    }}
    [data-testid="stColumn"] h4 {{
        line-height: 1.35;
        overflow-wrap: anywhere;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# INISIALISASI MODEL
# ==========================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Menyiapkan dataset dan melatih model awal..."):
        jalankan_training()

@st.cache_resource
def load_svm_model():
    return joblib.load(MODEL_PATH)

artifact = load_svm_model()
svm_model = artifact["model"]
nama_svm = artifact["nama_model"]

# ==========================================
# SIDEBAR KONFIGURASI
# ==========================================
with st.sidebar:
    st.markdown("### Konfigurasi Model")
    pilihan_engine = st.radio(
        "Arsitektur Pemrosesan",
        [
            "Linear SVM + TF-IDF",
            "IndoBERT (Fine-Tuned)",
            "Pre-trained Sentiment (Siap Pakai)",
        ],
        label_visibility="collapsed",
    )
    gunakan_indobert = pilihan_engine.startswith("IndoBERT")
    gunakan_pretrained = pilihan_engine.startswith("Pre-trained")

    selected_pretrained_id = DEFAULT_SENTIMENT_PRETRAINED
    if gunakan_pretrained:
        nama_varian = st.selectbox(
            "Varian Model Hugging Face",
            list(PRETRAINED_SENTIMENT_CATALOG.keys()),
        )
        selected_pretrained_id = PRETRAINED_SENTIMENT_CATALOG[nama_varian]

    st.divider()
    st.markdown("### Status Sistem")
    if gunakan_pretrained:
        if not TRANSFORMER_AVAILABLE:
            st.caption("Status: **Dependensi Belum Terpasang**")
        else:
            st.caption("Status: **Siap Digunakan (Zero-Shot)**")
            st.caption(f"Repo: `{selected_pretrained_id.split('/')[-1]}`")
    elif gunakan_indobert:
        if not TRANSFORMER_AVAILABLE:
            st.caption("Status: **Dependensi Belum Terpasang**")
            st.code("pip install transformers torch sentencepiece protobuf", language="powershell")
        elif not model_indobert_siap():
            st.caption("Status: **Menunggu Fine-Tuning**")
            st.caption("Buka tab **Dataset & Pelatihan** untuk melatih bobot IndoBERT.")
        else:
            st.caption("Status: **Siap Digunakan**")
            st.caption("Backbone: `indobert-base-p1`")
    else:
        st.caption("Status: **Siap Digunakan**")
        st.caption("Vektorisasi: `TF-IDF (1, 2)-gram`")

    if os.path.exists(DATASET_PATH):
        df_info = pd.read_csv(DATASET_PATH)
        st.metric("Korpus Latih", f"{len(df_info)} ulasan")

    st.divider()
    st.markdown("### Tampilan")
    pilihan_tema = ["Otomatis (Sistem)", "Gelap (Dark)", "Terang (Light)"]
    ikon_tema = {
        "Otomatis (Sistem)": ":material/brightness_auto:",
        "Gelap (Dark)": ":material/dark_mode:",
        "Terang (Light)": ":material/light_mode:",
    }
    if st.button(
        tema_aktif,
        key="ganti_mode_tema",
        icon=ikon_tema[tema_aktif],
        help="Klik untuk beralih ke mode tema berikutnya.",
        width="stretch",
    ):
        index_tema = pilihan_tema.index(tema_aktif)
        st.session_state["mode_tema"] = pilihan_tema[(index_tema + 1) % len(pilihan_tema)]
        st.rerun()

# ==========================================
# HEADER UTAMA
# ==========================================
if gunakan_pretrained:
    dot_color = "#3B82F6" if TRANSFORMER_AVAILABLE else "#F59E0B"
    badge_text = selected_pretrained_id.split("/")[-1].upper()
elif gunakan_indobert:
    dot_color = "#10B981" if model_indobert_siap() else "#F59E0B"
    badge_text = "INDOBERT-BASE-P1"
else:
    dot_color = "#10B981"
    badge_text = "LINEAR-SVC · TF-IDF"

st.markdown(f"""
<div class="app-header">
    <div>
        <h1 class="app-title">SentimenLab</h1>
        <p class="app-subtitle">Klasifikasi Polaritas Opini & Aspect-Based Sentiment Analysis Bahasa Indonesia</p>
    </div>
    <div class="engine-badge">
        <span class="engine-dot" style="background:{dot_color};"></span>{badge_text}
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FUNGSI PREDIKSI
# ==========================================
def prediksi_detail(teks: str):
    selected_engine = "pretrained" if gunakan_pretrained else "indobert" if gunakan_indobert else "svm"
    reason = alasan_model_belum_siap(selected_engine)
    if reason:
        raise RuntimeError(reason)

    if gunakan_pretrained:
        return prediksi_pretrained_detail(teks, model_id=selected_pretrained_id)

    if gunakan_indobert:
        return prediksi_indobert_detail(teks)

    clean = preprocess_text(teks)
    vektor = svm_model.named_steps["tfidf"].transform([clean])
    if vektor.nnz == 0:
        return "NETRAL", 0.0, clean, True, {"POSITIF": 0.0, "NETRAL": 100.0, "NEGATIF": 0.0}

    pred = svm_model.predict([clean])[0].upper()
    probs_arr = svm_model.predict_proba([clean])[0]
    class_map = {c.upper(): float(p) * 100 for c, p in zip(svm_model.classes_, probs_arr)}
    probs_dict = {
        "POSITIF": class_map.get("POSITIF", 0.0),
        "NETRAL": class_map.get("NETRAL", 0.0),
        "NEGATIF": class_map.get("NEGATIF", 0.0),
    }
    return pred, probs_dict[pred], clean, False, probs_dict

def prediksi_untuk_absa(teks: str):
    label, conf, clean, is_oov, _ = prediksi_detail(teks)
    return label, conf, clean, is_oov

# ==========================================
# NAVIGASI TAB
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "Analisis Teks & Aspek",
    "Pemrosesan Batch (CSV)",
    "Dataset & Pelatihan",
])

# ------------------------------------------
# TAB 1: ANALISIS TEKS & ASPEK
# ------------------------------------------
with tab1:
    if "input_komentar" not in st.session_state:
        st.session_state["input_komentar"] = ""

    st.caption("Skenario Uji Cepat:")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("Kontras Pengiriman", width="stretch"):
        st.session_state["input_komentar"] = "katanya pengiriman instan, tapi tiga hari baru sampai di rumah"
    if p2.button("Multi-Aspek Produk", width="stretch"):
        st.session_state["input_komentar"] = "Pengiriman cepat dan kurir ramah, tetapi kemasan penyok dan barang rusak"
    if p3.button("Ulasan Positif", width="stretch"):
        st.session_state["input_komentar"] = "Kualitas bahan premium, admin responsif, sangat puas belanja di sini"
    if p4.button("Pernyataan Netral", width="stretch"):
        st.session_state["input_komentar"] = "Paket sudah diterima oleh satpam komplek tadi siang pukul dua"

    komentar_input = st.text_area(
        "Teks Komentar atau Ulasan",
        key="input_komentar",
        placeholder="Ketik atau pilih skenario uji di atas untuk menganalisis sentimen keseluruhan dan per aspek...",
        height=115,
    )

    col_btn, _ = st.columns([1, 3])
    jalankan_uji = col_btn.button("Jalankan Analisis", type="primary", width="stretch")

    if jalankan_uji:
        if gunakan_pretrained and not TRANSFORMER_AVAILABLE:
            st.warning("Dependensi `transformers` dan `torch` belum terpasang untuk menjalankan model Pre-trained. Pasang dependensi atau pilih model lain di sidebar.")
        elif gunakan_indobert and (not TRANSFORMER_AVAILABLE or not model_indobert_siap()):
            st.warning("Model IndoBERT belum siap. Lakukan fine-tuning pada tab **Dataset & Pelatihan** atau beralih ke **Linear SVM** di sidebar.")
        elif komentar_input.strip():
            label, conf, clean, is_oov, probs_dict = prediksi_detail(komentar_input)

            warna_map = {
                "POSITIF": "#10B981",
                "NEGATIF": "#EF4444",
                "NETRAL": "#71717A",
            }
            accent = warna_map.get(label, "#71717A")

            col_kiri, col_kanan = st.columns([1.15, 1], gap="large")

            with col_kiri:
                catatan_oov = (
                    "<div class='verdict-conf' style='color:#F59E0B;'>Kosakata belum tercatat di korpus latih (OOV)</div>"
                    if is_oov
                    else f"<div class='verdict-conf'>Confidence Score: {conf:.1f}%</div>"
                )
                st.markdown(f"""
                <div class="verdict-card" style="border-left: 4px solid {accent};">
                    <div class="verdict-label-caption">Klasifikasi Keseluruhan</div>
                    <div class="verdict-value" style="color: {accent};">{label}</div>
                    {catatan_oov}
                </div>
                """, unsafe_allow_html=True)

            with col_kanan:
                st.markdown("<div class='verdict-label-caption' style='margin-top:0.6rem;'>Distribusi Probabilitas Kelas</div>", unsafe_allow_html=True)
                for k_name, k_color in [("POSITIF", "#10B981"), ("NETRAL", "#71717A"), ("NEGATIF", "#EF4444")]:
                    val = probs_dict[k_name]
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-header">
                            <span>{k_name.title()}</span>
                            <span class="mono-text">{val:.1f}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{val:.1f}%; background:{k_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("#### Segmentasi Opini per Aspek (ABSA)")
            hasil_absa = analisis_aspek(komentar_input, prediksi_untuk_absa)
            df_absa = pd.DataFrame(hasil_absa)

            st.dataframe(
                df_absa,
                width="stretch",
                hide_index=True,
                column_config={
                    "Klausa": st.column_config.TextColumn("Segmen Klausa", width="large"),
                    "Aspek": st.column_config.TextColumn("Kategori Aspek", width="medium"),
                    "Sentimen": st.column_config.TextColumn("Polaritas", width="small"),
                    "Keyakinan (%)": st.column_config.ProgressColumn(
                        "Keyakinan",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

            with st.expander("Inspeksi Pipeline Pemrosesan Teks"):
                tokens = clean.split()
                chips_html = "".join(f"<span class='token-chip'>{html.escape(t)}</span>" for t in tokens) if tokens else "<span class='verdict-conf'>(Kosong)</span>"
                st.markdown(f"""
                <div class="token-box">
                    <div class="verdict-label-caption">Input Aktual ke Model</div>
                    <div>{chips_html}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Masukkan teks komentar terlebih dahulu.")

# ------------------------------------------
# TAB 2: PEMROSESAN BATCH (CSV)
# ------------------------------------------
with tab2:
    st.caption("Unggah berkas CSV dengan kolom `komentar` untuk mengklasifikasikan seluruh baris secara otomatis.")
    batch_model_ready = True
    if gunakan_indobert and (not TRANSFORMER_AVAILABLE or not model_indobert_siap()):
        st.warning("Model IndoBERT belum siap. Lakukan fine-tuning pada tab **Dataset & Pelatihan** atau pilih model lain di sidebar.")
        batch_model_ready = False
    elif gunakan_pretrained and not TRANSFORMER_AVAILABLE:
        st.warning("Dependensi `transformers` dan `torch` belum terpasang untuk menjalankan model Pre-trained. Pasang dependensi atau pilih model lain di sidebar.")
        batch_model_ready = False

    uploaded_file = st.file_uploader("Unggah Berkas CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None and batch_model_ready:
        raw_bytes = uploaded_file.getvalue()
        if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw_bytes[:200]:
            raw_text = raw_bytes.decode("utf-16", errors="replace")
        else:
            raw_text = raw_bytes.decode("utf-8-sig", errors="replace")

        raw_text = raw_text.replace("\x00", "")
        df_upload = None

        for sep in (",", ";"):
            try:
                temp_df = pd.read_csv(io.StringIO(raw_text), sep=sep)
                temp_df.columns = [str(c).strip() for c in temp_df.columns]
                if "komentar" in temp_df.columns:
                    df_upload = temp_df
                    break
            except pd.errors.ParserError:
                pass

        if df_upload is None:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if not lines:
                df_upload = pd.DataFrame(columns=["komentar"])
            else:
                header_bersih = lines[0].rstrip(",;").strip('"').strip("'").strip().lower()
                if header_bersih == "komentar":
                    data_lines = [ln.rstrip(",;") for ln in lines[1:] if ln.rstrip(",;").strip()]
                else:
                    data_lines = [ln.rstrip(",;") for ln in lines if ln.rstrip(",;").strip()]
                df_upload = pd.DataFrame({"komentar": data_lines})

        if "komentar" not in df_upload.columns:
            st.error("Struktur CSV tidak valid: kolom `komentar` tidak ditemukan.")
        else:
            hasil_list = [prediksi_detail(str(k)) for k in df_upload["komentar"]]
            df_upload["sentimen"] = [h[0] for h in hasil_list]
            df_upload["keyakinan"] = [round(h[1], 1) for h in hasil_list]

            total = len(df_upload)
            n_pos = int((df_upload["sentimen"] == "POSITIF").sum())
            n_net = int((df_upload["sentimen"] == "NETRAL").sum())
            n_neg = int((df_upload["sentimen"] == "NEGATIF").sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Ulasan</div><div class='kpi-num'>{total}</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Positif</div><div class='kpi-num' style='color:#10B981;'>{n_pos}</div></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Netral</div><div class='kpi-num' style='color:#A1A1AA;'>{n_net}</div></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Negatif</div><div class='kpi-num' style='color:#EF4444;'>{n_neg}</div></div>", unsafe_allow_html=True)

            st.markdown("#### Distribusi Sentimen")
            st.bar_chart(df_upload["sentimen"].value_counts(), color="#3B82F6")

            st.markdown("#### Rincian Hasil Klasifikasi")
            st.dataframe(
                df_upload,
                width="stretch",
                hide_index=True,
                column_config={
                    "keyakinan": st.column_config.ProgressColumn(
                        "Keyakinan",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    )
                },
            )

            csv_output = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Ekspor Hasil ke CSV",
                data=csv_output,
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )

# ------------------------------------------
# TAB 3: DATASET & PELATIHAN MODEL
# ------------------------------------------
with tab3:
    if gunakan_pretrained:
        st.info(
            "Model **Pre-trained Sentiment** menggunakan bobot siap pakai dari Hugging Face sehingga tidak memerlukan pelatihan lokal. "
            "Anda tetap dapat menambah sampel di bawah ini untuk memperkaya korpus model lokal (**Linear SVM** dan **IndoBERT**)."
        )

    col_a, col_b = st.columns([1.15, 1], gap="large")

    with col_a:
        st.markdown("#### Tambah Sampel ke Korpus")
        st.caption("Perkaya data latih ketika menemukan pola kalimat atau slang baru.")
        new_comment = st.text_input("Teks Ulasan", placeholder="Contoh: barangnya oke banget sesuai pesanan")
        new_label = st.selectbox("Label Sentimen", ["positif", "netral", "negatif"])

        if st.button("Simpan & Perbarui Linear SVM"):
            if new_comment.strip():
                df_lama = pd.read_csv(DATASET_PATH, encoding="utf-8")
                df_baru = pd.DataFrame([[new_comment.strip(), new_label]], columns=["komentar", "sentimen"])
                df_gabung = pd.concat([df_lama, df_baru], ignore_index=True)
                df_gabung.to_csv(DATASET_PATH, index=False, encoding="utf-8")

                jalankan_training()
                st.cache_resource.clear()
                st.success("Sampel tersimpan dan model Linear SVM telah diperbarui.")
            else:
                st.warning("Teks ulasan tidak boleh kosong.")

    with col_b:
        st.markdown("#### Fine-Tuning IndoBERT")
        st.caption("Adaptasi bobot `indobenchmark/indobert-base-p1` terhadap seluruh data pada `dataset_ulasan.csv`.")
        epochs_input = st.slider("Epoch Pelatihan", min_value=1, max_value=5, value=3)

        if st.button("Latih Ulang IndoBERT"):
            if not TRANSFORMER_AVAILABLE:
                st.error("Dependensi `transformers` dan `torch` belum terpasang.")
            else:
                df_train = pd.read_csv(DATASET_PATH, encoding="utf-8").dropna()
                with st.spinner("Menjalankan fine-tuning IndoBERT..."):
                    latih_indobert(df_train, epochs=epochs_input)
                st.success("Fine-tuning selesai. Bobot terbaru siap digunakan.")