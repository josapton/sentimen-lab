import os
import importlib.util
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDOBERT_DIR = os.path.join(BASE_DIR, "model_indobert")
DEFAULT_PRETRAINED = "indobenchmark/indobert-base-p1"

# Katalog model yang sudah di-fine-tune khusus untuk analisis sentimen
PRETRAINED_SENTIMENT_CATALOG = {
    "Indo-RoBERTa SmSA (w11wo)": "w11wo/indonesian-roberta-base-sentiment-classifier",
    "XLM-RoBERTa Social (CardiffNLP)": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}
DEFAULT_SENTIMENT_PRETRAINED = "w11wo/indonesian-roberta-base-sentiment-classifier"

LABEL2ID = {"positif": 0, "netral": 1, "negatif": 2}
ID2LABEL = {0: "positif", 1: "netral", 2: "negatif"}

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None

# Cache terpisah untuk model pre-trained sentiment siap pakai
_PRETRAINED_CACHE = {}

def cek_dependensi_transformer() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )

TRANSFORMER_AVAILABLE = cek_dependensi_transformer()

class ModelNotReadyError(RuntimeError):
    pass


def alasan_model_belum_siap(engine: str) -> Optional[str]:
    if engine == "svm":
        return None
    if engine not in {"indobert", "pretrained"}:
        return f"Engine model tidak dikenal: {engine}."
    if not TRANSFORMER_AVAILABLE:
        return "Dependensi 'torch' dan 'transformers' belum terpasang."
    if engine == "indobert" and not model_indobert_siap():
        return "Model IndoBERT belum di-fine-tune."
    return None

def model_indobert_siap() -> bool:
    """Memastikan file konfigurasi dan bobot model IndoBERT hasil fine-tuning tersedia."""
    has_config = os.path.exists(os.path.join(INDOBERT_DIR, "config.json"))
    has_weights = (
        os.path.exists(os.path.join(INDOBERT_DIR, "model.safetensors"))
        or os.path.exists(os.path.join(INDOBERT_DIR, "pytorch_model.bin"))
    )
    return has_config and has_weights

def _normalisasi_label_hf(raw_label: str) -> str:
    """Menyeragamkan variasi label model Hugging Face ke POSITIF, NETRAL, atau NEGATIF."""
    lbl = str(raw_label).strip().lower()
    if "pos" in lbl or lbl == "label_0":
        return "POSITIF"
    if "neg" in lbl or lbl == "label_2":
        return "NEGATIF"
    if "neu" in lbl or "net" in lbl or lbl == "label_1":
        return "NETRAL"
    return "NETRAL"

def get_pretrained_sentiment_bundle(model_id: str = DEFAULT_SENTIMENT_PRETRAINED):
    """Memuat dan menyimpan model sentimen pre-trained di memori."""
    if not TRANSFORMER_AVAILABLE:
        raise RuntimeError("Library 'torch' dan 'transformers' belum terpasang.")
    if model_id not in PRETRAINED_SENTIMENT_CATALOG.values():
        raise ValueError(f"Model sentimen pre-trained tidak didukung: {model_id}")

    if model_id not in _PRETRAINED_CACHE:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        model.eval()
        _PRETRAINED_CACHE[model_id] = (model, tokenizer, device)

    return _PRETRAINED_CACHE[model_id]

def prediksi_pretrained_detail(teks: str, model_id: str = DEFAULT_SENTIMENT_PRETRAINED):
    """Prediksi sentimen menggunakan model pre-trained khusus sentimen dari Hugging Face."""
    import torch

    model, tokenizer, device = get_pretrained_sentiment_bundle(model_id)
    inputs = tokenizer(
        teks, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    id2label_cfg = getattr(model.config, "id2label", {0: "positive", 1: "neutral", 2: "negative"})
    probs_dict = {"POSITIF": 0.0, "NETRAL": 0.0, "NEGATIF": 0.0}

    for idx, prob_tensor in enumerate(probs):
        raw_name = id2label_cfg.get(idx, id2label_cfg.get(str(idx), "neutral"))
        mapped_name = _normalisasi_label_hf(raw_name)
        probs_dict[mapped_name] += float(prob_tensor.item()) * 100

    label_terbaik = max(probs_dict, key=probs_dict.get)
    confidence = probs_dict[label_terbaik]
    return label_terbaik, confidence, teks.strip(), False, probs_dict

def latih_indobert(df, model_name: str = DEFAULT_PRETRAINED, epochs: int = 3, lr: float = 3e-5):
    """Melakukan fine-tuning IndoBERT menggunakan dataset ulasan."""
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    if not TRANSFORMER_AVAILABLE:
        raise RuntimeError("Library 'torch' dan 'transformers' belum terpasang.")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    texts = df["komentar"].astype(str).tolist()
    labels = [LABEL2ID[s.lower()] for s in df["sentimen"]]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    class SentimentDataset(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in encodings.items()}
            item["labels"] = torch.tensor(labels[idx], dtype=torch.long)
            return item

    loader = DataLoader(SentimentDataset(), batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    os.makedirs(INDOBERT_DIR, exist_ok=True)
    model.save_pretrained(INDOBERT_DIR)
    tokenizer.save_pretrained(INDOBERT_DIR)

    _CACHED_MODEL = None
    _CACHED_TOKENIZER = None
    _CACHED_DEVICE = None
    return True

def prediksi_indobert(teks: str):
    label, conf, clean, is_oov, _ = prediksi_indobert_detail(teks)
    return label, conf, clean, is_oov

def prediksi_indobert_detail(teks: str):
    """Mengembalikan prediksi IndoBERT fine-tuned beserta distribusi probabilitas ketiga kelas."""
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    if not model_indobert_siap():
        raise FileNotFoundError("Model IndoBERT belum dilatih.")

    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    if _CACHED_MODEL is None or _CACHED_TOKENIZER is None:
        _CACHED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _CACHED_TOKENIZER = BertTokenizer.from_pretrained(INDOBERT_DIR)
        _CACHED_MODEL = BertForSequenceClassification.from_pretrained(INDOBERT_DIR).to(_CACHED_DEVICE)
        _CACHED_MODEL.eval()

    inputs = _CACHED_TOKENIZER(
        teks, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(_CACHED_DEVICE)

    with torch.no_grad():
        logits = _CACHED_MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item()) * 100
        probs_dict = {
            "POSITIF": float(probs[0].item()) * 100,
            "NETRAL": float(probs[1].item()) * 100,
            "NEGATIF": float(probs[2].item()) * 100,
        }

    return ID2LABEL[pred_id].upper(), confidence, teks.strip(), False, probs_dict