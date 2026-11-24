import os
from contextlib import asynccontextmanager
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from utils import preprocess_text, analisis_aspek
from train_model import DATASET_PATH, MODEL_PATH, jalankan_training
from transformer_model import (
    TRANSFORMER_AVAILABLE,
    INDOBERT_DIR,
    DEFAULT_SENTIMENT_PRETRAINED,
    PRETRAINED_SENTIMENT_CATALOG,
    ID2LABEL,
    alasan_model_belum_siap,
    model_indobert_siap,
    prediksi_indobert_detail,
    prediksi_pretrained_detail,
    get_pretrained_sentiment_bundle,
    _normalisasi_label_hf,
)

MAX_TEXT_LENGTH = 5000
MAX_BATCH_SIZE = 100
MAX_ABSA_CLAUSES = 100
SUPPORTED_PRETRAINED_MODELS = set(PRETRAINED_SENTIMENT_CATALOG.values())
ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]


def _load_svm_model():
    if not os.path.exists(MODEL_PATH):
        jalankan_training()
    artifact = joblib.load(MODEL_PATH)
    if "model" not in artifact:
        raise RuntimeError(f"Artefak model '{MODEL_PATH}' tidak berisi model.")
    return artifact["model"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svm_model = await run_in_threadpool(_load_svm_model)
    yield


app = FastAPI(title="SentimenLab Local API", version="1.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_BERT_MODEL = None
_BERT_TOKENIZER = None
_BERT_DEVICE = None


def get_indobert_bundle():
    global _BERT_MODEL, _BERT_TOKENIZER, _BERT_DEVICE
    if _BERT_MODEL is None or _BERT_TOKENIZER is None:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification

        _BERT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _BERT_TOKENIZER = BertTokenizer.from_pretrained(INDOBERT_DIR)
        _BERT_MODEL = BertForSequenceClassification.from_pretrained(INDOBERT_DIR).to(_BERT_DEVICE)
        _BERT_MODEL.eval()
    return _BERT_MODEL, _BERT_TOKENIZER, _BERT_DEVICE


class PredictRequest(BaseModel):
    text: str
    engine: Literal["indobert", "svm", "pretrained"] = "indobert"
    pretrained_model: str = DEFAULT_SENTIMENT_PRETRAINED
    include_absa: bool = True


class BatchPredictRequest(BaseModel):
    texts: List[str]
    engine: Literal["indobert", "svm", "pretrained"] = "indobert"
    pretrained_model: str = DEFAULT_SENTIMENT_PRETRAINED


def _validate_model_id(model_id: str):
    if model_id not in SUPPORTED_PRETRAINED_MODELS:
        raise HTTPException(status_code=422, detail="Model pre-trained tidak termasuk katalog yang didukung.")


def _validate_text(text: str):
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Teks melebihi batas {MAX_TEXT_LENGTH} karakter.",
        )


def _engine_label(engine: str, pretrained_model: str) -> str:
    if engine == "pretrained":
        return next(
            name for name, model_id in PRETRAINED_SENTIMENT_CATALOG.items()
            if model_id == pretrained_model
        )
    return {"indobert": "IndoBERT", "svm": "Linear SVM"}[engine]


def run_single_prediction(
    text: str,
    engine: str,
    pretrained_model: str = DEFAULT_SENTIMENT_PRETRAINED,
):
    reason = alasan_model_belum_siap(engine)
    if reason:
        raise HTTPException(status_code=503, detail=reason)

    if engine == "pretrained":
        label, conf, clean, is_oov, probs = prediksi_pretrained_detail(
            text, model_id=pretrained_model
        )
        return label, round(conf, 1), probs, is_oov, _engine_label(engine, pretrained_model)

    if engine == "indobert":
        label, conf, clean, is_oov, probs = prediksi_indobert_detail(text)
        return label, round(conf, 1), probs, is_oov, "IndoBERT"

    svm_model = getattr(app.state, "svm_model", None)
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Model Linear SVM belum siap.")

    clean = preprocess_text(text)
    vektor = svm_model.named_steps["tfidf"].transform([clean])
    if vektor.nnz == 0:
        return "NETRAL", 0.0, {"POSITIF": 0.0, "NETRAL": 100.0, "NEGATIF": 0.0}, True, "Linear SVM"

    pred = svm_model.predict([clean])[0].upper()
    probs_arr = svm_model.predict_proba([clean])[0]
    class_map = {c.upper(): float(p) * 100 for c, p in zip(svm_model.classes_, probs_arr)}
    probs_dict = {
        "POSITIF": round(class_map.get("POSITIF", 0.0), 1),
        "NETRAL": round(class_map.get("NETRAL", 0.0), 1),
        "NEGATIF": round(class_map.get("NEGATIF", 0.0), 1),
    }
    return pred, probs_dict[pred], probs_dict, False, "Linear SVM"


@app.get("/health")
def health_check():
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 0:
        total_data = len(pd.read_csv(DATASET_PATH))
    else:
        total_data = 0
    return {
        "status": "online",
        "svm_ready": getattr(app.state, "svm_model", None) is not None,
        "indobert_ready": TRANSFORMER_AVAILABLE and model_indobert_siap(),
        "pretrained_ready": TRANSFORMER_AVAILABLE,
        "dataset_count": total_data,
    }


@app.post("/predict")
def predict_single(req: PredictRequest):
    teks = req.text.strip()
    if not teks:
        raise HTTPException(status_code=400, detail="Teks kosong")
    _validate_text(teks)
    _validate_model_id(req.pretrained_model)

    label, conf, probs, is_oov, used_engine = run_single_prediction(
        teks, req.engine, req.pretrained_model
    )

    absa_results = []
    if req.include_absa:
        absa_calls = 0

        def _absa_fn(klausa: str):
            nonlocal absa_calls
            if absa_calls >= MAX_ABSA_CLAUSES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Analisis ABSA dibatasi hingga {MAX_ABSA_CLAUSES} klausa.",
                )
            absa_calls += 1
            l, c, _, oov, _ = run_single_prediction(
                klausa, req.engine, req.pretrained_model
            )
            return l, c, klausa, oov

        absa_results = analisis_aspek(teks, _absa_fn)

    return {
        "sentiment": label,
        "confidence": conf,
        "probabilities": probs,
        "is_oov": is_oov,
        "engine": used_engine,
        "absa": absa_results,
    }


@app.post("/predict-batch")
def predict_batch(req: BatchPredictRequest):
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch melebihi batas {MAX_BATCH_SIZE} teks.",
        )
    if any(len(text) > MAX_TEXT_LENGTH for text in req.texts):
        _validate_text(next(text for text in req.texts if len(text) > MAX_TEXT_LENGTH))
    _validate_model_id(req.pretrained_model)

    texts = [text.strip() for text in req.texts if text and text.strip()]
    if not texts:
        return {"results": [], "engine": _engine_label(req.engine, req.pretrained_model)}

    reason = alasan_model_belum_siap(req.engine)
    if reason:
        raise HTTPException(status_code=503, detail=reason)

    results = []
    if req.engine == "pretrained":
        import torch

        model, tokenizer, device = get_pretrained_sentiment_bundle(req.pretrained_model)
        id2label_cfg = getattr(model.config, "id2label", {0: "positive", 1: "neutral", 2: "negative"})

        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                for row in probs:
                    pred_id = int(torch.argmax(row).item())
                    raw_name = id2label_cfg.get(pred_id, id2label_cfg.get(str(pred_id), "neutral"))
                    sentiment = _normalisasi_label_hf(raw_name)
                    conf = round(float(row[pred_id].item()) * 100, 1)
                    results.append({"sentiment": sentiment, "confidence": conf})

    elif req.engine == "indobert":
        import torch

        model, tokenizer, device = get_indobert_bundle()
        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                for row in probs:
                    pred_id = int(torch.argmax(row).item())
                    conf = round(float(row[pred_id].item()) * 100, 1)
                    results.append({"sentiment": ID2LABEL[pred_id].upper(), "confidence": conf})

    else:
        for text in texts:
            label, conf, _, _, _ = run_single_prediction(text, "svm")
            results.append({"sentiment": label, "confidence": conf})

    return {"results": results, "engine": _engine_label(req.engine, req.pretrained_model)}
