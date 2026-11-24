import os
import sys
import json
import argparse
import joblib
import pandas as pd

from utils import preprocess_text, analisis_aspek
from train_model import MODEL_PATH, jalankan_training
from transformer_model import (
    DEFAULT_SENTIMENT_PRETRAINED,
    PRETRAINED_SENTIMENT_CATALOG,
    ModelNotReadyError,
    alasan_model_belum_siap,
    prediksi_indobert_detail,
    prediksi_pretrained_detail,
)

# Kode warna ANSI untuk Windows Terminal / PowerShell
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
GRAY = "\033[90m"
BLUE = "\033[94m"
YELLOW = "\033[93m"

COLOR_MAP = {
    "POSITIF": GREEN,
    "NEGATIF": RED,
    "NETRAL": GRAY,
}

def muat_svm():
    if not os.path.exists(MODEL_PATH):
        jalankan_training()
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"]

def buat_prediktor(engine: str):
    reason = alasan_model_belum_siap(engine)
    if reason:
        raise ModelNotReadyError(reason)

    use_pretrained = engine == "pretrained"
    use_bert = engine == "indobert"
    svm_model = muat_svm() if engine == "svm" else None
    active_engine = {
        "svm": "Linear SVM",
        "indobert": "IndoBERT",
        "pretrained": next(
            name for name, model_id in PRETRAINED_SENTIMENT_CATALOG.items()
            if model_id == DEFAULT_SENTIMENT_PRETRAINED
        ),
    }[engine]

    def predict_detail(teks: str):
        if use_pretrained:
            return prediksi_pretrained_detail(teks)
        if use_bert:
            return prediksi_indobert_detail(teks)

        clean = preprocess_text(teks)
        vektor = svm_model.named_steps["tfidf"].transform([clean])
        if vektor.nnz == 0:
            return "NETRAL", 0.0, clean, True, {"POSITIF": 0.0, "NETRAL": 100.0, "NEGATIF": 0.0}

        pred = svm_model.predict([clean])[0].upper()
        probs_arr = svm_model.predict_proba([clean])[0]
        class_map = {c.upper(): float(p) * 100 for c, p in zip(svm_model.classes_, probs_arr)}
        probs_dict = {
            "POSITIF": round(class_map.get("POSITIF", 0.0), 1),
            "NETRAL": round(class_map.get("NETRAL", 0.0), 1),
            "NEGATIF": round(class_map.get("NEGATIF", 0.0), 1),
        }
        return pred, probs_dict[pred], clean, False, probs_dict

    return predict_detail, active_engine

def render_bar(persen: float, lebar: int = 20) -> str:
    terisi = int(round((persen / 100.0) * lebar))
    return "█" * terisi + "░" * (lebar - terisi)

def cetak_hasil(teks: str, predict_fn, engine_name: str, tampilkan_absa: bool = True, as_json: bool = False):
    label, conf, clean, is_oov, probs = predict_fn(teks)

    absa_list = []
    if tampilkan_absa:
        def _absa_wrapper(klausa: str):
            l, c, cl, oov, _ = predict_fn(klausa)
            return l, c, cl, oov
        absa_list = analisis_aspek(teks, _absa_wrapper)

    if as_json:
        output = {
            "text": teks,
            "engine": engine_name,
            "sentiment": label,
            "confidence": round(conf, 1),
            "probabilities": {k: round(v, 1) for k, v in probs.items()},
            "is_oov": is_oov,
            "absa": absa_list,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    warna = COLOR_MAP.get(label, GRAY)
    print(f"\n{DIM}{'─' * 56}{RESET}")
    print(f"{BOLD}Sentimen :{RESET} {warna}{BOLD}{label}{RESET} {DIM}(Keyakinan: {conf:.1f}% · {engine_name}){RESET}")
    if is_oov:
        print(f"{YELLOW}Catatan  : Kata belum terdaftar di korpus latih (Out-of-Vocabulary){RESET}")

    print(f"\n{BOLD}Distribusi Probabilitas:{RESET}")
    for k_name in ("POSITIF", "NETRAL", "NEGATIF"):
        val = probs[k_name]
        c = COLOR_MAP[k_name]
        print(f"  {k_name:<8} {c}{render_bar(val)}{RESET} {val:5.1f}%")

    if tampilkan_absa and absa_list:
        print(f"\n{BOLD}Segmentasi Aspek (ABSA):{RESET}")
        for item in absa_list:
            s = item["Sentimen"]
            c = COLOR_MAP.get(s, GRAY)
            print(f"  • [{item['Aspek']}] {c}{s}{RESET} ({item['Keyakinan (%)']}%) — {DIM}\"{item['Klausa']}\"{RESET}")

    print(f"{DIM}{'─' * 56}{RESET}\n")

def baca_csv_aman(path: str) -> pd.DataFrame:
    import io

    with open(path, "rb") as f:
        raw_bytes = f.read()

    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw_bytes[:200]:
        text = raw_bytes.decode("utf-16", errors="replace")
    else:
        text = raw_bytes.decode("utf-8-sig", errors="replace")

    text = text.replace("\x00", "")

    for sep in (",", ";"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            df.columns = [str(c).strip() for c in df.columns]
            if "komentar" in df.columns:
                return df
        except pd.errors.ParserError:
            pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame(columns=["komentar"])

    header_bersih = lines[0].rstrip(",;").strip('"').strip("'").strip().lower()
    if header_bersih == "komentar":
        data_lines = [ln.rstrip(",;") for ln in lines[1:] if ln.rstrip(",;").strip()]
    else:
        data_lines = [ln.rstrip(",;") for ln in lines if ln.rstrip(",;").strip()]

    return pd.DataFrame({"komentar": data_lines})

def proses_csv(input_path: str, output_path: str, predict_fn, engine_name: str):
    if not os.path.exists(input_path):
        print(f"{RED}[ERROR] File '{input_path}' tidak ditemukan.{RESET}")
        sys.exit(1)

    df = baca_csv_aman(input_path)
    if "komentar" not in df.columns:
        print(f"{RED}[ERROR] File CSV wajib memiliki kolom bernama 'komentar'.{RESET}")
        sys.exit(1)

    print(f"{BLUE}[INFO] Memproses {len(df)} baris menggunakan {engine_name}...{RESET}")
    hasil = [predict_fn(str(k)) for k in df["komentar"]]
    df["sentimen"] = [h[0] for h in hasil]
    df["keyakinan_%"] = [round(h[1], 1) for h in hasil]

    df.to_csv(output_path, index=False, encoding="utf-8")
    distribusi = df["sentimen"].value_counts().to_dict()
    print(f"{GREEN}[SELESAI] Hasil disimpan ke '{output_path}'{RESET}")
    print(f"Ringkasan: Positif={distribusi.get('POSITIF', 0)}, Netral={distribusi.get('NETRAL', 0)}, Negatif={distribusi.get('NEGATIF', 0)}")

def mode_interaktif(engine_awal: str, tampilkan_absa: bool):
    current_engine = engine_awal
    predict_fn, active_name = buat_prediktor(current_engine)

    print(f"\n{BOLD}SentimenLab CLI{RESET} {DIM}— Mode Interaktif ({active_name}){RESET}")
    print(f"{DIM}Perintah: /svm | /indobert | /pretrained | /absa | /keluar{RESET}\n")

    while True:
        try:
            teks = input(f"{BLUE}sentimen>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar.")
            break

        if not teks:
            continue
        if teks.lower() in ("/keluar", "/exit", "keluar", "exit"):
            break
        if teks.lower() == "/svm":
            try:
                predict_fn, active_name = buat_prediktor("svm")
            except ModelNotReadyError as exc:
                print(f"{YELLOW}[PERINGATAN] {exc}{RESET}\n")
                continue
            current_engine = "svm"
            print(f"{GREEN}Model aktif diubah ke: {active_name}{RESET}\n")
            continue
        if teks.lower() == "/indobert":
            try:
                predict_fn, active_name = buat_prediktor("indobert")
            except ModelNotReadyError as exc:
                print(f"{YELLOW}[PERINGATAN] {exc}{RESET}\n")
                continue
            current_engine = "indobert"
            print(f"{GREEN}Model aktif diubah ke: {active_name}{RESET}\n")
            continue
        if teks.lower() == "/pretrained":
            try:
                predict_fn, active_name = buat_prediktor("pretrained")
            except ModelNotReadyError as exc:
                print(f"{YELLOW}[PERINGATAN] {exc}{RESET}\n")
                continue
            current_engine = "pretrained"
            print(f"{GREEN}Model aktif diubah ke: {active_name}{RESET}\n")
            continue
        if teks.lower() == "/absa":
            tampilkan_absa = not tampilkan_absa
            status = "AKTIF" if tampilkan_absa else "NONAKTIF"
            print(f"{GREEN}Segmentasi ABSA: {status}{RESET}\n")
            continue

        cetak_hasil(teks, predict_fn, active_name, tampilkan_absa=tampilkan_absa)

def main():
    default_engine = "svm" if alasan_model_belum_siap("indobert") else "indobert"

    parser = argparse.ArgumentParser(
        description="SentimenLab CLI — Analisis Sentimen & ABSA Bahasa Indonesia"
    )
    parser.add_argument("-t", "--text", type=str, help="Analisis satu teks komentar langsung lalu keluar")
    parser.add_argument("-f", "--file", type=str, help="Path file CSV input (harus memiliki kolom 'komentar')")
    parser.add_argument("-o", "--output", type=str, default="hasil_cli.csv", help="Path file CSV output (default: hasil_cli.csv)")
    parser.add_argument("-e", "--engine", choices=["indobert", "pretrained", "svm"], default=default_engine, help="Pilih model yang tersedia; model yang belum siap tidak diganti otomatis")
    parser.add_argument("--no-absa", action="store_true", help="Nonaktifkan rincian segmentasi aspek (ABSA)")
    parser.add_argument("--json", action="store_true", help="Keluarkan output dalam format JSON")

    args = parser.parse_args()
    try:
        predict_fn, active_name = buat_prediktor(args.engine)
    except ModelNotReadyError as exc:
        parser.error(str(exc))

    if args.file:
        proses_csv(args.file, args.output, predict_fn, active_name)
    elif args.text:
        cetak_hasil(args.text, predict_fn, active_name, tampilkan_absa=not args.no_absa, as_json=args.json)
    else:
        mode_interaktif(args.engine, tampilkan_absa=not args.no_absa)

if __name__ == "__main__":
    main()