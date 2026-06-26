#!/usr/bin/env python3
"""
PIPELINE COMPLET — Script de démarrage rapide
=============================================
Lance tous les modules en séquence sur un dossier de fichiers audio.

Usage :
    python pipeline.py --input /chemin/vers/audios/
    python pipeline.py --input corpus/ --model whisper --mode zeroshot

Durée estimée sur CPU (pour 30 min d'audio, ~60 segments) :
  Module 1 (segmentation)   : ~30 secondes
  Module 2 (embeddings)     : ~5-10 minutes (Whisper-small CPU)
  Module 3 (index FAISS)    : ~2 secondes
  Module 5 (criticité)      : ~2-3 minutes (LaBSE)
  Total                     : ~10-15 minutes
"""

import argparse
import json
import os
import time

from module1_ingestion  import process_directory, process_file, init_db
from module2_embeddings import compute_embeddings
from module3_index      import build_index
from module5_criticite  import score_corpus

import numpy as np


def banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def run_pipeline(input_path: str,
                 model: str  = "whisper",
                 mode: str   = "zeroshot",
                 labels: str = None,
                 seg_dir: str  = "corpus/segments",
                 db_path: str  = "corpus/metadata.db",
                 seg_json: str = "corpus/segments.json",
                 emb_path: str = "index/embeddings.npy",
                 idx_path: str = "index/faiss.index",
                 ids_path: str = "index/segment_ids.json"):

    t_start = time.time()
    os.makedirs("corpus/segments", exist_ok=True)
    os.makedirs("index", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Module 1 : Ingestion & Segmentation ──────────────────────────────────
    banner("MODULE 1 — Ingestion & Segmentation")
    conn = init_db(db_path)
    if os.path.isdir(input_path):
        metas = process_directory(input_path, seg_dir, conn)
    else:
        metas = process_file(input_path, seg_dir, conn)
    conn.close()

    with open(seg_json, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(metas)} segments produits")
    if not metas:
        print("✗ Aucun segment produit. Vérifiez vos fichiers audio.")
        return

    # ── Module 2 : Embeddings ─────────────────────────────────────────────────
    banner("MODULE 2 — Extraction d'Embeddings")
    result = compute_embeddings(seg_json, model, db_path, emb_path)
    if not result:
        print("✗ Embeddings échoués.")
        return
    print(f"✓ {len(result['ids'])} embeddings ({result['dim']}-dim)")

    # ── Module 3 : Index FAISS ────────────────────────────────────────────────
    banner("MODULE 3 — Indexation FAISS")
    emb_matrix = np.load(emb_path)
    emb_ids_path = emb_path.replace(".npy", "_ids.json")
    with open(emb_ids_path) as f:
        emb_ids = json.load(f)

    index = build_index(emb_matrix, emb_ids, idx_path, ids_path)
    print(f"✓ Index construit : {index.ntotal} vecteurs")

    # ── Module 5 : Scoring criticité ─────────────────────────────────────────
    banner("MODULE 5 — Scoring de Criticité")
    scored = score_corpus(seg_json, mode, db_path, labels)
    print(f"✓ {len(scored)} segments scorés")

    # ── Résumé ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner(f"PIPELINE TERMINÉ EN {elapsed:.0f}s")

    alertes = [s for s in scored if s["criticite"]["niveau"] in ("ROUGE", "ORANGE")]
    if alertes:
        print(f"\n🚨 {len(alertes)} alertes détectées :\n")
        for a in alertes[:10]:
            niv = a["criticite"]["niveau"]
            ico = "🔴" if niv == "ROUGE" else "🟠"
            dims = ", ".join(a["criticite"].get("alertes", []))
            print(f"  {ico} {a['source']}  [{a['start_sec']}s–{a['end_sec']}s]  → {dims}")

    print(f"\n▶ Lancer l'interface : streamlit run app.py")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Audio Sémantique Souverain")
    parser.add_argument("--input",   required=True,  help="Fichier audio ou dossier")
    parser.add_argument("--model",   default="whisper", choices=["whisper", "wav2vec2"])
    parser.add_argument("--mode",    default="zeroshot", choices=["zeroshot", "svm"])
    parser.add_argument("--labels",  default=None,    help="CSV labels (mode svm)")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        model=args.model,
        mode=args.mode,
        labels=args.labels,
    )
