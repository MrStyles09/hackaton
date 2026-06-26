"""
MODULE 3 — Indexation Vectorielle FAISS
=========================================
Construit et persiste un index FAISS local à partir des embeddings du module 2.
Supporte la recherche par similarité cosinus (produit scalaire sur vecteurs normalisés).
100% CPU, open-source, déployable sans cloud.

Usage :
    python module3_index.py --embeddings index/embeddings.npy --ids index/embeddings_ids.json
    python module3_index.py --rebuild   # recrée l'index depuis zéro
"""

import argparse
import json
import os
import sqlite3
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np


# ─── Constantes ────────────────────────────────────────────────────────────────

INDEX_PATH   = "index/faiss.index"
IDS_PATH     = "index/segment_ids.json"
DB_PATH      = "corpus/metadata.db"


# ─── Construction de l'index ───────────────────────────────────────────────────

def build_index(embeddings: np.ndarray, ids: List[str],
                index_path: str = INDEX_PATH,
                ids_path: str   = IDS_PATH) -> faiss.IndexFlatIP:
    """
    Construit un index FAISS IndexFlatIP (inner product = cosine sur vecteurs L2-normalisés).
    C'est l'index le plus simple, exact et sans paramètre — parfait pour <100k segments.

    Pour un corpus plus grand (>100k), utiliser IndexIVFFlat avec nlist=sqrt(N).
    """
    n, dim = embeddings.shape
    print(f"[M3] Construction index : {n} vecteurs × {dim} dims")

    # Vérifier la normalisation L2
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        print("[M3] ⚠ Normalisation L2 appliquée automatiquement")
        embeddings = embeddings / norms[:, None]

    # Créer l'index
    if n < 10_000:
        # Index exact (brute-force) — optimal pour petits corpus hackathon
        index = faiss.IndexFlatIP(dim)
    else:
        # Index approximatif (IVF) pour grand corpus
        nlist = min(int(np.sqrt(n)), 256)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = min(nlist // 4, 32)  # compromis précision/vitesse

    index.add(embeddings)

    # Sauvegarder l'index
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"[M3] ✓ Index sauvegardé : {index_path} ({index.ntotal} vecteurs)")

    # Sauvegarder le mapping index_position → segment_id
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    return index


def load_index(index_path: str = INDEX_PATH,
               ids_path:   str = IDS_PATH) -> Tuple[faiss.Index, List[str]]:
    """Charge un index FAISS existant et son mapping d'IDs."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index introuvable : {index_path}")
    index = faiss.read_index(index_path)
    with open(ids_path) as f:
        ids = json.load(f)
    print(f"[M3] Index chargé : {index.ntotal} vecteurs")
    return index, ids


# ─── Recherche ─────────────────────────────────────────────────────────────────

def search(query_vector: np.ndarray,
           index: faiss.Index,
           ids: List[str],
           conn: sqlite3.Connection,
           top_k: int = 5) -> List[Dict]:
    """
    Recherche les top_k segments les plus proches du vecteur requête.
    Retourne une liste de dicts avec métadonnées enrichies depuis SQLite.
    """
    # Normaliser le vecteur requête
    query = query_vector.astype(np.float32).reshape(1, -1)
    norm  = np.linalg.norm(query)
    if norm > 0:
        query /= norm

    # Recherche FAISS
    scores, indices = index.search(query, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0 or idx >= len(ids):
            continue   # position invalide (peut arriver avec IVF)

        seg_id = ids[idx]

        # Récupérer les métadonnées depuis SQLite
        row = conn.execute("""
            SELECT id, file, path, source, start_sec, end_sec, duration_sec,
                   transcription, criticite_json
            FROM segments WHERE id = ?
        """, (seg_id,)).fetchone()

        if row:
            results.append({
                "rank"         : rank + 1,
                "score"        : float(score),
                "id"           : row[0],
                "file"         : row[1],
                "path"         : row[2],
                "source"       : row[3],
                "start_sec"    : row[4],
                "end_sec"      : row[5],
                "duration_sec" : row[6],
                "transcription": row[7],
                "criticite"    : json.loads(row[8]) if row[8] else None,
            })

    return results


def search_by_text(query_text: str,
                   index: faiss.Index,
                   ids: List[str],
                   conn: sqlite3.Connection,
                   top_k: int = 5) -> List[Dict]:
    """
    Recherche à partir d'une requête textuelle (français / mooré translittéré).
    Encode via LaBSE puis interroge l'index FAISS.

    Note : fonctionne bien si les embeddings audio ont été projetés dans l'espace
    LaBSE via le CrossModalBridge du module 2. Si non, retourne des résultats
    basés sur la similarité acoustique brute (moins précis sémantiquement).
    """
    from module2_embeddings import LaBSEEmbedder
    embedder = LaBSEEmbedder()
    query_vector = embedder.embed(query_text)
    return search(query_vector, index, ids, conn, top_k)


# ─── Ajout incrémental de nouveaux segments ───────────────────────────────────

def add_to_index(new_embeddings: np.ndarray, new_ids: List[str],
                  index: faiss.Index, ids: List[str],
                  index_path: str = INDEX_PATH,
                  ids_path:   str = IDS_PATH):
    """Ajoute de nouveaux vecteurs à un index existant (mise à jour incrémentale)."""
    norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    new_embeddings = new_embeddings / np.where(norms > 0, norms, 1)

    index.add(new_embeddings.astype(np.float32))
    ids.extend(new_ids)

    faiss.write_index(index, index_path)
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    print(f"[M3] +{len(new_ids)} vecteurs ajoutés → total {index.ntotal}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation FAISS")
    parser.add_argument("--embeddings", default="index/embeddings.npy")
    parser.add_argument("--ids",        default="index/embeddings_ids.json")
    parser.add_argument("--index",      default=INDEX_PATH)
    parser.add_argument("--out-ids",    default=IDS_PATH)
    parser.add_argument("--query",      type=str, help="Requête texte de test")
    parser.add_argument("--db",         default=DB_PATH)
    args = parser.parse_args()

    # Construire l'index
    embeddings = np.load(args.embeddings)
    with open(args.ids) as f:
        ids = json.load(f)

    index = build_index(embeddings, ids, args.index, args.out_ids)

    # Test de recherche optionnel
    if args.query:
        conn = sqlite3.connect(args.db)
        print(f"\n[M3] Requête test : '{args.query}'")
        results = search_by_text(args.query, index, ids, conn)
        for r in results:
            print(f"  #{r['rank']} score={r['score']:.3f}  {r['source']}  "
                  f"[{r['start_sec']}s–{r['end_sec']}s]  {r['transcription'] or '(pas de transcription)'}")
        conn.close()
