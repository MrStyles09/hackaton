"""
ÉVALUATION DU PIPELINE — CITADEL 2026
======================================
Calcule les métriques de pertinence pour la présentation :

  Recherche TEXTE (LaBSE + textuel mooré) :
    - Precision@1, @3, @5
    - MRR (Mean Reciprocal Rank)
    - Temps de réponse moyen

  Recherche AUDIO→AUDIO (Whisper + FAISS) :
    - Precision@1, @3, @5 (self-retrieval : le fichier doit se retrouver lui-même)
    - Score de similarité moyen

Usage :
    python evaluate_pipeline.py --n 50
    python evaluate_pipeline.py --n 100 --out resultats_eval.json
"""

import argparse
import json
import os
import sqlite3
import time
import random
import sys

import numpy as np

DB_PATH    = "corpus/metadata.db"
INDEX_PATH = "index/faiss.index"
IDS_PATH   = "index/segment_ids.json"
SEG_JSON   = "corpus/segments.json"


# ── Chargement ────────────────────────────────────────────────────────────────

def load_corpus():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, file, path, transcription, translation_fr
        FROM segments WHERE transcription != '' OR translation_fr != ''
    """).fetchall()
    conn.close()
    return [{"id":r[0],"file":r[1],"path":r[2],
             "transcription":r[3] or "","translation_fr":r[4] or ""} for r in rows]


def load_labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")


def load_whisper():
    from module2_embeddings import WhisperEmbedder
    return WhisperEmbedder()


def load_faiss():
    import faiss
    index = faiss.read_index(INDEX_PATH)
    with open(IDS_PATH) as f:
        ids = json.load(f)
    return index, ids


def compute_labse_embs(segs, model):
    texts = [s["translation_fr"].strip() or s["transcription"].strip() or "vide"
             for s in segs]
    return model.encode(texts, batch_size=64,
                        normalize_embeddings=True, show_progress_bar=False)


# ── Métriques ─────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids, relevant_id, k):
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids, relevant_id):
    for i, rid in enumerate(retrieved_ids, 1):
        if rid == relevant_id:
            return 1.0 / i
    return 0.0


# ── Évaluation recherche texte ────────────────────────────────────────────────

def eval_text_search(segs, labse_embs, labse_model, n_queries=50):
    """
    Self-retrieval sur les traductions françaises :
    pour chaque segment, on cherche avec son propre texte FR
    et on vérifie qu'il revient en top-k.
    """
    print(f"\n{'─'*50}")
    print(f"ÉVALUATION RECHERCHE TEXTE (n={n_queries})")
    print(f"{'─'*50}")

    sample = random.sample([s for s in segs if s["translation_fr"].strip()],
                           min(n_queries, len(segs)))

    seg_by_id = {s["id"]: i for i, s in enumerate(segs)}

    p1 = p3 = p5 = mrr = 0.0
    times = []

    for seg in sample:
        query = seg["translation_fr"].strip()
        t0 = time.time()

        q_emb = labse_model.encode([query], normalize_embeddings=True)[0]
        sims  = labse_embs @ q_emb
        top_ids = [segs[i]["id"] for i in np.argsort(sims)[::-1][:10]]

        elapsed = time.time() - t0
        times.append(elapsed)

        p1  += precision_at_k(top_ids, seg["id"], 1)
        p3  += precision_at_k(top_ids, seg["id"], 3)
        p5  += precision_at_k(top_ids, seg["id"], 5)
        mrr += reciprocal_rank(top_ids, seg["id"])

    n = len(sample)
    results = {
        "type"         : "text_fr_self_retrieval",
        "n_queries"    : n,
        "precision@1"  : round(p1/n, 3),
        "precision@3"  : round(p3/n, 3),
        "precision@5"  : round(p5/n, 3),
        "MRR"          : round(mrr/n, 3),
        "latency_ms_mean": round(np.mean(times)*1000, 1),
        "latency_ms_p95" : round(np.percentile(times, 95)*1000, 1),
    }

    print(f"  Precision@1  : {results['precision@1']:.1%}")
    print(f"  Precision@3  : {results['precision@3']:.1%}")
    print(f"  Precision@5  : {results['precision@5']:.1%}")
    print(f"  MRR          : {results['MRR']:.3f}")
    print(f"  Latence moy  : {results['latency_ms_mean']} ms")
    print(f"  Latence P95  : {results['latency_ms_p95']} ms")

    return results


# ── Évaluation recherche audio ────────────────────────────────────────────────

def eval_audio_search(segs, faiss_index, faiss_ids, whisper, n_queries=30):
    """
    Self-retrieval audio : encode chaque fichier WAV et vérifie
    qu'il retrouve lui-même en top-k dans FAISS.
    """
    import librosa

    print(f"\n{'─'*50}")
    print(f"ÉVALUATION RECHERCHE AUDIO (n={n_queries})")
    print(f"{'─'*50}")

    # Garder seulement les segments avec fichier audio existant
    valid = [s for s in segs if os.path.exists(s["path"])]
    sample = random.sample(valid, min(n_queries, len(valid)))

    id_to_faiss_pos = {sid: i for i, sid in enumerate(faiss_ids)}

    p1 = p3 = p5 = mrr = 0.0
    sim_exact = []
    times = []
    not_in_index = 0

    for seg in sample:
        if seg["id"] not in id_to_faiss_pos:
            not_in_index += 1
            continue

        t0 = time.time()
        try:
            audio, sr = librosa.load(seg["path"], sr=16000, mono=True)
            peak = max(np.abs(audio).max(), 1e-8)
            audio = (audio / peak * 0.95).astype(np.float32)

            emb = whisper.embed(audio, sr=16000).astype(np.float32)
            emb = emb / np.linalg.norm(emb)

            scores, indices = faiss_index.search(emb.reshape(1,-1), 10)
            top_ids = [faiss_ids[i] for i in indices[0] if 0 <= i < len(faiss_ids)]

            elapsed = time.time() - t0
            times.append(elapsed)

            # Score de similarité avec lui-même
            for i, tid in enumerate(top_ids):
                if tid == seg["id"]:
                    sim_exact.append(float(scores[0][i]))
                    break

            p1  += precision_at_k(top_ids, seg["id"], 1)
            p3  += precision_at_k(top_ids, seg["id"], 3)
            p5  += precision_at_k(top_ids, seg["id"], 5)
            mrr += reciprocal_rank(top_ids, seg["id"])

        except Exception as e:
            print(f"  ⚠ Erreur {seg['file']} : {e}")

    n = len(sample) - not_in_index
    if n == 0:
        print("  Aucun segment valide trouvé.")
        return {}

    results = {
        "type"           : "audio_self_retrieval",
        "n_queries"      : n,
        "not_in_index"   : not_in_index,
        "precision@1"    : round(p1/n, 3),
        "precision@3"    : round(p3/n, 3),
        "precision@5"    : round(p5/n, 3),
        "MRR"            : round(mrr/n, 3),
        "sim_exact_mean" : round(np.mean(sim_exact), 4) if sim_exact else 0,
        "latency_ms_mean": round(np.mean(times)*1000, 1),
        "latency_ms_p95" : round(np.percentile(times,95)*1000, 1),
    }

    print(f"  Precision@1  : {results['precision@1']:.1%}")
    print(f"  Precision@3  : {results['precision@3']:.1%}")
    print(f"  Precision@5  : {results['precision@5']:.1%}")
    print(f"  MRR          : {results['MRR']:.3f}")
    print(f"  Sim. exacte  : {results['sim_exact_mean']:.4f}")
    print(f"  Latence moy  : {results['latency_ms_mean']} ms")
    print(f"  Latence P95  : {results['latency_ms_p95']} ms")
    if not_in_index:
        print(f"  ⚠ {not_in_index} segments absents de l'index")

    return results


# ── Stats corpus ──────────────────────────────────────────────────────────────

def corpus_stats(segs):
    print(f"\n{'─'*50}")
    print(f"STATISTIQUES DU CORPUS")
    print(f"{'─'*50}")

    conn = sqlite3.connect(DB_PATH)
    niveaux = conn.execute("""
        SELECT json_extract(criticite_json,'$.niveau'), COUNT(*)
        FROM segments WHERE criticite_json IS NOT NULL
        GROUP BY 1
    """).fetchall()
    conn.close()

    total = len(segs)
    with_moore = sum(1 for s in segs if s["transcription"].strip())
    with_fr    = sum(1 for s in segs if s["translation_fr"].strip())
    with_audio = sum(1 for s in segs if os.path.exists(s["path"]))

    print(f"  Total segments     : {total}")
    print(f"  Avec texte mooré   : {with_moore} ({with_moore/total:.1%})")
    print(f"  Avec traduction FR : {with_fr} ({with_fr/total:.1%})")
    print(f"  Fichiers audio OK  : {with_audio} ({with_audio/total:.1%})")
    print(f"  Niveaux criticité  : {dict(niveaux)}")

    return {
        "total": total, "with_moore": with_moore,
        "with_fr": with_fr, "with_audio": with_audio,
        "criticite": dict(niveaux),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=50, help="Nb requêtes test")
    parser.add_argument("--out",  default="resultats_eval.json")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Ignorer l'évaluation audio (plus rapide)")
    args = parser.parse_args()

    print("=" * 50)
    print("  ÉVALUATION PIPELINE CITADEL 2026")
    print("=" * 50)

    # Charger
    print("\nChargement du corpus...")
    segs = load_corpus()
    print(f"  {len(segs)} segments chargés")

    stats = corpus_stats(segs)

    print("\nChargement LaBSE...")
    labse = load_labse()
    print("  LaBSE OK")

    print("Calcul embeddings LaBSE corpus...")
    labse_embs = compute_labse_embs(segs, labse)
    print(f"  {labse_embs.shape} embeddings")

    # Éval texte
    text_results = eval_text_search(segs, labse_embs, labse, n_queries=args.n)

    # Éval audio
    audio_results = {}
    if not args.skip_audio:
        print("\nChargement Whisper + FAISS...")
        whisper = load_whisper()
        faiss_index, faiss_ids = load_faiss()
        print(f"  FAISS : {faiss_index.ntotal} vecteurs")
        audio_results = eval_audio_search(
            segs, faiss_index, faiss_ids, whisper,
            n_queries=min(30, args.n)
        )

    # Résumé
    print(f"\n{'═'*50}")
    print("  RÉSUMÉ POUR LA PRÉSENTATION")
    print(f"{'═'*50}")
    print(f"""
Corpus : {stats['total']} segments mooré
  · {stats['with_moore']} transcriptions mooré
  · {stats['with_fr']} traductions françaises
  · {stats['with_audio']} fichiers audio

Recherche textuelle (LaBSE) :
  · Precision@1 = {text_results.get('precision@1',0):.1%}
  · Precision@5 = {text_results.get('precision@5',0):.1%}
  · MRR         = {text_results.get('MRR',0):.3f}
  · Latence     = {text_results.get('latency_ms_mean',0)} ms/requête
""")
    if audio_results:
        print(f"""Recherche vocale (Whisper+FAISS) :
  · Precision@1 = {audio_results.get('precision@1',0):.1%}
  · Precision@5 = {audio_results.get('precision@5',0):.1%}
  · MRR         = {audio_results.get('MRR',0):.3f}
  · Latence     = {audio_results.get('latency_ms_mean',0)} ms/requête
""")

    # Sauvegarder
    output = {
        "corpus_stats"   : stats,
        "text_search"    : text_results,
        "audio_search"   : audio_results,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Résultats sauvegardés → {args.out}")


if __name__ == "__main__":
    main()
