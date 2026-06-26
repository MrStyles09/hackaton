"""
MOTEUR DE RECHERCHE HYBRIDE v2 — CITADEL 2026
==============================================
Stratégie simplifiée et fiable :
  - Toujours faire les DEUX recherches (textuelle mooré + sémantique FR)
  - Fusionner intelligemment les scores
  - Pas de détection de langue automatique (non fiable sur mots courts)
"""

import re
import unicodedata
from typing import List, Dict, Tuple
import numpy as np


def normalize_moore(text: str) -> str:
    """Minuscules + suppression ponctuation, garde diacritiques."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    return ' '.join(text.split())


def normalize_relaxed(text: str) -> str:
    """Supprime aussi les diacritiques — 'ned same' matche 'ned sãame'."""
    text = text.lower().strip()
    nfd  = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def token_overlap(query_tokens, text_tokens) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    q, t = set(query_tokens), set(text_tokens)
    inter = q & t
    if not inter:
        return 0.0
    return sum(len(w) for w in inter) / sum(len(w) for w in q)


def substring_score(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    if query in text:
        return 1.0
    tokens = [t for t in query.split() if len(t) > 1]
    if not tokens:
        return 0.0
    found = sum(1 for t in tokens if t in text)
    return found / len(tokens)


def score_moore_text(query: str, moore_text: str) -> float:
    """
    Score de correspondance textuelle mooré sur 4 niveaux.
    Retourne 0.0–1.0.
    """
    if not moore_text.strip():
        return 0.0

    q_norm    = normalize_moore(query)
    q_relaxed = normalize_relaxed(query)
    t_norm    = normalize_moore(moore_text)
    t_relaxed = normalize_relaxed(moore_text)

    q_tok     = [w for w in q_norm.split()    if len(w) > 1]
    q_tok_rel = [w for w in q_relaxed.split() if len(w) > 1]
    t_tok     = [w for w in t_norm.split()    if len(w) > 1]
    t_tok_rel = [w for w in t_relaxed.split() if len(w) > 1]

    s = 0.0
    s = max(s, substring_score(q_norm,    t_norm)    * 1.00)  # exact + diacritiques
    s = max(s, token_overlap(q_tok, t_tok)            * 0.90)  # tokens + diacritiques
    s = max(s, substring_score(q_relaxed, t_relaxed)  * 0.75)  # exact sans diacritiques
    s = max(s, token_overlap(q_tok_rel, t_tok_rel)    * 0.65)  # tokens sans diacritiques
    return round(s, 4)


def search_all(query: str,
               segments: List[Dict],
               corpus_embs: np.ndarray,
               labse_model,
               top_k: int = 5,
               lang: str = "both") -> List[Dict]:
    """
    Recherche unifiée :
      lang='fr'    → sémantique LaBSE sur traductions FR uniquement
      lang='moore' → textuel mooré uniquement  
      lang='both'  → fusion des deux (défaut)

    Retourne les top_k résultats fusionnés.
    """
    # ── Embeddings sémantiques (LaBSE) ───────────────────────────────────────
    q_emb = labse_model.encode([query], normalize_embeddings=True)[0]
    sem_sims = corpus_embs @ q_emb  # shape (N,)

    seg_by_id = {s["id"]: s for s in segments}
    scores    = {}  # id → {sem, text, final}

    for i, seg in enumerate(segments):
        sid = seg["id"]
        sem_score  = float(sem_sims[i])
        text_score = score_moore_text(query, seg.get("transcription", ""))

        if lang == "fr":
            final = sem_score
        elif lang == "moore":
            final = text_score
        else:  # both
            if text_score > 0.1:
                # Bonne correspondance textuelle : priorité au texte
                final = text_score * 0.70 + sem_score * 0.30
            elif text_score > 0:
                # Correspondance textuelle partielle
                final = text_score * 0.50 + sem_score * 0.50
            else:
                # Pas de correspondance textuelle : sémantique seul
                final = sem_score * 0.80

        scores[sid] = {
            "sem" : round(sem_score, 4),
            "text": round(text_score, 4),
            "final": round(final, 4),
        }

    # Trier par score final
    ranked = sorted(scores.items(), key=lambda x: -x[1]["final"])[:top_k]

    results = []
    for sid, sc in ranked:
        if sc["final"] < 0.05:
            continue
        seg = dict(seg_by_id[sid])
        seg["score"]       = sc["final"]
        seg["score_text"]  = sc["text"]
        seg["score_sem"]   = sc["sem"]
        seg["match_type"]  = (
            "textuel mooré"   if sc["text"] > 0.3 and sc["sem"] < 0.3 else
            "hybride"         if sc["text"] > 0.1 and sc["sem"] > 0.1 else
            "sémantique FR"
        )
        results.append(seg)

    return results
