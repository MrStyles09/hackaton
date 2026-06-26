"""
MOTEUR DE RECHERCHE HYBRIDE — CITADEL 2026
==========================================
Combine 3 stratégies selon la langue détectée :

  1. Recherche EXACTE mooré  : matching direct sur transcriptions (tokens)
  2. Recherche FLOUE mooré   : similarité de caractères (pour les diacritiques)
  3. Recherche SÉMANTIQUE FR : LaBSE sur traductions françaises

Fusionne les scores pour retourner les top-k résultats pertinents.
"""

import re
import unicodedata
from typing import List, Dict, Tuple
import numpy as np


# ── Détection de langue ───────────────────────────────────────────────────────

# Caractères typiques du mooré (diacritiques spécifiques)
MOORE_CHARS = set("ãõẽĩũỹāōēīūŋɩʋɛɔẽõãб")
MOORE_WORDS = {"ned", "yaa", "sẽ", "ka", "n", "tɩ", "a", "b", "roog", "naab",
               "yẽ", "wã", "pɛ", "sã", "kũ", "bõ", "zĩ", "tõ"}

def detect_language(text: str) -> str:
    """
    Détecte si le texte est en mooré ou en français.
    Retourne 'moore' ou 'french'.
    """
    text_lower = text.lower().strip()
    
    # Présence de caractères mooré spécifiques
    moore_char_count = sum(1 for c in text_lower if c in MOORE_CHARS)
    if moore_char_count >= 1:
        return "moore"
    
    # Mots mooré courants
    words = set(text_lower.split())
    if words & MOORE_WORDS:
        return "moore"
    
    # Mots français courants
    french_markers = {"le", "la", "les", "de", "du", "des", "un", "une",
                      "est", "sont", "qui", "que", "dans", "sur", "avec",
                      "pour", "pas", "plus", "très", "tout", "bien", "donc"}
    if words & french_markers:
        return "french"
    
    # Par défaut : français (meilleur avec LaBSE)
    return "french"


# ── Normalisation pour comparaison ───────────────────────────────────────────

def normalize_moore(text: str) -> str:
    """
    Normalise un texte mooré pour la comparaison :
    - Minuscules
    - Garde les diacritiques (ã, ẽ, etc. sont significatifs en mooré)
    - Supprime ponctuation
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    return text


def normalize_relaxed(text: str) -> str:
    """
    Normalisation relâchée : supprime aussi les diacritiques.
    Utile quand l'utilisateur tape sans les caractères spéciaux mooré.
    Ex: "ned same" → match "ned sãame"
    """
    text = text.lower().strip()
    # Supprimer les diacritiques (NFD puis garder ASCII)
    nfd = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


def token_overlap_score(query_tokens: List[str], text_tokens: List[str]) -> float:
    """Score basé sur le chevauchement de tokens (Jaccard modifié)."""
    if not query_tokens or not text_tokens:
        return 0.0
    q_set = set(query_tokens)
    t_set = set(text_tokens)
    intersection = q_set & t_set
    if not intersection:
        return 0.0
    # Pondérer par la longueur des tokens (tokens longs = plus significatifs)
    score = sum(len(tok) for tok in intersection) / sum(len(tok) for tok in q_set)
    return min(1.0, score)


def substring_score(query: str, text: str) -> float:
    """Score si la requête apparaît comme sous-chaîne."""
    if query in text:
        return 1.0
    # Vérifier chaque token de la requête
    tokens = query.split()
    found = sum(1 for t in tokens if t in text and len(t) > 2)
    return found / len(tokens) if tokens else 0.0


# ── Recherche mooré ───────────────────────────────────────────────────────────

def search_moore(query: str, segments: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Recherche en mooré : combine matching exact + matching relâché (sans diacritiques).
    
    Stratégie à 3 niveaux :
    1. Match exact avec diacritiques (score 1.0)
    2. Match tokens avec diacritiques (score 0.5-1.0)
    3. Match relâché sans diacritiques (score 0.2-0.6)
    """
    query_norm    = normalize_moore(query)
    query_relaxed = normalize_relaxed(query)
    query_tokens  = [t for t in query_norm.split() if len(t) > 1]
    query_tokens_relaxed = [t for t in query_relaxed.split() if len(t) > 1]
    
    results = []
    
    for seg in segments:
        moore_text = seg.get("transcription", "") or ""
        if not moore_text.strip():
            continue
        
        text_norm    = normalize_moore(moore_text)
        text_relaxed = normalize_relaxed(moore_text)
        text_tokens  = [t for t in text_norm.split() if len(t) > 1]
        text_tokens_relaxed = [t for t in text_relaxed.split() if len(t) > 1]
        
        score = 0.0
        
        # Niveau 1 : sous-chaîne exacte avec diacritiques
        s1 = substring_score(query_norm, text_norm)
        score = max(score, s1 * 1.0)
        
        # Niveau 2 : overlap de tokens avec diacritiques
        s2 = token_overlap_score(query_tokens, text_tokens)
        score = max(score, s2 * 0.9)
        
        # Niveau 3 : sous-chaîne relâchée (sans diacritiques)
        s3 = substring_score(query_relaxed, text_relaxed)
        score = max(score, s3 * 0.7)
        
        # Niveau 4 : overlap tokens relâché
        s4 = token_overlap_score(query_tokens_relaxed, text_tokens_relaxed)
        score = max(score, s4 * 0.6)
        
        if score > 0.05:
            seg_copy = dict(seg)
            seg_copy["score"] = round(score, 4)
            seg_copy["match_type"] = "moore_text"
            results.append(seg_copy)
    
    # Trier par score décroissant
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ── Recherche sémantique FR (LaBSE) ──────────────────────────────────────────

def search_french_semantic(query: str, segments: List[Dict],
                            corpus_embs: np.ndarray,
                            labse_model, top_k: int = 10) -> List[Dict]:
    """Recherche sémantique LaBSE sur les traductions françaises."""
    q_emb = labse_model.encode([query], normalize_embeddings=True)[0]
    sims  = corpus_embs @ q_emb
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if sims[idx] > 0.1:  # seuil minimal
            seg = dict(segments[idx])
            seg["score"] = round(float(sims[idx]), 4)
            seg["match_type"] = "semantic_fr"
            results.append(seg)
    return results


# ── Recherche hybride mooré : texte + sémantique FR ─────────────────────────

def search_moore_hybrid(query: str, segments: List[Dict],
                         corpus_embs: np.ndarray,
                         labse_model, top_k: int = 10) -> List[Dict]:
    """
    Pour une requête mooré :
    1. Recherche textuelle directe sur transcriptions mooré (poids 0.7)
    2. Recherche sémantique LaBSE sur traductions FR (poids 0.3)
       → LaBSE encode la requête mooré approximativement mais peut aider
    3. Fusion et déduplication par ID
    """
    # Recherche textuelle mooré
    text_results = search_moore(query, segments, top_k=top_k * 2)
    text_scores  = {r["id"]: r["score"] for r in text_results}
    
    # Recherche sémantique (LaBSE sur mooré → approximatif mais utile)
    sem_results = search_french_semantic(query, segments, corpus_embs,
                                          labse_model, top_k=top_k * 2)
    sem_scores  = {r["id"]: r["score"] for r in sem_results}
    
    # Fusion
    all_ids = set(text_scores.keys()) | set(sem_scores.keys())
    fused = []
    
    seg_by_id = {s["id"]: s for s in segments}
    
    for sid in all_ids:
        t_score = text_scores.get(sid, 0.0)
        s_score = sem_scores.get(sid, 0.0)
        
        # Score fusionné : textuel prioritaire
        if t_score > 0:
            final_score = t_score * 0.75 + s_score * 0.25
            match_type  = "hybrid_moore"
        else:
            final_score = s_score * 0.5  # pénalité si pas de match textuel
            match_type  = "semantic_only"
        
        if final_score > 0.05 and sid in seg_by_id:
            seg = dict(seg_by_id[sid])
            seg["score"]      = round(final_score, 4)
            seg["match_type"] = match_type
            seg["score_text"] = round(t_score, 4)
            seg["score_sem"]  = round(s_score, 4)
            fused.append(seg)
    
    fused.sort(key=lambda x: -x["score"])
    return fused[:top_k]


# ── Point d'entrée principal ──────────────────────────────────────────────────

def smart_search(query: str, segments: List[Dict],
                  corpus_embs: np.ndarray,
                  labse_model, top_k: int = 5) -> Tuple[List[Dict], str]:
    """
    Recherche intelligente : détecte la langue et applique la bonne stratégie.
    
    Retourne (résultats, langue_détectée).
    """
    lang = detect_language(query)
    
    if lang == "moore":
        results = search_moore_hybrid(query, segments, corpus_embs,
                                       labse_model, top_k)
    else:
        results = search_french_semantic(query, segments, corpus_embs,
                                          labse_model, top_k)
    
    return results, lang
