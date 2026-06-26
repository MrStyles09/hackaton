"""
MODULE 5 — Scoring de Criticité Multidimensionnel
===================================================
Attribue à chaque segment audio un score sur 5 dimensions :
  1. urgence_sanitaire  — signalement maladie, décès, rupture médicaments
  2. tension_sociale    — mobilisation, accusations, rumeurs
  3. alerte_agricole    — invasion acridienne, sécheresse, intrants
  4. desinformation     — contradiction faits vérifiables, fake news
  5. detresse_individu  — demande d'aide, danger immédiat

Deux modes complémentaires :
  A. Zero-shot via embeddings LaBSE (aucun exemple annoté requis)
  B. SVM supervisé (entraîné sur ~50-100 exemples annotés au hackathon)

Usage :
    python module5_criticite.py --json corpus/segments.json --mode zeroshot
    python module5_criticite.py --json corpus/segments.json --mode svm --labels corpus/labels.csv
"""

import argparse
import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Prototypes sémantiques zero-shot ──────────────────────────────────────────
# Ces phrases définissent chaque catégorie dans l'espace LaBSE.
# Elles sont en FRANÇAIS (+ translittérations mooré optionnelles) pour
# maximiser la couverture multilingue de LaBSE.

CRITICITE_PROTOTYPES = {
    "urgence_sanitaire": [
        # Français
        "des gens sont malades dans notre village",
        "il y a une épidémie de paludisme",
        "les médicaments sont en rupture de stock au centre de santé",
        "un enfant est mort de la rougeole",
        "nous avons besoin d'aide médicale urgente",
        "cas de choléra signalé dans le quartier",
        "la femme a accouché sans sage-femme il y a un problème",
        # Moore translitéré
        "ned sãame ne zĩnga pãnga",         # quelqu'un est très malade
        "rogom sẽed n tɩ pɛɛg ka be",        # les médicaments ne sont plus là
    ],
    "tension_sociale": [
        "il y a des tensions entre les communautés",
        "des gens appellent à manifester contre les autorités",
        "une rumeur circule sur cet homme qui aurait fait quelque chose",
        "des jeunes ont bloqué la route avec des barrières",
        "conflit entre agriculteurs et éleveurs dans la zone",
        "on accuse le chef du village d'avoir volé l'argent",
        "des tirs ont été entendus cette nuit au village",
        "il y a de la violence dans ce secteur",
    ],
    "alerte_agricole": [
        "les criquets ont envahi les champs de mil",
        "la sécheresse menace la récolte cette année",
        "les semences d'engrais ne sont pas disponibles",
        "attaque acridienne dans la région",
        "les pluies tardent et les champs sont secs",
        "récolte catastrophique à cause du manque de pluie",
        "les animaux meurent de soif et de faim",
        "invasion de chenilles légionnaires dans les cultures",
    ],
    "desinformation": [
        "c'est un mensonge ce qu'on dit sur le vaccin",
        "cette information est fausse elle a été inventée",
        "on raconte des choses fausses pour provoquer la peur",
        "cette rumeur n'est pas vraie j'ai vérifié",
        "les gens diffusent de fausses nouvelles sur la situation sécuritaire",
        "quelqu'un a inventé cette histoire pour nuire",
    ],
    "detresse_individu": [
        "je suis seul et j'ai besoin d'aide urgent",
        "ma famille est en danger nous n'avons plus rien à manger",
        "nous sommes bloqués et nous n'arrivons pas à partir",
        "s'il vous plaît aidez-nous nous avons besoin de secours",
        "des hommes armés sont venus et ont tout pris",
        "nous avons fui et nous n'avons nulle part où aller",
    ],
}

# Seuils de criticité (score cosinus minimum pour déclencher une alerte)
SEUILS = {
    "urgence_sanitaire" : 0.45,
    "tension_sociale"   : 0.42,
    "alerte_agricole"   : 0.40,
    "desinformation"    : 0.38,
    "detresse_individu" : 0.44,
}

# Couleurs pour l'affichage
COULEURS = {
    "urgence_sanitaire" : "#E53E3E",   # rouge
    "tension_sociale"   : "#ED8936",   # orange
    "alerte_agricole"   : "#38A169",   # vert
    "desinformation"    : "#805AD5",   # violet
    "detresse_individu" : "#3182CE",   # bleu
}


# ─── Scorer zero-shot ──────────────────────────────────────────────────────────

class ZeroShotScorer:
    """
    Calcule la similarité cosinus entre l'embedding du segment et
    les prototypes de chaque catégorie.
    Aucun entraînement requis, immédiatement opérationnel.
    """

    def __init__(self):
        from module2_embeddings import LaBSEEmbedder
        self.embedder = LaBSEEmbedder()
        print("[M5] Pré-calcul des embeddings prototypes...")
        self._proto_embeddings: Dict[str, np.ndarray] = {}
        for cat, phrases in CRITICITE_PROTOTYPES.items():
            embs = self.embedder.embed_batch(phrases)
            self._proto_embeddings[cat] = embs.mean(axis=0)
            # Normaliser
            n = np.linalg.norm(self._proto_embeddings[cat])
            if n > 0:
                self._proto_embeddings[cat] /= n
        print("[M5] Prototypes prêts.")

    def score(self, text_or_embedding) -> Dict[str, float]:
        """
        Calcule les scores pour un segment.
        Accepte soit un texte (transcription) soit un embedding LaBSE.
        """
        if isinstance(text_or_embedding, str):
            emb = self.embedder.embed(text_or_embedding)
        else:
            emb = text_or_embedding.astype(np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb /= n

        scores = {}
        for cat, proto in self._proto_embeddings.items():
            score = float(np.dot(emb, proto))
            scores[cat] = round(max(0.0, score), 4)

        return scores

    def criticite_globale(self, scores: Dict[str, float]) -> Dict:
        """
        Calcule un score global et un niveau d'alerte (VERT / ORANGE / ROUGE).
        """
        alertes = {
            cat: s for cat, s in scores.items()
            if s >= SEUILS.get(cat, 0.4)
        }

        score_max = max(scores.values()) if scores else 0.0

        if score_max >= 0.65 or len(alertes) >= 2:
            niveau = "ROUGE"
        elif score_max >= 0.45 or len(alertes) == 1:
            niveau = "ORANGE"
        else:
            niveau = "VERT"

        return {
            "scores"   : scores,
            "alertes"  : list(alertes.keys()),
            "niveau"   : niveau,
            "score_max": round(score_max, 4),
        }


# ─── Scorer SVM supervisé ─────────────────────────────────────────────────────

class SVMScorer:
    """
    Classificateur SVM multiclasse entraîné sur des exemples annotés manuellement.
    Opérationnel avec ~50 exemples par catégorie.
    Compatible avec les labels collectés pendant le hackathon.
    """

    def __init__(self):
        self.classifiers: Dict = {}
        self.trained = False

    def train(self, texts: List[str], labels: Dict[str, List[int]]):
        """
        Entraîne un SVM binaire par catégorie.
        labels : dict {categorie: [0/1, 0/1, ...]} de même longueur que texts
        """
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from module2_embeddings import LaBSEEmbedder

        embedder = LaBSEEmbedder()
        X = embedder.embed_batch(texts)

        for cat, y in labels.items():
            if len(set(y)) < 2:
                print(f"[M5] ⚠ Catégorie '{cat}' : une seule classe, skip")
                continue
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm",    LinearSVC(C=1.0, max_iter=1000, class_weight="balanced"))
            ])
            clf.fit(X, y)
            self.classifiers[cat] = clf
            print(f"[M5] SVM '{cat}' entraîné")

        self.trained = True

    def score(self, text_or_embedding) -> Dict[str, float]:
        """Retourne les probabilités calibrées par catégorie."""
        if not self.trained:
            raise RuntimeError("SVMScorer non entraîné. Appeler train() d'abord.")

        from module2_embeddings import LaBSEEmbedder
        if isinstance(text_or_embedding, str):
            emb = LaBSEEmbedder().embed(text_or_embedding).reshape(1, -1)
        else:
            emb = text_or_embedding.reshape(1, -1)

        scores = {}
        for cat, clf in self.classifiers.items():
            decision = clf.decision_function(emb)[0]
            # Calibration sigmoid simple
            prob = 1 / (1 + np.exp(-decision))
            scores[cat] = round(float(prob), 4)

        return scores

    def save(self, path: str = "models/svm_scorer.pkl"):
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.classifiers, f)
        print(f"[M5] SVM sauvegardé : {path}")

    def load(self, path: str = "models/svm_scorer.pkl"):
        import pickle
        with open(path, "rb") as f:
            self.classifiers = pickle.load(f)
        self.trained = True
        print(f"[M5] SVM chargé : {path}")


# ─── Application au corpus ────────────────────────────────────────────────────

def score_corpus(segments_json: str,
                 mode: str = "zeroshot",
                 db_path: str = "corpus/metadata.db",
                 labels_csv: Optional[str] = None) -> List[Dict]:
    """
    Applique le scoring de criticité à tous les segments du corpus.
    Mode 'zeroshot' : immédiat, sans données annotées.
    Mode 'svm'      : nécessite labels_csv.
    """
    with open(segments_json, encoding="utf-8") as f:
        segments = json.load(f)

    conn = sqlite3.connect(db_path)

    if mode == "zeroshot":
        scorer = ZeroShotScorer()
    elif mode == "svm":
        scorer = SVMScorer()
        if labels_csv and os.path.exists(labels_csv):
            import pandas as pd
            df = pd.read_csv(labels_csv)
            texts  = df["transcription"].tolist()
            labels = {
                cat: df[cat].tolist()
                for cat in CRITICITE_PROTOTYPES.keys()
                if cat in df.columns
            }
            scorer.train(texts, labels)
        else:
            print("[M5] ⚠ Pas de labels CSV → fallback zero-shot")
            scorer = ZeroShotScorer()
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    results = []
    for seg in segments:
        # Utiliser la transcription si disponible, sinon passer l'embedding
        row = conn.execute(
            "SELECT transcription FROM segments WHERE id=?", (seg["id"],)
        ).fetchone()

        text = (row[0] if row and row[0] else
                f"[audio segment {seg['start_sec']}s-{seg['end_sec']}s]")

        scores = scorer.score(text)
        crit   = scorer.criticite_globale(scores) if hasattr(scorer, "criticite_globale") else {
            "scores": scores, "niveau": "VERT", "alertes": [], "score_max": max(scores.values())
        }

        # Mise à jour SQLite
        conn.execute(
            "UPDATE segments SET criticite_json=? WHERE id=?",
            (json.dumps(crit), seg["id"])
        )

        seg["criticite"] = crit
        results.append(seg)

    conn.commit()
    conn.close()

    alertes_rouge  = sum(1 for r in results if r["criticite"]["niveau"] == "ROUGE")
    alertes_orange = sum(1 for r in results if r["criticite"]["niveau"] == "ORANGE")
    print(f"\n[M5] ✓ {len(results)} segments scorés")
    print(f"     🔴 ROUGE  : {alertes_rouge}")
    print(f"     🟠 ORANGE : {alertes_orange}")
    print(f"     🟢 VERT   : {len(results) - alertes_rouge - alertes_orange}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring de Criticité")
    parser.add_argument("--json",   default="corpus/segments.json")
    parser.add_argument("--mode",   default="zeroshot", choices=["zeroshot", "svm"])
    parser.add_argument("--labels", default=None, help="CSV avec colonnes transcription + catégories")
    parser.add_argument("--db",     default="corpus/metadata.db")
    args = parser.parse_args()

    results = score_corpus(args.json, args.mode, args.db, args.labels)

    # Afficher les alertes ROUGE
    print("\n─── ALERTES ROUGE ─────────────────────────────────────────────")
    for r in results:
        if r["criticite"]["niveau"] == "ROUGE":
            print(f"  {r['source']}  [{r['start_sec']}s–{r['end_sec']}s]")
            print(f"  Catégories : {r['criticite']['alertes']}")
            print(f"  Score max  : {r['criticite']['score_max']}")
            print()
