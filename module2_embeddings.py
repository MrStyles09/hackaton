"""
MODULE 2 — Extraction d'Embeddings Audio
==========================================
Stratégie hybride souveraine :
  1. Whisper encoder  → embedding acoustique 512-dim (moyen de toutes les frames)
  2. wav2vec2-XLSR-53 → embedding acoustique 1024-dim (alternative/complémentaire)
  3. LaBSE (text)     → embedding sémantique 768-dim pour les requêtes textuelles

Toujours local, zéro API externe.

Usage :
    python module2_embeddings.py --json corpus/segments.json --model whisper
    python module2_embeddings.py --json corpus/segments.json --model wav2vec2
"""

import argparse
import json
import os
import sqlite3
from typing import List, Dict, Tuple

import numpy as np
import torch

# ─── Configuration ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[M2] Device : {DEVICE}")

WHISPER_MODEL   = "openai/whisper-small"   # ~240 MB — frugal CPU
WAV2VEC2_MODEL  = "facebook/wav2vec2-large-xlsr-53"  # ~1.2 GB
LABSE_MODEL     = "sentence-transformers/LaBSE"       # ~1.8 GB


# ─── Chargeur Whisper (embeddings via encoder) ─────────────────────────────────

class WhisperEmbedder:
    """
    Utilise l'encodeur Whisper pour extraire un embedding acoustique par segment.
    On moyenne les hidden states de la dernière couche de l'encodeur.
    Avantage : multilingue, robuste au bruit, fonctionne sans transcription.
    """

    def __init__(self, model_name: str = WHISPER_MODEL):
        from transformers import WhisperProcessor, WhisperModel
        print(f"[M2] Chargement Whisper : {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model     = WhisperModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.dim = 512  # whisper-small

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        """Retourne un vecteur numpy de dimension 512."""
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)

        encoder_output = self.model.encoder(input_features)
        # Moyenne sur la dimension temporelle → vecteur fixe
        embedding = encoder_output.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.cpu().numpy().astype(np.float32)


# ─── Chargeur wav2vec2-XLSR ────────────────────────────────────────────────────

class Wav2Vec2Embedder:
    """
    Utilise wav2vec2-large-XLSR-53 (53 langues dont plusieurs africaines).
    Extrait les features de la dernière couche cachée → moyenne temporelle.
    Plus lourd que Whisper-small mais potentiellement plus robuste sur mooré/dioula.
    """

    def __init__(self, model_name: str = WAV2VEC2_MODEL):
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        print(f"[M2] Chargement wav2vec2 : {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model     = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.dim = 1024

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        """Retourne un vecteur numpy de dimension 1024."""
        inputs = self.processor(
            audio, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(DEVICE)

        outputs = self.model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.cpu().numpy().astype(np.float32)


# ─── Chargeur LaBSE (embeddings texte multilingues) ──────────────────────────

class LaBSEEmbedder:
    """
    LaBSE (Language-agnostic BERT Sentence Embedding) pour encoder les requêtes
    textuelles en français ou en mooré translittéré.
    Couvre 109 langues, léger et CPU-friendly.
    """

    def __init__(self, model_name: str = LABSE_MODEL):
        from sentence_transformers import SentenceTransformer
        print(f"[M2] Chargement LaBSE : {model_name}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.dim = 768

    def embed(self, text: str) -> np.ndarray:
        """Retourne un vecteur numpy de dimension 768."""
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes → matrice (N, 768)."""
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)


# ─── Pont audio→texte pour l'alignement cross-modal ──────────────────────────

class CrossModalBridge:
    """
    Stratégie d'alignement audio ↔ texte :

    Option A (recommandée pour le hackathon) :
      Whisper génère une transcription approximative (même imparfaite),
      LaBSE encode cette transcription → embedding dans l'espace texte.
      → Permet une recherche textuelle directe sur le corpus audio.

    Option B (plus robuste, moins dépendant de la transcription) :
      On entraîne un projecteur linéaire W : R^512 → R^768 qui mappe
      l'espace Whisper vers l'espace LaBSE à partir de quelques paires
      audio/texte annotées manuellement pendant le hackathon.
    """

    def __init__(self, whisper_embedder: WhisperEmbedder,
                 text_embedder: LaBSEEmbedder):
        self.audio_emb = whisper_embedder
        self.text_emb  = text_embedder
        # Projecteur W (initialisé à l'identité projetée, affiné si des paires existent)
        self.W = None

    def fit_projector(self, audio_list: List[np.ndarray],
                      text_list: List[str], epochs: int = 50):
        """
        Entraîne W en minimisant ||W @ a - t||² sur des paires (audio, texte).
        Même 20-30 paires annotées au hackathon suffisent pour amorcer.
        """
        from sklearn.linear_model import Ridge

        A = np.stack([self.audio_emb.embed(a) for a in audio_list])   # (N, 512)
        T = self.text_emb.embed_batch(text_list)                        # (N, 768)

        # Régression Ridge : T ≈ A @ W^T
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(A, T)
        self.W = reg.coef_.T.astype(np.float32)   # (512, 768)
        print(f"[M2] Projecteur entraîné sur {len(audio_list)} paires")

    def project_audio(self, audio_emb: np.ndarray) -> np.ndarray:
        """Projette un embedding audio dans l'espace LaBSE."""
        if self.W is not None:
            projected = audio_emb @ self.W
        else:
            # Sans projecteur : padding/troncature naïf
            if len(audio_emb) < 768:
                projected = np.pad(audio_emb, (0, 768 - len(audio_emb)))
            else:
                projected = audio_emb[:768]
        # Normalisation L2
        norm = np.linalg.norm(projected)
        return (projected / norm).astype(np.float32) if norm > 0 else projected


# ─── Pipeline principal d'embedding ───────────────────────────────────────────

def compute_embeddings(segments_json: str, model_type: str = "whisper",
                        db_path: str = "corpus/metadata.db",
                        out_embeddings: str = "index/embeddings.npy") -> Dict:
    """
    Charge la liste des segments, calcule leurs embeddings et les sauvegarde.

    Retourne un dict :
      {
        "ids"        : List[str],          # identifiants des segments
        "embeddings" : np.ndarray (N, D),  # matrice d'embeddings
        "dim"        : int
      }
    """
    import soundfile as sf
    from tqdm import tqdm

    with open(segments_json, encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[M2] {len(segments)} segments à encoder (modèle : {model_type})")

    # Initialiser l'embedder
    if model_type == "whisper":
        embedder = WhisperEmbedder()
    elif model_type == "wav2vec2":
        embedder = Wav2Vec2Embedder()
    else:
        raise ValueError(f"Modèle inconnu : {model_type}")

    ids        = []
    embeddings = []
    errors     = []

    for seg in tqdm(segments, desc="Embedding"):
        path = seg["path"]
        if not os.path.exists(path):
            errors.append(seg["id"])
            continue
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio, sr)
            ids.append(seg["id"])
            embeddings.append(emb)
        except Exception as e:
            print(f"[M2] ⚠ Erreur sur {path} : {e}")
            errors.append(seg["id"])

    if not embeddings:
        print("[M2] ✗ Aucun embedding produit !")
        return {}

    emb_matrix = np.stack(embeddings).astype(np.float32)

    # Normalisation L2 (cosine similarity via produit scalaire)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix /= norms

    os.makedirs(os.path.dirname(out_embeddings) or ".", exist_ok=True)
    np.save(out_embeddings, emb_matrix)

    # Sauvegarder les IDs correspondants
    ids_file = out_embeddings.replace(".npy", "_ids.json")
    with open(ids_file, "w") as f:
        json.dump(ids, f)

    # Marquer comme indexés dans SQLite
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "UPDATE segments SET indexed=1 WHERE id=?", [(i,) for i in ids]
    )
    conn.commit()
    conn.close()

    print(f"\n[M2] ✓ {len(ids)} embeddings ({embedder.dim}-dim) → {out_embeddings}")
    if errors:
        print(f"     ⚠ {len(errors)} erreurs ignorées")

    return {"ids": ids, "embeddings": emb_matrix, "dim": embedder.dim}


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction d'Embeddings Audio")
    parser.add_argument("--json",   default="corpus/segments.json")
    parser.add_argument("--model",  default="whisper", choices=["whisper", "wav2vec2"])
    parser.add_argument("--db",     default="corpus/metadata.db")
    parser.add_argument("--out",    default="index/embeddings.npy")
    args = parser.parse_args()

    compute_embeddings(args.json, args.model, args.db, args.out)
