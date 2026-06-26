"""
COMPOSANT ENREGISTREMENT AUDIO — CITADEL 2026
=============================================
Fournit deux modes d'entrée audio pour la recherche vocale :
  1. Upload d'un fichier WAV/MP3/OGG
  2. Enregistrement micro via streamlit-audio-recorder

Usage dans Streamlit :
    from audio_recorder import get_query_audio
    audio_bytes, sr = get_query_audio()
    if audio_bytes:
        # encoder et chercher...
"""

import io
import numpy as np
from typing import Optional, Tuple


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Charge des bytes audio (WAV/MP3/OGG/FLAC) en array numpy float32 mono.
    Rééchantillonne à target_sr si nécessaire.
    """
    import soundfile as sf
    import librosa

    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf, dtype='float32', always_2d=False)
    except Exception:
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=None, mono=True)

    # Convertir en mono si stéréo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Rééchantillonner si nécessaire
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normaliser
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), sr


def render_audio_input(key_prefix: str = "rec") -> Optional[bytes]:
    """
    Affiche l'interface d'entrée audio (upload + enregistrement micro).
    Retourne les bytes audio ou None si rien n'est fourni.
    """
    import streamlit as st

    tab_upload, tab_micro = st.tabs(["📁 Uploader un fichier", "🎤 Enregistrer"])

    audio_bytes = None

    with tab_upload:
        st.caption("Formats acceptés : WAV, MP3, OGG, FLAC, M4A")
        uploaded = st.file_uploader(
            "Fichier audio",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key=f"{key_prefix}_upload",
            label_visibility="collapsed"
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes)
            st.success(f"Fichier chargé : {uploaded.name} ({len(audio_bytes)//1024} Ko)")

    with tab_micro:
        # Tentative avec streamlit-audio-recorder
        recorder_available = False
        try:
            from audiorecorder import audiorecorder
            recorder_available = True
        except ImportError:
            pass

        if recorder_available:
            st.caption("Parle en mooré puis clique Stop.")
            audio_seg = audiorecorder(
                start_prompt="🎤 Enregistrer",
                stop_prompt="⏹ Stop",
                key=f"{key_prefix}_recorder"
            )
            if len(audio_seg) > 0:
                buf = io.BytesIO()
                audio_seg.export(buf, format="wav")
                audio_bytes = buf.getvalue()
                st.audio(audio_bytes, format="audio/wav")
                st.success(f"Enregistrement : {len(audio_seg)/1000:.1f}s")
        else:
            st.warning("Module `audiorecorder` non installé.")
            st.code("pip install streamlit-audio-recorder")
            st.caption("En attendant, utilise l'onglet 'Uploader un fichier'.")

            # Fallback : st.audio_input (Streamlit ≥ 1.32)
            try:
                recorded = st.audio_input(
                    "Enregistre ta requête vocale",
                    key=f"{key_prefix}_native"
                )
                if recorded:
                    audio_bytes = recorded.read()
                    st.audio(audio_bytes, format="audio/wav")
                    st.success("Enregistrement prêt.")
            except AttributeError:
                st.info("Mise à jour Streamlit recommandée : `pip install --upgrade streamlit`")

    return audio_bytes
