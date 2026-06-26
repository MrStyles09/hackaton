"""
COMPOSANT ENREGISTREMENT AUDIO — CITADEL 2026 (v2)
===================================================
Gère l'entrée audio via :
  1. Upload fichier (WAV/MP3/OGG/FLAC)
  2. Enregistrement micro via audio-recorder-streamlit

Usage :
    from audio_recorder import render_audio_input, load_audio_bytes
    audio_bytes = render_audio_input()
    if audio_bytes:
        audio, sr = load_audio_bytes(audio_bytes)
"""

import io
import numpy as np
from typing import Optional, Tuple


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Charge bytes audio → array numpy float32 mono à target_sr Hz."""
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf, dtype='float32', always_2d=False)
    except Exception:
        import librosa
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=None, mono=True)

    # Mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Rééchantillonnage
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalisation
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), sr


def render_audio_input(key_prefix: str = "rec") -> Optional[bytes]:
    """
    Affiche les deux modes d'entrée audio.
    Retourne les bytes audio bruts ou None.
    """
    import streamlit as st

    audio_bytes = None

    tab_upload, tab_micro = st.tabs(["📁 Uploader un fichier", "🎤 Enregistrer au micro"])

    with tab_upload:
        st.caption("Formats : WAV · MP3 · OGG · FLAC · M4A")
        uploaded = st.file_uploader(
            "Fichier audio",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key=f"{key_prefix}_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes)
            st.success(f"✅ {uploaded.name} · {len(audio_bytes)//1024} Ko")

    with tab_micro:
        # Tentative 1 : audio-recorder-streamlit (pip install audio-recorder-streamlit)
        try:
            from audio_recorder_streamlit import audio_recorder

            st.caption("Clique sur le micro 🎤, parle en mooré, reclique pour stopper.")
            recorded = audio_recorder(
                text="",
                recording_color="#E53E3E",
                neutral_color="#4A5568",
                icon_size="2x",
                key=f"{key_prefix}_arec",
            )
            if recorded:
                audio_bytes = recorded
                st.audio(audio_bytes, format="audio/wav")
                st.success(f"✅ Enregistrement capturé · {len(audio_bytes)//1024} Ko")

        except ImportError:
            # Tentative 2 : st.audio_input natif (Streamlit ≥ 1.32)
            try:
                st.caption("Clique sur le micro pour enregistrer.")
                recorded = st.audio_input(
                    "Enregistre ta requête vocale",
                    key=f"{key_prefix}_native",
                )
                if recorded:
                    audio_bytes = recorded.read()
                    st.success(f"✅ Enregistrement capturé · {len(audio_bytes)//1024} Ko")
            except AttributeError:
                st.warning(
                    "`audio-recorder-streamlit` non trouvé et Streamlit < 1.32 détecté.\n\n"
                    "Installe le composant micro :\n"
                    "```\npip install audio-recorder-streamlit\n```\n"
                    "Ou utilise l'onglet **📁 Uploader un fichier** à la place."
                )

    return audio_bytes
