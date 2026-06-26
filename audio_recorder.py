"""
COMPOSANT ENREGISTREMENT AUDIO — CITADEL 2026 (v3)
===================================================
Utilise librosa.load directement (même pipeline que module2)
pour garantir que les embeddings Whisper sont identiques.
"""

import io
import numpy as np
from typing import Optional, Tuple


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Charge bytes audio → float32 mono à target_sr Hz.
    Utilise librosa.load exactement comme module2_embeddings.py
    pour garantir des embeddings identiques à ceux du corpus indexé.
    """
    import librosa

    buf = io.BytesIO(audio_bytes)
    # Même appel exact que dans module1_ingestion.py load_audio()
    audio, sr = librosa.load(buf, sr=target_sr, mono=True)

    # Même normalisation que module1
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), target_sr


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
        # Tentative 1 : audio-recorder-streamlit
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
                    "Installe le composant micro :\n"
                    "```\npip install audio-recorder-streamlit\n```"
                )

    return audio_bytes
