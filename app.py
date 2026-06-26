"""
MODULE 6 — Interface Streamlit
================================
Interface web locale (http://localhost:8501) pour interagir avec le système.
100% souverain : aucune donnée audio ne quitte la machine.

Lancement : streamlit run app.py
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import streamlit as st

# ─── Configuration ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Audio Sémantique Souverain",
    page_icon="🎙️",
    layout="wide",
)

DB_PATH    = "corpus/metadata.db"
INDEX_PATH = "index/faiss.index"
IDS_PATH   = "index/segment_ids.json"
SEG_JSON   = "corpus/segments.json"

# ─── CSS léger ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.alerte-rouge  { background:#FED7D7; border-left:4px solid #E53E3E; padding:8px 12px; border-radius:4px; }
.alerte-orange { background:#FEEBC8; border-left:4px solid #DD6B20; padding:8px 12px; border-radius:4px; }
.alerte-vert   { background:#C6F6D5; border-left:4px solid #38A169; padding:8px 12px; border-radius:4px; }
.score-bar { display:inline-block; height:8px; border-radius:4px; }
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─── Fonctions utilitaires ─────────────────────────────────────────────────────

@st.cache_resource
def load_faiss_index():
    """Charge l'index FAISS (mis en cache pour éviter les rechargements)."""
    try:
        import faiss
        from module3_index import load_index
        index, ids = load_index(INDEX_PATH, IDS_PATH)
        return index, ids
    except Exception as e:
        return None, []


@st.cache_resource
def load_text_embedder():
    """Charge LaBSE (mis en cache)."""
    try:
        from module2_embeddings import LaBSEEmbedder
        return LaBSEEmbedder()
    except Exception as e:
        st.error(f"Impossible de charger LaBSE : {e}")
        return None


def get_db():
    """Retourne une connexion SQLite (lecture seule pour l'interface)."""
    if os.path.exists(DB_PATH):
        return sqlite3.connect(DB_PATH, check_same_thread=False)
    return None


def get_all_segments():
    """Récupère tous les segments depuis SQLite."""
    conn = get_db()
    if not conn:
        return []
    rows = conn.execute("""
        SELECT id, file, path, source, start_sec, end_sec, duration_sec,
               transcription, criticite_json, created_at
        FROM segments ORDER BY source, start_sec
    """).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r[0], "file": r[1], "path": r[2], "source": r[3],
            "start_sec": r[4], "end_sec": r[5], "duration_sec": r[6],
            "transcription": r[7],
            "criticite": json.loads(r[8]) if r[8] else None,
            "created_at": r[9],
        })
    conn.close()
    return results


def search_segments(query: str, top_k: int = 5) -> List[Dict]:
    """Recherche sémantique dans le corpus audio."""
    index, ids = load_faiss_index()
    if index is None:
        return []

    embedder = load_text_embedder()
    if embedder is None:
        return []

    from module3_index import search
    conn = get_db()
    if not conn:
        return []

    query_vec = embedder.embed(query)
    results   = search(query_vec, index, ids, conn, top_k)
    conn.close()
    return results


def niveau_badge(niveau: str) -> str:
    colors = {"ROUGE": "#E53E3E", "ORANGE": "#DD6B20", "VERT": "#38A169"}
    icons  = {"ROUGE": "🔴", "ORANGE": "🟠", "VERT": "🟢"}
    c = colors.get(niveau, "#718096")
    i = icons.get(niveau, "⚪")
    return f'<span class="badge" style="background:{c}20;color:{c}">{i} {niveau}</span>'


def score_bars(scores: Dict[str, float]) -> str:
    labels = {
        "urgence_sanitaire" : ("🏥 Sanitaire",  "#E53E3E"),
        "tension_sociale"   : ("⚡ Sociale",    "#DD6B20"),
        "alerte_agricole"   : ("🌾 Agricole",   "#38A169"),
        "desinformation"    : ("🔍 Désinf.",    "#805AD5"),
        "detresse_individu" : ("🆘 Détresse",   "#3182CE"),
    }
    html = ""
    for cat, val in sorted(scores.items(), key=lambda x: -x[1]):
        label, color = labels.get(cat, (cat, "#718096"))
        pct = int(val * 100)
        html += f"""
        <div style="margin:4px 0;display:flex;align-items:center;gap:8px;font-size:12px">
          <span style="width:90px;color:#4A5568">{label}</span>
          <div style="flex:1;background:#EDF2F7;border-radius:4px;height:8px">
            <div class="score-bar" style="width:{pct}%;background:{color};"></div>
          </div>
          <span style="width:35px;text-align:right;color:{color};font-weight:600">{pct}%</span>
        </div>"""
    return html


def render_segment_card(seg: Dict, rank: Optional[int] = None):
    """Affiche une carte de résultat pour un segment."""
    crit   = seg.get("criticite")
    niveau = crit["niveau"] if crit else "VERT"
    score_sim = seg.get("score", None)

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            header = f"**{seg['source']}**  ·  {seg['start_sec']}s → {seg['end_sec']}s"
            if rank:
                header = f"#{rank} · " + header
            if score_sim is not None:
                header += f"  ·  similarité : **{score_sim:.2%}**"
            st.markdown(header)

            if seg.get("transcription"):
                st.caption(f"📝 {seg['transcription']}")
            else:
                st.caption("_(transcription non disponible)_")

            if crit:
                st.markdown(
                    score_bars(crit["scores"]),
                    unsafe_allow_html=True
                )

        with col2:
            if crit:
                st.markdown(niveau_badge(niveau), unsafe_allow_html=True)

            # Lecteur audio
            if seg.get("path") and os.path.exists(seg["path"]):
                with open(seg["path"], "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            else:
                st.caption("_fichier audio introuvable_")

        st.divider()


# ─── Pipeline rapide (pour le bouton "Indexer maintenant") ─────────────────────

def run_pipeline(audio_file_path: str, progress_bar):
    """Lance le pipeline complet sur un fichier audio uploadé."""
    import subprocess, sys

    steps = [
        ("Segmentation...",    f"python module1_ingestion.py --input {audio_file_path} --out corpus/segments"),
        ("Embeddings...",      f"python module2_embeddings.py --json corpus/segments.json --model whisper"),
        ("Indexation FAISS...", f"python module3_index.py --embeddings index/embeddings.npy --ids index/embeddings_ids.json"),
        ("Scoring criticité...", f"python module5_criticite.py --json corpus/segments.json --mode zeroshot"),
    ]

    for i, (label, cmd) in enumerate(steps):
        progress_bar.progress((i + 1) / len(steps), text=label)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Erreur à l'étape {label}:\n{result.stderr}")
            return False

    progress_bar.progress(1.0, text="✓ Pipeline terminé")
    return True


# ─── Interface principale ──────────────────────────────────────────────────────

def main():
    # ── En-tête ──
    st.title("🎙️ Audio Sémantique Souverain")
    st.caption(
        "Recherche sémantique et veille critique dans les messages vocaux "
        "en mooré, dioula et fulfuldé · 100% local · Hackathon CITADEL 2026"
    )

    # ── Tabs ──
    tab_search, tab_corpus, tab_upload, tab_about = st.tabs([
        "🔍 Recherche", "📂 Corpus", "⬆️ Ajouter des audios", "ℹ️ À propos"
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 : Recherche sémantique
    # ──────────────────────────────────────────────────────────────────────────
    with tab_search:
        st.subheader("Recherche sémantique dans le corpus audio")

        col_q, col_k = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "Requête",
                placeholder="Ex : signalement de maladie dans le village · yãmb yaa fo sẽed zĩnga",
                label_visibility="collapsed"
            )
        with col_k:
            top_k = st.selectbox("Résultats", [3, 5, 10], index=1, label_visibility="collapsed")

        # Filtres criticité
        with st.expander("🎛 Filtres"):
            filtre_niveau = st.multiselect(
                "Niveau d'alerte",
                ["ROUGE", "ORANGE", "VERT"],
                default=["ROUGE", "ORANGE", "VERT"]
            )
            filtre_dim = st.multiselect(
                "Dimension de criticité",
                ["urgence_sanitaire", "tension_sociale", "alerte_agricole",
                 "desinformation", "detresse_individu"],
                default=[]
            )

        if st.button("🔍 Rechercher", type="primary", disabled=not query):
            index, ids = load_faiss_index()
            if index is None:
                st.warning("⚠️ Index FAISS non trouvé. Ajoutez des audios et lancez le pipeline d'abord.")
            else:
                with st.spinner("Recherche en cours..."):
                    results = search_segments(query, top_k)

                # Appliquer les filtres
                if filtre_niveau:
                    results = [r for r in results
                               if (r.get("criticite") or {}).get("niveau", "VERT") in filtre_niveau]
                if filtre_dim:
                    results = [r for r in results
                               if any(d in (r.get("criticite") or {}).get("alertes", [])
                                      for d in filtre_dim)]

                if not results:
                    st.info("Aucun résultat trouvé.")
                else:
                    st.success(f"{len(results)} segment(s) trouvé(s)")
                    for r in results:
                        render_segment_card(r, rank=r.get("rank"))

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 : Vue corpus
    # ──────────────────────────────────────────────────────────────────────────
    with tab_corpus:
        st.subheader("Corpus indexé")
        segments = get_all_segments()

        if not segments:
            st.info("Corpus vide. Ajoutez des fichiers audio dans l'onglet '⬆️ Ajouter des audios'.")
        else:
            # Statistiques
            n_rouge  = sum(1 for s in segments if (s["criticite"] or {}).get("niveau") == "ROUGE")
            n_orange = sum(1 for s in segments if (s["criticite"] or {}).get("niveau") == "ORANGE")
            n_vert   = len(segments) - n_rouge - n_orange
            total_min = sum(s["duration_sec"] for s in segments) / 60

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Segments", len(segments))
            c2.metric("Durée totale", f"{total_min:.1f} min")
            c3.metric("🔴 Alertes", n_rouge + n_orange)
            c4.metric("Sources", len(set(s["source"] for s in segments)))

            # Filtre niveau
            niveau_filter = st.radio(
                "Filtrer",
                ["Tous", "🔴 ROUGE", "🟠 ORANGE", "🟢 VERT"],
                horizontal=True
            )
            filtered = segments
            if "ROUGE" in niveau_filter:
                filtered = [s for s in segments if (s["criticite"] or {}).get("niveau") == "ROUGE"]
            elif "ORANGE" in niveau_filter:
                filtered = [s for s in segments if (s["criticite"] or {}).get("niveau") == "ORANGE"]
            elif "VERT" in niveau_filter:
                filtered = [s for s in segments if (s["criticite"] or {}).get("niveau") == "VERT"]

            for seg in filtered[:50]:  # Limiter l'affichage à 50
                render_segment_card(seg)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 : Upload et pipeline
    # ──────────────────────────────────────────────────────────────────────────
    with tab_upload:
        st.subheader("Ajouter des messages vocaux au corpus")

        st.info(
            "**Consentement requis** : les fichiers audio doivent être enregistrés "
            "avec l'accord informé des locuteurs, ou provenir d'espaces publics "
            "(émissions radio, groupes communautaires publics)."
        )

        uploaded = st.file_uploader(
            "Déposer des fichiers audio (WAV, MP3, OGG, M4A)",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            accept_multiple_files=True
        )

        modele = st.radio(
            "Modèle d'embeddings",
            ["whisper (recommandé, ~240 MB)", "wav2vec2-XLSR (~1.2 GB)"],
            horizontal=True
        )
        model_key = "whisper" if "whisper" in modele else "wav2vec2"

        if uploaded and st.button("▶️ Lancer le pipeline", type="primary"):
            os.makedirs("corpus/segments", exist_ok=True)

            prog = st.progress(0, text="Préparation...")

            # Sauvegarder les fichiers uploadés
            for f in uploaded:
                save_path = f"corpus/{f.name}"
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())

            # Lancer le pipeline
            from module1_ingestion import process_directory, init_db
            conn = init_db(DB_PATH)
            metas = process_directory("corpus", "corpus/segments", conn)
            conn.close()

            with open(SEG_JSON, "w") as f:
                json.dump(metas, f, ensure_ascii=False, indent=2)

            prog.progress(0.4, text=f"✓ {len(metas)} segments créés")

            # Embeddings
            from module2_embeddings import compute_embeddings
            compute_embeddings(SEG_JSON, model_key, DB_PATH, "index/embeddings.npy")
            prog.progress(0.7, text="✓ Embeddings calculés")

            # Index FAISS
            from module3_index import build_index
            import numpy as np
            embs = np.load("index/embeddings.npy")
            with open("index/embeddings_ids.json") as f:
                ids = json.load(f)
            build_index(embs, ids, INDEX_PATH, IDS_PATH)
            prog.progress(0.9, text="✓ Index FAISS construit")

            # Scoring criticité
            from module5_criticite import score_corpus
            score_corpus(SEG_JSON, "zeroshot", DB_PATH)
            prog.progress(1.0, text="✓ Pipeline complet !")

            st.success(f"🎉 {len(metas)} segments indexés et scorés !")
            st.balloons()

            # Invalider le cache
            st.cache_resource.clear()
            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 : À propos
    # ──────────────────────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("Audio Sémantique Souverain")
        st.markdown("""
**Hackathon IA CITADEL Summer School 2026** · Défi "Audio Sémantique Souverain"

### Architecture technique
| Composante | Technologie | Raison |
|---|---|---|
| Segmentation | librosa + VAD énergie | CPU frugal, zéro dépendance |
| Embeddings audio | Whisper-small encoder | Multilingue, open weights |
| Embeddings texte | LaBSE | 109 langues, CPU-friendly |
| Index vectoriel | FAISS IndexFlatIP | Open source, local |
| Scoring criticité | Zero-shot embeddings | Aucun entraînement requis |
| Interface | Streamlit | Déployable en <2h |

### Cadre éthique VERTUS
- ✅ **Consentement** : aucune collecte de messages privés sans accord
- ✅ **Souveraineté** : zéro donnée envoyée à des APIs externes
- ✅ **Transparence** : scores explicables, contestables et corrigeables
- ✅ **Non-discrimination** : pas de biais systématiques selon genre/région/dialecte

### Ressources utiles
- Dataset mooré-français : [HuggingFace sawadogosalif/MooreFRCollections](https://huggingface.co/datasets/sawadogosalif/MooreFRCollections)
- Dioula dans Mozilla Common Voice 19
- Code source : modules 1-5 + app.py (ce fichier)
        """)


if __name__ == "__main__":
    main()
