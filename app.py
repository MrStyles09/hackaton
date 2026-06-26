"""
INTERFACE STREAMLIT — CITADEL 2026  (v2 — recherche corrigée)
==============================================================
Correction principale : la recherche utilise LaBSE sur les TEXTES stockés
en base (transcription mooré + traduction française), pas FAISS sur les
embeddings Whisper. Les deux espaces vectoriels sont ainsi cohérents.

Architecture de recherche v2 :
  Requête texte (FR ou mooré)
      ↓ LaBSE embed
  Comparaison cosinus contre LaBSE embeddings des transcriptions en cache
      ↓ top-k
  Résultats avec audio + scores criticité

Lancement : streamlit run app.py
"""

import json
import os
import sqlite3
from typing import List, Dict, Optional

import numpy as np
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CITADEL 2026 — Audio Sémantique Mooré",
    page_icon="🎙️",
    layout="wide",
)

DB_PATH  = "corpus/metadata.db"
SEG_JSON = "corpus/segments.json"
AUDIO_DIR_MOORE = "audios_moore"

st.markdown("""
<style>
.card-rouge  { background:#FFF5F5; border-left:4px solid #E53E3E; padding:10px 14px; border-radius:6px; margin:6px 0; }
.card-orange { background:#FFFAF0; border-left:4px solid #DD6B20; padding:10px 14px; border-radius:6px; margin:6px 0; }
.card-vert   { background:#F0FFF4; border-left:4px solid #38A169; padding:10px 14px; border-radius:6px; margin:6px 0; }
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:700; }
.sim-score { font-size:11px; color:#718096; }
</style>
""", unsafe_allow_html=True)


# ── Chargement des ressources (cachées) ───────────────────────────────────────

@st.cache_resource(show_spinner="Chargement de LaBSE...")
def load_labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")


@st.cache_data(show_spinner="Chargement du corpus...")
def load_corpus() -> List[Dict]:
    """Charge tous les segments depuis SQLite avec leurs textes et scores."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, file, path, source, start_sec, end_sec, duration_sec,
               transcription, translation_fr, criticite_json
        FROM segments
        ORDER BY source, start_sec
    """).fetchall()
    conn.close()
    segments = []
    for r in rows:
        segments.append({
            "id"           : r[0],
            "file"         : r[1],
            "path"         : r[2],
            "source"       : r[3],
            "start_sec"    : r[4],
            "end_sec"      : r[5],
            "duration_sec" : r[6],
            "transcription": r[7] or "",
            "translation_fr": r[8] or "",
            "criticite"    : json.loads(r[9]) if r[9] else None,
        })
    return segments


@st.cache_data(show_spinner="Calcul des embeddings du corpus...")
def compute_corpus_embeddings(texts: tuple) -> np.ndarray:
    """
    Encode tous les textes du corpus avec LaBSE.
    Utilise la TRADUCTION FRANÇAISE si disponible (meilleure qualité),
    sinon le texte mooré.
    Mis en cache — ne se recalcule qu'une fois par session.
    """
    model = load_labse()
    embeddings = model.encode(
        list(texts),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def get_search_texts(segments: List[Dict]) -> tuple:
    """Retourne les textes à indexer : FR si dispo, sinon mooré."""
    texts = []
    for s in segments:
        t = s["translation_fr"].strip() if s["translation_fr"].strip() else s["transcription"].strip()
        texts.append(t if t else "texte non disponible")
    return tuple(texts)  # tuple pour que st.cache_data puisse le hasher


def search_semantic(query: str, segments: List[Dict],
                    corpus_embs: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    Recherche sémantique LaBSE : requête → similarité cosinus contre corpus.
    Compatible avec requêtes en français ET en mooré (LaBSE est multilingue).
    """
    model = load_labse()
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    # Similarité cosinus (embeddings déjà normalisés → produit scalaire)
    sims = corpus_embs @ q_emb

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        seg = dict(segments[idx])
        seg["score"] = float(sims[idx])
        results.append(seg)
    return results


def search_by_criticite(segments: List[Dict],
                         dimension: str, top_k: int = 10) -> List[Dict]:
    """Filtre les segments par score de criticité sur une dimension."""
    scored = []
    for s in segments:
        if s["criticite"] and dimension in s["criticite"].get("scores", {}):
            score = s["criticite"]["scores"][dimension]
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_k]]


# ── Rendus ────────────────────────────────────────────────────────────────────

DIM_LABELS = {
    "urgence_sanitaire" : "🏥 Urgence sanitaire",
    "tension_sociale"   : "⚡ Tension sociale",
    "alerte_agricole"   : "🌾 Alerte agricole",
    "desinformation"    : "🔍 Désinformation",
    "detresse_individu" : "🆘 Détresse individu",
    "sagesse_menace"    : "⚠️ Sagesse / menace",
}

DIM_COLORS = {
    "urgence_sanitaire" : "#E53E3E",
    "tension_sociale"   : "#DD6B20",
    "alerte_agricole"   : "#38A169",
    "desinformation"    : "#805AD5",
    "detresse_individu" : "#3182CE",
    "sagesse_menace"    : "#D69E2E",
}

NIVEAU_COLORS = {"ROUGE": "#E53E3E", "ORANGE": "#DD6B20", "VERT": "#38A169"}
NIVEAU_ICONS  = {"ROUGE": "🔴", "ORANGE": "🟠", "VERT": "🟢"}


def badge(niveau):
    c = NIVEAU_COLORS.get(niveau, "#718096")
    i = NIVEAU_ICONS.get(niveau, "⚪")
    return f'<span class="badge" style="background:{c}20;color:{c}">{i} {niveau}</span>'


def score_bars_html(scores: Dict) -> str:
    html = ""
    for cat, val in sorted(scores.items(), key=lambda x: -x[1]):
        label = DIM_LABELS.get(cat, cat)
        color = DIM_COLORS.get(cat, "#718096")
        # Normaliser sur 0-100% en fonction du max observé (0.3 = 100%)
        pct = min(100, int(val / 0.30 * 100))
        html += f"""
        <div style="margin:3px 0;display:flex;align-items:center;gap:8px;font-size:12px">
          <span style="width:120px;color:#4A5568">{label}</span>
          <div style="flex:1;background:#EDF2F7;border-radius:4px;height:7px">
            <div style="width:{pct}%;background:{color};height:7px;border-radius:4px"></div>
          </div>
          <span style="width:40px;text-align:right;color:{color};font-weight:600">{val:.3f}</span>
        </div>"""
    return html


def render_card(seg: Dict, rank: Optional[int] = None):
    crit   = seg.get("criticite")
    niveau = crit["niveau"] if crit else "VERT"
    css    = f"card-{niveau.lower()}"

    # Trouver le fichier audio
    audio_path = seg.get("path", "")
    if not os.path.exists(audio_path):
        # Chercher dans audios_moore/
        fname = seg.get("file", "")
        alt = os.path.join(AUDIO_DIR_MOORE, fname)
        audio_path = alt if os.path.exists(alt) else ""

    rank_str = f"#{rank} · " if rank else ""
    score_str = f" · sim: **{seg['score']:.3f}**" if "score" in seg else ""

    moore_text = seg.get("transcription", "")
    fr_text    = seg.get("translation_fr", "")

    st.markdown(f'<div class="{css}">', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"{rank_str}**{seg['source']}** · {seg['start_sec']}s–{seg['end_sec']}s{score_str}")
        if moore_text:
            st.caption(f"🔤 Mooré : {moore_text}")
        if fr_text:
            st.caption(f"🇫🇷 FR    : {fr_text}")
        if crit:
            st.markdown(score_bars_html(crit["scores"]), unsafe_allow_html=True)

    with col2:
        if crit:
            st.markdown(badge(niveau), unsafe_allow_html=True)
        if audio_path:
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        else:
            st.caption("_audio introuvable_")

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


# ── Interface principale ──────────────────────────────────────────────────────

def main():
    st.title("🎙️ CITADEL 2026 — Audio Sémantique Souverain")
    st.caption("Recherche sémantique sur corpus audio mooré · 100% local · Aucune donnée ne quitte la machine")

    # Charger le corpus
    segments = load_corpus()
    if not segments:
        st.error(f"Corpus introuvable ({DB_PATH}). Lance d'abord le pipeline complet.")
        return

    # Pré-calculer les embeddings du corpus (une fois par session)
    search_texts = get_search_texts(segments)
    with st.spinner(f"Encodage de {len(segments)} segments avec LaBSE..."):
        corpus_embs = compute_corpus_embeddings(search_texts)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Corpus")
        st.metric("Segments total", len(segments))
        if segments[0]["criticite"]:
            niveaux = [s["criticite"]["niveau"] for s in segments if s["criticite"]]
            st.metric("🔴 ROUGE",  niveaux.count("ROUGE"))
            st.metric("🟠 ORANGE", niveaux.count("ORANGE"))
            st.metric("🟢 VERT",   niveaux.count("VERT"))

        st.divider()
        st.header("⚙️ Paramètres")
        top_k = st.slider("Nombre de résultats", 3, 20, 5)

        st.divider()
        st.caption("💡 **Astuce** : la recherche est multilingue — tape en français ou en mooré, LaBSE comprend les deux.")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Recherche sémantique",
        "🚨 Alertes criticité",
        "📚 Corpus complet",
    ])

    # ── Tab 1 : Recherche ─────────────────────────────────────────────────────
    with tab1:
        st.subheader("Recherche sémantique multilingue")
        st.info(
            "Tape une requête en **français** ou en **mooré** — LaBSE retrouve les segments "
            "sémantiquement proches, même si les mots exacts ne correspondent pas.",
            icon="ℹ️"
        )

        col_q, col_btn = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "Requête",
                placeholder="Ex: maladie village  /  danger animal  /  ned sãame  /  conflict route",
                label_visibility="collapsed"
            )
        with col_btn:
            search_btn = st.button("🔍 Chercher", use_container_width=True)

        # Exemples cliquables
        st.caption("Exemples :")
        ex_cols = st.columns(5)
        examples = [
            "maladie urgence",
            "danger animal scorpion",
            "conflict route village",
            "mensonge rumeur",
            "ned sãame pãnga",
        ]
        for i, ex in enumerate(examples):
            if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex
                search_btn = True

        if query and (search_btn or query):
            with st.spinner(f"Recherche de '{query}'..."):
                results = search_semantic(query, segments, corpus_embs, top_k)

            if results:
                st.success(f"Top {len(results)} résultats pour **{query}**")
                for i, seg in enumerate(results, 1):
                    render_card(seg, rank=i)
            else:
                st.warning("Aucun résultat trouvé.")

    # ── Tab 2 : Alertes ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("Segments les plus critiques par dimension")

        dim = st.selectbox(
            "Dimension de criticité",
            options=list(DIM_LABELS.keys()),
            format_func=lambda x: DIM_LABELS[x]
        )

        results_crit = search_by_criticite(segments, dim, top_k)
        if results_crit:
            st.write(f"**Top {len(results_crit)} segments — {DIM_LABELS[dim]}**")
            for i, seg in enumerate(results_crit, 1):
                render_card(seg, rank=i)
        else:
            st.info("Aucun segment scoré pour cette dimension.")

        # Vue globale
        st.divider()
        st.subheader("Distribution des niveaux d'alerte")
        if segments[0]["criticite"]:
            niveaux = [s["criticite"]["niveau"] for s in segments if s["criticite"]]
            rouge  = niveaux.count("ROUGE")
            orange = niveaux.count("ORANGE")
            vert   = niveaux.count("VERT")
            total  = len(niveaux)
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 ROUGE",  rouge,  f"{rouge/total:.1%}")
            c2.metric("🟠 ORANGE", orange, f"{orange/total:.1%}")
            c3.metric("🟢 VERT",   vert,   f"{vert/total:.1%}")

    # ── Tab 3 : Corpus ────────────────────────────────────────────────────────
    with tab3:
        st.subheader(f"Corpus complet — {len(segments)} segments")

        col_f1, col_f2 = st.columns(2)
        filtre_niveau = col_f1.selectbox("Filtrer par niveau", ["Tous", "ROUGE", "ORANGE", "VERT"])
        filtre_texte  = col_f2.text_input("Filtrer par texte (mooré ou FR)", "")

        filtered = segments
        if filtre_niveau != "Tous":
            filtered = [s for s in filtered
                        if s.get("criticite") and s["criticite"]["niveau"] == filtre_niveau]
        if filtre_texte:
            ft = filtre_texte.lower()
            filtered = [s for s in filtered
                        if ft in (s["transcription"] or "").lower()
                        or ft in (s["translation_fr"] or "").lower()]

        st.caption(f"{len(filtered)} segments affichés")

        for seg in filtered[:50]:  # limiter à 50 pour la perf
            render_card(seg)

        if len(filtered) > 50:
            st.info(f"Affichage limité à 50 segments. Utilise le filtre pour affiner.")


if __name__ == "__main__":
    main()
