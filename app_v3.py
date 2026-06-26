"""
INTERFACE STREAMLIT v3 — CITADEL 2026
======================================
Intègre le moteur de recherche hybride (search_engine.py) :
  - Requête française  → LaBSE sémantique sur traductions FR
  - Requête mooré      → matching textuel direct + LaBSE fusionné
  - Détection de langue automatique
  - Indicateur de type de match affiché

Lancement : streamlit run app_v3.py
"""

import json
import os
import sqlite3
from typing import List, Dict, Optional
import numpy as np
import streamlit as st

from search_engine import smart_search, detect_language

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CITADEL 2026 — Audio Sémantique Mooré",
    page_icon="🎙️",
    layout="wide",
)

DB_PATH         = "corpus/metadata.db"
AUDIO_DIR_MOORE = "audios_moore"

st.markdown("""
<style>
.card-rouge  { background:#FFF5F5; border-left:4px solid #E53E3E; padding:10px 14px; border-radius:6px; margin:6px 0; }
.card-orange { background:#FFFAF0; border-left:4px solid #DD6B20; padding:10px 14px; border-radius:6px; margin:6px 0; }
.card-vert   { background:#F0FFF4; border-left:4px solid #38A169; padding:10px 14px; border-radius:6px; margin:6px 0; }
.badge  { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:700; }
.mtag   { display:inline-block; padding:1px 7px; border-radius:8px; font-size:11px; background:#EBF8FF; color:#2B6CB0; }
</style>
""", unsafe_allow_html=True)

# ── Ressources ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de LaBSE...")
def load_labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")


@st.cache_data(show_spinner="Chargement du corpus...")
def load_corpus() -> List[Dict]:
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, file, path, source, start_sec, end_sec, duration_sec,
               transcription, translation_fr, criticite_json
        FROM segments ORDER BY source, start_sec
    """).fetchall()
    conn.close()
    return [{
        "id": r[0], "file": r[1], "path": r[2], "source": r[3],
        "start_sec": r[4], "end_sec": r[5], "duration_sec": r[6],
        "transcription": r[7] or "",
        "translation_fr": r[8] or "",
        "criticite": json.loads(r[9]) if r[9] else None,
    } for r in rows]


@st.cache_data(show_spinner="Encodage LaBSE du corpus...")
def compute_corpus_embeddings(texts: tuple) -> np.ndarray:
    model = load_labse()
    return model.encode(list(texts), batch_size=64,
                        normalize_embeddings=True, show_progress_bar=False)


def get_search_texts(segments):
    return tuple(
        s["translation_fr"].strip() or s["transcription"].strip() or "vide"
        for s in segments
    )


# ── Rendus ────────────────────────────────────────────────────────────────────
DIM_LABELS = {
    "urgence_sanitaire" : "🏥 Sanitaire",
    "tension_sociale"   : "⚡ Sociale",
    "alerte_agricole"   : "🌾 Agricole",
    "desinformation"    : "🔍 Désinf.",
    "detresse_individu" : "🆘 Détresse",
    "sagesse_menace"    : "⚠️ Sagesse",
}
DIM_COLORS = {
    "urgence_sanitaire":"#E53E3E","tension_sociale":"#DD6B20",
    "alerte_agricole":"#38A169","desinformation":"#805AD5",
    "detresse_individu":"#3182CE","sagesse_menace":"#D69E2E",
}
NIVEAU_C = {"ROUGE":"#E53E3E","ORANGE":"#DD6B20","VERT":"#38A169"}
NIVEAU_I = {"ROUGE":"🔴","ORANGE":"🟠","VERT":"🟢"}

MATCH_LABELS = {
    "moore_text"   : "✅ Match textuel mooré",
    "hybrid_moore" : "🔀 Hybride mooré+sémantique",
    "semantic_only": "🔍 Sémantique seul",
    "semantic_fr"  : "🔍 Sémantique FR",
}


def score_bars(scores):
    html = ""
    for cat, val in sorted(scores.items(), key=lambda x: -x[1]):
        label = DIM_LABELS.get(cat, cat)
        color = DIM_COLORS.get(cat, "#718096")
        pct   = min(100, int(val / 0.30 * 100))
        html += f"""<div style="margin:3px 0;display:flex;align-items:center;gap:8px;font-size:12px">
          <span style="width:90px;color:#4A5568">{label}</span>
          <div style="flex:1;background:#EDF2F7;border-radius:4px;height:7px">
            <div style="width:{pct}%;background:{color};height:7px;border-radius:4px"></div>
          </div>
          <span style="width:40px;text-align:right;color:{color};font-weight:600">{val:.3f}</span>
        </div>"""
    return html


def find_audio(seg):
    p = seg.get("path", "")
    if os.path.exists(p):
        return p
    alt = os.path.join(AUDIO_DIR_MOORE, seg.get("file", ""))
    return alt if os.path.exists(alt) else ""


def render_card(seg: Dict, rank: Optional[int] = None, show_debug: bool = False):
    crit   = seg.get("criticite")
    niveau = crit["niveau"] if crit else "VERT"
    css    = f"card-{niveau.lower()}"
    audio  = find_audio(seg)
    
    match_type  = seg.get("match_type", "")
    match_label = MATCH_LABELS.get(match_type, "")
    score_text  = seg.get("score_text")
    score_sem   = seg.get("score_sem")

    st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])

    with col1:
        rank_str = f"**#{rank}** · " if rank else ""
        score_str = f"sim={seg['score']:.3f}" if "score" in seg else ""
        st.markdown(f"{rank_str}**{seg['source']}** · {seg['start_sec']}s–{seg['end_sec']}s · {score_str}")

        if match_label:
            detail = ""
            if show_debug and score_text is not None:
                detail = f" (texte={score_text:.3f}, sem={score_sem:.3f})"
            st.markdown(f'<span class="mtag">{match_label}{detail}</span>', unsafe_allow_html=True)

        moore = seg.get("transcription", "")
        fr    = seg.get("translation_fr", "")
        if moore: st.caption(f"🔤 {moore}")
        if fr:    st.caption(f"🇫🇷 {fr}")

        if crit:
            st.markdown(score_bars(crit["scores"]), unsafe_allow_html=True)

    with col2:
        if crit:
            c = NIVEAU_C.get(niveau, "#718096")
            i = NIVEAU_I.get(niveau, "⚪")
            st.markdown(f'<span class="badge" style="background:{c}20;color:{c}">{i} {niveau}</span>',
                        unsafe_allow_html=True)
        if audio:
            with open(audio, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        else:
            st.caption("_audio introuvable_")

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("🎙️ CITADEL 2026 — Audio Sémantique Souverain")
    st.caption("Mooré · Dioula · Fulfuldé · 100% local · Souverain")

    segments = load_corpus()
    if not segments:
        st.error("Corpus introuvable. Lance le pipeline d'abord.")
        return

    search_texts = get_search_texts(segments)
    with st.spinner("Encodage LaBSE..."):
        corpus_embs = compute_corpus_embeddings(search_texts)

    with st.sidebar:
        st.header("📊 Corpus")
        st.metric("Segments", len(segments))
        niveaux = [s["criticite"]["niveau"] for s in segments if s.get("criticite")]
        if niveaux:
            st.metric("🔴 ROUGE",  niveaux.count("ROUGE"))
            st.metric("🟠 ORANGE", niveaux.count("ORANGE"))
            st.metric("🟢 VERT",   niveaux.count("VERT"))
        st.divider()
        top_k      = st.slider("Nb résultats", 3, 20, 5)
        show_debug = st.checkbox("Afficher scores détaillés", False)
        st.divider()
        st.info("**Tip mooré** : si les résultats sont mauvais, essaie sans les diacritiques — `ned same` au lieu de `ned sãame`")

    tab1, tab2, tab3 = st.tabs([
        "🔍 Recherche", "🚨 Alertes", "📚 Corpus"
    ])

    with tab1:
        st.subheader("Recherche multilingue — mooré & français")

        col_q, col_b = st.columns([4, 1])
        with col_q:
            query = st.text_input("", placeholder="Ex: ned sãame   /   maladie village   /   danger animal", label_visibility="collapsed")
        with col_b:
            go = st.button("🔍 Chercher", use_container_width=True)

        # Exemples en deux langues
        st.caption("Exemples mooré :")
        ecols = st.columns(4)
        moore_ex = ["ned sãame", "baag yaa", "zĩrẽ zoeta", "bõnpoak yãnd"]
        for i, ex in enumerate(moore_ex):
            if ecols[i].button(ex, key=f"mex_{i}"):
                query, go = ex, True

        st.caption("Exemples français :")
        fcols = st.columns(4)
        fr_ex = ["maladie urgence", "conflit village", "mensonge rumeur", "danger animal"]
        for i, ex in enumerate(fr_ex):
            if fcols[i].button(ex, key=f"fex_{i}"):
                query, go = ex, True

        if query and go:
            lang = detect_language(query)
            lang_label = "🔤 Mooré détecté" if lang == "moore" else "🇫🇷 Français détecté"
            st.info(f"{lang_label} · Stratégie : {'matching textuel + sémantique' if lang == 'moore' else 'recherche sémantique LaBSE'}")

            with st.spinner(f"Recherche '{query}'..."):
                results, _ = smart_search(query, segments, corpus_embs,
                                           load_labse(), top_k)

            if results:
                st.success(f"**{len(results)} résultats** pour *{query}*")
                for i, seg in enumerate(results, 1):
                    render_card(seg, rank=i, show_debug=show_debug)
            else:
                st.warning("Aucun résultat. Essaie une variante orthographique.")
                # Suggestion automatique
                if lang == "moore":
                    from search_engine import normalize_relaxed
                    st.caption(f"Version sans diacritiques testée : `{normalize_relaxed(query)}`")

    with tab2:
        st.subheader("Segments les plus critiques")
        dim = st.selectbox("Dimension", list(DIM_LABELS.keys()),
                           format_func=lambda x: DIM_LABELS[x])
        top_crit = sorted(
            [s for s in segments if s.get("criticite") and dim in s["criticite"].get("scores", {})],
            key=lambda s: -s["criticite"]["scores"][dim]
        )[:top_k]
        for i, seg in enumerate(top_crit, 1):
            render_card(seg, rank=i, show_debug=show_debug)

    with tab3:
        st.subheader(f"Corpus — {len(segments)} segments")
        fn = st.selectbox("Niveau", ["Tous", "ROUGE", "ORANGE", "VERT"])
        ft = st.text_input("Filtrer texte", "")
        filtered = [s for s in segments
                    if (fn == "Tous" or (s.get("criticite") and s["criticite"]["niveau"] == fn))
                    and (not ft or ft.lower() in (s["transcription"]+s["translation_fr"]).lower())]
        st.caption(f"{len(filtered)} segments")
        for seg in filtered[:50]:
            render_card(seg, show_debug=show_debug)
        if len(filtered) > 50:
            st.info("Affichage limité à 50. Utilise le filtre pour affiner.")


if __name__ == "__main__":
    main()
