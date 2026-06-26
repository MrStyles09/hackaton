"""
INTERFACE STREAMLIT v3 — CITADEL 2026
======================================
Lancement : streamlit run app_v3.py
"""

import json, os, sqlite3
from typing import List, Dict, Optional
import numpy as np
import streamlit as st
from search_engine import search_all

st.set_page_config(page_title="CITADEL 2026", page_icon="🎙️", layout="wide")

DB_PATH         = "corpus/metadata.db"
AUDIO_DIR_MOORE = "audios_moore"

st.markdown("""
<style>
.card-rouge  {background:#FFF5F5;border-left:4px solid #E53E3E;padding:10px 14px;border-radius:6px;margin:6px 0}
.card-orange {background:#FFFAF0;border-left:4px solid #DD6B20;padding:10px 14px;border-radius:6px;margin:6px 0}
.card-vert   {background:#F0FFF4;border-left:4px solid #38A169;padding:10px 14px;border-radius:6px;margin:6px 0}
.badge {display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700}
.mtag  {display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;background:#EBF8FF;color:#2B6CB0;margin-right:4px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Chargement LaBSE (~1.8 Go, une seule fois)...")
def load_labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")

@st.cache_data(show_spinner="Chargement du corpus...")
def load_corpus() -> List[Dict]:
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, file, path, source, start_sec, end_sec,
               transcription, translation_fr, criticite_json
        FROM segments ORDER BY source, start_sec
    """).fetchall()
    conn.close()
    return [{"id":r[0],"file":r[1],"path":r[2],"source":r[3],
             "start_sec":r[4],"end_sec":r[5],
             "transcription":r[6] or "","translation_fr":r[7] or "",
             "criticite":json.loads(r[8]) if r[8] else None} for r in rows]

@st.cache_data(show_spinner="Encodage LaBSE du corpus...")
def compute_embs(texts: tuple) -> np.ndarray:
    m = load_labse()
    return m.encode(list(texts), batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False)

def get_texts(segs):
    return tuple(s["translation_fr"].strip() or s["transcription"].strip() or "vide"
                 for s in segs)

DIM_LABELS = {
    "urgence_sanitaire":"🏥 Sanitaire","tension_sociale":"⚡ Sociale",
    "alerte_agricole":"🌾 Agricole","desinformation":"🔍 Désinf.",
    "detresse_individu":"🆘 Détresse","sagesse_menace":"⚠️ Sagesse",
}
DIM_COLORS = {
    "urgence_sanitaire":"#E53E3E","tension_sociale":"#DD6B20",
    "alerte_agricole":"#38A169","desinformation":"#805AD5",
    "detresse_individu":"#3182CE","sagesse_menace":"#D69E2E",
}
NVC = {"ROUGE":"#E53E3E","ORANGE":"#DD6B20","VERT":"#38A169"}
NVI = {"ROUGE":"🔴","ORANGE":"🟠","VERT":"🟢"}

def score_bars(scores):
    html = ""
    for cat, val in sorted(scores.items(), key=lambda x: -x[1]):
        lbl = DIM_LABELS.get(cat, cat)
        col = DIM_COLORS.get(cat, "#718096")
        pct = min(100, int(val / 0.30 * 100))
        html += f"""<div style="margin:3px 0;display:flex;align-items:center;gap:8px;font-size:12px">
          <span style="width:95px;color:#4A5568">{lbl}</span>
          <div style="flex:1;background:#EDF2F7;border-radius:4px;height:7px">
            <div style="width:{pct}%;background:{col};height:7px;border-radius:4px"></div>
          </div>
          <span style="width:42px;text-align:right;color:{col};font-weight:600">{val:.3f}</span>
        </div>"""
    return html

def find_audio(seg):
    p = seg.get("path","")
    if os.path.exists(p): return p
    alt = os.path.join(AUDIO_DIR_MOORE, seg.get("file",""))
    return alt if os.path.exists(alt) else ""

def render_card(seg, rank=None, debug=False):
    crit  = seg.get("criticite")
    niv   = crit["niveau"] if crit else "VERT"
    audio = find_audio(seg)
    mt    = seg.get("match_type","")

    st.markdown(f'<div class="card-{niv.lower()}">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        r = f"**#{rank}** · " if rank else ""
        sc = f"score={seg['score']:.3f}" if "score" in seg else ""
        st.markdown(f"{r}**{seg['source']}** · {seg['start_sec']}s–{seg['end_sec']}s · {sc}")
        if mt:
            detail = f" | texte={seg.get('score_text',0):.3f} sem={seg.get('score_sem',0):.3f}" if debug else ""
            st.markdown(f'<span class="mtag">🔎 {mt}{detail}</span>', unsafe_allow_html=True)
        mo = seg.get("transcription","")
        fr = seg.get("translation_fr","")
        if mo: st.caption(f"🔤 {mo}")
        if fr: st.caption(f"🇫🇷 {fr}")
        if crit: st.markdown(score_bars(crit["scores"]), unsafe_allow_html=True)
    with c2:
        if crit:
            c = NVC.get(niv,"#718096"); i = NVI.get(niv,"⚪")
            st.markdown(f'<span class="badge" style="background:{c}20;color:{c}">{i} {niv}</span>', unsafe_allow_html=True)
        if audio:
            with open(audio,"rb") as f: st.audio(f.read(), format="audio/wav")
        else:
            st.caption("_audio non trouvé_")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


def main():
    st.title("🎙️ CITADEL 2026 — Audio Sémantique Souverain")
    st.caption("Mooré · 100% local · Souverain")

    segs = load_corpus()
    if not segs:
        st.error("Corpus introuvable.")
        return

    embs = compute_embs(get_texts(segs))

    with st.sidebar:
        st.header("📊 Corpus")
        st.metric("Segments", len(segs))
        niveaux = [s["criticite"]["niveau"] for s in segs if s.get("criticite")]
        if niveaux:
            st.metric("🔴 ROUGE",  niveaux.count("ROUGE"))
            st.metric("🟠 ORANGE", niveaux.count("ORANGE"))
            st.metric("🟢 VERT",   niveaux.count("VERT"))
        st.divider()
        top_k = st.slider("Nb résultats", 3, 20, 5)
        debug = st.checkbox("Scores détaillés", False)
        st.divider()
        st.markdown("""
**💡 Conseils de recherche**

🇫🇷 **Français** → résultats sémantiques
*(maladie, danger, conflit...)*

🔤 **Mooré** → résultats textuels
*(koom, ned, baag, yaa...)*

✏️ Tu peux taper **sans diacritiques** :
`ned same` → matche `ned sãame`
        """)

    tab1, tab2, tab3 = st.tabs(["🔍 Recherche", "🚨 Alertes", "📚 Corpus"])

    with tab1:
        st.subheader("Recherche dans le corpus audio mooré")

        # Sélecteur de langue EXPLICITE
        col_lang, col_q, col_btn = st.columns([1.2, 4, 1])
        with col_lang:
            lang = st.selectbox("Langue", ["🔀 Auto", "🔤 Mooré", "🇫🇷 Français"],
                                label_visibility="collapsed")
            lang_key = "both" if "Auto" in lang else ("moore" if "Mooré" in lang else "fr")

        with col_q:
            query = st.text_input("", placeholder="koom  /  ned sãame  /  maladie village  /  baag",
                                  label_visibility="collapsed")
        with col_btn:
            go = st.button("🔍 Chercher", use_container_width=True)

        # Exemples cliquables
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Exemples mooré :")
            cols = st.columns(3)
            for i, ex in enumerate(["koom", "ned sãame", "baag yaa"]):
                if cols[i].button(ex, key=f"m{i}"):
                    query, go, lang_key = ex, True, "moore"
        with col_b:
            st.caption("Exemples français :")
            cols = st.columns(3)
            for i, ex in enumerate(["maladie", "danger animal", "conflit"]):
                if cols[i].button(ex, key=f"f{i}"):
                    query, go, lang_key = ex, True, "fr"

        if query and go:
            lang_labels = {"fr":"🇫🇷 Français → sémantique LaBSE",
                           "moore":"🔤 Mooré → matching textuel + sémantique",
                           "both":"🔀 Auto → fusion textuel + sémantique"}
            st.info(lang_labels.get(lang_key, ""))

            with st.spinner(f"Recherche '{query}'..."):
                results = search_all(query, segs, embs, load_labse(), top_k, lang_key)

            if results:
                st.success(f"**{len(results)} résultats** pour *{query}*")
                for i, seg in enumerate(results, 1):
                    render_card(seg, rank=i, debug=debug)
            else:
                st.warning("Aucun résultat trouvé.")
                if lang_key == "moore":
                    st.caption("💡 Essaie sans diacritiques ou change la langue en 🔀 Auto.")

    with tab2:
        st.subheader("Segments les plus critiques")
        dim = st.selectbox("Dimension", list(DIM_LABELS.keys()),
                           format_func=lambda x: DIM_LABELS[x])
        top_crit = sorted(
            [s for s in segs if s.get("criticite") and dim in s["criticite"].get("scores",{})],
            key=lambda s: -s["criticite"]["scores"][dim]
        )[:top_k]
        for i, seg in enumerate(top_crit, 1):
            render_card(seg, rank=i, debug=debug)

    with tab3:
        st.subheader(f"Corpus — {len(segs)} segments")
        fn = st.selectbox("Niveau", ["Tous","ROUGE","ORANGE","VERT"])
        ft = st.text_input("Filtrer texte","")
        filtered = [s for s in segs
                    if (fn=="Tous" or (s.get("criticite") and s["criticite"]["niveau"]==fn))
                    and (not ft or ft.lower() in (s["transcription"]+s["translation_fr"]).lower())]
        st.caption(f"{len(filtered)} segments")
        for seg in filtered[:50]:
            render_card(seg, debug=debug)
        if len(filtered) > 50:
            st.info("Limité à 50. Utilise le filtre pour affiner.")

if __name__ == "__main__":
    main()
