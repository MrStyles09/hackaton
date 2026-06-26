"""
INTERFACE STREAMLIT v3 — CITADEL 2026
======================================
4 onglets :
  1. 🔍 Recherche texte (mooré + français)
  2. 🎤 Recherche vocale (audio→audio via Whisper + FAISS)
  3. 🚨 Alertes criticité
  4. 📚 Corpus complet

Lancement : streamlit run app_v3.py
"""

import json, os, sqlite3, io
from typing import List, Dict, Optional
import numpy as np
import streamlit as st
from search_engine import search_all
from audio_recorder import render_audio_input, load_audio_bytes

st.set_page_config(page_title="CITADEL 2026", page_icon="🎙️", layout="wide")

DB_PATH         = "corpus/metadata.db"
AUDIO_DIR_MOORE = "audios_moore"
INDEX_PATH      = "index/faiss.index"
IDS_PATH        = "index/segment_ids.json"

st.markdown("""
<style>
.card-rouge  {background:#FFF5F5;border-left:4px solid #E53E3E;padding:10px 14px;border-radius:6px;margin:6px 0}
.card-orange {background:#FFFAF0;border-left:4px solid #DD6B20;padding:10px 14px;border-radius:6px;margin:6px 0}
.card-vert   {background:#F0FFF4;border-left:4px solid #38A169;padding:10px 14px;border-radius:6px;margin:6px 0}
.badge {display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700}
.mtag  {display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;
        background:#EBF8FF;color:#2B6CB0;margin-right:4px}
.atag  {display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;
        background:#FAF5FF;color:#6B46C1;margin-right:4px}
</style>
""", unsafe_allow_html=True)


# ── Ressources cachées ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement LaBSE...")
def load_labse():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")


@st.cache_resource(show_spinner="Chargement Whisper encoder...")
def load_whisper_embedder():
    """Charge le WhisperEmbedder du Module 2."""
    try:
        from module2_embeddings import WhisperEmbedder
        return WhisperEmbedder()
    except Exception as e:
        st.error(f"Whisper embedder indisponible : {e}")
        return None


@st.cache_resource(show_spinner="Chargement index FAISS...")
def load_faiss():
    """Charge l'index FAISS et les IDs de segments."""
    try:
        import faiss
        if not os.path.exists(INDEX_PATH):
            return None, []
        index = faiss.read_index(INDEX_PATH)
        ids_f = IDS_PATH
        # Chercher aussi segment_ids.json ou embeddings_ids.json
        if not os.path.exists(ids_f):
            for alt in ["index/embeddings_ids.json", "index/ids.json"]:
                if os.path.exists(alt):
                    ids_f = alt
                    break
        with open(ids_f) as f:
            ids = json.load(f)
        return index, ids
    except Exception as e:
        st.warning(f"FAISS non disponible : {e}")
        return None, []


@st.cache_data(show_spinner="Chargement corpus...")
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


@st.cache_data(show_spinner="Encodage LaBSE corpus...")
def compute_labse_embs(texts: tuple) -> np.ndarray:
    m = load_labse()
    return m.encode(list(texts), batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False)


def get_texts(segs):
    return tuple(s["translation_fr"].strip() or s["transcription"].strip() or "vide"
                 for s in segs)


# ── Recherche audio→audio via FAISS ──────────────────────────────────────────

def search_by_audio(audio_bytes: bytes, top_k: int = 5) -> List[Dict]:
    """
    Pipeline recherche vocale :
      audio bytes → float32 array → Whisper encoder → embedding 768-dim
          → similarité cosinus FAISS → top-k segments + métadonnées SQLite
    """
    # 1. Charger l'audio avec librosa (même pipeline que module2)
    import librosa, io as _io
    audio, sr = librosa.load(_io.BytesIO(audio_bytes), sr=16000, mono=True)
    peak = max(abs(audio).max(), 1e-8)
    audio = (audio / peak * 0.95).astype('float32')

    # 2. Encoder avec Whisper
    embedder = load_whisper_embedder()
    if embedder is None:
        return []

    query_emb = embedder.embed(audio, sr=sr)  # shape (768,)
    query_emb = query_emb.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb /= norm

    # 3. Recherche FAISS
    index, ids = load_faiss()
    if index is None or not ids:
        st.error("Index FAISS introuvable. Vérifie index/faiss.index")
        return []

    scores, indices = index.search(query_emb, top_k * 2)

    # 4. Récupérer les métadonnées SQLite
    conn = sqlite3.connect(DB_PATH)
    results = []
    seg_by_id = {s["id"]: s for s in load_corpus()}

    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(ids):
            continue
        sid = ids[idx]
        if sid in seg_by_id:
            seg = dict(seg_by_id[sid])
            seg["score"]      = round(float(score), 4)
            seg["match_type"] = "audio→audio"
            results.append(seg)
        if len(results) >= top_k:
            break

    conn.close()
    return results


# ── Rendus ────────────────────────────────────────────────────────────────────

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
    p = seg.get("path", "")
    if os.path.exists(p):
        return p
    alt = os.path.join(AUDIO_DIR_MOORE, seg.get("file", ""))
    return alt if os.path.exists(alt) else ""


def render_card(seg, rank=None, debug=False, audio_mode=False):
    crit  = seg.get("criticite")
    niv   = crit["niveau"] if crit else "VERT"
    audio = find_audio(seg)
    mt    = seg.get("match_type", "")

    st.markdown(f'<div class="card-{niv.lower()}">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])

    with c1:
        r  = f"**#{rank}** · " if rank else ""
        sc = f"score={seg['score']:.4f}" if "score" in seg else ""
        st.markdown(f"{r}**{seg['source']}** · {seg['start_sec']}s–{seg['end_sec']}s · {sc}")

        # Badge type de match
        if mt:
            tag_class = "atag" if audio_mode else "mtag"
            icon = "🎤" if audio_mode else "🔎"
            detail = ""
            if debug and not audio_mode:
                detail = f" | txt={seg.get('score_text',0):.3f} sem={seg.get('score_sem',0):.3f}"
            st.markdown(f'<span class="{tag_class}">{icon} {mt}{detail}</span>',
                        unsafe_allow_html=True)

        mo = seg.get("transcription", "")
        fr = seg.get("translation_fr", "")
        if mo: st.caption(f"🔤 {mo}")
        if fr: st.caption(f"🇫🇷 {fr}")
        if crit:
            st.markdown(score_bars(crit["scores"]), unsafe_allow_html=True)

    with c2:
        if crit:
            c = NVC.get(niv, "#718096")
            i = NVI.get(niv, "⚪")
            st.markdown(
                f'<span class="badge" style="background:{c}20;color:{c}">{i} {niv}</span>',
                unsafe_allow_html=True
            )
        if audio:
            with open(audio, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        else:
            st.caption("_audio non trouvé_")

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("🎙️ CITADEL 2026 — Audio Sémantique Souverain")
    st.caption("Mooré · 100% local · Souverain · Aucune donnée ne quitte la machine")

    segs = load_corpus()
    if not segs:
        st.error("Corpus introuvable. Lance le pipeline d'abord.")
        return

    labse_embs = compute_labse_embs(get_texts(segs))

    # ── Sidebar ──────────────────────────────────────────────────────────────
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
**💡 Recherche texte**

🔤 **Mooré** → matching textuel direct
*(koom, ned, baag...)*

🇫🇷 **Français** → sémantique LaBSE
*(maladie, danger, conflit...)*

---
**🎤 Recherche vocale**

Parle en mooré → Whisper encode →
FAISS retrouve les segments
acoustiquement proches
        """)

    # ── 4 onglets ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Recherche texte",
        "🎤 Recherche vocale",
        "🚨 Alertes",
        "📚 Corpus",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Recherche texte
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Recherche textuelle — mooré & français")

        col_lang, col_q, col_btn = st.columns([1.2, 4, 1])
        with col_lang:
            lang = st.selectbox(
                "Langue",
                ["🔀 Auto", "🔤 Mooré", "🇫🇷 Français"],
                label_visibility="collapsed"
            )
            lang_key = "both" if "Auto" in lang else ("moore" if "Mooré" in lang else "fr")

        with col_q:
            query = st.text_input(
                "Requête", label_visibility="collapsed",
                placeholder="koom  /  ned sãame  /  maladie village  /  baag"
            )
        with col_btn:
            go = st.button("🔍 Chercher", use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Exemples mooré :")
            cols = st.columns(4)
            for i, ex in enumerate(["koom", "ned sãame", "baag yaa", "zĩrẽ"]):
                if cols[i].button(ex, key=f"m{i}"):
                    query, go, lang_key = ex, True, "moore"
        with col_b:
            st.caption("Exemples français :")
            cols = st.columns(4)
            for i, ex in enumerate(["maladie", "danger", "conflit", "mensonge"]):
                if cols[i].button(ex, key=f"f{i}"):
                    query, go, lang_key = ex, True, "fr"

        if query and go:
            labels = {
                "fr":    "🇫🇷 Français → sémantique LaBSE",
                "moore": "🔤 Mooré → matching textuel + fusion sémantique",
                "both":  "🔀 Auto → fusion textuel mooré + sémantique FR",
            }
            st.info(labels.get(lang_key, ""))
            with st.spinner(f"Recherche '{query}'..."):
                results = search_all(query, segs, labse_embs,
                                     load_labse(), top_k, lang_key)
            if results:
                st.success(f"**{len(results)} résultats** pour *{query}*")
                for i, seg in enumerate(results, 1):
                    render_card(seg, rank=i, debug=debug)
            else:
                st.warning("Aucun résultat. Essaie sans diacritiques ou change de langue.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Recherche vocale
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🎤 Recherche par la voix — audio→audio")
        st.info(
            "Parle en **mooré** (ou toute autre langue) — "
            "Whisper encode ton audio et retrouve les segments "
            "**acoustiquement proches** dans le corpus, "
            "sans transcription préalable.",
            icon="🎙️"
        )

        # Vérifier que FAISS est dispo
        index, ids = load_faiss()
        if index is None:
            st.warning(
                "Index FAISS introuvable (`index/faiss.index`). "
                "Lance d'abord `python module3_index.py`."
            )
        else:
            st.caption(f"Index FAISS prêt · {index.ntotal} vecteurs Whisper indexés")

        st.divider()

        # Interface d'entrée audio
        audio_bytes = render_audio_input(key_prefix="vocal_search")

        st.divider()

        col_search, col_info = st.columns([1, 2])
        with col_search:
            search_audio_btn = st.button(
                "🎤 Lancer la recherche vocale",
                disabled=(audio_bytes is None or index is None),
                use_container_width=True,
                type="primary"
            )
        with col_info:
            if audio_bytes is None:
                st.caption("👆 Upload ou enregistre un audio d'abord")
            elif index is None:
                st.caption("⚠️ Index FAISS manquant")
            else:
                # Estimer la durée
                try:
                    import soundfile as sf
                    info = sf.info(io.BytesIO(audio_bytes))
                    st.caption(f"Audio prêt · {info.duration:.1f}s · {info.samplerate} Hz")
                except Exception:
                    st.caption(f"Audio prêt · {len(audio_bytes)//1024} Ko")

        if search_audio_btn and audio_bytes:
            with st.spinner("Encodage Whisper + recherche FAISS..."):
                results = search_by_audio(audio_bytes, top_k)

            if results:
                st.success(f"**{len(results)} segments acoustiquement proches** trouvés")
                st.caption(
                    "Les scores FAISS (similarité cosinus dans l'espace Whisper) "
                    "reflètent la proximité acoustique, pas sémantique."
                )
                for i, seg in enumerate(results, 1):
                    render_card(seg, rank=i, debug=debug, audio_mode=True)
            else:
                st.warning(
                    "Aucun résultat. "
                    "Vérifie que l'index FAISS correspond bien aux embeddings Whisper du corpus."
                )

        # Note pédagogique
        with st.expander("ℹ️ Comment fonctionne la recherche vocale ?"):
            st.markdown("""
**Pipeline audio→audio (sans transcription) :**

```
Ta voix (WAV/MP3/enregistrement)
    ↓
Whisper encoder (couche interne, pas de décodage texte)
    ↓
Vecteur 768 dimensions (représentation acoustique)
    ↓
Recherche FAISS (cosinus) dans les 1159 vecteurs du corpus
    ↓
Top-k segments les plus proches acoustiquement
```

**Pourquoi c'est intéressant :**
- Fonctionne **sans reconnaître les mots** — compare les patterns acoustiques
- Retrouve des prononciations similaires même pour des locuteurs différents
- 100% local, zéro API externe, zéro transcription

**Limites actuelles :**
- Les segments du corpus sont courts (3-4s) : une requête longue peut moins bien matcher
- Whisper n'est pas fine-tuné sur le mooré : les embeddings sont approximatifs
- Pour de meilleurs résultats, fais des requêtes courtes (1-5s)
            """)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Alertes
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Segments les plus critiques par dimension")
        dim = st.selectbox(
            "Dimension", list(DIM_LABELS.keys()),
            format_func=lambda x: DIM_LABELS[x]
        )
        top_crit = sorted(
            [s for s in segs
             if s.get("criticite") and dim in s["criticite"].get("scores", {})],
            key=lambda s: -s["criticite"]["scores"][dim]
        )[:top_k]
        for i, seg in enumerate(top_crit, 1):
            render_card(seg, rank=i, debug=debug)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Corpus
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader(f"Corpus complet — {len(segs)} segments")
        fn = st.selectbox("Niveau", ["Tous", "ROUGE", "ORANGE", "VERT"])
        ft = st.text_input("Filtrer texte", "")
        filtered = [
            s for s in segs
            if (fn == "Tous" or (s.get("criticite") and s["criticite"]["niveau"] == fn))
            and (not ft or ft.lower() in (s["transcription"] + s["translation_fr"]).lower())
        ]
        st.caption(f"{len(filtered)} segments")
        for seg in filtered[:50]:
            render_card(seg, debug=debug)
        if len(filtered) > 50:
            st.info("Limité à 50. Utilise le filtre pour affiner.")


if __name__ == "__main__":
    main()
