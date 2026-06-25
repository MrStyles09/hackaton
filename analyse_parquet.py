"""
ANALYSE DU DATASET PARQUET — CITADEL 2026
==========================================
Lance ce script sur ta machine locale :
    python analyse_parquet.py --file full-00000-of-00001.parquet

Il n'écrit rien, ne charge pas tout en mémoire, et te donne
exactement ce qu'il faut pour configurer le pipeline.
"""

import argparse
import sys

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import numpy as np
except ImportError:
    print("Installation manquante. Lance :")
    print("  pip install pandas pyarrow numpy")
    sys.exit(1)


def analyse(file_path: str, n_sample: int = 500):

    print(f"\n{'═'*55}")
    print(f"  ANALYSE : {file_path}")
    print(f"{'═'*55}\n")

    # ── 1. Métadonnées sans tout charger ──────────────────────
    pf = pq.ParquetFile(file_path)
    meta = pf.schema_arrow
    total_rows = pf.metadata.num_rows
    total_size_mb = pf.metadata.serialized_size / 1e6

    print(f"Lignes totales   : {total_rows:,}")
    print(f"Taille sérialisée: {total_size_mb:.1f} MB")
    print(f"\nColonnes ({len(meta.names)}) :")
    for name, dtype in zip(meta.names, meta.types):
        print(f"  [{dtype}]  {name}")

    # ── 2. Échantillon pour explorer les valeurs ───────────────
    print(f"\n{'─'*55}")
    print(f"Chargement d'un échantillon ({n_sample} lignes)...")
    df = pf.read_row_group(0).to_pandas()
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=42)

    print(f"\n── APERÇU DES COLONNES ──")
    for col in df.columns:
        dtype = df[col].dtype
        sample_vals = df[col].dropna().head(3).tolist()

        # Détecter les colonnes audio (bytes)
        if dtype == object:
            first = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first, (bytes, bytearray, dict)):
                print(f"\n  🎵 {col}  [AUDIO/BINAIRE]")
                if isinstance(first, dict):
                    print(f"     Clés du dict : {list(first.keys())}")
                    # Cas HuggingFace : {'bytes': b'...', 'path': '...'}
                    if 'bytes' in first and first['bytes']:
                        print(f"     Taille sample audio : {len(first['bytes']):,} bytes")
                    if 'path' in first:
                        print(f"     Chemin sample : {first['path']}")
                else:
                    print(f"     Taille sample : {len(first):,} bytes")
                continue

        print(f"\n  📝 {col}  [{dtype}]")
        for v in sample_vals:
            preview = str(v)[:80].replace('\n', ' ')
            print(f"     → {preview}")

    # ── 3. Stats sur les colonnes texte ───────────────────────
    text_cols = [c for c in df.columns
                 if df[c].dtype == object
                 and not isinstance(df[c].dropna().iloc[0] if not df[c].dropna().empty else None,
                                    (bytes, bytearray, dict))]

    if text_cols:
        print(f"\n── STATS TEXTE ──")
        for col in text_cols:
            lengths = df[col].dropna().str.len()
            print(f"  {col} : min={lengths.min():.0f}  moy={lengths.mean():.0f}  "
                  f"max={lengths.max():.0f} chars  nulls={df[col].isna().sum()}")

    # ── 4. Détecter la colonne audio et estimer les durées ────
    print(f"\n── ESTIMATION DURÉES AUDIO ──")
    audio_col = None
    for col in df.columns:
        first = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if isinstance(first, (bytes, bytearray)):
            audio_col = col
            break
        if isinstance(first, dict) and 'bytes' in first:
            audio_col = col
            break

    if audio_col:
        print(f"  Colonne audio détectée : '{audio_col}'")
        try:
            import io, wave, struct

            durations = []
            for val in df[audio_col].dropna().head(100):
                raw = val['bytes'] if isinstance(val, dict) else val
                if not raw:
                    continue
                try:
                    # Essai WAV header
                    with io.BytesIO(raw) as buf:
                        with wave.open(buf) as wf:
                            dur = wf.getnframes() / wf.getframerate()
                            durations.append(dur)
                except Exception:
                    # Estimer depuis la taille (approximatif, ~16kHz 16bit mono)
                    est = len(raw) / (16000 * 2)
                    durations.append(est)

            if durations:
                print(f"  Durée min  : {min(durations):.2f}s")
                print(f"  Durée moy  : {np.mean(durations):.2f}s")
                print(f"  Durée max  : {max(durations):.2f}s")
                print(f"  Médiane    : {np.median(durations):.2f}s")
                short = sum(1 for d in durations if d < 5)
                print(f"  < 5s       : {short}/{len(durations)} ({short/len(durations)*100:.0f}%)")
                long_ = sum(1 for d in durations if d > 30)
                print(f"  > 30s      : {long_}/{len(durations)} ({long_/len(durations)*100:.0f}%)")

                # Extrapoler au dataset complet
                total_audio_h = np.mean(durations) * total_rows / 3600
                print(f"\n  ⏱ Estimation dataset complet : ~{total_audio_h:.1f}h d'audio")
                emb_time_h = np.mean(durations) * total_rows / 0.6 / 3600
                print(f"  ⏳ Temps embeddings Whisper CPU : ~{emb_time_h:.0f}h "
                      f"(prendre un sous-ensemble !)")
        except Exception as e:
            print(f"  ⚠ Impossible d'estimer les durées : {e}")
    else:
        print("  ⚠ Aucune colonne audio binaire détectée.")
        print("    Les audios sont peut-être des chemins de fichiers (colonne path).")

    # ── 5. Recommandations ────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  RECOMMANDATIONS PIPELINE")
    print(f"{'═'*55}")
    print(f"""
1. EXTRAIRE les audios du Parquet vers des .wav :
   → Utilise le script extract_from_parquet.py (généré séparément)
   → Cible : 2 000-3 000 fichiers pour le hackathon

2. FORMAT TSV à produire pour module1 --parallel :
   audio_file \\t texte_moore \\t traduction_fr

3. Si la colonne texte est en mooré seulement :
   → module5 utilisera LaBSE sur le mooré directement (OK)
   → Si traduction FR disponible, résultats criticité meilleurs

4. Commande de démarrage après extraction :
   python module1_ingestion_v2.py \\
     --parallel \\
     --input ./audios_extraits/ \\
     --text-file ./transcriptions.tsv \\
     --out corpus/segments \\
     --db corpus/metadata.db \\
     --json corpus/segments.json
""")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Chemin vers le fichier .parquet")
    parser.add_argument("--sample", type=int, default=500, help="Taille de l'échantillon")
    args = parser.parse_args()
    analyse(args.file, args.sample)
