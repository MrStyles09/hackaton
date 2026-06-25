"""
EXTRACTION AUDIOS + TSV DEPUIS LE PARQUET — CITADEL 2026
=========================================================
Lance APRÈS analyse_parquet.py, une fois que tu sais les noms de colonnes.

Usage :
    # Cas HuggingFace standard (audio dict + sentence + translation)
    python extract_from_parquet.py \\
        --file full-00000-of-00001.parquet \\
        --col-audio audio \\
        --col-moore sentence \\
        --col-french translation \\
        --out-dir ./audios/ \\
        --out-tsv ./transcriptions.tsv \\
        --max 3000

    # Adapter --col-moore et --col-french aux vrais noms de colonnes
    # détectés par analyse_parquet.py
"""

import argparse
import io
import os
import sys
import wave

try:
    import pyarrow.parquet as pq
    import numpy as np
except ImportError:
    print("pip install pyarrow numpy")
    sys.exit(1)


def extract(file_path: str,
            col_audio: str,
            col_moore: str,
            col_french: str,
            out_dir: str,
            out_tsv: str,
            max_files: int,
            min_dur: float,
            max_dur: float):

    os.makedirs(out_dir, exist_ok=True)

    pf = pq.ParquetFile(file_path)
    total_rows = pf.metadata.num_rows
    print(f"Dataset : {total_rows:,} lignes")
    print(f"Extraction : max {max_files} fichiers vers {out_dir}/\n")

    extracted = 0
    skipped_dur = 0
    skipped_err = 0
    tsv_lines = ["fichier\ttexte_moore\ttraduction_fr"]

    # Lire par batch pour ne pas tout charger en RAM
    batch_size = 500
    for batch in pf.iter_batches(batch_size=batch_size):
        if extracted >= max_files:
            break

        df = batch.to_pandas()

        for _, row in df.iterrows():
            if extracted >= max_files:
                break

            try:
                # ── Récupérer l'audio ──────────────────────────
                audio_val = row.get(col_audio)
                if audio_val is None:
                    skipped_err += 1
                    continue

                # Formats possibles selon la source HuggingFace
                if isinstance(audio_val, dict):
                    raw_bytes = audio_val.get('bytes') or audio_val.get('array')
                    path_hint = audio_val.get('path', '')
                    sampling_rate = audio_val.get('sampling_rate', 16000)
                elif isinstance(audio_val, (bytes, bytearray)):
                    raw_bytes = audio_val
                    path_hint = ''
                    sampling_rate = 16000
                else:
                    skipped_err += 1
                    continue

                if raw_bytes is None or len(raw_bytes) == 0:
                    skipped_err += 1
                    continue

                # ── Estimer la durée ───────────────────────────
                dur = None
                audio_array = None

                if isinstance(raw_bytes, np.ndarray):
                    # Cas où HuggingFace donne directement un array numpy
                    audio_array = raw_bytes.astype(np.float32)
                    dur = len(audio_array) / sampling_rate
                else:
                    try:
                        with io.BytesIO(raw_bytes) as buf:
                            with wave.open(buf) as wf:
                                dur = wf.getnframes() / wf.getframerate()
                    except Exception:
                        dur = len(raw_bytes) / (16000 * 2)  # estimation

                if dur < min_dur or dur > max_dur:
                    skipped_dur += 1
                    continue

                # ── Nom du fichier de sortie ───────────────────
                filename = f"audio_{extracted:05d}.wav"
                if path_hint:
                    base = os.path.splitext(os.path.basename(path_hint))[0]
                    filename = f"{base}.wav"

                out_path = os.path.join(out_dir, filename)

                # ── Écrire le WAV ──────────────────────────────
                if audio_array is not None:
                    # Array numpy → WAV 16bit
                    import struct
                    sr = sampling_rate
                    pcm = (audio_array * 32767).astype(np.int16)
                    with wave.open(out_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(pcm.tobytes())
                elif isinstance(raw_bytes, (bytes, bytearray)):
                    # Bytes WAV → écriture directe
                    try:
                        with io.BytesIO(raw_bytes) as buf:
                            wave.open(buf)  # valider que c'est un WAV
                        with open(out_path, 'wb') as f:
                            f.write(raw_bytes)
                    except Exception:
                        # Pas un WAV → écrire quand même (MP3/OGG possible)
                        ext = '.mp3' if raw_bytes[:3] == b'ID3' else '.wav'
                        out_path = out_path.replace('.wav', ext)
                        filename = filename.replace('.wav', ext)
                        with open(out_path, 'wb') as f:
                            f.write(raw_bytes)

                # ── Textes ────────────────────────────────────
                text_moore  = str(row.get(col_moore,  '') or '').strip().replace('\t', ' ')
                text_french = str(row.get(col_french, '') or '').strip().replace('\t', ' ')

                tsv_lines.append(f"{filename}\t{text_moore}\t{text_french}")
                extracted += 1

                if extracted % 100 == 0:
                    print(f"  {extracted}/{max_files} extraits "
                          f"(ignorés durée: {skipped_dur}, erreurs: {skipped_err})...")

            except Exception as e:
                skipped_err += 1
                if skipped_err <= 5:
                    print(f"  ⚠ Erreur ligne {extracted + skipped_dur + skipped_err} : {e}")

    # ── Écrire le TSV ─────────────────────────────────────────
    with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))

    print(f"""
{'═'*50}
  EXTRACTION TERMINÉE
{'═'*50}
  Fichiers extraits  : {extracted}
  Ignorés (durée)    : {skipped_dur}
  Erreurs            : {skipped_err}
  TSV produit        : {out_tsv}
{'═'*50}

Prochaine étape :
  python module1_ingestion_v2.py \\
    --parallel \\
    --input {out_dir} \\
    --text-file {out_tsv} \\
    --out corpus/segments \\
    --db corpus/metadata.db \\
    --json corpus/segments.json
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",       required=True)
    parser.add_argument("--col-audio",  default="audio",
                        help="Nom de la colonne audio dans le Parquet")
    parser.add_argument("--col-moore",  default="sentence",
                        help="Nom de la colonne texte mooré")
    parser.add_argument("--col-french", default="translation",
                        help="Nom de la colonne traduction française")
    parser.add_argument("--out-dir",    default="./audios/")
    parser.add_argument("--out-tsv",    default="./transcriptions.tsv")
    parser.add_argument("--max",        type=int,   default=3000,
                        help="Nombre max de fichiers à extraire")
    parser.add_argument("--min-dur",    type=float, default=2.0,
                        help="Durée minimale en secondes")
    parser.add_argument("--max-dur",    type=float, default=60.0,
                        help="Durée maximale en secondes")
    args = parser.parse_args()

    extract(
        file_path  = args.file,
        col_audio  = args.col_audio,
        col_moore  = args.col_moore,
        col_french = args.col_french,
        out_dir    = args.out_dir,
        out_tsv    = args.out_tsv,
        max_files  = args.max,
        min_dur    = args.min_dur,
        max_dur    = args.max_dur,
    )
