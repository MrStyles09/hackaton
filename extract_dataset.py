"""
EXTRACTION DATASET MOORÉ → WAV + TSV
======================================
Adapté exactement au format détecté :
  moore_audio   : struct{bytes: binary, path: string}  (FLAC)
  moore_text    : string
  french_text   : string
  french_audio  : struct{bytes: binary, path: string}  (FLAC)
  duration_moore: float
  duration_french: float
  chapter       : string

Produit :
  ./audios_moore/   → fichiers WAV mooré
  ./audios_french/  → fichiers WAV français (optionnel)
  ./transcriptions.tsv → TSV pour module1 --parallel

Usage :
    python extract_dataset.py --file full-00000-of-00001.parquet
    python extract_dataset.py --file full-00000-of-00001.parquet --both-audios
"""

import argparse
import io
import os
import sys
import wave
import struct

try:
    import pyarrow.parquet as pq
    import numpy as np
except ImportError:
    print("pip install pyarrow numpy")
    sys.exit(1)


def flac_bytes_to_wav(raw_bytes: bytes) -> bytes:
    """
    Convertit des bytes FLAC en WAV via soundfile si disponible,
    sinon tente une écriture directe (si déjà WAV).
    """
    try:
        import soundfile as sf
        buf_in  = io.BytesIO(raw_bytes)
        audio, sr = sf.read(buf_in, dtype='int16')
        buf_out = io.BytesIO()
        sf.write(buf_out, audio, sr, format='WAV', subtype='PCM_16')
        return buf_out.getvalue()
    except ImportError:
        # soundfile absent : écrire le FLAC tel quel avec extension .flac
        return None
    except Exception as e:
        return None


def save_audio(raw_bytes: bytes, out_path: str) -> bool:
    """
    Sauvegarde l'audio.
    Essaie WAV d'abord, replie sur FLAC si soundfile absent.
    """
    if not raw_bytes:
        return False

    # Tentative conversion WAV
    wav_bytes = flac_bytes_to_wav(raw_bytes)
    if wav_bytes:
        with open(out_path, 'wb') as f:
            f.write(wav_bytes)
        return True
    else:
        # Écrire en FLAC directement
        flac_path = out_path.replace('.wav', '.flac')
        with open(flac_path, 'wb') as f:
            f.write(raw_bytes)
        return True


def extract(file_path: str, out_dir_moore: str, out_dir_french: str,
            out_tsv: str, both_audios: bool, min_dur: float, max_dur: float):

    os.makedirs(out_dir_moore, exist_ok=True)
    if both_audios:
        os.makedirs(out_dir_french, exist_ok=True)

    # Vérifier si soundfile est disponible
    try:
        import soundfile
        has_soundfile = True
        print("✓ soundfile disponible → conversion FLAC→WAV activée")
    except ImportError:
        has_soundfile = False
        print("⚠ soundfile absent → audios sauvegardés en .flac")
        print("  Pour installer : pip install soundfile")

    pf  = pq.ParquetFile(file_path)
    total = pf.metadata.num_rows
    print(f"\nDataset : {total} lignes")
    print(f"Extraction vers {out_dir_moore}/\n")

    tsv_lines   = ["fichier_moore\ttexte_moore\ttraduction_fr\tfichier_french\tduration_moore\tchapter"]
    extracted   = 0
    skipped_dur = 0
    skipped_err = 0

    ext = '.wav' if has_soundfile else '.flac'

    for batch in pf.iter_batches(batch_size=200):
        df = batch.to_pandas()

        for idx, row in df.iterrows():
            try:
                dur_moore = float(row.get('duration_moore', 0) or 0)

                # Filtre durée
                if dur_moore < min_dur or dur_moore > max_dur:
                    skipped_dur += 1
                    continue

                # Audio mooré
                moore_audio = row.get('moore_audio')
                if not moore_audio or not isinstance(moore_audio, dict):
                    skipped_err += 1
                    continue

                raw_moore = moore_audio.get('bytes')
                path_hint = moore_audio.get('path', f'seg_{extracted:05d}.flac')
                base_name = os.path.splitext(os.path.basename(path_hint))[0]

                filename_moore  = f"{base_name}_moore{ext}"
                out_path_moore  = os.path.join(out_dir_moore, filename_moore)

                if not save_audio(raw_moore, out_path_moore):
                    skipped_err += 1
                    continue

                # Audio français (optionnel)
                filename_french = ''
                if both_audios:
                    french_audio = row.get('french_audio')
                    if french_audio and isinstance(french_audio, dict):
                        raw_french = french_audio.get('bytes')
                        filename_french = f"{base_name}_french{ext}"
                        out_path_french = os.path.join(out_dir_french, filename_french)
                        save_audio(raw_french, out_path_french)

                # Textes
                moore_text  = str(row.get('moore_text',  '') or '').strip().replace('\t', ' ')
                french_text = str(row.get('french_text', '') or '').strip().replace('\t', ' ')
                chapter     = str(row.get('chapter',     '') or '').strip()

                tsv_lines.append(
                    f"{filename_moore}\t{moore_text}\t{french_text}"
                    f"\t{filename_french}\t{dur_moore:.2f}\t{chapter}"
                )
                extracted += 1

                if extracted % 100 == 0:
                    print(f"  {extracted}/{total} extraits "
                          f"(ignorés durée: {skipped_dur}, erreurs: {skipped_err})...")

            except Exception as e:
                skipped_err += 1
                if skipped_err <= 3:
                    print(f"  ⚠ Erreur ligne {idx} : {e}")

    # Écrire le TSV
    with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))

    print(f"""
{'═'*55}
  EXTRACTION TERMINÉE
{'═'*55}
  Fichiers mooré extraits : {extracted}
  Ignorés (durée)         : {skipped_dur}
  Erreurs                 : {skipped_err}
  TSV produit             : {out_tsv}
  Format audio            : {ext}
{'═'*55}

ÉTAPE SUIVANTE — colle cette commande :

  python module1_ingestion_v2.py \\
    --parallel \\
    --input {out_dir_moore} \\
    --text-file {out_tsv} \\
    --col-moore 1 \\
    --col-french 2 \\
    --out corpus/segments \\
    --db corpus/metadata.db \\
    --json corpus/segments.json
""")

    # Afficher 3 exemples du TSV
    print("── Aperçu TSV (3 premières lignes) ──")
    for line in tsv_lines[1:4]:
        cols = line.split('\t')
        print(f"  fichier : {cols[0]}")
        print(f"  mooré   : {cols[1][:60]}...")
        print(f"  français: {cols[2][:60]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction dataset mooré depuis Parquet")
    parser.add_argument("--file",        required=True, help="Chemin vers le .parquet")
    parser.add_argument("--out-moore",   default="./audios_moore/")
    parser.add_argument("--out-french",  default="./audios_french/")
    parser.add_argument("--out-tsv",     default="./transcriptions.tsv")
    parser.add_argument("--both-audios", action="store_true",
                        help="Extraire aussi les audios français")
    parser.add_argument("--min-dur",     type=float, default=1.0,
                        help="Durée min mooré (s) — défaut 1.0s")
    parser.add_argument("--max-dur",     type=float, default=60.0,
                        help="Durée max mooré (s) — défaut 60s")
    args = parser.parse_args()

    extract(
        file_path      = args.file,
        out_dir_moore  = args.out_moore,
        out_dir_french = args.out_french,
        out_tsv        = args.out_tsv,
        both_audios    = args.both_audios,
        min_dur        = args.min_dur,
        max_dur        = args.max_dur,
    )
