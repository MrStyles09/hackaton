"""
EXTRACTION DATASET MOORÉ → WAV + TSV  (v3 — noms uniques garantis)
====================================================================
Chaque fichier WAV reçoit un préfixe numérique global (00001_, 00002_...)
qui garantit l'unicité même si les noms originaux se répètent entre chapitres.

Usage :
    python extract_dataset.py --file full-00000-of-00001.parquet --both-audios
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


def flac_bytes_to_wav(raw_bytes: bytes, sampling_rate: int = 16000):
    try:
        import soundfile as sf
        buf_in  = io.BytesIO(raw_bytes)
        audio, sr = sf.read(buf_in, dtype='int16')
        buf_out = io.BytesIO()
        sf.write(buf_out, audio, sr, format='WAV', subtype='PCM_16')
        return buf_out.getvalue()
    except ImportError:
        return None
    except Exception:
        return None


def save_audio(raw_bytes: bytes, out_path: str) -> bool:
    if not raw_bytes:
        return False
    wav_bytes = flac_bytes_to_wav(raw_bytes)
    if wav_bytes:
        with open(out_path, 'wb') as f:
            f.write(wav_bytes)
        return True
    flac_path = out_path.replace('.wav', '.flac')
    with open(flac_path, 'wb') as f:
        f.write(raw_bytes)
    return True


def extract(file_path, out_dir_moore, out_dir_french, out_tsv,
            both_audios, min_dur, max_dur):

    os.makedirs(out_dir_moore, exist_ok=True)
    if both_audios:
        os.makedirs(out_dir_french, exist_ok=True)

    try:
        import soundfile
        ext = '.wav'
        print("OK soundfile disponible -> conversion FLAC->WAV activee")
    except ImportError:
        ext = '.flac'
        print("ATTENTION soundfile absent -> audios en .flac  (pip install soundfile)")

    pf    = pq.ParquetFile(file_path)
    total = pf.metadata.num_rows
    print(f"Dataset : {total} lignes\n")

    tsv_lines   = ["fichier_moore\ttexte_moore\ttraduction_fr\tfichier_french\tduration_moore\tchapter"]
    extracted   = 0
    skipped_dur = 0
    skipped_err = 0

    for batch in pf.iter_batches(batch_size=200):
        df = batch.to_pandas()

        for _, row in df.iterrows():
            try:
                dur_moore = float(row.get('duration_moore') or 0)

                if dur_moore < min_dur or dur_moore > max_dur:
                    skipped_dur += 1
                    continue

                moore_audio = row.get('moore_audio')
                if not moore_audio or not isinstance(moore_audio, dict):
                    skipped_err += 1
                    continue

                raw_moore = moore_audio.get('bytes')
                path_hint = moore_audio.get('path', f'seg_{extracted:05d}.flac')
                base_name = os.path.splitext(os.path.basename(path_hint))[0]

                # Nom unique : index global + nom original
                idx_str        = f"{extracted:05d}"
                filename_moore = f"{idx_str}_{base_name}_moore{ext}"
                out_path_moore = os.path.join(out_dir_moore, filename_moore)

                if not save_audio(raw_moore, out_path_moore):
                    skipped_err += 1
                    continue

                filename_french = ''
                if both_audios:
                    french_audio = row.get('french_audio')
                    if french_audio and isinstance(french_audio, dict):
                        raw_french      = french_audio.get('bytes')
                        filename_french = f"{idx_str}_{base_name}_french{ext}"
                        save_audio(raw_french, os.path.join(out_dir_french, filename_french))

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
                          f"(ignores duree: {skipped_dur}, erreurs: {skipped_err})...")

            except Exception as e:
                skipped_err += 1
                if skipped_err <= 3:
                    print(f"  Erreur : {e}")

    with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))

    print(f"""
===================================================
  EXTRACTION TERMINEE
===================================================
  Fichiers moore extraits : {extracted}
  Ignores (duree hors [{min_dur}s-{max_dur}s]) : {skipped_dur}
  Erreurs                 : {skipped_err}
  TSV produit             : {out_tsv}
  Format audio            : {ext}
===================================================

Verification rapide (PowerShell) :
  (Get-ChildItem {out_dir_moore} -Filter *{ext}).Count   # doit etre {extracted}

Prochaine etape :
  python module1_ingestion_v2.py --parallel --input {out_dir_moore} --text-file {out_tsv} --col-moore 1 --col-french 2 --out corpus/segments --db corpus/metadata.db --json corpus/segments.json
""")

    print("-- Apercu TSV (3 premieres lignes) --")
    for line in tsv_lines[1:4]:
        cols = line.split('\t')
        print(f"  [{cols[5]}]  {cols[0]}")
        print(f"    moore   : {cols[1][:70]}")
        print(f"    francais: {cols[2][:70]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",        required=True)
    parser.add_argument("--out-moore",   default="./audios_moore/")
    parser.add_argument("--out-french",  default="./audios_french/")
    parser.add_argument("--out-tsv",     default="./transcriptions.tsv")
    parser.add_argument("--both-audios", action="store_true")
    parser.add_argument("--min-dur",     type=float, default=1.0)
    parser.add_argument("--max-dur",     type=float, default=60.0)
    args = parser.parse_args()

    extract(args.file, args.out_moore, args.out_french, args.out_tsv,
            args.both_audios, args.min_dur, args.max_dur)
