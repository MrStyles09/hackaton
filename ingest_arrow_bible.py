"""
INGESTION DATASET BIBLE MOORÉ (FORMAT ARROW HuggingFace)
=========================================================
Lit les bytes audio bruts SANS passer par torchcodec/FFmpeg.
Structure détectée : 471 lignes/shard, colonnes 'audio' + 'text', 24kHz.

Usage :
    # Diagnostic (vérifie la structure sans décoder l'audio)
    python ingest_arrow_bible.py --diagnose --shard moore_audio_data-train-00000-of-00030.arrow

    # Extraire 3000 fichiers depuis tous les shards
    python ingest_arrow_bible.py --dir . --max 3000 --out-dir audios_bible/ --out-tsv transcriptions_bible.tsv

    # Extraire depuis un seul shard (test)
    python ingest_arrow_bible.py --dir . --shard-pattern "*00000*" --max 471
"""

import argparse
import io
import os
import sys
import wave
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import numpy as np
except ImportError:
    print("pip install pyarrow numpy")
    sys.exit(1)


def read_shard_raw(shard_path: str):
    """
    Lit un fichier Arrow en accédant aux bytes bruts de l'audio
    SANS passer par le décodeur HuggingFace (évite torchcodec/FFmpeg).
    Retourne un itérateur de dicts {'audio_bytes': bytes, 'text': str}.
    """
    with pa.memory_map(str(shard_path), 'r') as source:
        reader = ipc.open_file(source)
        for b_idx in range(reader.num_record_batches):
            batch = reader.get_batch(b_idx)
            n = batch.num_rows
            for i in range(n):
                row = {}
                for col in batch.schema.names:
                    val = batch.column(col)[i]
                    # Extraire la valeur Python native (sans décodage audio HF)
                    if hasattr(val, 'as_py'):
                        row[col] = val.as_py()
                    else:
                        row[col] = val
                yield row


def extract_audio_bytes(audio_val) -> bytes:
    """
    Extrait les bytes bruts depuis la valeur audio HuggingFace Arrow.
    Format Arrow HF audio : struct avec champ 'bytes' (bytes FLAC/MP3/WAV).
    """
    if audio_val is None:
        return b''
    if isinstance(audio_val, (bytes, bytearray)):
        return bytes(audio_val)
    if isinstance(audio_val, dict):
        # Format HuggingFace : {'bytes': b'...', 'path': '...'}
        raw = audio_val.get('bytes') or audio_val.get('array')
        if raw is not None:
            return bytes(raw) if isinstance(raw, (bytearray, memoryview)) else raw
    return b''


def convert_to_wav(audio_bytes: bytes, target_sr: int = 16000) -> bytes:
    """
    Convertit bytes audio (FLAC/MP3/WAV à 24kHz) en WAV 16kHz mono.
    Utilise soundfile + librosa pour le rééchantillonnage.
    """
    try:
        import soundfile as sf
        import librosa

        buf = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buf, dtype='float32', always_2d=False)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Rééchantillonnage 24kHz → 16kHz
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Normalisation
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        # Écrire en WAV
        buf_out = io.BytesIO()
        sf.write(buf_out, audio.astype(np.float32), sr, format='WAV', subtype='PCM_16')
        return buf_out.getvalue()

    except Exception as e:
        # Fallback : écrire les bytes bruts tels quels
        return audio_bytes


def diagnose(shard_path: str):
    """Examine la structure sans décoder l'audio."""
    print(f"\nDiagnostic (raw) : {shard_path}")
    print(f"Taille : {os.path.getsize(shard_path)//1024//1024} Mo")

    with pa.memory_map(str(shard_path), 'r') as source:
        reader = ipc.open_file(source)
        total = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        print(f"Lignes totales : {total}")
        print(f"Batches : {reader.num_record_batches}")
        print(f"Colonnes : {reader.schema_arrow.names}")
        print(f"Schéma :")
        for field in reader.schema_arrow:
            print(f"  [{field.type}]  {field.name}")

    # Premier exemple (raw)
    print("\nPremier exemple (raw) :")
    for row in read_shard_raw(shard_path):
        for k, v in row.items():
            if isinstance(v, (bytes, bytearray)):
                print(f"  {k} : <bytes {len(v)//1024} Ko>")
            elif isinstance(v, dict):
                print(f"  {k} : dict keys={list(v.keys())}")
                for dk, dv in v.items():
                    if isinstance(dv, (bytes, bytearray)):
                        print(f"    {dk} : <bytes {len(dv)//1024} Ko>")
                    else:
                        preview = str(dv)[:80]
                        print(f"    {dk} : {preview}")
            else:
                preview = str(v)[:100]
                print(f"  {k} : {preview}")
        break

    print("\nDiagnostic terminé. Lance l'ingestion avec --dir .")


def ingest(shard_dir: str, out_dir: str, out_tsv: str,
           max_files: int, shard_pattern: str,
           col_audio: str, col_text: str):
    """Extrait les audios et transcriptions de tous les shards."""
    os.makedirs(out_dir, exist_ok=True)

    shard_files = sorted(Path(shard_dir).glob(shard_pattern))
    if not shard_files:
        shard_files = sorted(Path(shard_dir).glob("**/*.arrow"))

    print(f"Shards trouvés : {len(shard_files)}")
    if not shard_files:
        print(f"ERREUR : aucun .arrow dans {shard_dir}")
        sys.exit(1)

    try:
        import soundfile
        has_sf = True
        ext = '.wav'
        print("soundfile disponible → conversion 24kHz→16kHz WAV")
    except ImportError:
        has_sf = False
        ext = '.flac'
        print("soundfile absent → bytes bruts (pip install soundfile pour WAV)")

    tsv_lines = ["fichier_moore\ttexte_moore\ttraduction_fr\tshard"]
    extracted = 0
    skipped   = 0

    for shard_path in shard_files:
        if extracted >= max_files:
            break

        shard_name = shard_path.stem
        shard_extracted = 0

        try:
            for row in read_shard_raw(str(shard_path)):
                if extracted >= max_files:
                    break

                audio_val  = row.get(col_audio)
                text_moore = str(row.get(col_text, '') or '').strip().replace('\t', ' ')

                raw_bytes = extract_audio_bytes(audio_val)
                if not raw_bytes or len(raw_bytes) < 500:
                    skipped += 1
                    continue

                # Nom unique
                filename = f"{extracted:06d}_{shard_name[:20]}{ext}"
                out_path = os.path.join(out_dir, filename)

                # Convertir et sauvegarder
                if has_sf:
                    wav_bytes = convert_to_wav(raw_bytes, target_sr=16000)
                    with open(out_path, 'wb') as f:
                        f.write(wav_bytes)
                else:
                    # Écrire brut avec extension adaptée
                    flac_path = out_path.replace('.wav', '.flac')
                    with open(flac_path, 'wb') as f:
                        f.write(raw_bytes)
                    filename = filename.replace('.wav', '.flac')
                    out_path = flac_path

                tsv_lines.append(f"{filename}\t{text_moore}\t\t{shard_name}")
                extracted += 1
                shard_extracted += 1

            print(f"  {shard_path.name} : {shard_extracted} extraits")

        except Exception as e:
            print(f"  ERREUR {shard_path.name} : {e}")

        if extracted % 500 == 0 and extracted > 0:
            print(f"  → Total : {extracted}/{max_files}")

    with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))

    est_total = len(shard_files) * (extracted / max(len(shard_files), 1))

    print(f"""
{'='*55}
  INGESTION TERMINÉE
{'='*55}
  Shards traités    : {len(shard_files)}
  Fichiers extraits : {extracted}
  Ignorés           : {skipped}
  TSV               : {out_tsv}
  Format audio      : {ext}
  Fréq. originale   : 24000 Hz → resample 16000 Hz
{'='*55}

Estimation dataset complet (~30 shards × 471) : ~14 130 segments

Prochaine étape :
  python module1_ingestion_v2.py --parallel \\
    --input {out_dir} \\
    --text-file {out_tsv} \\
    --col-moore 1 \\
    --col-french 2 \\
    --out corpus_bible/segments \\
    --db corpus_bible/metadata.db \\
    --json corpus_bible/segments.json
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Bible mooré Arrow")
    parser.add_argument("--shard",         help="Chemin vers un seul shard .arrow")
    parser.add_argument("--dir",           help="Dossier contenant les shards .arrow")
    parser.add_argument("--diagnose",      action="store_true")
    parser.add_argument("--out-dir",       default="./audios_bible/")
    parser.add_argument("--out-tsv",       default="./transcriptions_bible.tsv")
    parser.add_argument("--max",           type=int, default=3000)
    parser.add_argument("--shard-pattern", default="*.arrow")
    parser.add_argument("--col-audio",     default="audio")
    parser.add_argument("--col-text",      default="text")
    args = parser.parse_args()

    if args.diagnose:
        target = args.shard or (list(Path(args.dir or '.').glob("*.arrow"))[0]
                                if args.dir else None)
        if not target:
            print("Spécifie --shard ou --dir pour le diagnostic")
            sys.exit(1)
        diagnose(str(target))
    elif args.dir or args.shard:
        d = args.dir or str(Path(args.shard).parent)
        pat = f"*{Path(args.shard).name}*" if args.shard and not args.dir else args.shard_pattern
        ingest(d, args.out_dir, args.out_tsv,
               args.max, pat, args.col_audio, args.col_text)
    else:
        parser.print_help()
