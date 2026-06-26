"""
INGESTION BIBLE MOORÉ — FORMAT ARROW STREAM (v2)
=================================================
Format détecté : Arrow Stream (ipc.open_stream)
Structure      : audio{bytes, path} + text (mooré)
Sampling rate  : 24000 Hz → resample 16000 Hz
Taille/shard   : ~591 Mo · ~471 lignes · batches de 100

Usage :
    # Test sur un shard
    python ingest_arrow_bible.py --diagnose --shard "moore_audio_data-train-00000-of-00030.arrow"

    # Extraire 3000 fichiers depuis tous les shards
    python ingest_arrow_bible.py --dir . --max 3000 ^
      --out-dir "C:/Users/HP/.../datasets/audios_bible/" ^
      --out-tsv  "C:/Users/HP/.../datasets/transcriptions_bible.tsv"
"""

import argparse
import io
import os
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import numpy as np
except ImportError:
    print("pip install pyarrow numpy")
    sys.exit(1)


# ── Lecture Arrow stream ──────────────────────────────────────────────────────

def iter_shard(shard_path: str):
    """Itère les lignes d'un shard Arrow stream sans décoder l'audio."""
    with open(str(shard_path), 'rb') as f:
        reader = ipc.open_stream(f)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            for i in range(batch.num_rows):
                row = {}
                for col in batch.schema.names:
                    v = batch.column(col)[i].as_py()
                    row[col] = v
                yield row


# ── Conversion audio ──────────────────────────────────────────────────────────

def convert_to_wav(raw_bytes: bytes, orig_sr: int = 24000,
                   target_sr: int = 16000) -> bytes:
    """
    FLAC/WAV bytes 24kHz → WAV PCM 16kHz mono.
    Utilise soundfile + librosa (même pipeline que module2).
    """
    import soundfile as sf
    import librosa

    buf = io.BytesIO(raw_bytes)
    audio, sr = sf.read(buf, dtype='float32', always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    buf_out = io.BytesIO()
    sf.write(buf_out, audio.astype(np.float32), target_sr,
             format='WAV', subtype='PCM_16')
    return buf_out.getvalue()


# ── Diagnostic ────────────────────────────────────────────────────────────────

def diagnose(shard_path: str):
    print(f"\nDiagnostic : {shard_path}")
    print(f"Taille     : {os.path.getsize(shard_path)//1024//1024} Mo")

    rows = list(iter_shard(shard_path))
    print(f"Lignes     : {len(rows)}")

    if rows:
        r = rows[0]
        print("\nPremier exemple :")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k} : dict{list(v.keys())}")
                for dk, dv in v.items():
                    if isinstance(dv, (bytes, bytearray)):
                        print(f"    {dk} : {len(dv)//1024} Ko")
                    else:
                        print(f"    {dk} : {str(dv)[:80]}")
            elif isinstance(v, (bytes, bytearray)):
                print(f"  {k} : {len(v)//1024} Ko bytes")
            else:
                print(f"  {k} : {str(v)[:100]}")

    print("\nDiagnostic OK — lance l'ingestion avec --dir .")


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(shard_dir: str, out_dir: str, out_tsv: str,
           max_files: int, shard_pattern: str):

    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(Path(shard_dir).glob(shard_pattern))
    if not shards:
        shards = sorted(Path(shard_dir).glob("**/*.arrow"))
    print(f"Shards : {len(shards)}")

    try:
        import soundfile
        ext = '.wav'
        print("soundfile OK → WAV 16kHz")
    except ImportError:
        ext = '.flac'
        print("soundfile absent → bytes bruts .flac (pip install soundfile)")

    tsv_lines = ["fichier_moore\ttexte_moore\ttraduction_fr\tshard"]
    extracted = skipped = 0

    for shard_path in shards:
        if extracted >= max_files:
            break
        shard_name   = shard_path.stem[:30]
        shard_count  = 0

        try:
            for row in iter_shard(str(shard_path)):
                if extracted >= max_files:
                    break

                audio_val  = row.get('audio', {})
                text_moore = str(row.get('text', '') or '').strip().replace('\t', ' ')

                # Extraire bytes bruts
                if isinstance(audio_val, dict):
                    raw = audio_val.get('bytes', b'')
                    path_hint = audio_val.get('path', f'seg_{extracted:06d}.wav')
                else:
                    raw = audio_val if isinstance(audio_val, (bytes, bytearray)) else b''
                    path_hint = f'seg_{extracted:06d}.wav'

                if not raw or len(raw) < 1000:
                    skipped += 1
                    continue

                base     = os.path.splitext(os.path.basename(path_hint))[0]
                filename = f"{extracted:06d}_{base}{ext}"
                out_path = os.path.join(out_dir, filename)

                try:
                    if ext == '.wav':
                        wav = convert_to_wav(bytes(raw))
                        with open(out_path, 'wb') as f:
                            f.write(wav)
                    else:
                        with open(out_path, 'wb') as f:
                            f.write(bytes(raw))
                except Exception as e:
                    # Fallback : écrire brut
                    flac_path = out_path.replace('.wav', '.flac')
                    with open(flac_path, 'wb') as f:
                        f.write(bytes(raw))
                    filename = filename.replace('.wav', '.flac')

                tsv_lines.append(f"{filename}\t{text_moore}\t\t{shard_name}")
                extracted  += 1
                shard_count += 1

        except Exception as e:
            print(f"  ERREUR shard {shard_path.name} : {e}")

        pct = extracted / max_files * 100
        print(f"  {shard_path.name[:40]} : {shard_count} extraits "
              f"({extracted}/{max_files} total, {pct:.0f}%)")

    with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))

    print(f"""
{'='*55}
  INGESTION BIBLE MOORÉ TERMINÉE
{'='*55}
  Fichiers extraits : {extracted}
  Ignorés           : {skipped}
  TSV               : {out_tsv}
  Format            : {ext} · 16000 Hz
{'='*55}

Prochaine étape dans datasets/ :
  python module1_ingestion_v2.py --parallel ^
    --input {out_dir} ^
    --text-file {out_tsv} ^
    --col-moore 1 --col-french 2 ^
    --out corpus_bible/segments ^
    --db corpus_bible/metadata.db ^
    --json corpus_bible/segments.json

Puis Module 2 (embeddings Whisper) :
  python module2_embeddings.py ^
    --json corpus_bible/segments.json ^
    --model whisper ^
    --db corpus_bible/metadata.db ^
    --out index_bible/embeddings.npy
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",         help="Shard unique (.arrow)")
    parser.add_argument("--dir",           help="Dossier contenant les shards")
    parser.add_argument("--diagnose",      action="store_true")
    parser.add_argument("--out-dir",       default="./audios_bible/")
    parser.add_argument("--out-tsv",       default="./transcriptions_bible.tsv")
    parser.add_argument("--max",           type=int, default=3000)
    parser.add_argument("--shard-pattern", default="*.arrow")
    args = parser.parse_args()

    if args.diagnose:
        target = args.shard
        if not target and args.dir:
            found = list(Path(args.dir).glob("*.arrow"))
            target = str(found[0]) if found else None
        if not target:
            print("Spécifie --shard <fichier.arrow>")
            sys.exit(1)
        diagnose(target)

    elif args.dir or args.shard:
        d = args.dir or str(Path(args.shard).parent)
        ingest(d, args.out_dir, args.out_tsv, args.max, args.shard_pattern)

    else:
        parser.print_help()
