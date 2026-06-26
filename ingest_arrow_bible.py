"""
INGESTION DATASET BIBLE MOORÉ (FORMAT ARROW)
=============================================
Traite les 30 shards Arrow (~15 Go) du dataset mooré sans transcription française.
Format attendu : audio mooré + transcription mooré (pas de traduction FR)

Usage :
    # D'abord voir la structure d'un shard
    python ingest_arrow_bible.py --diagnose --shard moore_audio_data-Train-00000-of-00030.arrow

    # Ingérer tous les shards (échantillon de 3000 pour le hackathon)
    python ingest_arrow_bible.py --dir /chemin/vers/shards/ --max 3000 --out-dir audios_bible/ --out-tsv transcriptions_bible.tsv
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


def diagnose_shard(shard_path: str):
    """Examine la structure d'un fichier Arrow sans tout charger."""
    print(f"\nDiagnostic : {shard_path}")
    print(f"Taille : {os.path.getsize(shard_path) // 1024 // 1024} Mo")

    with pa.memory_map(shard_path, 'r') as source:
        reader = ipc.open_file(source)
        schema = reader.schema_arrow
        n_batches = reader.num_record_batches
        total_rows = sum(reader.get_batch(i).num_rows for i in range(min(3, n_batches)))

        print(f"\nSchéma ({len(schema)} colonnes) :")
        for field in schema:
            print(f"  [{field.type}]  {field.name}")

        print(f"\nBatches : {n_batches}")
        print(f"Rows (estimé sur 3 premiers batches) : {total_rows}")

        # Aperçu du premier batch
        batch = reader.get_batch(0)
        df = batch.to_pandas()
        print(f"\nAperçu première ligne :")
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, (bytes, bytearray)):
                print(f"  {col} : <bytes {len(val)//1024} Ko>")
            elif isinstance(val, dict):
                print(f"  {col} : dict keys={list(val.keys())}")
                if 'bytes' in val and val['bytes']:
                    print(f"    → bytes size: {len(val['bytes'])//1024} Ko")
                if 'path' in val:
                    print(f"    → path: {val['path']}")
            else:
                preview = str(val)[:100]
                print(f"  {col} : {preview}")


def save_wav(raw_bytes: bytes, out_path: str) -> bool:
    """Convertit FLAC/audio bytes en WAV via soundfile."""
    try:
        import soundfile as sf
        buf_in = io.BytesIO(raw_bytes)
        audio, sr = sf.read(buf_in, dtype='int16', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1).astype('int16')
        with wave.open(out_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        return True
    except Exception as e:
        # Fallback : écrire en FLAC
        flac_path = out_path.replace('.wav', '.flac')
        with open(flac_path, 'wb') as f:
            f.write(raw_bytes)
        return True


def ingest_shards(shard_dir: str, out_dir: str, out_tsv: str,
                  max_files: int, col_audio: str, col_text: str):
    """Ingère tous les shards Arrow du répertoire."""
    os.makedirs(out_dir, exist_ok=True)

    shard_files = sorted(Path(shard_dir).glob("*.arrow"))
    if not shard_files:
        # Chercher aussi dans le répertoire courant
        shard_files = sorted(Path(shard_dir).glob("**/*.arrow"))

    print(f"Shards trouvés : {len(shard_files)}")
    if not shard_files:
        print(f"ERREUR : aucun fichier .arrow dans {shard_dir}")
        sys.exit(1)

    try:
        import soundfile
        ext = '.wav'
        print("✓ soundfile disponible → WAV")
    except ImportError:
        ext = '.flac'
        print("⚠ soundfile absent → FLAC (pip install soundfile)")

    tsv_lines = ["fichier_moore\ttexte_moore\ttraduction_fr\tshard"]
    extracted = 0
    skipped   = 0

    for shard_path in shard_files:
        if extracted >= max_files:
            break

        shard_name = shard_path.stem
        print(f"\nShard : {shard_path.name} ...", end=" ", flush=True)

        try:
            with pa.memory_map(str(shard_path), 'r') as source:
                reader = ipc.open_file(source)

                for b_idx in range(reader.num_record_batches):
                    if extracted >= max_files:
                        break

                    batch = reader.get_batch(b_idx)
                    df    = batch.to_pandas()

                    for _, row in df.iterrows():
                        if extracted >= max_files:
                            break

                        try:
                            # Récupérer l'audio
                            audio_val = row.get(col_audio)
                            if audio_val is None:
                                skipped += 1
                                continue

                            if isinstance(audio_val, dict):
                                raw = audio_val.get('bytes') or audio_val.get('array')
                                path_hint = audio_val.get('path', f'seg_{extracted:06d}.flac')
                                sr = audio_val.get('sampling_rate', 16000)
                            elif isinstance(audio_val, (bytes, bytearray)):
                                raw = audio_val
                                path_hint = f'seg_{extracted:06d}.flac'
                                sr = 16000
                            else:
                                skipped += 1
                                continue

                            if not raw or len(raw) < 100:
                                skipped += 1
                                continue

                            # Nom unique
                            base = os.path.splitext(os.path.basename(path_hint))[0]
                            filename = f"{extracted:06d}_{base}{ext}"
                            out_path = os.path.join(out_dir, filename)

                            # Sauvegarder
                            if isinstance(raw, np.ndarray):
                                import soundfile as sf
                                pcm = (raw * 32767).astype(np.int16)
                                buf = io.BytesIO()
                                sf.write(buf, pcm, sr, format='WAV', subtype='PCM_16')
                                with open(out_path, 'wb') as f:
                                    f.write(buf.getvalue())
                            else:
                                save_wav(bytes(raw), out_path)

                            # Texte
                            text_moore = str(row.get(col_text, '') or '').strip().replace('\t', ' ')

                            tsv_lines.append(
                                f"{filename}\t{text_moore}\t\t{shard_name}"
                            )
                            extracted += 1

                        except Exception as e:
                            skipped += 1
                            if skipped <= 3:
                                print(f"\n  ⚠ Erreur ligne : {e}")

            print(f"{extracted} extraits", end="")

        except Exception as e:
            print(f"ERREUR shard : {e}")

        if extracted % 500 == 0 and extracted > 0:
            print(f"\n  → {extracted}/{max_files} extraits au total")

    with open(out_tsv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))

    print(f"""
{'═'*55}
  INGESTION TERMINÉE
{'═'*55}
  Fichiers extraits : {extracted}
  Ignorés           : {skipped}
  TSV               : {out_tsv}
  Format            : {ext}
{'═'*55}

Prochaine étape — Module 1 :
  python module1_ingestion_v2.py --parallel \\
    --input {out_dir} \\
    --text-file {out_tsv} \\
    --col-moore 1 \\
    --col-french 2 \\
    --out corpus_bible/segments \\
    --db corpus_bible/metadata.db \\
    --json corpus_bible/segments.json

Note : col-french=2 est vide pour ce dataset (pas de traduction FR).
Le scoring criticité utilisera uniquement le mooré.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",       help="Dossier contenant les fichiers .arrow")
    parser.add_argument("--shard",     help="Un seul shard (pour --diagnose)")
    parser.add_argument("--diagnose",  action="store_true")
    parser.add_argument("--out-dir",   default="./audios_bible/")
    parser.add_argument("--out-tsv",   default="./transcriptions_bible.tsv")
    parser.add_argument("--max",       type=int, default=3000)
    parser.add_argument("--col-audio", default="audio",
                        help="Nom colonne audio (défaut: 'audio')")
    parser.add_argument("--col-text",  default="sentence",
                        help="Nom colonne texte mooré (défaut: 'sentence')")
    args = parser.parse_args()

    if args.diagnose:
        if not args.shard:
            print("--diagnose nécessite --shard <fichier.arrow>")
            sys.exit(1)
        diagnose_shard(args.shard)
    elif args.dir:
        ingest_shards(args.dir, args.out_dir, args.out_tsv,
                      args.max, args.col_audio, args.col_text)
    else:
        parser.print_help()
