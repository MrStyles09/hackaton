"""
MODULE 1 — Ingestion & Segmentation Audio
==========================================
Charge un fichier audio (WAV/MP3/OGG/M4A), applique une détection
d'activité vocale (VAD), découpe en segments de 5-30 secondes et
produit un JSON de métadonnées horodatées.

PATCHÉ v2 :
  - SEGMENT_MIN_SEC abaissé à 5s (messages WhatsApp courts)
  - Fichiers courts (<SEGMENT_MIN_SEC) conservés tels quels au lieu d'être dropés
  - Logs enrichis : avertissement explicite si segment trop court conservé
  - process_directory : retourne aussi les stats de rejet

Usage :
    python module1_ingestion.py --input corpus/message.wav --out corpus/
    python module1_ingestion.py --input corpus/ --out corpus/  # traitement batch
"""

import argparse
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


# ─── Constantes ────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16_000   # Hz requis par Whisper et wav2vec2
SEGMENT_MIN_SEC    = 5.0      # durée minimale d'un segment (abaissé de 10→5s pour WhatsApp)
SEGMENT_MAX_SEC    = 30.0     # durée maximale d'un segment
SILENCE_THRESHOLD  = 0.01     # énergie RMS en dessous = silence
SILENCE_MIN_SEC    = 0.3      # pause de silence déclenchant une coupure
SUPPORTED_FORMATS  = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".opus"}


# ─── VAD maison (frugale, sans modèle externe) ─────────────────────────────────

def vad_simple(audio: np.ndarray, sr: int, threshold: float = SILENCE_THRESHOLD,
               min_silence_sec: float = SILENCE_MIN_SEC) -> List[tuple]:
    """
    Voice Activity Detection basée sur l'énergie RMS par frames.
    Retourne une liste de (start_sample, end_sample) pour les régions vocales.
    Frugale : zéro dépendance externe, tourne sur CPU en millisecondes.
    """
    frame_length = int(sr * 0.025)   # 25 ms par frame
    hop_length   = int(sr * 0.010)   # 10 ms de saut

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    voiced = rms > threshold

    min_frames = int(min_silence_sec * sr / hop_length)
    silence_run = 0
    result = []
    in_speech = False
    seg_start = 0

    for i, v in enumerate(voiced):
        if v:
            if not in_speech:
                in_speech = True
                seg_start = i
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_frames:
                    result.append((seg_start * hop_length, i * hop_length))
                    in_speech = False
                    silence_run = 0

    if in_speech:
        result.append((seg_start * hop_length, len(audio)))

    return result


def merge_short_segments(segments: List[tuple], sr: int,
                          min_sec: float = SEGMENT_MIN_SEC,
                          max_sec: float = SEGMENT_MAX_SEC) -> List[tuple]:
    """
    Fusionne les segments trop courts et découpe les trop longs.
    PATCH v2 : les segments résiduels trop courts sont quand même conservés
    (rattachés au dernier ou gardés seuls si unique) plutôt que dropés.
    """
    merged = []
    current_start = None
    current_end   = None

    for start, end in segments:
        dur = (end - start) / sr

        if current_start is None:
            current_start, current_end = start, end
            continue

        current_dur = (current_end - current_start) / sr
        combined    = current_dur + dur

        if combined <= max_sec:
            current_end = end
        else:
            if current_dur >= min_sec:
                merged.append((current_start, current_end))
            else:
                # PATCH : segment court mais on le garde quand même (rattaché au suivant si possible)
                current_end = end  # étendre avec le suivant
                if (current_end - current_start) / sr >= min_sec:
                    merged.append((current_start, current_end))
                    current_start, current_end = None, None
                    continue
            current_start, current_end = start, end

    if current_start is not None:
        dur = (current_end - current_start) / sr
        if dur >= min_sec:
            merged.append((current_start, current_end))
        elif merged:
            # Rattacher à la fin du dernier segment si ça ne dépasse pas max
            last_start, last_end = merged[-1]
            if (current_end - last_start) / sr <= max_sec:
                merged[-1] = (last_start, current_end)
            else:
                # PATCH v2 : conserver quand même (mieux que dropper silencieusement)
                merged.append((current_start, current_end))
        else:
            # Fichier entier < min_sec → conserver tel quel
            merged.append((current_start, current_end))

    # Découper les segments encore trop longs en parts égales
    final = []
    for start, end in merged:
        dur = (end - start) / sr
        if dur > max_sec:
            n_parts = int(np.ceil(dur / max_sec))
            part_len = int((end - start) / n_parts)
            for k in range(n_parts):
                s = start + k * part_len
                e = start + (k + 1) * part_len if k < n_parts - 1 else end
                final.append((s, e))
        else:
            final.append((start, end))

    return final


# ─── Chargement et normalisation audio ─────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Charge et normalise un fichier audio en float32 mono à SAMPLE_RATE Hz."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


# ─── Sauvegarde des segments ───────────────────────────────────────────────────

def save_segment(audio: np.ndarray, start_s: float, end_s: float,
                 source_file: str, out_dir: str) -> Dict:
    """Sauvegarde un segment WAV et retourne ses métadonnées."""
    seg_id   = str(uuid.uuid4())[:8]
    filename = f"seg_{seg_id}.wav"
    out_path = os.path.join(out_dir, filename)

    start_sample = int(start_s * SAMPLE_RATE)
    end_sample   = int(end_s   * SAMPLE_RATE)
    segment_audio = audio[start_sample:end_sample]

    sf.write(out_path, segment_audio, SAMPLE_RATE, subtype="PCM_16")

    dur = round(end_s - start_s, 2)
    return {
        "id"          : seg_id,
        "file"        : filename,
        "path"        : out_path,
        "source"      : os.path.basename(source_file),
        "start_sec"   : round(start_s, 2),
        "end_sec"     : round(end_s,   2),
        "duration_sec": dur,
        "sample_rate" : SAMPLE_RATE,
        "short"       : dur < SEGMENT_MIN_SEC,  # flag pour les très courts
        "indexed"     : False,
        "criticite"   : None,
    }


# ─── Base SQLite pour les métadonnées ──────────────────────────────────────────

def init_db(db_path: str = "corpus/metadata.db") -> sqlite3.Connection:
    """Initialise la base SQLite de métadonnées."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id            TEXT PRIMARY KEY,
            file          TEXT NOT NULL,
            path          TEXT NOT NULL,
            source        TEXT,
            start_sec     REAL,
            end_sec       REAL,
            duration_sec  REAL,
            sample_rate   INTEGER,
            short         INTEGER DEFAULT 0,
            indexed       INTEGER DEFAULT 0,
            transcription TEXT,
            translation_fr TEXT,
            criticite_json TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def insert_segment(conn: sqlite3.Connection, meta: Dict):
    """Insère un segment dans la base."""
    conn.execute("""
        INSERT OR REPLACE INTO segments
        (id, file, path, source, start_sec, end_sec, duration_sec, sample_rate, short)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta["id"], meta["file"], meta["path"], meta["source"],
        meta["start_sec"], meta["end_sec"], meta["duration_sec"],
        meta["sample_rate"], int(meta.get("short", False))
    ))
    conn.commit()


# ─── NOUVEAU : Chargement des paires audio+texte depuis ton dataset ────────────

def ingest_parallel_dataset(audio_dir: str,
                             text_file: str,
                             out_dir: str,
                             conn: sqlite3.Connection,
                             delimiter: str = "\t",
                             col_audio: int = 0,
                             col_moore: int = 1,
                             col_french: int = 2,
                             has_header: bool = True) -> List[Dict]:
    """
    Charge un dataset parallèle (audio + texte mooré + traduction française).

    Format attendu du fichier texte (TSV ou CSV) :
        nom_audio.wav  \\t  texte_mooré  \\t  traduction_française

    Insère la transcription mooré ET la traduction française directement
    en base (champ translation_fr), ce qui active le scoring de criticité
    dès le Module 5 sans avoir besoin de Whisper !

    Retourne la liste des métadonnées (même format que process_file).
    """
    os.makedirs(out_dir, exist_ok=True)
    all_metas = []
    skipped = 0

    with open(text_file, encoding="utf-8") as f:
        lines = f.readlines()

    if has_header:
        lines = lines[1:]

    print(f"[M1-PARALLEL] {len(lines)} lignes à ingérer depuis {text_file}")

    for i, line in enumerate(lines):
        parts = line.strip().split(delimiter)
        if len(parts) < 2:
            skipped += 1
            continue

        audio_name = parts[col_audio].strip()
        text_moore = parts[col_moore].strip() if col_moore < len(parts) else ""
        text_french = parts[col_french].strip() if col_french < len(parts) else ""

        audio_path = os.path.join(audio_dir, audio_name)
        if not os.path.exists(audio_path):
            # Essayer sans extension, avec .wav, etc.
            for ext in [".wav", ".mp3", ".ogg", ".flac"]:
                alt = audio_path + ext if not audio_path.endswith(ext) else audio_path
                if os.path.exists(alt):
                    audio_path = alt
                    break
            else:
                skipped += 1
                continue

        try:
            audio = load_audio(audio_path)
            total_sec = len(audio) / SAMPLE_RATE

            # Pour les audios courts du dataset (< SEGMENT_MAX_SEC) :
            # on les garde entiers, pas besoin de segmenter
            if total_sec <= SEGMENT_MAX_SEC:
                meta = save_segment(audio, 0.0, total_sec, audio_path, out_dir)
            else:
                # Segmenter les longs
                raw_segs = vad_simple(audio, SAMPLE_RATE)
                if not raw_segs:
                    raw_segs = [(0, len(audio))]
                segs = merge_short_segments(raw_segs, SAMPLE_RATE)
                # On prend juste le premier segment pour un dataset parallèle
                # (chaque ligne = un enregistrement court en général)
                s, e = segs[0]
                meta = save_segment(audio, s / SAMPLE_RATE, e / SAMPLE_RATE, audio_path, out_dir)

            # Stocker transcription mooré + traduction française directement en DB
            conn.execute("""
                INSERT OR REPLACE INTO segments
                (id, file, path, source, start_sec, end_sec, duration_sec,
                 sample_rate, short, transcription, translation_fr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meta["id"], meta["file"], meta["path"], meta["source"],
                meta["start_sec"], meta["end_sec"], meta["duration_sec"],
                meta["sample_rate"], int(meta.get("short", False)),
                text_moore, text_french
            ))
            conn.commit()
            meta["transcription"] = text_moore
            meta["translation_fr"] = text_french
            all_metas.append(meta)

        except Exception as e:
            print(f"[M1-PARALLEL] ⚠ Erreur {audio_name} : {e}")
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"[M1-PARALLEL] {i+1}/{len(lines)} traités ({skipped} ignorés)...")

    print(f"[M1-PARALLEL] ✓ {len(all_metas)} segments ingérés, {skipped} ignorés")
    return all_metas


# ─── Diagnostic du dataset ────────────────────────────────────────────────────

def diagnose_dataset(audio_dir: str, sample_size: int = 50) -> Dict:
    """
    Analyse un échantillon du dataset audio pour détecter les problèmes
    avant de lancer le pipeline complet.
    Retourne un rapport de diagnostic.
    """
    import random

    audio_files = [
        p for p in Path(audio_dir).rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]
    total = len(audio_files)
    sample = random.sample(audio_files, min(sample_size, total))

    durations = []
    errors = []
    formats = {}
    short_files = []   # < 5s
    long_files  = []   # > 30s

    print(f"[DIAG] Analyse de {len(sample)}/{total} fichiers...")

    for p in sample:
        fmt = p.suffix.lower()
        formats[fmt] = formats.get(fmt, 0) + 1
        try:
            dur = librosa.get_duration(path=str(p))
            durations.append(dur)
            if dur < SEGMENT_MIN_SEC:
                short_files.append((str(p), dur))
            if dur > SEGMENT_MAX_SEC:
                long_files.append((str(p), dur))
        except Exception as e:
            errors.append((str(p), str(e)))

    report = {
        "total_files"   : total,
        "sample_size"   : len(sample),
        "formats"       : formats,
        "errors"        : len(errors),
        "error_samples" : errors[:5],
        "duration_stats": {
            "min_sec"  : round(min(durations), 2) if durations else 0,
            "max_sec"  : round(max(durations), 2) if durations else 0,
            "mean_sec" : round(np.mean(durations), 2) if durations else 0,
            "median_sec": round(float(np.median(durations)), 2) if durations else 0,
        },
        "short_files_pct": round(len(short_files) / len(sample) * 100, 1) if sample else 0,
        "long_files_pct" : round(len(long_files)  / len(sample) * 100, 1) if sample else 0,
        "short_samples"  : short_files[:3],
        "long_samples"   : long_files[:3],
    }

    print(f"\n{'─'*50}")
    print(f"  RAPPORT DE DIAGNOSTIC — {audio_dir}")
    print(f"{'─'*50}")
    print(f"  Fichiers totaux     : {total}")
    print(f"  Formats             : {formats}")
    print(f"  Erreurs de lecture  : {len(errors)}")
    print(f"  Durée min/moy/max   : {report['duration_stats']['min_sec']}s / "
          f"{report['duration_stats']['mean_sec']}s / "
          f"{report['duration_stats']['max_sec']}s")
    print(f"  Médiane             : {report['duration_stats']['median_sec']}s")
    print(f"  Trop courts (<5s)   : {report['short_files_pct']}%")
    print(f"  Trop longs  (>30s)  : {report['long_files_pct']}%")
    if errors:
        print(f"  Exemples d'erreurs  : {errors[:2]}")
    print(f"{'─'*50}\n")

    return report


# ─── Fonctions originales process_file / process_directory ───────────────────

def process_file(audio_path: str, out_dir: str, conn: sqlite3.Connection,
                 verbose: bool = True) -> List[Dict]:
    """
    Traite un fichier audio : charge → VAD → segmentation → sauvegarde.
    PATCH v2 : les fichiers courts sont conservés (plus de drop silencieux).
    """
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[M1] Chargement : {audio_path}")

    audio = load_audio(audio_path)
    total_sec = len(audio) / SAMPLE_RATE

    if verbose:
        print(f"     Durée totale : {total_sec:.1f}s")

    # Fichier entier < min_sec : conserver tel quel
    if total_sec < SEGMENT_MIN_SEC:
        print(f"     ⚠ Fichier court ({total_sec:.1f}s < {SEGMENT_MIN_SEC}s) — conservé tel quel")
        meta = save_segment(audio, 0.0, total_sec, audio_path, out_dir)
        insert_segment(conn, meta)
        return [meta]

    raw_segs = vad_simple(audio, SAMPLE_RATE)

    if not raw_segs:
        print(f"[M1] ⚠ Aucune activité vocale détectée dans {audio_path}")
        return []

    segments = merge_short_segments(raw_segs, SAMPLE_RATE)

    if verbose:
        print(f"     {len(segments)} segments détectés")

    metas = []
    for start_s_sample, end_s_sample in segments:
        start_s = start_s_sample / SAMPLE_RATE
        end_s   = end_s_sample   / SAMPLE_RATE
        meta    = save_segment(audio, start_s, end_s, audio_path, out_dir)
        insert_segment(conn, meta)
        metas.append(meta)

        if verbose:
            flag = " ⚠ court" if meta["short"] else ""
            print(f"     → {meta['file']}  [{meta['start_sec']}s – {meta['end_sec']}s]{flag}")

    return metas


def process_directory(input_dir: str, out_dir: str,
                       conn: sqlite3.Connection) -> List[Dict]:
    """Traitement batch d'un répertoire."""
    all_metas = []
    audio_files = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]
    print(f"[M1] {len(audio_files)} fichiers audio trouvés dans {input_dir}")

    for af in audio_files:
        metas = process_file(str(af), out_dir, conn)
        all_metas.extend(metas)

    return all_metas


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion & Segmentation Audio v2")
    parser.add_argument("--input",     required=False, help="Fichier ou dossier audio")
    parser.add_argument("--out",       default="corpus/segments")
    parser.add_argument("--db",        default="corpus/metadata.db")
    parser.add_argument("--json",      default="corpus/segments.json")
    # Mode dataset parallèle
    parser.add_argument("--parallel",  action="store_true",
                        help="Mode dataset parallèle audio+texte")
    parser.add_argument("--text-file", help="Fichier TSV avec texte mooré + traduction")
    parser.add_argument("--col-moore", type=int, default=1)
    parser.add_argument("--col-french",type=int, default=2)
    parser.add_argument("--no-header", action="store_true")
    # Diagnostic
    parser.add_argument("--diagnose",  action="store_true",
                        help="Analyser le dataset sans rien écrire")
    parser.add_argument("--sample",    type=int, default=50,
                        help="Taille de l'échantillon pour le diagnostic")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_dataset(args.input, args.sample)
        exit(0)

    conn = init_db(args.db)

    if args.parallel and args.text_file:
        metas = ingest_parallel_dataset(
            audio_dir  = args.input,
            text_file  = args.text_file,
            out_dir    = args.out,
            conn       = conn,
            col_moore  = args.col_moore,
            col_french = args.col_french,
            has_header = not args.no_header,
        )
    elif args.input and os.path.isdir(args.input):
        metas = process_directory(args.input, args.out, conn)
    elif args.input:
        metas = process_file(args.input, args.out, conn)
    else:
        parser.print_help()
        exit(1)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    short_count = sum(1 for m in metas if m.get("short"))
    print(f"\n[M1] ✓ {len(metas)} segments produits → {args.json}")
    if short_count:
        print(f"     ⚠ {short_count} segments courts (<{SEGMENT_MIN_SEC}s) conservés")
    conn.close()
