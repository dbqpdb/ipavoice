"""Audio preprocessing and dataset generation for VITS training.

Resamples all audio segments to a uniform format (22050 Hz, mono, 16-bit),
trims silence, filters outliers, builds the IPA vocabulary, and generates
Coqui TTS-compatible metadata CSV files with train/val split.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from processing.database import get_connection, init_db
from training.ipa_tokenizer import build_vocabulary, normalize_ipa, tokenize

# --- Constants ---

TARGET_SR: int = 22050
MIN_DURATION_S: float = 0.3
MAX_DURATION_S: float = 15.0
TRIM_TOP_DB: int = 30
VAL_FRACTION: float = 0.1

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR: Path = DATA_DIR / "training"
WAVS_DIR: Path = TRAINING_DIR / "wavs"


def _process_segment(args: tuple[str, str, str]) -> tuple[str, float] | None:
    """Process a single audio segment: resample, normalize, trim, filter.

    Args:
        args: (source_path, dest_path, segment_key)

    Returns:
        (segment_key, duration_s) on success, None if filtered out.
    """
    source_path, dest_path, segment_key = args

    if not os.path.isfile(source_path):
        return None

    try:
        # Load and resample to target sample rate, mono
        y: np.ndarray
        sr: int
        y, sr = librosa.load(source_path, sr=TARGET_SR, mono=True)
    except Exception:
        return None

    if len(y) == 0:
        return None

    # Trim silence from edges
    y_trimmed: np.ndarray
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    if len(y_trimmed) == 0:
        return None

    duration_s: float = len(y_trimmed) / TARGET_SR

    # Filter by duration
    if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
        return None

    # Peak normalize to [-1, 1]
    peak: float = float(np.max(np.abs(y_trimmed)))
    if peak > 0:
        y_trimmed = y_trimmed / peak

    # Write as 16-bit PCM WAV
    sf.write(dest_path, y_trimmed, TARGET_SR, subtype="PCM_16")

    return (segment_key, duration_s)


def preprocess_audio(conn: sqlite3.Connection, workers: int | None = None) -> dict[str, Any]:
    """Resample and normalize all audio segments for training.

    Args:
        conn: Database connection.
        workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        Dict with processing statistics.
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)

    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # Query all segments with their IPA text and language code
    rows: list[sqlite3.Row] = conn.execute("""
        SELECT
            s.segment_file,
            e.ipa,
            e.language_code,
            r.audio_filename,
            e.entry_number
        FROM segments s
        JOIN entries e ON s.entry_id = e.id
        JOIN recordings r ON s.recording_id = r.id
        WHERE e.ipa IS NOT NULL AND e.ipa != ''
              AND s.segment_file IS NOT NULL
        ORDER BY e.language_code, r.audio_filename, e.entry_number
    """).fetchall()

    print(f"Found {len(rows):,} segments with IPA transcriptions")

    # Build work items
    tasks: list[tuple[str, str, str]] = []
    # Track metadata for manifest generation
    segment_meta: dict[str, dict[str, str]] = {}

    for row in rows:
        source: str = row["segment_file"]
        lang: str = row["language_code"]
        audio_fn: str = row["audio_filename"]
        entry_num: int = row["entry_number"] or 0
        ipa: str = row["ipa"]

        # Generate segment key: e.g. abk_word-list_1970_01_001
        segment_key: str = f"{audio_fn}_{entry_num:03d}"
        dest: str = str(WAVS_DIR / f"{segment_key}.wav")

        tasks.append((source, dest, segment_key))
        segment_meta[segment_key] = {
            "ipa": ipa,
            "language_code": lang,
        }

    # Process in parallel
    print(f"Preprocessing audio with {workers} workers...")
    results: list[tuple[str, float] | None] = []

    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_segment, tasks, chunksize=64),
            total=len(tasks),
            desc="Resampling audio",
        ):
            results.append(result)

    # Collect successful results
    successful: dict[str, float] = {}
    for r in results:
        if r is not None:
            successful[r[0]] = r[1]

    # Duration statistics
    durations: list[float] = list(successful.values())
    stats: dict[str, Any] = {
        "total_segments": len(tasks),
        "successful": len(successful),
        "filtered_out": len(tasks) - len(successful),
        "total_duration_hours": sum(durations) / 3600,
    }

    if durations:
        dur_arr: np.ndarray = np.array(durations)
        stats["duration_mean_s"] = float(np.mean(dur_arr))
        stats["duration_median_s"] = float(np.median(dur_arr))
        stats["duration_std_s"] = float(np.std(dur_arr))
        stats["duration_min_s"] = float(np.min(dur_arr))
        stats["duration_max_s"] = float(np.max(dur_arr))

    print(f"\nPreprocessing complete:")
    print(f"  Segments processed: {stats['total_segments']:,}")
    print(f"  Segments kept: {stats['successful']:,}")
    print(f"  Filtered out: {stats['filtered_out']:,}")
    print(f"  Total duration: {stats['total_duration_hours']:.1f} hours")
    if durations:
        print(f"  Duration: mean={stats['duration_mean_s']:.2f}s, "
              f"median={stats['duration_median_s']:.2f}s, "
              f"std={stats['duration_std_s']:.2f}s")
        print(f"  Range: [{stats['duration_min_s']:.2f}s, {stats['duration_max_s']:.2f}s]")

    return stats | {"successful_keys": set(successful.keys()), "segment_meta": segment_meta}


def build_speaker_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Build language_code → speaker_id mapping.

    Returns:
        Dict mapping language codes to integer speaker IDs.
    """
    rows: list[sqlite3.Row] = conn.execute(
        "SELECT DISTINCT code FROM languages ORDER BY code"
    ).fetchall()
    speaker_map: dict[str, int] = {row["code"]: i for i, row in enumerate(rows)}

    path: Path = TRAINING_DIR / "speakers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2)

    print(f"Speaker map: {len(speaker_map)} languages → {path}")
    return speaker_map


def build_vocab_from_db(conn: sqlite3.Connection, min_count: int = 2) -> Path:
    """Build IPA vocabulary from all entries in the database.

    Args:
        conn: Database connection.
        min_count: Minimum token frequency to include.

    Returns:
        Path to saved vocabulary JSON.
    """
    # Fetch all IPA strings
    rows: list[sqlite3.Row] = conn.execute(
        "SELECT ipa FROM entries WHERE ipa IS NOT NULL AND ipa != ''"
    ).fetchall()
    ipa_strings: list[str] = [row["ipa"] for row in rows]
    print(f"Building vocabulary from {len(ipa_strings):,} IPA strings...")

    # Fetch language codes
    lang_rows: list[sqlite3.Row] = conn.execute(
        "SELECT DISTINCT code FROM languages ORDER BY code"
    ).fetchall()
    language_codes: list[str] = [row["code"] for row in lang_rows]

    vocab = build_vocabulary(ipa_strings, language_codes=language_codes, min_count=min_count)

    vocab_path: Path = TRAINING_DIR / "ipa_vocab.json"
    vocab.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    return vocab_path


def generate_manifests(
    successful_keys: set[str],
    segment_meta: dict[str, dict[str, str]],
) -> tuple[int, int]:
    """Generate Coqui TTS-compatible metadata CSV files.

    Uses the ``coqui`` formatter format (pipe-delimited CSV with header)::

        audio_file|text|speaker_name
        wavs/abk_word-list_1970_01_001.wav|aˈkə|ABK

    IPA text is normalized (NFD, ASCII→IPA fixes, bracket stripping) to
    match the character set built from the vocabulary.

    Args:
        successful_keys: Set of segment keys that passed preprocessing.
        segment_meta: Dict of segment_key → {ipa, language_code}.

    Returns:
        (train_count, val_count) tuple.
    """
    # Build entries grouped by language for stratified splitting
    by_language: dict[str, list[tuple[str, str, str]]] = {}
    for key in sorted(successful_keys):
        meta: dict[str, str] = segment_meta[key]
        ipa_raw: str = meta["ipa"]
        lang: str = meta["language_code"]

        # Normalize IPA to match the character set used in config
        ipa: str = normalize_ipa(ipa_raw)

        # Validate IPA produces tokens
        tokens: list[str] = tokenize(ipa, normalize=False)
        if not tokens:
            continue

        entry: tuple[str, str, str] = (key, ipa, lang)
        by_language.setdefault(lang, []).append(entry)

    # Stratified train/val split
    train_entries: list[tuple[str, str, str]] = []
    val_entries: list[tuple[str, str, str]] = []

    for lang, entries in sorted(by_language.items()):
        n_val: int = max(1, int(len(entries) * VAL_FRACTION))
        # Last n_val entries go to validation (deterministic, no shuffle)
        val_entries.extend(entries[-n_val:])
        train_entries.extend(entries[:-n_val])

    # Write CSVs — coqui formatter format with header
    train_path: Path = TRAINING_DIR / "metadata_train.csv"
    val_path: Path = TRAINING_DIR / "metadata_val.csv"

    for path, entries in [(train_path, train_entries), (val_path, val_entries)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writerow(["audio_file", "text", "speaker_name"])
            for key, ipa, speaker_name in entries:
                writer.writerow([f"wavs/{key}.wav", ipa, speaker_name])

    print(f"\nManifests generated:")
    print(f"  Train: {len(train_entries):,} entries → {train_path}")
    print(f"  Val:   {len(val_entries):,} entries → {val_path}")

    return len(train_entries), len(val_entries)


def run_preprocessing(workers: int | None = None) -> None:
    """Run the full preprocessing pipeline.

    1. Resample and normalize audio
    2. Build speaker map
    3. Build IPA vocabulary
    4. Generate train/val metadata CSVs
    """
    conn: sqlite3.Connection = get_connection()
    init_db(conn)

    try:
        # Step 1: Audio preprocessing
        result: dict[str, Any] = preprocess_audio(conn, workers=workers)

        # Step 2: Speaker map
        speaker_map: dict[str, int] = build_speaker_map(conn)

        # Step 3: IPA vocabulary
        build_vocab_from_db(conn)

        # Step 4: Generate manifests
        generate_manifests(
            successful_keys=result["successful_keys"],
            segment_meta=result["segment_meta"],
        )

        print("\nPreprocessing pipeline complete.")
    finally:
        conn.close()
