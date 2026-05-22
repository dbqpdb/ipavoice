"""Audio preprocessing and dataset generation for Common Voice Spontaneous Speech.

Converts Common Voice MP3 files to training-ready WAV format, builds IPA
vocabulary from Allosaurus phone recognition output, and generates Coqui TTS
compatible metadata CSV files.
"""

from __future__ import annotations

import csv
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from training.ipa_tokenizer import IPAVocabulary, SPECIAL_TOKENS, normalize_ipa, tokenize

# --- Constants ---

TARGET_SR: int = 22050
MIN_DURATION_S: float = 0.3
MAX_DURATION_S: float = 15.0
TRIM_TOP_DB: int = 30
VAL_SPLIT: str = "dev"
TRAIN_SPLIT: str = "train"

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
TRAINING_CV_DIR: Path = DATA_DIR / "training_cv"
WAVS_DIR: Path = TRAINING_CV_DIR / "wavs"


def convert_allosaurus_to_ipa(phones: str) -> str:
    """Convert space-separated Allosaurus phones to concatenated IPA Unicode.

    Allosaurus outputs phones like "t ɛ s t", which we convert to "tɛst"
    so the existing tokenizer works unchanged.

    Args:
        phones: Space-separated phone string from ipa_audio_universal column.

    Returns:
        Concatenated IPA string with no spaces.
    """
    if not phones:
        return ""
    # Join phones without spaces
    return "".join(phones.split())


def load_cv_dataset(
    parquet_path: str | Path,
    splits: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and filter the Common Voice Spontaneous Speech dataset.

    Args:
        parquet_path: Path to unified.parquet file.
        splits: List of splits to include (e.g., ["train", "dev"]).
                If None, includes all splits.

    Returns:
        List of dicts with filtered rows.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    print(f"Loading Common Voice dataset from {parquet_path}")
    table = pq.read_table(parquet_path)

    # Convert to list of dicts for filtering
    columns: list[str] = table.column_names
    rows: list[dict[str, Any]] = []

    for i in range(table.num_rows):
        row: dict[str, Any] = {col: table.column(col)[i].as_py() for col in columns}
        rows.append(row)

    print(f"Loaded {len(rows):,} total rows")

    # Filter by split
    if splits:
        rows = [r for r in rows if r.get("split") in splits]
        print(f"After split filter ({splits}): {len(rows):,} rows")

    # Filter by non-empty IPA
    rows = [r for r in rows if r.get("ipa_audio_universal")]
    print(f"After IPA filter: {len(rows):,} rows")

    # Filter by duration
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        duration_ms: str | int = row.get("duration_ms", "0")
        try:
            duration_s: float = float(duration_ms) / 1000.0
        except (ValueError, TypeError):
            continue

        if MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
            row["duration_s"] = duration_s
            filtered_rows.append(row)

    print(f"After duration filter ({MIN_DURATION_S}s-{MAX_DURATION_S}s): {len(filtered_rows):,} rows")

    return filtered_rows


def _find_audio_file(audio_filename: str, locale: str, cv_base_dir: Path) -> Path | None:
    """Find the MP3 file in the extracted directory structure.

    The CV corpus structure is:
        data/extracted/sps-corpus-3.0-YYYY-MM-DD-{locale}/audios/{audio_filename}

    Args:
        audio_filename: Filename from the audio_file column.
        locale: Locale code (e.g., "en", "fr").
        cv_base_dir: Path to the CommonVoiceSpontaneous directory.

    Returns:
        Path to MP3 file if found, None otherwise.
    """
    extracted_dir: Path = cv_base_dir / "data" / "extracted"

    if not extracted_dir.exists():
        return None

    # Find the locale directory (has variable date in name)
    for dirname in extracted_dir.iterdir():
        if dirname.is_dir() and dirname.name.endswith(f"-{locale}"):
            audio_path: Path = dirname / "audios" / audio_filename
            if audio_path.exists():
                return audio_path

    return None


def _process_cv_audio(args: tuple[str, str, str, str, Path]) -> tuple[str, float] | None:
    """Process a single CV audio file: convert MP3 to WAV, resample, normalize, trim.

    Args:
        args: (source_path, dest_path, audio_id, locale, cv_base_dir)

    Returns:
        (audio_id, duration_s) on success, None if failed.
    """
    audio_filename, dest_path, audio_id, locale, cv_base_dir = args

    # Find the source file
    source_path: Path | None = _find_audio_file(audio_filename, locale, cv_base_dir)
    if source_path is None:
        return None

    try:
        # Load and resample to target sample rate, mono
        y: np.ndarray
        sr: int
        y, sr = librosa.load(str(source_path), sr=TARGET_SR, mono=True)
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

    # Re-check duration after trimming
    if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
        return None

    # Peak normalize to [-1, 1]
    peak: float = float(np.max(np.abs(y_trimmed)))
    if peak > 0:
        y_trimmed = y_trimmed / peak

    # Write as 16-bit PCM WAV
    sf.write(dest_path, y_trimmed, TARGET_SR, subtype="PCM_16")

    return (audio_id, duration_s)


def convert_audio(
    rows: list[dict[str, Any]],
    cv_base_dir: Path,
    workers: int | None = None,
) -> dict[str, float]:
    """Convert MP3 files to WAV format with resampling and normalization.

    Args:
        rows: List of dataset rows.
        cv_base_dir: Path to CommonVoiceSpontaneous directory.
        workers: Number of parallel workers.

    Returns:
        Dict mapping audio_id to duration_s for successfully converted files.
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)

    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # Build work items
    tasks: list[tuple[str, str, str, str, Path]] = []
    for row in rows:
        audio_filename: str = row["audio_file"]
        audio_id: str = row["audio_id"]
        locale: str = row["locale"]
        dest_path: str = str(WAVS_DIR / f"cv_{locale}_{audio_id}.wav")

        tasks.append((audio_filename, dest_path, audio_id, locale, cv_base_dir))

    print(f"Converting {len(tasks):,} audio files with {workers} workers...")

    # Process in parallel
    results: list[tuple[str, float] | None] = []
    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_cv_audio, tasks, chunksize=32),
            total=len(tasks),
            desc="Converting audio",
        ):
            results.append(result)

    # Collect successful results
    successful: dict[str, float] = {}
    for r in results:
        if r is not None:
            successful[r[0]] = r[1]

    # Statistics
    print(f"\nAudio conversion complete:")
    print(f"  Total files: {len(tasks):,}")
    print(f"  Successfully converted: {len(successful):,}")
    print(f"  Failed/filtered: {len(tasks) - len(successful):,}")

    if successful:
        durations: list[float] = list(successful.values())
        dur_arr: np.ndarray = np.array(durations)
        print(f"  Total duration: {sum(durations) / 3600:.1f} hours")
        print(f"  Duration: mean={np.mean(dur_arr):.2f}s, median={np.median(dur_arr):.2f}s")

    return successful


def build_cv_vocabulary(
    rows: list[dict[str, Any]],
    min_count: int = 2,
) -> IPAVocabulary:
    """Build IPA vocabulary from Allosaurus phone recognition output.

    Args:
        rows: List of dataset rows with ipa_audio_universal column.
        min_count: Minimum token frequency to include.

    Returns:
        IPAVocabulary with token-to-ID mappings.
    """
    from collections import Counter

    # Collect unique locales as language codes
    locales: set[str] = {row["locale"].upper() for row in rows}
    language_codes: list[str] = sorted(locales)

    # Count all phonemes
    counter: Counter[str] = Counter()
    for row in rows:
        ipa_spaced: str = row.get("ipa_audio_universal", "")
        ipa: str = convert_allosaurus_to_ipa(ipa_spaced)
        if ipa:
            # Normalize and tokenize
            ipa_norm: str = normalize_ipa(ipa)
            tokens: list[str] = tokenize(ipa_norm, normalize=False)
            counter.update(tokens)

    # Filter by minimum count
    filtered: list[tuple[str, int]] = [
        (token, count) for token, count in counter.most_common()
        if count >= min_count
    ]

    total_tokens: int = sum(counter.values())
    filtered_tokens: list[str] = [t for t, _ in filtered]
    dropped: int = len(counter) - len(filtered_tokens)

    print(f"\nCV IPA vocabulary statistics:")
    print(f"  Total tokens in corpus: {total_tokens:,}")
    print(f"  Unique tokens: {len(counter):,}")
    print(f"  Tokens with count >= {min_count}: {len(filtered_tokens):,}")
    print(f"  Dropped (rare): {dropped:,}")
    print(f"  Language conditioning tokens: {len(language_codes):,}")
    print(f"  Final vocabulary size: {len(SPECIAL_TOKENS) + len(filtered_tokens) + len(language_codes):,}")

    # Show top 20 tokens
    print(f"\n  Top 20 tokens:")
    for token, count in filtered[:20]:
        pct: float = 100.0 * count / total_tokens
        if any(ord(c) > 127 for c in token):
            codepoints: str = " ".join(f"U+{ord(c):04X}" for c in token)
            print(f"    {token!r:12s} ({codepoints}): {count:>8,} ({pct:.1f}%)")
        else:
            print(f"    {token!r:12s}: {count:>8,} ({pct:.1f}%)")

    return IPAVocabulary(tokens=filtered_tokens, language_codes=language_codes)


def generate_cv_manifests(
    rows: list[dict[str, Any]],
    successful_ids: set[str],
) -> tuple[int, int]:
    """Generate Coqui TTS-compatible metadata CSV files.

    Creates separate train and validation manifests based on the split column.

    Args:
        rows: List of dataset rows.
        successful_ids: Set of audio_ids that were successfully converted.

    Returns:
        (train_count, val_count) tuple.
    """
    train_entries: list[tuple[str, str, str]] = []
    val_entries: list[tuple[str, str, str]] = []

    for row in rows:
        audio_id: str = row["audio_id"]
        if audio_id not in successful_ids:
            continue

        # Convert IPA
        ipa_spaced: str = row.get("ipa_audio_universal", "")
        ipa: str = convert_allosaurus_to_ipa(ipa_spaced)
        if not ipa:
            continue

        # Normalize IPA
        ipa_norm: str = normalize_ipa(ipa)
        tokens: list[str] = tokenize(ipa_norm, normalize=False)
        if not tokens:
            continue

        locale: str = row["locale"]
        speaker: str = locale.upper()
        wav_path: str = f"wavs/cv_{locale}_{audio_id}.wav"

        split: str = row.get("split", "")
        entry: tuple[str, str, str] = (wav_path, ipa_norm, speaker)

        if split == TRAIN_SPLIT:
            train_entries.append(entry)
        elif split == VAL_SPLIT:
            val_entries.append(entry)

    # Write CSVs
    train_path: Path = TRAINING_CV_DIR / "metadata_train.csv"
    val_path: Path = TRAINING_CV_DIR / "metadata_val.csv"

    for path, entries in [(train_path, train_entries), (val_path, val_entries)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
            writer.writerow(["audio_file", "text", "speaker_name"])
            for wav_path, ipa, speaker in entries:
                writer.writerow([wav_path, ipa, speaker])

    print(f"\nManifests generated:")
    print(f"  Train: {len(train_entries):,} entries -> {train_path}")
    print(f"  Val:   {len(val_entries):,} entries -> {val_path}")

    return len(train_entries), len(val_entries)


def generate_speaker_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Generate speaker (locale) to ID mapping.

    Args:
        rows: List of dataset rows.

    Returns:
        Dict mapping locale codes (uppercase) to integer IDs.
    """
    locales: set[str] = {row["locale"].upper() for row in rows}
    speaker_map: dict[str, int] = {loc: i for i, loc in enumerate(sorted(locales))}

    path: Path = TRAINING_CV_DIR / "speakers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2)

    print(f"Speaker map: {len(speaker_map)} locales -> {path}")
    return speaker_map


def run_cv_preprocessing(
    cv_parquet: str | Path,
    cv_base_dir: str | Path | None = None,
    workers: int | None = None,
) -> None:
    """Run the full CV preprocessing pipeline.

    1. Load and filter parquet data
    2. Convert audio files
    3. Build speaker map
    4. Build IPA vocabulary
    5. Generate train/val manifests

    Args:
        cv_parquet: Path to unified.parquet file.
        cv_base_dir: Path to CommonVoiceSpontaneous directory.
                     If None, inferred from cv_parquet path.
        workers: Number of parallel workers for audio conversion.
    """
    cv_parquet = Path(cv_parquet)

    # Infer CV base directory from parquet path
    if cv_base_dir is None:
        # Expect: .../CommonVoiceSpontaneous/data/processed/unified.parquet
        cv_base_dir = cv_parquet.parent.parent.parent
    else:
        cv_base_dir = Path(cv_base_dir)

    print(f"Common Voice base directory: {cv_base_dir}")

    # Create output directory
    TRAINING_CV_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and filter dataset
    rows: list[dict[str, Any]] = load_cv_dataset(
        cv_parquet,
        splits=[TRAIN_SPLIT, VAL_SPLIT],
    )

    if not rows:
        print("No rows passed filtering. Check parquet file and splits.")
        return

    # Step 2: Convert audio files
    successful: dict[str, float] = convert_audio(rows, cv_base_dir, workers=workers)

    if not successful:
        print("No audio files successfully converted.")
        return

    successful_ids: set[str] = set(successful.keys())

    # Step 3: Build speaker map
    generate_speaker_map(rows)

    # Step 4: Build vocabulary
    vocab: IPAVocabulary = build_cv_vocabulary(rows, min_count=2)
    vocab_path: Path = TRAINING_CV_DIR / "ipa_vocab.json"
    vocab.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    # Step 5: Generate manifests
    generate_cv_manifests(rows, successful_ids)

    print("\nCV preprocessing pipeline complete.")
    print(f"Output directory: {TRAINING_CV_DIR}")
