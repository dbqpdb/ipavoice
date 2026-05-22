"""Extract phone-level durations from MFA TextGrid alignments.

Converts MFA TextGrid files to frame-level duration arrays that can be used
for duration supervision in VITS training.
"""

from __future__ import annotations

import csv
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from praatio import textgrid
from tqdm import tqdm

from training.ipa_tokenizer import tokenize


# --- Constants ---

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
TRAINING_CV_DIR: Path = DATA_DIR / "training_cv"
MFA_DIR: Path = DATA_DIR / "mfa"
MFA_TEXTGRID_DIR: Path = MFA_DIR / "textgrids"
DURATIONS_DIR: Path = TRAINING_CV_DIR / "durations"

# Audio parameters (must match training config)
SAMPLE_RATE: int = 22050
HOP_LENGTH: int = 256
FRAMES_PER_SECOND: float = SAMPLE_RATE / HOP_LENGTH  # ~86.13


def extract_durations_from_textgrid(
    tg_path: Path,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> tuple[list[str], list[int]] | None:
    """Extract phone labels and frame durations from a TextGrid file.

    Args:
        tg_path: Path to TextGrid file.
        sample_rate: Audio sample rate.
        hop_length: STFT hop length in samples.

    Returns:
        Tuple of (phones, durations) where phones is a list of phone labels
        and durations is a list of frame counts. Returns None if extraction fails.
    """
    try:
        tg: textgrid.Textgrid = textgrid.openTextgrid(
            str(tg_path),
            includeEmptyIntervals=True,
        )
    except Exception as e:
        print(f"Error reading {tg_path}: {e}")
        return None

    # MFA uses "words" tier for our phone-level alignments
    # (since each "word" in our corpus is a single phone)
    try:
        phones_tier = tg.getTier("words")
    except KeyError:
        # Try "phones" tier as fallback
        try:
            phones_tier = tg.getTier("phones")
        except KeyError:
            print(f"No words or phones tier in {tg_path}")
            return None

    phones: list[str] = []
    durations: list[int] = []

    for interval in phones_tier.entries:
        phone: str = interval.label

        # Skip silence intervals (empty labels)
        if not phone:
            continue

        # Calculate duration in frames
        duration_sec: float = interval.end - interval.start
        duration_frames: int = int(duration_sec * sample_rate / hop_length)

        # Ensure at least 1 frame
        duration_frames = max(1, duration_frames)

        phones.append(phone)
        durations.append(duration_frames)

    if not phones:
        return None

    return phones, durations


def _process_textgrid(
    args: tuple[Path, Path, int, int],
) -> tuple[str, list[str], list[int]] | None:
    """Process a single TextGrid file.

    Args:
        args: (tg_path, output_dir, sample_rate, hop_length) tuple.

    Returns:
        Tuple of (audio_id, phones, durations) on success, None on failure.
    """
    tg_path, output_dir, sample_rate, hop_length = args

    result = extract_durations_from_textgrid(tg_path, sample_rate, hop_length)
    if result is None:
        return None

    phones, durations = result

    # Get audio ID from filename (e.g., "cv_ady_62977.TextGrid" -> "cv_ady_62977")
    audio_id: str = tg_path.stem

    # Save durations to JSON
    duration_path: Path = output_dir / f"{audio_id}.json"
    with open(duration_path, "w", encoding="utf-8") as f:
        json.dump({
            "phones": phones,
            "durations": durations,
            "total_frames": sum(durations),
        }, f)

    return audio_id, phones, durations


def extract_all_durations(
    textgrid_dir: Path | None = None,
    output_dir: Path | None = None,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    workers: int | None = None,
) -> dict[str, list[int]]:
    """Extract durations from all TextGrid files.

    Args:
        textgrid_dir: Directory containing TextGrid files (can be nested).
        output_dir: Output directory for duration JSON files.
        sample_rate: Audio sample rate.
        hop_length: STFT hop length.
        workers: Number of parallel workers.

    Returns:
        Dict mapping audio_id to duration list.
    """
    if textgrid_dir is None:
        textgrid_dir = MFA_TEXTGRID_DIR
    if output_dir is None:
        output_dir = DURATIONS_DIR
    if workers is None:
        workers = max(1, cpu_count() - 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TextGrid files
    tg_files: list[Path] = list(textgrid_dir.rglob("*.TextGrid"))
    print(f"Found {len(tg_files):,} TextGrid files")

    if not tg_files:
        return {}

    # Prepare work items
    tasks: list[tuple[Path, Path, int, int]] = [
        (tg_path, output_dir, sample_rate, hop_length)
        for tg_path in tg_files
    ]

    # Process in parallel
    print(f"Extracting durations with {workers} workers...")
    results: dict[str, list[int]] = {}
    failed: int = 0

    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_textgrid, tasks, chunksize=100),
            total=len(tasks),
            desc="Extracting durations",
        ):
            if result is not None:
                audio_id, phones, durations = result
                results[audio_id] = durations
            else:
                failed += 1

    print(f"\nDuration extraction complete:")
    print(f"  Total TextGrids: {len(tg_files):,}")
    print(f"  Successfully extracted: {len(results):,}")
    print(f"  Failed: {failed:,}")
    print(f"  Output directory: {output_dir}")

    if results:
        # Statistics
        all_durations: list[int] = []
        for durs in results.values():
            all_durations.extend(durs)

        dur_arr: np.ndarray = np.array(all_durations)
        print(f"\nDuration statistics (in frames):")
        print(f"  Total phones: {len(all_durations):,}")
        print(f"  Mean: {np.mean(dur_arr):.2f}")
        print(f"  Median: {np.median(dur_arr):.0f}")
        print(f"  Std: {np.std(dur_arr):.2f}")
        print(f"  Min: {np.min(dur_arr)}, Max: {np.max(dur_arr)}")

        # Frame to ms conversion
        ms_per_frame: float = 1000.0 * hop_length / sample_rate
        print(f"\n  (1 frame = {ms_per_frame:.2f} ms)")
        print(f"  Mean duration: {np.mean(dur_arr) * ms_per_frame:.1f} ms")
        print(f"  Median duration: {np.median(dur_arr) * ms_per_frame:.1f} ms")

    return results


def update_metadata_with_durations(
    metadata_path: Path,
    durations: dict[str, list[int]],
    output_path: Path | None = None,
) -> int:
    """Update metadata CSV with duration information.

    Adds a 'durations' column with JSON-encoded duration arrays.

    Args:
        metadata_path: Path to input metadata CSV.
        durations: Dict mapping audio_id to duration list.
        output_path: Output path. If None, overwrites input.

    Returns:
        Number of rows with durations added.
    """
    if output_path is None:
        output_path = metadata_path

    # Read existing metadata
    rows: list[dict[str, str]] = []
    with open(metadata_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        fieldnames: list[str] = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    # Add durations column
    if "durations" not in fieldnames:
        fieldnames.append("durations")

    updated: int = 0
    for row in rows:
        audio_file: str = row.get("audio_file", "")
        # Extract audio ID from path (e.g., "wavs/cv_ady_62977.wav" -> "cv_ady_62977")
        audio_id: str = Path(audio_file).stem

        if audio_id in durations:
            row["durations"] = json.dumps(durations[audio_id])
            updated += 1
        else:
            row["durations"] = ""

    # Write updated metadata
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated:,}/{len(rows):,} rows with durations -> {output_path}")
    return updated


def validate_durations(
    metadata_path: Path,
    durations_dir: Path | None = None,
) -> tuple[int, int, list[str]]:
    """Validate that durations match transcript phone counts.

    Args:
        metadata_path: Path to metadata CSV.
        durations_dir: Directory containing duration JSON files.

    Returns:
        Tuple of (matched, mismatched, mismatch_ids).
    """
    if durations_dir is None:
        durations_dir = DURATIONS_DIR

    matched: int = 0
    mismatched: int = 0
    mismatch_ids: list[str] = []

    with open(metadata_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            audio_file: str = row.get("audio_file", "")
            text: str = row.get("text", "")
            audio_id: str = Path(audio_file).stem

            # Get expected phone count from transcript
            expected_phones: list[str] = tokenize(text)
            expected_count: int = len(expected_phones)

            # Get actual phone count from durations
            duration_path: Path = durations_dir / f"{audio_id}.json"
            if not duration_path.exists():
                continue

            with open(duration_path, encoding="utf-8") as df:
                data: dict[str, Any] = json.load(df)
                actual_phones: list[str] = data.get("phones", [])
                actual_count: int = len(actual_phones)

            if expected_count == actual_count:
                matched += 1
            else:
                mismatched += 1
                if len(mismatch_ids) < 10:  # Limit examples
                    mismatch_ids.append(
                        f"{audio_id}: expected {expected_count}, got {actual_count}"
                    )

    print(f"\nDuration validation:")
    print(f"  Matched: {matched:,}")
    print(f"  Mismatched: {mismatched:,}")
    if mismatch_ids:
        print(f"  Examples of mismatches:")
        for example in mismatch_ids:
            print(f"    {example}")

    return matched, mismatched, mismatch_ids


def run_duration_extraction(
    textgrid_dir: Path | None = None,
    training_dir: Path | None = None,
    workers: int | None = None,
    update_metadata: bool = True,
    validate: bool = True,
) -> dict[str, list[int]]:
    """Run full duration extraction pipeline.

    Args:
        textgrid_dir: Directory containing TextGrid files.
        training_dir: Directory containing training data.
        workers: Number of parallel workers.
        update_metadata: Whether to update metadata CSVs with durations.
        validate: Whether to validate phone counts match.

    Returns:
        Dict mapping audio_id to duration list.
    """
    if training_dir is None:
        training_dir = TRAINING_CV_DIR

    print("=" * 60)
    print("Duration Extraction from MFA TextGrids")
    print("=" * 60)

    # Extract durations
    durations: dict[str, list[int]] = extract_all_durations(
        textgrid_dir=textgrid_dir,
        workers=workers,
    )

    if not durations:
        print("No durations extracted.")
        return {}

    # Update metadata files
    if update_metadata:
        for split in ["train", "val"]:
            metadata_path: Path = training_dir / f"metadata_{split}.csv"
            if metadata_path.exists():
                update_metadata_with_durations(metadata_path, durations)

    # Validate
    if validate:
        for split in ["train", "val"]:
            metadata_path = training_dir / f"metadata_{split}.csv"
            if metadata_path.exists():
                print(f"\nValidating {split} split:")
                validate_durations(metadata_path)

    print("\n" + "=" * 60)
    print("Duration extraction complete.")
    print("=" * 60)

    return durations
