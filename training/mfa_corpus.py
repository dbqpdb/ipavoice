"""MFA corpus preparation for phone-level duration extraction.

Creates the corpus structure that Montreal Forced Aligner expects:
    corpus/
    ├── speaker1/
    │   ├── audio1.wav
    │   ├── audio1.txt  # space-separated phones
    │   └── ...
    └── speaker2/
        └── ...

Also generates the identity dictionary where each IPA phone maps to itself.
"""

from __future__ import annotations

import csv
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from tqdm import tqdm

from training.ipa_tokenizer import IPAVocabulary, tokenize


# --- Constants ---

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
TRAINING_CV_DIR: Path = DATA_DIR / "training_cv"
MFA_DIR: Path = DATA_DIR / "mfa"
MFA_CORPUS_DIR: Path = MFA_DIR / "corpus"
MFA_OUTPUT_DIR: Path = MFA_DIR / "textgrids"


def load_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """Load metadata CSV file.

    Args:
        metadata_path: Path to metadata_train.csv or metadata_val.csv.

    Returns:
        List of dicts with audio_file, text, speaker_name keys.
    """
    rows: list[dict[str, str]] = []
    with open(metadata_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(dict(row))
    return rows


def _create_corpus_entry(args: tuple[dict[str, str], Path, Path]) -> str | None:
    """Create a single corpus entry (symlink + transcript).

    Args:
        args: (row, training_dir, corpus_dir) tuple.

    Returns:
        Audio ID on success, None on failure.
    """
    row, training_dir, corpus_dir = args

    audio_rel: str = row["audio_file"]  # e.g., "wavs/cv_ady_62977.wav"
    audio_path: Path = training_dir / audio_rel
    text: str = row["text"]
    speaker: str = row["speaker_name"]

    if not audio_path.exists():
        return None

    # Tokenize IPA into space-separated phones
    tokens: list[str] = tokenize(text)
    if not tokens:
        return None

    # Create speaker directory
    speaker_dir: Path = corpus_dir / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # Get audio filename (without path)
    audio_name: str = Path(audio_rel).stem

    # Create symlink to audio file
    wav_link: Path = speaker_dir / f"{audio_name}.wav"
    if not wav_link.exists():
        try:
            wav_link.symlink_to(audio_path.resolve())
        except OSError:
            # Fall back to relative symlink if absolute fails
            rel_path: Path = os.path.relpath(audio_path.resolve(), speaker_dir)
            wav_link.symlink_to(rel_path)

    # Write transcript (space-separated phones)
    txt_path: Path = speaker_dir / f"{audio_name}.txt"
    transcript: str = " ".join(tokens)
    txt_path.write_text(transcript, encoding="utf-8")

    return audio_name


def prepare_mfa_corpus(
    metadata_paths: list[Path] | None = None,
    training_dir: Path | None = None,
    output_dir: Path | None = None,
    workers: int | None = None,
) -> int:
    """Prepare MFA corpus from training metadata.

    Creates symlinks to audio files and space-separated phone transcripts
    in the structure MFA expects.

    Args:
        metadata_paths: Paths to metadata CSV files. Defaults to train + val.
        training_dir: Directory containing wavs/. Defaults to data/training_cv.
        output_dir: Output corpus directory. Defaults to data/mfa/corpus.
        workers: Number of parallel workers.

    Returns:
        Number of successfully prepared files.
    """
    if training_dir is None:
        training_dir = TRAINING_CV_DIR
    if output_dir is None:
        output_dir = MFA_CORPUS_DIR
    if metadata_paths is None:
        metadata_paths = [
            training_dir / "metadata_train.csv",
            training_dir / "metadata_val.csv",
        ]
    if workers is None:
        workers = max(1, cpu_count() - 1)

    # Clean output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metadata
    all_rows: list[dict[str, str]] = []
    for path in metadata_paths:
        if path.exists():
            rows: list[dict[str, str]] = load_metadata(path)
            all_rows.extend(rows)
            print(f"Loaded {len(rows):,} entries from {path.name}")

    if not all_rows:
        print("No metadata entries found.")
        return 0

    print(f"Total entries: {len(all_rows):,}")

    # Prepare work items
    tasks: list[tuple[dict[str, str], Path, Path]] = [
        (row, training_dir, output_dir) for row in all_rows
    ]

    # Process in parallel
    print(f"Preparing corpus with {workers} workers...")
    successful: int = 0

    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_create_corpus_entry, tasks, chunksize=100),
            total=len(tasks),
            desc="Creating corpus entries",
        ):
            if result is not None:
                successful += 1

    # Count speakers
    speakers: list[str] = [d.name for d in output_dir.iterdir() if d.is_dir()]

    print(f"\nCorpus preparation complete:")
    print(f"  Total entries: {len(all_rows):,}")
    print(f"  Successfully prepared: {successful:,}")
    print(f"  Speakers (locales): {len(speakers)}")
    print(f"  Output directory: {output_dir}")

    return successful


def create_phone_dictionary(
    vocab_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    """Create MFA identity dictionary from IPA vocabulary.

    Each phone maps to itself: phone<TAB>phone

    Args:
        vocab_path: Path to ipa_vocab.json. Defaults to training_cv/ipa_vocab.json.
        output_path: Output dictionary path. Defaults to data/mfa/phone_dict.txt.

    Returns:
        Number of phones in dictionary.
    """
    if vocab_path is None:
        vocab_path = TRAINING_CV_DIR / "ipa_vocab.json"
    if output_path is None:
        output_path = MFA_DIR / "phone_dict.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab: IPAVocabulary = IPAVocabulary.load(vocab_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for phone in vocab.phoneme_tokens:
            # MFA format: word<TAB>pronunciation
            # For us: each "word" is a single phone
            f.write(f"{phone}\t{phone}\n")

    print(f"Dictionary created: {len(vocab.phoneme_tokens)} phones -> {output_path}")
    return len(vocab.phoneme_tokens)


def run_mfa_preparation(
    training_dir: Path | None = None,
    workers: int | None = None,
) -> tuple[int, int]:
    """Run full MFA preparation: corpus + dictionary.

    Args:
        training_dir: Directory containing training data.
        workers: Number of parallel workers.

    Returns:
        (corpus_count, dict_count) tuple.
    """
    if training_dir is None:
        training_dir = TRAINING_CV_DIR

    print("=" * 60)
    print("MFA Corpus Preparation")
    print("=" * 60)

    # Create corpus
    corpus_count: int = prepare_mfa_corpus(
        training_dir=training_dir,
        workers=workers,
    )

    # Create dictionary
    dict_count: int = create_phone_dictionary(
        vocab_path=training_dir / "ipa_vocab.json",
    )

    print("\n" + "=" * 60)
    print("MFA preparation complete.")
    print(f"  Corpus entries: {corpus_count:,}")
    print(f"  Dictionary phones: {dict_count:,}")
    print(f"  Corpus dir: {MFA_CORPUS_DIR}")
    print(f"  Dictionary: {MFA_DIR / 'phone_dict.txt'}")
    print("=" * 60)

    return corpus_count, dict_count
