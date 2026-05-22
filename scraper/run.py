"""CLI entry point for the UCLA Phonetics Archive scraper.

Usage:
    python -m scraper.run metadata    [--language CODE]
    python -m scraper.run download    [--language CODE] [--delay SECONDS] [--workers N]
    python -m scraper.run segment     [--language CODE]
    python -m scraper.run export      [--language CODE] [--output FILE]
    python -m scraper.run preprocess  [--workers N]
    python -m scraper.run train       [--test-run] [--resume PATH]

MFA alignment pipeline:
    python -m scraper.run mfa-prepare   [--workers N]
    python -m scraper.run mfa-align     [--jobs N] [--retrain]
    python -m scraper.run mfa-extract   [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import requests
from tqdm import tqdm

from processing.database import (
    get_connection,
    init_db,
    upsert_language,
    upsert_recording,
    insert_entries,
    get_languages,
    get_recordings,
    get_entries,
)
from scraper.index_parser import LanguageInfo, fetch_languages
from scraper.language_parser import RecordingInfo, parse_language_page
from scraper.wordlist_parser import WordEntry, parse_wordlist
from scraper.downloader import download_recordings
from processing.segmenter import segment_all
from training.preprocess import run_preprocessing
from training.preprocess_cv import run_cv_preprocessing
from training.mfa_corpus import run_mfa_preparation
from training.mfa_align import run_mfa_alignment
from training.extract_durations import run_duration_extraction
from ipavoice.train import train as run_training
from training.config import DATASET_UCLA, VALID_DATASETS


def cmd_metadata(args: argparse.Namespace) -> None:
    """Scrape metadata: languages, recordings, and word list entries."""
    conn: sqlite3.Connection = get_connection()
    init_db(conn)
    session: requests.Session = requests.Session()
    session.headers["User-Agent"] = "IPAVoice-Scraper/1.0 (academic research)"

    # Phase 1: Fetch language index
    if args.language:
        # If a specific language is requested but not in DB, we still need the index
        existing: list[sqlite3.Row] = get_languages(conn, args.language)
        if not existing:
            print("Language not in database, fetching index first...")
            _scrape_index(conn, session)
        existing = get_languages(conn, args.language)
        if not existing:
            print(f"Language code '{args.language}' not found in archive.")
            return
        langs: list[sqlite3.Row] = existing
    else:
        _scrape_index(conn, session)
        langs = get_languages(conn)

    print(f"\nProcessing {len(langs)} language(s)...")

    # Phase 2: Parse each language page for recordings
    for lang in tqdm(langs, desc="Parsing language pages"):
        try:
            recordings: list[RecordingInfo] = parse_language_page(lang["url"], session)
            for rec in recordings:
                upsert_recording(
                    conn,
                    lang["code"],
                    rec["audio_filename"],
                    recording_type=rec["recording_type"],
                    year=rec["year"],
                    sequence=rec["sequence"],
                    wav_url=rec["wav_url"],
                    entry_start=rec["entry_start"],
                    entry_end=rec["entry_end"],
                    additional_info=rec["additional_info"],
                    wordlist_url=rec["wordlist_url"],
                )
            conn.commit()
        except Exception as e:
            print(f"\n  Error parsing {lang['name']}: {e}")
        time.sleep(args.delay)

    # Phase 3: Parse word list pages for entries
    recs: list[sqlite3.Row] = get_recordings(conn, language_code=args.language, recording_type="word-list")
    print(f"\nParsing {len(recs)} word list(s)...")

    for rec in tqdm(recs, desc="Parsing word lists"):
        if not rec["wordlist_url"]:
            continue
        try:
            entries: list[WordEntry] = parse_wordlist(rec["wordlist_url"], session)
            if entries:
                insert_entries(conn, rec["id"], rec["language_code"], entries)
                conn.commit()
        except Exception as e:
            print(f"\n  Error parsing word list {rec['audio_filename']}: {e}")
        time.sleep(args.delay)

    # Summary
    n_langs: int = conn.execute("SELECT COUNT(*) FROM languages").fetchone()[0]
    n_recs: int = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    n_entries: int = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    print(f"\nDone. {n_langs} languages, {n_recs} recordings, {n_entries} entries.")
    conn.close()


def _scrape_index(conn: sqlite3.Connection, session: requests.Session) -> None:
    """Fetch and store the language index."""
    print("Fetching language index...")
    languages: list[LanguageInfo] = fetch_languages(session)
    print(f"Found {len(languages)} languages.")
    for lang in languages:
        upsert_language(conn, lang["code"], lang["name"], lang["url"])
    conn.commit()


def cmd_download(args: argparse.Namespace) -> None:
    """Download WAV audio files."""
    conn: sqlite3.Connection = get_connection()
    init_db(conn)
    download_recordings(
        conn,
        language_code=args.language,
        delay=args.delay,
        workers=args.workers,
    )
    conn.close()


def cmd_segment(args: argparse.Namespace) -> None:
    """Segment word-list audio into per-entry files."""
    conn: sqlite3.Connection = get_connection()
    init_db(conn)
    segment_all(
        conn,
        language_code=args.language,
        force=args.force,
        workers=args.workers or None,
        use_ffmpeg=args.ffmpeg,
    )
    conn.close()


def cmd_export(args: argparse.Namespace) -> None:
    """Export training-ready JSON manifest."""
    conn: sqlite3.Connection = get_connection()
    init_db(conn)

    query: str = """
        SELECT
            e.id, e.entry_number, e.ipa, e.english, e.orthography,
            e.language_code, l.name as language_name,
            r.recording_type, r.audio_filename,
            s.segment_file
        FROM entries e
        JOIN recordings r ON e.recording_id = r.id
        JOIN languages l ON e.language_code = l.code
        LEFT JOIN segments s ON s.entry_id = e.id
        WHERE e.ipa IS NOT NULL AND e.ipa != ''
    """
    params: list[str] = []
    if args.language:
        query += " AND e.language_code = ?"
        params.append(args.language.upper())
    query += " ORDER BY e.language_code, e.entry_number"

    rows: list[sqlite3.Row] = conn.execute(query, params).fetchall()

    manifest: list[dict[str, Any]] = []
    for row in rows:
        code: str = row["language_code"].lower()
        entry_num: int = row["entry_number"] or 0
        item: dict[str, Any] = {
            "id": f"{code}_{entry_num:03d}",
            "language": row["language_name"],
            "language_code": row["language_code"],
            "ipa": row["ipa"],
            "english": row["english"],
            "recording_type": row["recording_type"],
        }
        if row["segment_file"]:
            # Store as relative path from project root
            try:
                item["audio_path"] = str(Path(row["segment_file"]).relative_to(Path.cwd()))
            except ValueError:
                item["audio_path"] = row["segment_file"]
        else:
            item["audio_path"] = f"data/audio/{row['language_code']}/{row['audio_filename']}.wav"

        if row["orthography"]:
            item["orthography"] = row["orthography"]

        manifest.append(item)

    output: str = args.output or "data/manifest.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(manifest)} entries to {output}")
    conn.close()


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess audio and build training dataset."""
    run_preprocessing(workers=args.workers or None)


def cmd_preprocess_cv(args: argparse.Namespace) -> None:
    """Preprocess Common Voice Spontaneous Speech dataset."""
    run_cv_preprocessing(
        cv_parquet=args.cv_parquet,
        cv_base_dir=args.cv_base,
        workers=args.workers or None,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train VITS model."""
    run_training(
        test_run=args.test_run,
        resume_path=args.resume,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        mixed_precision=args.mixed_precision,
        num_loader_workers=args.workers,
        dataset=args.dataset,
    )


def cmd_mfa_prepare(args: argparse.Namespace) -> None:
    """Prepare MFA corpus structure and phone dictionary."""
    run_mfa_preparation(workers=args.workers or None)


def cmd_mfa_align(args: argparse.Namespace) -> None:
    """Run MFA alignment to generate TextGrids with phone boundaries."""
    run_mfa_alignment(
        num_jobs=args.jobs or None,
        retrain=args.retrain,
    )


def cmd_mfa_extract(args: argparse.Namespace) -> None:
    """Extract phone durations from MFA TextGrid alignments."""
    run_duration_extraction(
        workers=args.workers or None,
        update_metadata=not args.no_update,
        validate=not args.no_validate,
    )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="UCLA Phonetics Archive scraper for IPA-to-speech training data"
    )
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="command", required=True
    )

    # metadata
    p_meta: argparse.ArgumentParser = sub.add_parser(
        "metadata", help="Scrape all metadata (languages, recordings, entries)"
    )
    p_meta.add_argument("--language", type=str, help="Limit to a specific language code (e.g. ABQ)")
    p_meta.add_argument("--delay", type=float, default=0.5, help="Seconds between requests (default: 0.5)")

    # download
    p_dl: argparse.ArgumentParser = sub.add_parser("download", help="Download WAV audio files")
    p_dl.add_argument("--language", type=str, help="Limit to a specific language code")
    p_dl.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads (default: 1.0)")
    p_dl.add_argument("--workers", type=int, default=1, help="Number of parallel downloads (default: 1)")

    # segment
    p_seg: argparse.ArgumentParser = sub.add_parser(
        "segment", help="Segment word-list audio into per-entry files"
    )
    p_seg.add_argument("--language", type=str, help="Limit to a specific language code")
    p_seg.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1)")
    p_seg.add_argument("--ffmpeg", action="store_true", help="Use ffmpeg backend (faster, fewer segments)")
    p_seg.add_argument("--force", action="store_true", help="Re-segment even if segments already exist")

    # export
    p_exp: argparse.ArgumentParser = sub.add_parser("export", help="Export training-ready JSON manifest")
    p_exp.add_argument("--language", type=str, help="Limit to a specific language code")
    p_exp.add_argument("--output", type=str, help="Output file path (default: data/manifest.json)")

    # preprocess
    p_pre: argparse.ArgumentParser = sub.add_parser(
        "preprocess", help="Preprocess audio and build training dataset"
    )
    p_pre.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1)")

    # preprocess-cv
    p_pre_cv: argparse.ArgumentParser = sub.add_parser(
        "preprocess-cv", help="Preprocess Common Voice Spontaneous Speech dataset"
    )
    p_pre_cv.add_argument(
        "--cv-parquet", type=str, required=True,
        help="Path to unified.parquet file from Common Voice Spontaneous corpus"
    )
    p_pre_cv.add_argument(
        "--cv-base", type=str, default=None,
        help="Path to CommonVoiceSpontaneous directory (inferred from parquet path if not set)"
    )
    p_pre_cv.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1)")

    # train
    p_train: argparse.ArgumentParser = sub.add_parser("train", help="Train VITS model")
    p_train.add_argument("--test-run", action="store_true", help="Run 1000 steps to validate pipeline")
    p_train.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    p_train.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    p_train.add_argument("--eval-batch-size", type=int, default=16, help="Eval batch size (default: 16)")
    p_train.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training")
    p_train.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    p_train.add_argument(
        "--dataset", type=str, default=DATASET_UCLA, choices=VALID_DATASETS,
        help=f"Dataset source: {', '.join(VALID_DATASETS)} (default: {DATASET_UCLA})"
    )

    # mfa-prepare
    p_mfa_prep: argparse.ArgumentParser = sub.add_parser(
        "mfa-prepare", help="Prepare MFA corpus structure and phone dictionary"
    )
    p_mfa_prep.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1)")

    # mfa-align
    p_mfa_align: argparse.ArgumentParser = sub.add_parser(
        "mfa-align", help="Run MFA alignment to generate TextGrids"
    )
    p_mfa_align.add_argument("--jobs", type=int, default=0, help="MFA parallel jobs (default: auto)")
    p_mfa_align.add_argument("--retrain", action="store_true", help="Force retraining even if model exists")

    # mfa-extract
    p_mfa_ext: argparse.ArgumentParser = sub.add_parser(
        "mfa-extract", help="Extract phone durations from MFA TextGrids"
    )
    p_mfa_ext.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1)")
    p_mfa_ext.add_argument("--no-update", action="store_true", help="Don't update metadata CSVs")
    p_mfa_ext.add_argument("--no-validate", action="store_true", help="Don't validate phone counts")

    args: argparse.Namespace = parser.parse_args()

    commands: dict[str, Callable[[argparse.Namespace], None]] = {
        "metadata": cmd_metadata,
        "download": cmd_download,
        "segment": cmd_segment,
        "export": cmd_export,
        "preprocess": cmd_preprocess,
        "preprocess-cv": cmd_preprocess_cv,
        "train": cmd_train,
        "mfa-prepare": cmd_mfa_prepare,
        "mfa-align": cmd_mfa_align,
        "mfa-extract": cmd_mfa_extract,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
