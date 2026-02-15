"""CLI entry point for the UCLA Phonetics Archive scraper.

Usage:
    python -m scraper.run metadata [--language CODE]
    python -m scraper.run download [--language CODE] [--delay SECONDS] [--workers N]
    python -m scraper.run segment  [--language CODE]
    python -m scraper.run export   [--language CODE] [--output FILE]
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

    args: argparse.Namespace = parser.parse_args()

    commands: dict[str, Callable[[argparse.Namespace], None]] = {
        "metadata": cmd_metadata,
        "download": cmd_download,
        "segment": cmd_segment,
        "export": cmd_export,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
