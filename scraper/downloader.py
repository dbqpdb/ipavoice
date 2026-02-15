"""Audio downloader with rate limiting, resume support, and retries."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "audio"


def download_recordings(
    conn: sqlite3.Connection,
    language_code: str | None = None,
    delay: float = 1.0,
    workers: int = 1,
    data_dir: str | Path | None = None,
) -> None:
    """Download WAV files for recordings in the database."""
    resolved_dir: Path = Path(data_dir) if data_dir else DATA_DIR

    query: str = "SELECT * FROM recordings WHERE wav_url IS NOT NULL AND downloaded=0"
    params: list[str] = []
    if language_code:
        query += " AND language_code=?"
        params.append(language_code.upper())
    query += " ORDER BY language_code, audio_filename"

    recordings: list[sqlite3.Row] = conn.execute(query, params).fetchall()
    if not recordings:
        print("No recordings to download.")
        return

    session: requests.Session = requests.Session()
    session.headers["User-Agent"] = "IPAVoice-Scraper/1.0 (academic research)"

    for rec in tqdm(recordings, desc="Downloading WAV files"):
        lang_dir: Path = resolved_dir / rec["language_code"]
        lang_dir.mkdir(parents=True, exist_ok=True)

        filename: str = rec["audio_filename"] + ".wav"
        filepath: Path = lang_dir / filename

        if filepath.exists():
            # Check file size via HEAD request
            try:
                head: requests.Response = session.head(rec["wav_url"], timeout=10)
                remote_size: int = int(head.headers.get("Content-Length", 0))
                if remote_size > 0 and filepath.stat().st_size >= remote_size:
                    conn.execute("UPDATE recordings SET downloaded=1 WHERE id=?", (rec["id"],))
                    conn.commit()
                    continue
            except requests.RequestException:
                pass

        success: bool = _download_file(session, rec["wav_url"], filepath)
        if success:
            conn.execute("UPDATE recordings SET downloaded=1 WHERE id=?", (rec["id"],))
            conn.commit()

        time.sleep(delay)


def _download_file(session: requests.Session, url: str, filepath: Path, max_retries: int = 3) -> bool:
    """Download a single file with exponential backoff retries."""
    for attempt in range(max_retries):
        try:
            resp: requests.Response = session.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total: int = int(resp.headers.get("Content-Length", 0))
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if total > 0 and filepath.stat().st_size < total:
                print(f"\n  Warning: incomplete download for {filepath.name}")
                filepath.unlink(missing_ok=True)
                continue

            return True

        except requests.RequestException as e:
            wait: int = 2 ** attempt
            print(f"\n  Retry {attempt + 1}/{max_retries} for {filepath.name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print(f"\n  Failed to download {filepath.name}")
                filepath.unlink(missing_ok=True)

    return False
