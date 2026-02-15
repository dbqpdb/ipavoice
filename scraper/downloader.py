"""Audio downloader with rate limiting, resume support, retries, and parallel downloads."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "audio"

# SQLite busy timeout (ms) for concurrent writers
_BUSY_TIMEOUT_MS: int = 30000


def _get_worker_conn() -> sqlite3.Connection:
    """Get a SQLite connection configured for concurrent worker use."""
    from processing.database import DEFAULT_DB_PATH
    conn: sqlite3.Connection = sqlite3.connect(str(DEFAULT_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.row_factory = sqlite3.Row
    return conn


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

    rec_dicts: list[dict[str, Any]] = [dict(rec) for rec in recordings]

    if workers > 1:
        _download_parallel(rec_dicts, resolved_dir, delay, workers)
    else:
        _download_serial(rec_dicts, resolved_dir, delay)


def _download_serial(
    recordings: list[dict[str, Any]],
    data_dir: Path,
    delay: float,
) -> None:
    """Download recordings one at a time (original behavior)."""
    conn: sqlite3.Connection = _get_worker_conn()
    session: requests.Session = _make_session()

    for rec in tqdm(recordings, desc="Downloading WAV files"):
        _process_one(conn, session, rec, data_dir)
        time.sleep(delay)

    conn.close()


def _download_parallel(
    recordings: list[dict[str, Any]],
    data_dir: Path,
    delay: float,
    workers: int,
) -> None:
    """Download recordings across multiple threads with rate limiting."""
    print(f"  Downloading {len(recordings)} files with {workers} workers")

    pbar: tqdm[None] = tqdm(total=len(recordings), desc="Downloading WAV files")

    def _worker(rec: dict[str, Any]) -> tuple[str, bool]:
        # Each thread gets its own session and DB connection
        conn: sqlite3.Connection = _get_worker_conn()
        session: requests.Session = _make_session()
        try:
            _process_one(conn, session, rec, data_dir)
            return (rec["audio_filename"], True)
        except Exception as e:
            print(f"\n  Error downloading {rec['audio_filename']}: {e}")
            return (rec["audio_filename"], False)
        finally:
            conn.close()
            time.sleep(delay)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, rec): rec for rec in recordings}
        for future in as_completed(futures):
            future.result()
            pbar.update(1)

    pbar.close()


def _make_session() -> requests.Session:
    """Create a requests session with appropriate headers."""
    session: requests.Session = requests.Session()
    session.headers["User-Agent"] = "IPAVoice-Scraper/1.0 (academic research)"
    return session


def _process_one(
    conn: sqlite3.Connection,
    session: requests.Session,
    rec: dict[str, Any],
    data_dir: Path,
) -> None:
    """Download a single recording and mark it in the database."""
    lang_dir: Path = data_dir / rec["language_code"]
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
                return
        except requests.RequestException:
            pass

    success: bool = _download_file(session, rec["wav_url"], filepath)
    if success:
        conn.execute("UPDATE recordings SET downloaded=1 WHERE id=?", (rec["id"],))
        conn.commit()


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
