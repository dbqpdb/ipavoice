"""SQLite database schema and operations for the IPA Voice project."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "db" / "ipavoice.db"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path: Path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn: sqlite3.Connection = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS languages (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language_code TEXT NOT NULL REFERENCES languages(code),
            recording_type TEXT,
            year INTEGER,
            sequence INTEGER,
            audio_filename TEXT,
            wav_url TEXT,
            entry_start INTEGER,
            entry_end INTEGER,
            additional_info TEXT,
            wordlist_url TEXT,
            downloaded BOOLEAN DEFAULT 0,
            UNIQUE(language_code, audio_filename)
        );

        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL REFERENCES recordings(id),
            entry_number INTEGER,
            ipa TEXT,
            english TEXT,
            orthography TEXT,
            sound_illustrated TEXT,
            language_code TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL REFERENCES entries(id),
            recording_id INTEGER NOT NULL REFERENCES recordings(id),
            segment_file TEXT,
            start_ms INTEGER,
            end_ms INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_recordings_lang ON recordings(language_code);
        CREATE INDEX IF NOT EXISTS idx_entries_recording ON entries(recording_id);
        CREATE INDEX IF NOT EXISTS idx_segments_entry ON segments(entry_id);
    """)
    conn.commit()


def upsert_language(conn: sqlite3.Connection, code: str, name: str, url: str) -> None:
    conn.execute(
        "INSERT INTO languages (code, name, url) VALUES (?, ?, ?) "
        "ON CONFLICT(code) DO UPDATE SET name=excluded.name, url=excluded.url",
        (code, name, url),
    )


def upsert_recording(conn: sqlite3.Connection, language_code: str, audio_filename: str, **kwargs: Any) -> int:
    existing: sqlite3.Row | None = conn.execute(
        "SELECT id FROM recordings WHERE language_code=? AND audio_filename=?",
        (language_code, audio_filename),
    ).fetchone()

    if existing:
        sets: str = ", ".join(f"{k}=?" for k in kwargs)
        vals: list[Any] = list(kwargs.values()) + [existing["id"]]
        conn.execute(f"UPDATE recordings SET {sets} WHERE id=?", vals)
        return existing["id"]
    else:
        cols: list[str] = ["language_code", "audio_filename"] + list(kwargs.keys())
        placeholders: str = ", ".join(["?"] * len(cols))
        vals = [language_code, audio_filename] + list(kwargs.values())
        cur: sqlite3.Cursor = conn.execute(
            f"INSERT INTO recordings ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        return cur.lastrowid


def insert_entries(conn: sqlite3.Connection, recording_id: int, language_code: str, entries: list[dict[str, Any]]) -> None:
    """Insert word list entries. entries is a list of dicts with keys:
    entry_number, ipa, english, orthography, sound_illustrated."""
    # Delete existing entries for this recording to allow re-scraping
    conn.execute("DELETE FROM entries WHERE recording_id=?", (recording_id,))
    for entry in entries:
        conn.execute(
            "INSERT INTO entries (recording_id, entry_number, ipa, english, "
            "orthography, sound_illustrated, language_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                recording_id,
                entry.get("entry_number"),
                entry.get("ipa"),
                entry.get("english"),
                entry.get("orthography"),
                entry.get("sound_illustrated"),
                language_code,
            ),
        )


def insert_segment(conn: sqlite3.Connection, entry_id: int, recording_id: int, segment_file: str, start_ms: int, end_ms: int) -> None:
    conn.execute(
        "INSERT INTO segments (entry_id, recording_id, segment_file, start_ms, end_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (entry_id, recording_id, segment_file, start_ms, end_ms),
    )


def mark_downloaded(conn: sqlite3.Connection, recording_id: int) -> None:
    conn.execute("UPDATE recordings SET downloaded=1 WHERE id=?", (recording_id,))


def get_languages(conn: sqlite3.Connection, code: str | None = None) -> list[sqlite3.Row]:
    if code:
        return conn.execute("SELECT * FROM languages WHERE code=?", (code.upper(),)).fetchall()
    return conn.execute("SELECT * FROM languages ORDER BY name").fetchall()


def get_recordings(
    conn: sqlite3.Connection,
    language_code: str | None = None,
    recording_type: str | None = None,
    downloaded: bool | None = None,
) -> list[sqlite3.Row]:
    query: str = "SELECT * FROM recordings WHERE 1=1"
    params: list[Any] = []
    if language_code:
        query += " AND language_code=?"
        params.append(language_code.upper())
    if recording_type:
        query += " AND recording_type=?"
        params.append(recording_type)
    if downloaded is not None:
        query += " AND downloaded=?"
        params.append(int(downloaded))
    return conn.execute(query, params).fetchall()


def get_entries(
    conn: sqlite3.Connection,
    recording_id: int | None = None,
    language_code: str | None = None,
) -> list[sqlite3.Row]:
    query: str = "SELECT * FROM entries WHERE 1=1"
    params: list[Any] = []
    if recording_id:
        query += " AND recording_id=?"
        params.append(recording_id)
    if language_code:
        query += " AND language_code=?"
        params.append(language_code.upper())
    query += " ORDER BY entry_number"
    return conn.execute(query, params).fetchall()
