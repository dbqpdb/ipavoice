"""Parse per-language recording tables from UCLA Phonetics Lab Archive."""

from __future__ import annotations

import re
from typing import TypedDict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag


class RecordingInfo(TypedDict):
    audio_filename: str | None
    wav_url: str | None
    wordlist_url: str | None
    entry_start: int | None
    entry_end: int | None
    additional_info: str | None
    recording_type: str | None
    year: int | None
    sequence: int | None


def parse_language_page(url: str, session: requests.Session | None = None) -> list[RecordingInfo]:
    """Parse a language page and extract recording metadata."""
    session = session or requests.Session()
    resp: requests.Response = session.get(url)
    resp.raise_for_status()
    soup: BeautifulSoup = BeautifulSoup(resp.content, "lxml")

    recordings: list[RecordingInfo] = []
    table: Tag | None = _find_recordings_table(soup)
    if not table:
        return recordings

    headers: list[str] = _parse_headers(table)
    rows: list[Tag] = table.find_all("tr")[1:]  # skip header row

    for row in rows:
        cells: list[Tag] = row.find_all("td")
        if len(cells) < 4:
            continue

        rec: RecordingInfo | None = _parse_row(cells, headers, url)
        if rec and rec.get("audio_filename"):
            recordings.append(rec)

    return recordings


def _find_recordings_table(soup: BeautifulSoup) -> Tag | None:
    """Find the main recordings table on the page."""
    # Look for a table that contains WAV links
    for table in soup.find_all("table"):
        text: str = table.get_text(separator=" ").lower()
        if "wav" in text and ("word" in text or "audio" in text or "entries" in text):
            return table
    # Fallback: largest table on the page
    tables: list[Tag] = soup.find_all("table")
    if tables:
        return max(tables, key=lambda t: len(t.find_all("tr")))
    return None


def _parse_headers(table: Tag) -> list[str]:
    """Extract header column names from the first row."""
    header_row: Tag | None = table.find("tr")
    if not header_row:
        return []
    headers: list[str] = []
    for cell in header_row.find_all(["th", "td"]):
        text: str = cell.get_text(strip=True).lower()
        headers.append(text)
    return headers


def _find_col(headers: list[str], *keywords: str) -> int | None:
    """Find the index of a column whose header contains any of the keywords."""
    for i, h in enumerate(headers):
        for kw in keywords:
            if kw in h:
                return i
    return None


def _parse_row(cells: list[Tag], headers: list[str], base_url: str) -> RecordingInfo:
    """Parse a single table row into a recording dict."""
    rec: RecordingInfo = {
        "audio_filename": None,
        "wav_url": None,
        "wordlist_url": None,
        "entry_start": None,
        "entry_end": None,
        "additional_info": None,
        "recording_type": None,
        "year": None,
        "sequence": None,
    }

    # Strategy: find WAV link, word-list link, and entry range from cells
    for i, cell in enumerate(cells):
        text: str = cell.get_text(strip=True)
        link: Tag | None = cell.find("a", href=True)

        # WAV link
        if link and link["href"].lower().endswith(".wav"):
            rec["wav_url"] = urljoin(base_url, link["href"])
            # Audio filename is often in an adjacent cell or the link text
            filename: str = link["href"].rsplit("/", 1)[-1].replace(".wav", "")
            rec["audio_filename"] = filename

        # Word list / recording link (HTML page)
        if link and re.search(r"(word-list|sentence|story|conversation|narrative).*\.html?", link["href"], re.IGNORECASE):
            rec["wordlist_url"] = urljoin(base_url, link["href"])

        # Entry range like "1 - 27" or "1-27"
        range_match: re.Match[str] | None = re.match(r"(\d+)\s*[-–]\s*(\d+)", text)
        if range_match and rec["entry_start"] is None:
            rec["entry_start"] = int(range_match.group(1))
            rec["entry_end"] = int(range_match.group(2))

    # Try to get audio filename from a cell that looks like a filename
    if not rec["audio_filename"]:
        for cell in cells:
            text = cell.get_text(strip=True)
            if re.match(r"[a-z]{2,3}_\w+_\d{4}_\d{2}", text, re.IGNORECASE):
                rec["audio_filename"] = text
                break

    # Try to build WAV URL from filename if we have a filename but no WAV URL
    if rec["audio_filename"] and not rec["wav_url"]:
        rec["wav_url"] = urljoin(base_url, rec["audio_filename"] + ".wav")

    # Parse recording type and year/sequence from filename
    if rec["audio_filename"]:
        fn: str = rec["audio_filename"]
        type_match: re.Match[str] | None = re.search(r"_(word-list|sentence|story|conversation|narrative)_", fn, re.IGNORECASE)
        if type_match:
            rec["recording_type"] = type_match.group(1).lower()

        year_match: re.Match[str] | None = re.search(r"_(\d{4})_(\d{2})$", fn)
        if year_match:
            rec["year"] = int(year_match.group(1))
            rec["sequence"] = int(year_match.group(2))

    # Additional info: look for cells with longer text that aren't links/filenames
    for cell in cells:
        text = cell.get_text(strip=True)
        if len(text) > 15 and not cell.find("a") and not re.match(r"[\d\s\-–]+$", text):
            rec["additional_info"] = text
            break

    # Also check header-mapped columns for additional info
    info_col: int | None = _find_col(headers, "additional", "speaker", "info")
    if info_col is not None and info_col < len(cells):
        info_text: str = cells[info_col].get_text(strip=True)
        if info_text:
            rec["additional_info"] = info_text

    return rec
