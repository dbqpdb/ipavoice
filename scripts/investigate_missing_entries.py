"""Investigate why 7 word-list recordings have 0 parsed entries.

Recordings:
  - amh_word-list_1967_01: wordlist_url points to conversation page (404 with #1 anchor)
  - aqc_word-list_0000_02: wordlist_url is NULL
  - hye_word-list_1973_01: wordlist_url exists, page has data, parser returns 0
  - hye_word-list_1983_01: wordlist_url exists, page has data, parser returns 0
  - run_word-list_1965_01: wordlist_url is NULL
  - run_word-list_1972_03: wordlist_url is NULL
  - pav_word-list_1995_61: wordlist_url is NULL

Run: uv run python -m scripts.investigate_missing_entries

FINDINGS:
  Category 1 — AMH: wordlist_url scraped incorrectly (points to conversation page, 404)
  Category 2 — AQC, RUN×2, PAV: No word list HTML exists on the archive (NULL URL, 404 at expected path)
  Category 3 — HYE×2: Malformed HTML tables. Headers have 4 columns (Entry, Orthography,
    Transcription, English) but data rows only have 2 <td> cells:
      cell 0 = "1գիր" (entry_number + orthography merged)
      cell 1 = "kʰiɾ writing" (IPA + english merged with nbsp)
    Parser's column map assigns ipa_col=2 and english_col=3 based on headers, but those
    indices don't exist in the 2-cell rows, so _get() returns None for both.
    Fix would require detecting column-count mismatch and falling back to a regex-based
    split of the merged cells.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from processing.database import get_connection
from scraper.wordlist_parser import parse_wordlist

FILENAMES: list[str] = [
    "amh_word-list_1967_01",
    "aqc_word-list_0000_02",
    "hye_word-list_1973_01",
    "hye_word-list_1983_01",
    "run_word-list_1965_01",
    "run_word-list_1972_03",
    "pav_word-list_1995_61",
]


def main() -> None:
    conn: sqlite3.Connection = get_connection()
    session: requests.Session = requests.Session()
    session.headers["User-Agent"] = "IPAVoice-Scraper/1.0 (academic research)"

    print("=" * 70)
    print("STEP 1: Database state for each recording")
    print("=" * 70)

    for fn in FILENAMES:
        row = conn.execute(
            "SELECT id, language_code, wordlist_url, entry_start, entry_end "
            "FROM recordings WHERE audio_filename=?",
            (fn,),
        ).fetchone()
        if not row:
            print(f"\n{fn}: NOT IN DATABASE")
            continue
        cnt: int = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE recording_id=?", (row["id"],)
        ).fetchone()[0]
        print(f"\n{fn}:")
        print(f"  language_code:  {row['language_code']}")
        print(f"  wordlist_url:   {row['wordlist_url']!r}")
        print(f"  entry_range:    {row['entry_start']}-{row['entry_end']}")
        print(f"  entries in DB:  {cnt}")

    print("\n")
    print("=" * 70)
    print("STEP 2: Fetch and inspect pages that have URLs")
    print("=" * 70)

    urls_to_check: list[tuple[str, str]] = []
    for fn in FILENAMES:
        row = conn.execute(
            "SELECT wordlist_url FROM recordings WHERE audio_filename=?", (fn,)
        ).fetchone()
        if row and row["wordlist_url"]:
            # Strip fragment (#1) for clean fetch
            url: str = row["wordlist_url"].split("#")[0]
            urls_to_check.append((fn, url))

    for fn, url in urls_to_check:
        print(f"\n--- {fn} ---")
        print(f"  URL: {url}")
        try:
            resp: requests.Response = session.get(url, timeout=15)
            print(f"  Status: {resp.status_code}")
            if resp.status_code != 200:
                continue
        except Exception as e:
            print(f"  Fetch error: {e}")
            continue

        soup: BeautifulSoup = BeautifulSoup(resp.content, "lxml")
        tables: list[Tag] = soup.find_all("table")
        print(f"  Tables found: {len(tables)}")

        for i, table in enumerate(tables):
            rows = table.find_all("tr")
            print(f"  Table {i}: {len(rows)} rows")
            if rows:
                header_cells = rows[0].find_all(["th", "td"])
                print(f"    Headers ({len(header_cells)}): {[c.get_text(strip=True)[:30] for c in header_cells]}")
            if len(rows) > 1:
                data_cells = rows[1].find_all(["td", "th"])
                print(f"    Row 1 cells ({len(data_cells)}): {[c.get_text(strip=True)[:40] for c in data_cells]}")
                # Show raw HTML of first data row to see colspan / structure issues
                print(f"    Row 1 raw HTML: {str(rows[1])[:300]}")
            if len(rows) > 2:
                data_cells = rows[2].find_all(["td", "th"])
                print(f"    Row 2 cells ({len(data_cells)}): {[c.get_text(strip=True)[:40] for c in data_cells]}")

    print("\n")
    print("=" * 70)
    print("STEP 3: Test parser on URLs that returned 200")
    print("=" * 70)

    for fn, url in urls_to_check:
        print(f"\n--- {fn} ---")
        try:
            entries = parse_wordlist(url, session)
            print(f"  Parser returned: {len(entries)} entries")
            if entries:
                print(f"  First entry: {entries[0]}")
                print(f"  Last entry:  {entries[-1]}")
        except Exception as e:
            print(f"  Parser error: {e}")

    print("\n")
    print("=" * 70)
    print("STEP 4: Check language pages for recordings with NULL wordlist_url")
    print("=" * 70)

    null_url_fns: list[str] = [
        fn for fn in FILENAMES
        if conn.execute(
            "SELECT wordlist_url FROM recordings WHERE audio_filename=?", (fn,)
        ).fetchone()["wordlist_url"] is None
    ]

    for fn in null_url_fns:
        row = conn.execute(
            "SELECT r.language_code, l.url as lang_url "
            "FROM recordings r JOIN languages l ON r.language_code = l.code "
            "WHERE r.audio_filename=?",
            (fn,),
        ).fetchone()
        print(f"\n--- {fn} (language: {row['language_code']}) ---")
        print(f"  Language page: {row['lang_url']}")
        # Check if a wordlist HTML file exists at the expected location
        expected_url: str = row["lang_url"].rsplit("/", 1)[0] + f"/{fn}.html"
        print(f"  Expected wordlist URL: {expected_url}")
        try:
            resp = session.head(expected_url, timeout=10)
            print(f"  HEAD status: {resp.status_code}")
        except Exception as e:
            print(f"  HEAD error: {e}")

    conn.close()
    print("\n\nDone.")


if __name__ == "__main__":
    main()
