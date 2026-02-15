"""Parse word list HTML pages to extract IPA entries."""

from __future__ import annotations

import re
from typing import TypedDict

import requests
from bs4 import BeautifulSoup, Tag


# IPA character ranges — used to detect which column has IPA content
_IPA_PATTERN: re.Pattern[str] = re.compile(
    r"["
    r"\u0250-\u02AF"  # IPA Extensions
    r"\u0300-\u036F"  # Combining Diacritical Marks
    r"\u0190-\u01FF"  # Latin Extended-B (partial)
    r"\u02B0-\u02FF"  # Spacing Modifier Letters
    r"\u1D00-\u1D7F"  # Phonetic Extensions
    r"\u1D80-\u1DBF"  # Phonetic Extensions Supplement
    r"\u0294"         # glottal stop ʔ
    r"\u01C0-\u01C3"  # clicks
    r"ː"             # length mark
    r"\u2070-\u209F"  # superscript numbers (tone)
    r"]"
)


class WordEntry(TypedDict):
    entry_number: int | None
    ipa: str | None
    english: str | None
    orthography: str | None
    sound_illustrated: str | None


class _ColumnMap(TypedDict):
    entry_col: int | None
    ipa_col: int | None
    english_col: int | None
    ortho_col: int | None
    sound_col: int | None


def parse_wordlist(url: str, session: requests.Session | None = None) -> list[WordEntry]:
    """Parse a word list HTML page."""
    session = session or requests.Session()
    resp: requests.Response = session.get(url)
    resp.raise_for_status()
    soup: BeautifulSoup = BeautifulSoup(resp.content, "lxml")

    table: Tag | None = _find_wordlist_table(soup)
    if not table:
        return []

    rows: list[Tag] = table.find_all("tr")
    if len(rows) < 2:
        return []

    headers: list[str] = _get_headers(rows[0])
    data_rows: list[Tag] = rows[1:]

    # Detect merged-cell tables: more headers than actual data cells per row.
    # e.g. HYE pages have 4 headers but only 2 <td> per row (entry+ortho, ipa+english).
    max_data_cells: int = max(
        (len(row.find_all(["td", "th"]) ) for row in data_rows[:5]),
        default=0,
    )
    use_merged_parser: bool = len(headers) >= 4 and max_data_cells == 2

    col_map: _ColumnMap = _detect_columns(headers, data_rows)

    entries: list[WordEntry] = []
    for row in data_rows:
        cells: list[Tag] = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        if use_merged_parser:
            entry: WordEntry | None = _parse_merged_row(cells)
        else:
            entry = _parse_entry(cells, col_map)
        if entry and (entry.get("ipa") or entry.get("english")):
            entries.append(entry)

    return entries


def _find_wordlist_table(soup: BeautifulSoup) -> Tag | None:
    """Find the word list table — usually the largest table with numbered rows."""
    tables: list[Tag] = soup.find_all("table")
    best: Tag | None = None
    best_score: int = 0
    for table in tables:
        rows: list[Tag] = table.find_all("tr")
        score: int = len(rows)
        # Bonus if first column cells contain numbers
        for row in rows[1:3]:
            cells: list[Tag] = row.find_all("td")
            if cells and cells[0].get_text(strip=True).isdigit():
                score += 50
        if score > best_score:
            best_score = score
            best = table
    return best


def _get_headers(header_row: Tag) -> list[str]:
    """Extract header texts from the first row."""
    headers: list[str] = []
    for cell in header_row.find_all(["th", "td"]):
        text: str = cell.get_text(strip=True).lower()
        headers.append(text)
    return headers


def _detect_columns(headers: list[str], data_rows: list[Tag]) -> _ColumnMap:
    """Detect which column serves what purpose."""
    n_cols: int = len(headers) if headers else 0
    # Update from data rows if headers are sparse
    for row in data_rows[:3]:
        cells: list[Tag] = row.find_all(["td", "th"])
        n_cols = max(n_cols, len(cells))

    col_map: _ColumnMap = {
        "entry_col": None,
        "ipa_col": None,
        "english_col": None,
        "ortho_col": None,
        "sound_col": None,
    }

    # Step 1: Try header-based detection
    for i, h in enumerate(headers):
        if h in ("entry", "#", "no.", "no"):
            col_map["entry_col"] = i
        elif "transcription" in h or "phonetic" in h:
            col_map["ipa_col"] = i
        elif h == "english" or "english" in h or "gloss" in h:
            col_map["english_col"] = i
        elif "orthograph" in h:
            col_map["ortho_col"] = i
        elif "sound" in h or "phoneme" in h:
            col_map["sound_col"] = i

    # Step 2: If IPA column not found by header, detect by content
    if col_map["ipa_col"] is None:
        ipa_scores: list[int] = [0] * n_cols
        for row in data_rows[:10]:
            cells = row.find_all(["td", "th"])
            for i, cell in enumerate(cells):
                text: str = cell.get_text(strip=True)
                if _IPA_PATTERN.search(text):
                    ipa_scores[i] += 1
        if any(s > 0 for s in ipa_scores):
            col_map["ipa_col"] = ipa_scores.index(max(ipa_scores))

    # Step 3: Entry column — first column with numbers
    if col_map["entry_col"] is None:
        for row in data_rows[:5]:
            cells = row.find_all(["td", "th"])
            for i, cell in enumerate(cells):
                if cell.get_text(strip=True).isdigit():
                    col_map["entry_col"] = i
                    break
            if col_map["entry_col"] is not None:
                break

    # Step 4: English column — if not found by header, it's typically the last column
    if col_map["english_col"] is None:
        # English is usually the last non-entry, non-IPA column
        for i, h in enumerate(headers):
            if i != col_map["entry_col"] and i != col_map["ipa_col"] and i != col_map["sound_col"]:
                # Check if header looks like a language name (not "english")
                # If headers have a language name, the other text column is probably English
                pass
        # Default: last column
        if n_cols > 0:
            last: int = n_cols - 1
            if last != col_map["entry_col"] and last != col_map["ipa_col"]:
                col_map["english_col"] = last

    # Step 5: If IPA still not detected, use heuristic — non-entry, non-english column
    # with the most non-ASCII characters (IPA uses lots of Unicode)
    if col_map["ipa_col"] is None and n_cols >= 3:
        non_ascii_scores: list[int] = [0] * n_cols
        for row in data_rows[:10]:
            cells = row.find_all(["td", "th"])
            for i, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                non_ascii: int = sum(1 for c in text if ord(c) > 127)
                non_ascii_scores[i] += non_ascii
        # Pick the column with most non-ASCII that isn't entry or english
        for idx in sorted(range(n_cols), key=lambda i: non_ascii_scores[i], reverse=True):
            if idx != col_map["entry_col"] and idx != col_map["english_col"]:
                col_map["ipa_col"] = idx
                break

    # If IPA column is same as a language-name header column (e.g. "Abaza"), that's fine —
    # the language-name column often IS the IPA column
    # Also: if we found no IPA but there's a header matching a language name, use that
    if col_map["ipa_col"] is None:
        for i, h in enumerate(headers):
            if h and h not in ("entry", "english", "#", "no.", "no") and i != col_map["entry_col"] and i != col_map["english_col"]:
                col_map["ipa_col"] = i
                break

    return col_map


def _parse_merged_row(cells: list[Tag]) -> WordEntry | None:
    """Parse a row where 4 logical columns are packed into 2 <td> cells.

    Pattern seen in HYE word lists:
      cell[0] = "1գիր"           (entry_number + orthography)
      cell[1] = "kʰiɾ\xa0writing"  (IPA + english, often nbsp-separated)
    """
    text0: str = cells[0].get_text(strip=True)
    text1: str = cells[1].get_text(strip=True)

    # Cell 0: leading digits = entry number, rest = orthography
    m: re.Match[str] | None = re.match(r"(\d+)\s*(.*)", text0)
    if not m:
        return None
    entry_number: int = int(m.group(1))
    orthography: str | None = m.group(2).strip() or None

    # Cell 1: split IPA from English on nbsp, or fall back to IPA/ASCII boundary
    ipa: str | None = None
    english: str | None = None
    if "\xa0" in text1:
        parts: list[str] = text1.split("\xa0", 1)
        ipa = parts[0].strip() or None
        english = parts[1].strip() or None if len(parts) > 1 else None
    else:
        # Find last non-ASCII char followed by a space → split point
        for i in range(len(text1) - 1, 0, -1):
            if ord(text1[i - 1]) > 127 and text1[i] == " ":
                ipa = text1[:i].strip() or None
                english = text1[i:].strip() or None
                break
        if ipa is None:
            ipa = text1.strip() or None

    return {
        "entry_number": entry_number,
        "ipa": ipa,
        "english": english,
        "orthography": orthography,
        "sound_illustrated": None,
    }


def _parse_entry(cells: list[Tag], col_map: _ColumnMap) -> WordEntry | None:
    """Parse a single table row into an entry dict."""

    def _get(col_key: str) -> str | None:
        idx: int | None = col_map.get(col_key)  # type: ignore[arg-type]
        if idx is not None and idx < len(cells):
            return cells[idx].get_text(strip=True) or None
        return None

    entry_text: str | None = _get("entry_col")
    entry_number: int | None = None
    if entry_text:
        num_match: re.Match[str] | None = re.match(r"(\d+)", entry_text)
        if num_match:
            entry_number = int(num_match.group(1))
        else:
            return None  # not a data row

    return {
        "entry_number": entry_number,
        "ipa": _get("ipa_col"),
        "english": _get("english_col"),
        "orthography": _get("ortho_col"),
        "sound_illustrated": _get("sound_col"),
    }
