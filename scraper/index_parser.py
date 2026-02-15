"""Parse the UCLA Phonetics Lab Archive language index page."""

from __future__ import annotations

import re
from typing import TypedDict

import requests
from bs4 import BeautifulSoup

BASE_URL: str = "https://archive.phonetics.ucla.edu"
INDEX_URL: str = f"{BASE_URL}/Language%20Indices/index_available.htm"


class LanguageInfo(TypedDict):
    name: str
    code: str
    url: str


def fetch_languages(session: requests.Session | None = None) -> list[LanguageInfo]:
    """Fetch and parse the available languages index."""
    session = session or requests.Session()
    resp: requests.Response = session.get(INDEX_URL)
    resp.raise_for_status()
    soup: BeautifulSoup = BeautifulSoup(resp.content, "lxml")

    languages: list[LanguageInfo] = []
    seen_codes: set[str] = set()

    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        # Match language page links like ../Language/ABQ/abq.html
        match: re.Match[str] | None = re.match(r"\.\./Language/([A-Z0-9]+)/\w+\.html?", href, re.IGNORECASE)
        if not match:
            continue

        code: str = match.group(1).upper()
        if code in seen_codes:
            continue
        seen_codes.add(code)

        name: str = link.get_text(strip=True)
        if not name:
            continue

        # Build absolute URL
        # href is like ../Language/ABQ/abq.html relative to Language Indices/
        url: str = f"{BASE_URL}/Language/{code}/{href.split('/')[-1]}"

        languages.append({"name": name, "code": code, "url": url})

    return languages
