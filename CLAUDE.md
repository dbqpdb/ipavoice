# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IPA Voice scrapes the UCLA Phonetics Lab Archive (<https://archive.phonetics.ucla.edu>) to build IPA-to-speech training data. It extracts IPA transcriptions and audio from ~307 languages, segments word-list recordings into per-word WAV files, and exports a JSON manifest for ML training. Licensed CC BY-NC 2.0 matching the archive.

## Style

We should always prioritize accuracy and correctness over all other
considerations. Since we are conducting compute-heavy operations in this
project, we should optimize code to run as quickly as possible, caching data,
vectorizing operations, etc. when appropriate. We should retain a full paper
trail of what we've done as we go, saving scripts and analyses to files.

## Commands

```bash
# Install Python dependencies (ffmpeg + ffprobe must be on PATH for pydub audio segmentation)
uv sync

# All commands run from the project root via `uv run`
uv run python -m scraper.run metadata                      # Scrape all language metadata
uv run python -m scraper.run metadata --language ABQ        # Single language
uv run python -m scraper.run download --language ABQ        # Download WAV files
uv run python -m scraper.run segment --language ABQ         # Segment audio into per-word files
uv run python -m scraper.run export --language ABQ          # Export JSON training manifest

# Useful flags
#   --delay SECONDS   Rate limit between requests (default: 0.5 metadata, 1.0 download)
#   --workers N       Parallel downloads (default: 1, not yet implemented)
#   --output FILE     Export path (default: data/manifest.json)
```

There are no tests yet.

## Architecture

Two packages with a strict data flow: **scraper** (fetches + parses HTML) → **processing** (stores + transforms).

### Data pipeline (4 phases)

1. **metadata**: `index_parser` → `language_parser` → `wordlist_parser` — scrapes HTML into SQLite
2. **download**: `downloader` — fetches WAV files to `data/audio/{LANG_CODE}/`
3. **segment**: `segmenter` — splits word-list WAVs into per-entry files via silence detection
4. **export**: generates `data/manifest.json` joining entries with segment audio paths

### SQLite schema (processing/database.py)

Four tables with foreign key chain: **languages** → **recordings** → **entries** → **segments**. All DB access uses `sqlite3.Row` for dict-style access. Metadata uses upsert so re-running is safe.

### Key design patterns

- **Adaptive silence detection** (segmenter.py): Tries 12 parameter combinations (4 dBFS thresholds × 3 min-silence durations) and picks whichever produces segment count closest to the known entry count.
- **Column detection heuristic** (wordlist_parser.py): Word list tables vary from 3–5 columns across languages. Detection priority: header text match → IPA Unicode character frequency → non-ASCII density → positional fallback.
- **TypedDict contracts**: `LanguageInfo`, `RecordingInfo`, `WordEntry` define the dict shapes passed between parser modules and the database layer.

## Code Conventions

- Python 3.10+ with `from __future__ import annotations`
- Full type hints on all function signatures, return types, and local variables
- All CLI entry points go through `scraper/run.py` (argparse subcommands)
- Dependencies managed with uv; `pyproject.toml` is the single source of truth
- Data files are gitignored under `data/`; SQLite DB lives at `data/db/ipavoice.db`
- Rate limiting and `User-Agent: IPAVoice-Scraper/1.0` on all HTTP requests
