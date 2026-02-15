# IPA Voice — Development Log

## 2026-02-13: Initial Implementation

### What we built
Complete scraper pipeline for the UCLA Phonetics Lab Archive (https://archive.phonetics.ucla.edu), targeting IPA-to-speech training data.

### Architecture
```
ipavoice/
├── scraper/
│   ├── index_parser.py      — Parse 307 languages from the archive index
│   ├── language_parser.py   — Parse per-language recording tables (WAV URLs, entry ranges)
│   ├── wordlist_parser.py   — Parse word list HTML → IPA entries (handles 3-5 column variants)
│   ├── downloader.py        — WAV download with rate limiting, resume, retry
│   └── run.py               — CLI: metadata / download / segment / export
├── processing/
│   ├── database.py          — SQLite schema (languages, recordings, entries, segments)
│   └── segmenter.py         — Silence-based audio segmentation with adaptive parameters
├── data/                    — (gitignored) audio + SQLite DB
├── pyproject.toml           — uv-managed deps: requests, beautifulsoup4, lxml, pydub, tqdm
└── LICENSE                  — CC BY-NC 2.0
```

### CLI Usage
```bash
uv run python -m scraper.run metadata              # Scrape all metadata
uv run python -m scraper.run metadata --language ABQ   # Single language
uv run python -m scraper.run download --language ABQ   # Download WAVs
uv run python -m scraper.run segment --language ABQ    # Split into per-word segments
uv run python -m scraper.run export --language ABQ     # JSON manifest for ML training
```

### Verification (Abaza / ABQ)
- **307 languages** parsed from index (plan expected 312 — 5 may have broken links or non-standard format)
- **1 recording** for ABQ: `abq_word-list_1970_01` (entries 1–11)
- **11 IPA entries** correctly extracted (zəˈkʼə = one, ˀyˑba = two, ... ɦʕɔ̈ʒa = twenty)
- **WAV download**: 2.3 MB file downloaded with resume support
- **Segmentation**: Adaptive silence detection found exactly 11 segments matching 11 entries
- **Export**: Clean JSON manifest with IPA, English, language metadata, and audio paths

### Key design decisions
1. **Adaptive segmenter**: Tries 12 parameter combinations (4 thresholds × 3 min-silence-lengths) and picks the one that produces the closest segment count to the expected entry count. This handles the wide variation in recording quality across 307 languages.
2. **Column detection heuristic**: Word list tables vary from 3 to 5+ columns. The parser detects IPA columns by header name first, then falls back to Unicode IPA character frequency analysis.
3. **Upsert-based metadata**: Re-running `metadata` updates existing records rather than duplicating, allowing incremental scraping.

### Dependencies resolved
- `sqlite3` required upgrading conda's sqlite from 3.32 to 3.50 (needed `sqlite3_deserialize` symbol)
- `ffmpeg` must be on PATH for pydub WAV processing
- Switched from pip/conda to **uv** for dependency management (`pyproject.toml` + `uv.lock`)

### Next steps
- [x] Run full `metadata` scrape for all 307 languages
- [x] Run full `download` for all word-list recordings
- [x] Run full `segment` and evaluate mismatch rates across languages
- [ ] Test with a language that has 4-5 column word lists (e.g., Mandarin CMN)
- [ ] Add parallel download support (`--workers N`)

---

## 2026-02-14: Full Pipeline Run + Segmentation Optimization

### Full pipeline results

| Stage | Count |
|-------|-------|
| Languages scraped | 307 |
| Total recordings | 1,961 |
| Word-list recordings | 1,836 |
| Downloaded WAVs | 1,961 (62 GB) |
| Word-list entries | 213,065 |
| Segments created | 169,811 |
| Segment WAV files | 169,625 (17 GB) |
| Languages with segments | 292 |
| Recordings segmented | 1,828 / 1,836 |

### Unsegmented recordings (8 of 1,836)
- **7 recordings** had 0 parsed entries (word list HTML missing or unparseable): AMH, AQC, HYE (×2), RUN (×2), PAV
- **1 recording** failed due to corrupted WAV frames: APC `apc_word-list_1986_02` (213 entries, `audioop.error: not a whole number of frames`)

### Bug fixes during full run
1. **NULL entry_number** — 255 of 213,065 entries had `entry_number = NULL`, causing `TypeError` in `f"{None:03d}"`. Fixed with fallback to positional index.
2. **Corrupted WAV frames** — Some files cause `audioop.error` in pydub's silence detection. Wrapped in try/except to skip gracefully.
3. **Skip-existing optimization** — Added to `segment_all()` so re-runs don't reprocess already-segmented recordings.

### Segmenter v2 (`processing/segmenter_v2.py`)
Created an optimized segmenter with 4 improvements over v1, benchmarked on ARZ (10 recordings):

| Configuration | Time | Segments | Speedup |
|--------------|------|----------|---------|
| v1 pydub serial | 271s | 1,094 | 1.0× |
| v2 pydub 1 worker | 252s | 1,093 | 1.1× |
| v2 pydub 4 workers | 85s | 1,093 | 3.2× |
| v2 ffmpeg 1 worker | 50s | 908 | 5.4× |
| v2 ffmpeg 4 workers | 42s | 908 | 6.5× |

**Optimizations:**
1. **Multiprocessing** — `Pool` with `imap_unordered`, each worker gets its own SQLite connection (WAL mode + busy_timeout)
2. **detect_nonsilent for search** — Parameter grid search uses `detect_nonsilent` (returns timestamps only), then slices audio directly. Avoids redundant `split_on_silence` full-file scan.
3. **ffmpeg silencedetect** — Optional fast path using ffmpeg's native C-based silence detection + segment extraction. ~6× faster but produces fewer segments due to different silence calibration.
4. **Early exit** — Accepts segment count within ±2 of target instead of requiring exact match.

**Trade-off:** ffmpeg backend is significantly faster but produces ~17% fewer segments for the tested language. Pydub backend with multiprocessing is the best balance of speed and accuracy.

### Next steps
- [x] Wire `segmenter_v2` into the CLI (`run.py` still uses v1)
- [x] Run `export` to generate JSON training manifest for all 170K segments
- [x] Investigate the 7 recordings with 0 entries (missing word list HTML)

---

## 2026-02-14: Pipeline polish + model selection

### Completed
- **Segmenter v2 promoted** — replaced v1 with v2 as `processing/segmenter.py`. Added `--workers`, `--ffmpeg`, `--force` CLI flags.
- **Export** — generated `data/manifest.json`: 205,408 entries (162,928 with segment audio, 42,480 with full-recording fallback).
- **Missing entries investigation** — 3 root causes identified:
  - AMH: wrong URL scraped (points to conversation page, 404s)
  - AQC, RUN×2, PAV: no word list HTML exists on the archive
  - HYE×2: malformed HTML (4 headers, 2 `<td>` cells per row — entry+ortho and IPA+english merged)
- **HYE parser fix** — added `_parse_merged_row()` fallback for merged-cell tables. 106 new entries, 105 new segments.
- **Parser tests** — verified all column variants: 3-col (ABQ), 4-col+sound (CMN), 4-col+ortho (ABK), 5-col (AER, APE), merged (HYE).
- **Parallel downloads** — `ThreadPoolExecutor` with per-thread sessions and SQLite connections. Usage: `--workers N`.

### Model architecture decision: VITS

**Goal:** Build a universal IPA-to-speech synthesizer. Given an arbitrary IPA string, generate audio that produces recognizable phonetic output. Not expecting natural speech quality given the data diversity, but aiming for broad phonetic coverage across the IPA inventory.

**Dataset characteristics that drive the choice:**
- 170K audio segments from 307 languages
- IPA transcriptions as input (not orthographic text)
- Unknown, unlabeled speakers — only language code available as grouping
- Highly varied recording conditions (1960s–2000s, different equipment)
- Moderate dataset size (too small for large transformer models, right-sized for VAE-based)

**Why VITS over alternatives:**

| Model | Pros | Cons for this use case |
|-------|------|----------------------|
| **VITS** | Phoneme-native input; end-to-end (no separate vocoder); VAE handles data diversity; multi-speaker via embeddings; trains well at 170K scale | — |
| Matcha-TTS | Flow matching may train more stably | Less battle-tested for multilingual; fewer reference implementations |
| Tacotron 2 + HiFi-GAN | Well understood | Autoregressive attention unstable with 307 languages of speaker variation; two-stage pipeline |
| Bark / SpeechT5 | Very capable | Need significantly more data and compute; designed for text not IPA |

**Key VITS properties that fit:**
1. **Phoneme input** — VITS natively takes phoneme sequences. IPA *is* a phoneme representation, so no grapheme-to-phoneme conversion needed. We tokenize IPA at the character level.
2. **End-to-end** — single model learns alignment, acoustic features, and waveform generation jointly. Simpler pipeline, fewer failure modes.
3. **Variational autoencoder** — the VAE latent space absorbs speaker/recording variation, letting the model learn phoneme-to-sound mappings that generalize across speakers.
4. **Multi-speaker conditioning** — we'll use language code as a pseudo-speaker label. This gives the model 307 conditioning signals to help disentangle linguistic content from speaker identity.
5. **Scale match** — VITS trains well on datasets of this size (~170K utterances). Larger models (Bark) need millions; smaller datasets would underfit.

**Implementation plan:** Coqui TTS library (`TTS` package) — has a well-tested VITS implementation with multi-speaker support and configurable phoneme tokenization.

### Next steps
- [ ] Audio preprocessing (resample to 22050 Hz mono, normalize loudness)
- [ ] IPA tokenizer (character-level with combining diacritics, affricates, tone marks)
- [ ] Train/val/test split
- [ ] VITS training configuration and pipeline
