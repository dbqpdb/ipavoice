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
- [x] Audio preprocessing (resample to 22050 Hz mono, normalize loudness)
- [x] IPA tokenizer (character-level with combining diacritics, affricates, tone marks)
- [x] Train/val/test split
- [x] VITS training configuration and pipeline

---

## 2026-02-17: Training milestone — 3M steps

**Checkpoint:** `ipavoice_vits-February-17-2026_04+05PM-5505364/checkpoint_3060000.pth`

Highest step count reached in initial training run. Model trained on full dataset with `min_text_len: 1`.

---

## 2026-03-02: Config tweak — filter short entries

**Change:** Increased `min_text_len` from 1 to 3.

**Rationale:** Filter out single/double character IPA entries that may be noisy or unhelpful for training (isolated diacritics, incomplete transcriptions).

**Action:** Started new training run initialized from 3.06M checkpoint weights. Step counter reset to 0 in new output directory.

**Run directory:** `ipavoice_vits-March-02-2026_01+14PM-5505364/`

**Last checkpoint before hiatus:** `checkpoint_140000.pth` (effective ~3.2M total steps)

---

## 2026-05-05: Resume training after hiatus

**Context:** Project on ice since early March. Resumed training today.

**Training state:**
- Resumed from: `ipavoice_vits-March-02-2026_01+14PM-5505364/checkpoint_140000.pth`
- Effective training: ~3.2M steps (3.06M from Feb run + 140K from March run)
- GPU: RTX 4080 Laptop (12GB VRAM)
- Settings: `--mixed-precision --batch-size 4`

**Note on checkpoint numbering:** The March run loaded weights from the 3M checkpoint but reset the step counter. So `checkpoint_140000.pth` in the March directory represents a model with ~3.2M effective training steps.

### Reference: Training commands

```bash
# Resume from latest checkpoint with mixed precision
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m training.train \
  --resume "$(printf '%s\n' data/vits_output/ipavoice_vits-*/checkpoint_*.pth | sed 's/.*checkpoint_\([0-9]*\)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)" \
  --mixed-precision --batch-size 4

# Test run (1000 steps)
uv run python -m training.train --test-run
```

### Reference: Memory management
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prevents OOM on 12GB cards
- Batch size 4 is safe for 12GB VRAM with variable-length audio
- Mixed precision (`--mixed-precision`) halves memory usage

### Training data analysis

Generated comprehensive coverage report: `docs/TRAINING_DATA_REPORT.md`

**Summary statistics:**
- 205,408 audio entries
- 1,194,911 IPA tokens
- 1,855 unique tokens
- 291 languages

**Coverage by sound class:**

| Category | Tokens | Coverage |
|----------|--------|----------|
| Vowels (oral) | 496K | Excellent |
| Plosives | 245K | Excellent |
| Fricatives | 125K | Excellent |
| Nasals | 107K | Excellent |
| Laterals | 41K | Excellent |
| Approximants | 29K | Excellent |
| Trills | 26K | Excellent |
| Ejectives | 13K | Excellent |
| Clicks | 11K | Excellent |
| Vowels (nasalized) | 10K | Good |
| Taps/flaps | 3K | Good |
| Implosives | 2K | Good |
| Affricates | 1.5K | Good |
| **Tones** | **690** | **Limited** |

**Key limitation:** Tone data is severely underrepresented. The model will likely struggle with tonal languages despite good segmental coverage.

**Rare tokens:** 345 hapax legomena, 1,083 tokens with ≤10 occurrences. These cannot be learned reliably.

**Language coverage:** 98 languages have <50 entries — insufficient for reliable synthesis.
