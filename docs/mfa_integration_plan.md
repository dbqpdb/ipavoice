# MFA Integration Plan for Duration Supervision

## Goal
Add Montreal Forced Aligner to the CV preprocessing pipeline to extract phone-level durations, improving duration prediction in VITS training.

## Background
Current training shows duration loss plateaued at ~1.9 while mel loss dropped to ~24. The SDP learns durations implicitly from audio without explicit supervision. MFA can provide ground-truth phone boundaries.

## Approach: Phone-to-Phone Alignment

Since we already have IPA phones from Allosaurus, we use MFA in a simplified mode:

1. **Create identity dictionary** — Each Allosaurus phone maps to itself
2. **Train acoustic model** — On our CV data (or use pretrained multilingual)
3. **Align** — Get TextGrid files with phone boundaries
4. **Extract durations** — Convert to frame-level targets
5. **Train with supervision** — Add duration MSE loss

## Implementation Steps

### 1. Install MFA
```bash
# MFA 3.x via conda (recommended)
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa

# Or via pip (may have issues)
pip install montreal-forced-aligner
```

### 2. Create Phone Dictionary (`training/mfa_dict.py`)
```python
def create_phone_dictionary(vocab_path: Path, output_path: Path) -> None:
    """Create MFA dictionary where each phone maps to itself.

    MFA format: word\tphone1 phone2 ...
    For us: phone\tphone
    """
    vocab = IPAVocabulary.load(vocab_path)
    with open(output_path, "w") as f:
        for phone in vocab.phoneme_tokens:
            # Each "word" is a phone, pronunciation is itself
            f.write(f"{phone}\t{phone}\n")
```

### 3. Prepare Corpus Structure
MFA expects:
```
corpus/
├── speaker1/
│   ├── audio1.wav
│   ├── audio1.txt  # space-separated phones
│   ├── audio2.wav
│   └── audio2.txt
└── speaker2/
    └── ...
```

We'll create `training/prepare_mfa_corpus.py`:
- Read metadata CSV
- Create speaker directories (by locale)
- Symlink WAVs
- Write phone transcripts (space-separated)

### 4. Train Acoustic Model (Optional)
```bash
# If using pretrained (faster):
mfa model download acoustic english_mfa

# Or train on our data (better for IPA):
mfa train /path/to/corpus /path/to/dict.txt /path/to/output_model
```

### 5. Run Alignment
```bash
mfa align /path/to/corpus /path/to/dict.txt /path/to/acoustic_model /path/to/output_textgrids
```

### 6. Extract Durations (`training/extract_durations.py`)
```python
import textgrid

def extract_phone_durations(tg_path: Path, hop_length: int = 256, sr: int = 22050) -> list[int]:
    """Convert TextGrid phone intervals to frame counts."""
    tg = textgrid.TextGrid.fromFile(tg_path)
    phones_tier = tg.getFirst("phones")

    durations = []
    for interval in phones_tier:
        duration_sec = interval.maxTime - interval.minTime
        duration_frames = int(duration_sec * sr / hop_length)
        durations.append(max(1, duration_frames))

    return durations
```

### 7. Update Training Data
Add `durations` column to metadata CSV or create separate duration files:
```
data/training_cv/
├── durations/
│   ├── cv_ady_62977.json  # [12, 8, 15, ...]
│   └── ...
```

### 8. Modify VITS Training
In `training/config.py`, enable duration supervision:
```python
# Add duration loss weight
config.duration_loss_weight = 1.0  # or tune this
config.use_duration_supervision = True
```

## File Changes Summary

| File | Change |
|------|--------|
| `training/mfa_dict.py` | New: Generate phone dictionary |
| `training/prepare_mfa_corpus.py` | New: Prepare MFA corpus structure |
| `training/extract_durations.py` | New: TextGrid → frame durations |
| `training/preprocess_cv.py` | Add MFA steps to pipeline |
| `scraper/run.py` | Add `--mfa` flag to preprocess-cv |

## CLI Usage (Planned)
```bash
# Full pipeline with MFA
uv run python -m scraper.run preprocess-cv \
    --cv-parquet ~/gordo/Corpora/CommonVoiceSpontaneous/data/processed/unified.parquet \
    --mfa \
    --workers 8

# Or run MFA separately
uv run python -m scraper.run align-mfa --corpus data/training_cv/mfa_corpus
```

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| MFA fails on some files | Skip failures, log for review |
| Allosaurus phones not in MFA inventory | Use identity dictionary (our approach) |
| Alignment quality varies | Filter by alignment score |
| Slow on 10K files | Parallelize with `--jobs` flag |

## Timeline
1. Install MFA, test on 10 files — 30 min
2. Create dictionary generator — 30 min
3. Prepare corpus script — 1 hr
4. Run alignment (~10K files) — 2-3 hrs
5. Duration extraction — 1 hr
6. Training integration — 2 hrs
7. Test training run — overnight

## References
- [MFA Documentation](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/index.html)
- [MFA with IPA](https://memcauliffe.com/bootstrapping-an-ipa-dictionary-for-english-using-montreal-forced-aligner-20.html)
- [MFA Tutorial](https://eleanorchodroff.com/mfa_tutorial.html)
