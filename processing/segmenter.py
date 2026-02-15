"""Optimized silence-based audio segmentation for word-list recordings.

Improvements over segmenter.py:
1. Multiprocessing — process N recordings in parallel across CPU cores
2. detect_nonsilent for search — only count boundaries during parameter search,
   then slice audio directly from timestamps (no redundant split_on_silence call)
3. Early exit — accept diff <= 2 as "good enough" instead of only exact match
4. ffmpeg silencedetect — optional fast path using ffmpeg's native C-based
   silence detection instead of pydub's Python-based scanning
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "audio"

# Parameter grid for adaptive silence detection — ordered from conservative to aggressive
_PARAM_GRID: list[dict[str, int]] = [
    {"min_silence_len": ms, "silence_thresh": th}
    for th in [-45, -40, -35, -30]
    for ms in [300, 200, 150]
]

# Padding (ms) to keep around each segment so words aren't clipped
_KEEP_SILENCE_MS: int = 100

# Acceptable segment-count deviation to stop search early
_ACCEPTABLE_DIFF: int = 2

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


# ---------------------------------------------------------------------------
# ffmpeg-based silence detection (fast path)
# ---------------------------------------------------------------------------

def _ffmpeg_detect_nonsilent(
    wav_path: str | Path,
    min_silence_len_s: float = 0.2,
    silence_thresh_db: int = -35,
) -> list[tuple[float, float]]:
    """Use ffmpeg's silencedetect filter to find non-silent regions.

    Returns list of (start_s, end_s) for each non-silent region.
    """
    cmd: list[str] = [
        "ffmpeg", "-i", str(wav_path),
        "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_len_s}",
        "-f", "null", "-",
    ]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd, capture_output=True, text=True
    )
    stderr: str = result.stderr

    # Parse lines like:
    #   [silencedetect @ 0x...] silence_start: 1.234
    #   [silencedetect @ 0x...] silence_end: 2.567 | silence_duration: 1.333
    silence_starts: list[float] = []
    silence_ends: list[float] = []
    for line in stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            silence_starts.append(float(m_start.group(1)))
        if m_end:
            silence_ends.append(float(m_end.group(1)))

    # Get total duration from ffmpeg output
    duration: float = 0.0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
    if m_dur:
        h, mn, s, cs = (
            int(m_dur.group(1)), int(m_dur.group(2)),
            int(m_dur.group(3)), int(m_dur.group(4)),
        )
        duration = h * 3600 + mn * 60 + s + cs / 100.0

    # Build non-silent regions from silence boundaries
    nonsilent: list[tuple[float, float]] = []
    prev_end: float = 0.0
    for i, ss in enumerate(silence_starts):
        if ss > prev_end:
            nonsilent.append((prev_end, ss))
        if i < len(silence_ends):
            prev_end = silence_ends[i]
    # Trailing non-silent region after last silence
    if duration > 0 and prev_end < duration:
        nonsilent.append((prev_end, duration))
    elif not silence_starts and duration > 0:
        # No silence detected at all — whole file is one region
        nonsilent.append((0.0, duration))

    return nonsilent


_FFMPEG_PARAM_GRID: list[dict[str, Any]] = [
    {"min_silence_len_s": ms / 1000.0, "silence_thresh_db": th}
    for th in [-45, -40, -35, -30]
    for ms in [300, 200, 150]
]


def _ffmpeg_adaptive_detect(
    wav_path: str | Path, target_count: int
) -> list[tuple[float, float]]:
    """Try multiple ffmpeg silencedetect parameters and pick the best match."""
    best_regions: list[tuple[float, float]] = []
    best_diff: float = float("inf")

    for params in _FFMPEG_PARAM_GRID:
        regions = _ffmpeg_detect_nonsilent(wav_path, **params)
        diff: int = abs(len(regions) - target_count)
        if diff < best_diff:
            best_diff = diff
            best_regions = regions
        if diff <= _ACCEPTABLE_DIFF:
            break

    return best_regions


# ---------------------------------------------------------------------------
# pydub-based silence detection (optimized)
# ---------------------------------------------------------------------------

def _pydub_adaptive_split(
    audio: AudioSegment, target_count: int
) -> tuple[list[AudioSegment], list[tuple[int, int]]]:
    """Try parameter sets with detect_nonsilent, then slice with best params.

    Returns (chunks, regions) where regions are (start_ms, end_ms) pairs.
    """
    best_regions: list[tuple[int, int]] = []
    best_params: dict[str, int] = _PARAM_GRID[0]
    best_diff: float = float("inf")

    for params in _PARAM_GRID:
        # detect_nonsilent just returns timestamps — no audio copying
        regions: list[tuple[int, int]] = detect_nonsilent(
            audio,
            min_silence_len=params["min_silence_len"],
            silence_thresh=params["silence_thresh"],
        )
        diff: int = abs(len(regions) - target_count)
        if diff < best_diff:
            best_diff = diff
            best_regions = regions
            best_params = params
        if diff <= _ACCEPTABLE_DIFF:
            break

    # Slice audio directly from timestamps (avoids redundant split_on_silence)
    keep: int = _KEEP_SILENCE_MS
    chunks: list[AudioSegment] = []
    padded_regions: list[tuple[int, int]] = []
    for start_ms, end_ms in best_regions:
        padded_start: int = max(0, start_ms - keep)
        padded_end: int = min(len(audio), end_ms + keep)
        chunks.append(audio[padded_start:padded_end])
        padded_regions.append((padded_start, padded_end))

    return chunks, padded_regions


# ---------------------------------------------------------------------------
# Single-recording segmentation (runs in worker process)
# ---------------------------------------------------------------------------

def _segment_recording_worker(
    args: tuple[dict[str, Any], str, bool],
) -> tuple[int, str, int]:
    """Worker function for multiprocessing.

    Returns (recording_id, audio_filename, segments_created).
    """
    rec_dict, data_dir_str, use_ffmpeg = args
    recording_id: int = rec_dict["id"]
    audio_filename: str = rec_dict["audio_filename"]
    language_code: str = rec_dict["language_code"]

    resolved_dir: Path = Path(data_dir_str)
    conn: sqlite3.Connection = _get_worker_conn()

    entries: list[sqlite3.Row] = conn.execute(
        "SELECT * FROM entries WHERE recording_id=? ORDER BY entry_number",
        (recording_id,),
    ).fetchall()
    if not entries:
        conn.close()
        return (recording_id, audio_filename, 0)

    wav_path: Path = resolved_dir / language_code / (audio_filename + ".wav")
    if not wav_path.exists():
        print(f"  WAV not found: {wav_path}")
        conn.close()
        return (recording_id, audio_filename, -1)

    seg_dir: Path = resolved_dir / language_code / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    n_entries: int = len(entries)

    if use_ffmpeg:
        return _segment_with_ffmpeg(
            conn, rec_dict, entries, wav_path, seg_dir, n_entries
        )
    else:
        return _segment_with_pydub(
            conn, rec_dict, entries, wav_path, seg_dir, n_entries
        )


def _segment_with_pydub(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    entries: list[sqlite3.Row],
    wav_path: Path,
    seg_dir: Path,
    n_entries: int,
) -> tuple[int, str, int]:
    """Segment using optimized pydub (detect_nonsilent search + direct slice)."""
    recording_id: int = rec["id"]
    audio_filename: str = rec["audio_filename"]

    try:
        audio: AudioSegment = AudioSegment.from_wav(str(wav_path))
    except Exception as e:
        print(f"  Error loading {wav_path.name}: {e}")
        conn.close()
        return (recording_id, audio_filename, -1)

    try:
        chunks, regions = _pydub_adaptive_split(audio, n_entries)
    except Exception as e:
        print(f"  Error segmenting {wav_path.name}: {e}")
        conn.close()
        return (recording_id, audio_filename, -1)

    if not chunks:
        conn.close()
        return (recording_id, audio_filename, 0)

    n_chunks: int = len(chunks)
    if abs(n_chunks - n_entries) > max(3, n_entries * 0.3):
        print(
            f"  Warning: {wav_path.name} has {n_chunks} segments but {n_entries} entries "
            f"(mismatch > 30%)"
        )

    conn.execute("DELETE FROM segments WHERE recording_id=?", (recording_id,))
    created: int = 0
    for i, (chunk, (start_ms, end_ms)) in enumerate(zip(chunks, regions)):
        if i >= n_entries:
            break
        entry: sqlite3.Row = entries[i]
        entry_num: int = entry["entry_number"] if entry["entry_number"] is not None else i + 1
        seg_name: str = f"{audio_filename}_{entry_num:03d}.wav"
        seg_path: Path = seg_dir / seg_name

        chunk.export(str(seg_path), format="wav")

        conn.execute(
            "INSERT INTO segments (entry_id, recording_id, segment_file, start_ms, end_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry["id"], recording_id, str(seg_path), start_ms, end_ms),
        )
        created += 1

    conn.commit()
    conn.close()
    return (recording_id, audio_filename, created)


def _segment_with_ffmpeg(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    entries: list[sqlite3.Row],
    wav_path: Path,
    seg_dir: Path,
    n_entries: int,
) -> tuple[int, str, int]:
    """Segment using ffmpeg silencedetect + ffmpeg extraction."""
    recording_id: int = rec["id"]
    audio_filename: str = rec["audio_filename"]

    try:
        regions: list[tuple[float, float]] = _ffmpeg_adaptive_detect(wav_path, n_entries)
    except Exception as e:
        print(f"  Error detecting silence in {wav_path.name}: {e}")
        conn.close()
        return (recording_id, audio_filename, -1)

    if not regions:
        conn.close()
        return (recording_id, audio_filename, 0)

    n_regions: int = len(regions)
    if abs(n_regions - n_entries) > max(3, n_entries * 0.3):
        print(
            f"  Warning: {wav_path.name} has {n_regions} segments but {n_entries} entries "
            f"(mismatch > 30%)"
        )

    conn.execute("DELETE FROM segments WHERE recording_id=?", (recording_id,))
    created: int = 0

    # Add padding around each region
    keep_s: float = _KEEP_SILENCE_MS / 1000.0

    for i, (start_s, end_s) in enumerate(regions):
        if i >= n_entries:
            break
        entry: sqlite3.Row = entries[i]
        entry_num: int = entry["entry_number"] if entry["entry_number"] is not None else i + 1
        seg_name: str = f"{audio_filename}_{entry_num:03d}.wav"
        seg_path: Path = seg_dir / seg_name

        padded_start: float = max(0.0, start_s - keep_s)
        padded_end: float = end_s + keep_s

        # Re-encode to PCM WAV (stream copy fails on arbitrary cut points)
        cmd: list[str] = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-ss", f"{padded_start:.3f}", "-to", f"{padded_end:.3f}",
            "-acodec", "pcm_s16le",
            str(seg_path),
        ]
        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Error extracting {seg_name}: {result.stderr.strip()[:200]}")
            continue

        start_ms: int = int(padded_start * 1000)
        end_ms: int = int(padded_end * 1000)

        conn.execute(
            "INSERT INTO segments (entry_id, recording_id, segment_file, start_ms, end_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry["id"], recording_id, str(seg_path), start_ms, end_ms),
        )
        created += 1

    conn.commit()
    conn.close()
    return (recording_id, audio_filename, created)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def segment_recording(
    conn: sqlite3.Connection,
    recording_id: int,
    data_dir: str | Path | None = None,
    use_ffmpeg: bool = False,
) -> int:
    """Segment a single word-list recording. Returns segments created or -1 on error."""
    resolved_dir: Path = Path(data_dir) if data_dir else DATA_DIR

    rec: sqlite3.Row | None = conn.execute(
        "SELECT * FROM recordings WHERE id=?", (recording_id,)
    ).fetchone()
    if not rec:
        return -1
    if rec["recording_type"] != "word-list":
        return 0

    rec_dict: dict[str, Any] = dict(rec)
    _, _, created = _segment_recording_worker(
        (rec_dict, str(resolved_dir), use_ffmpeg)
    )
    return created


def segment_all(
    conn: sqlite3.Connection,
    language_code: str | None = None,
    data_dir: str | Path | None = None,
    force: bool = False,
    workers: int | None = None,
    use_ffmpeg: bool = False,
) -> int:
    """Segment all downloaded word-list recordings using multiprocessing."""
    resolved_dir: Path = Path(data_dir) if data_dir else DATA_DIR
    n_workers: int = workers if workers else max(1, cpu_count() - 1)

    query: str = (
        "SELECT * FROM recordings WHERE recording_type='word-list' AND downloaded=1"
    )
    params: list[str] = []
    if language_code:
        query += " AND language_code=?"
        params.append(language_code.upper())

    recordings: list[sqlite3.Row] = conn.execute(query, params).fetchall()

    # Filter out already-segmented recordings (unless force)
    to_process: list[dict[str, Any]] = []
    skipped_total: int = 0
    for rec in recordings:
        if not force:
            existing: int = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE recording_id=?", (rec["id"],)
            ).fetchone()[0]
            if existing > 0:
                skipped_total += existing
                continue
        to_process.append(dict(rec))

    print(
        f"  Segmenting {len(to_process)} recordings with {n_workers} workers "
        f"({'ffmpeg' if use_ffmpeg else 'pydub'} backend)"
    )
    if skipped_total > 0:
        print(f"  Skipping {len(recordings) - len(to_process)} already-segmented recordings")

    if not to_process:
        print(f"\nTotal segments (all pre-existing): {skipped_total}")
        return skipped_total

    worker_args: list[tuple[dict[str, Any], str, bool]] = [
        (rec_dict, str(resolved_dir), use_ffmpeg) for rec_dict in to_process
    ]

    total_new: int = 0
    with Pool(processes=n_workers) as pool:
        for recording_id, audio_filename, created in pool.imap_unordered(
            _segment_recording_worker, worker_args
        ):
            if created > 0:
                print(f"  {audio_filename}: {created} segments")
                total_new += created
            elif created == 0:
                print(f"  {audio_filename}: skipped (no entries or no segments detected)")

    total: int = skipped_total + total_new
    print(f"\nTotal segments: {total} ({total_new} new, {skipped_total} pre-existing)")
    return total
