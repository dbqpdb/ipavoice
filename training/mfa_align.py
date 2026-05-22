"""MFA alignment wrapper for phone-level duration extraction.

Runs Montreal Forced Aligner via subprocess (since MFA requires conda environment).
Trains an acoustic model on the corpus and generates TextGrid alignments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# --- Constants ---

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
MFA_DIR: Path = DATA_DIR / "mfa"
MFA_CORPUS_DIR: Path = MFA_DIR / "corpus"
MFA_OUTPUT_DIR: Path = MFA_DIR / "textgrids"
MFA_MODEL_PATH: Path = MFA_DIR / "acoustic_model.zip"
MFA_DICT_PATH: Path = MFA_DIR / "phone_dict.txt"

# Default MFA binary location (conda environment)
DEFAULT_MFA_CONDA_ENV: str = "mfa"
DEFAULT_CONDA_PATH: str = "/opt/miniconda3/condabin/conda"


def find_mfa_binary() -> tuple[str, list[str]]:
    """Find MFA binary, checking conda environments.

    Returns:
        Tuple of (binary_path, prefix_args) where prefix_args may include
        conda run arguments.

    Raises:
        FileNotFoundError: If MFA cannot be found.
    """
    # Check if mfa is directly on PATH
    mfa_path: str | None = shutil.which("mfa")
    if mfa_path is not None:
        return mfa_path, []

    # Check conda environments
    conda_paths: list[str] = [
        DEFAULT_CONDA_PATH,
        shutil.which("conda") or "",
        shutil.which("mamba") or "",
        shutil.which("micromamba") or "",
    ]

    for conda in conda_paths:
        if not conda or not Path(conda).exists():
            continue

        # Check if mfa env exists
        env_path: Path = Path(conda).parent.parent / "envs" / DEFAULT_MFA_CONDA_ENV
        if env_path.exists():
            mfa_in_env: Path = env_path / "bin" / "mfa"
            if mfa_in_env.exists():
                # Use conda run to ensure proper environment
                return conda, ["run", "-n", DEFAULT_MFA_CONDA_ENV, "mfa"]

    raise FileNotFoundError(
        "MFA not found. Install it with:\n"
        f"  conda create -n {DEFAULT_MFA_CONDA_ENV} -c conda-forge montreal-forced-aligner"
    )


def run_mfa_command(
    args: list[str],
    mfa_binary: str | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an MFA command.

    Args:
        args: MFA command arguments (e.g., ["train", "corpus", "dict", "model"]).
        mfa_binary: Path to MFA binary or conda. If None, auto-detect.
        timeout: Command timeout in seconds.

    Returns:
        CompletedProcess result.

    Raises:
        subprocess.CalledProcessError: If command fails.
        FileNotFoundError: If MFA not found.
    """
    if mfa_binary is None:
        binary, prefix = find_mfa_binary()
    else:
        binary = mfa_binary
        prefix = []

    cmd: list[str] = [binary] + prefix + args

    print(f"Running: {' '.join(cmd)}")

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        timeout=timeout,
    )

    return result


def validate_corpus(
    corpus_dir: Path | None = None,
    dict_path: Path | None = None,
    mfa_binary: str | None = None,
) -> bool:
    """Validate MFA corpus structure.

    Args:
        corpus_dir: Path to corpus directory.
        dict_path: Path to pronunciation dictionary.
        mfa_binary: Path to MFA binary.

    Returns:
        True if validation passes.
    """
    if corpus_dir is None:
        corpus_dir = MFA_CORPUS_DIR
    if dict_path is None:
        dict_path = MFA_DICT_PATH

    print("Validating MFA corpus...")

    result: subprocess.CompletedProcess[str] = run_mfa_command(
        ["validate", str(corpus_dir), str(dict_path)],
        mfa_binary=mfa_binary,
    )

    return result.returncode == 0


def train_and_align(
    corpus_dir: Path | None = None,
    dict_path: Path | None = None,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    mfa_binary: str | None = None,
    clean: bool = True,
    num_jobs: int | None = None,
    timeout: int = 86400,  # 24 hours default
) -> bool:
    """Train acoustic model and generate alignments.

    This is the main MFA pipeline: train a model on the corpus and output
    TextGrid files with phone boundaries.

    Args:
        corpus_dir: Path to corpus directory.
        dict_path: Path to pronunciation dictionary.
        model_path: Output path for trained acoustic model.
        output_dir: Output directory for TextGrid files.
        mfa_binary: Path to MFA binary.
        clean: Clean previous MFA temp files.
        num_jobs: Number of parallel jobs.
        timeout: Command timeout in seconds.

    Returns:
        True if successful.
    """
    if corpus_dir is None:
        corpus_dir = MFA_CORPUS_DIR
    if dict_path is None:
        dict_path = MFA_DICT_PATH
    if model_path is None:
        model_path = MFA_MODEL_PATH
    if output_dir is None:
        output_dir = MFA_OUTPUT_DIR

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    args: list[str] = [
        "train",
        str(corpus_dir),
        str(dict_path),
        str(model_path),
        "--output_directory", str(output_dir),
    ]

    if clean:
        args.append("--clean")

    if num_jobs is not None:
        args.extend(["--num_jobs", str(num_jobs)])

    print("=" * 60)
    print("MFA Training and Alignment")
    print("=" * 60)
    print(f"  Corpus: {corpus_dir}")
    print(f"  Dictionary: {dict_path}")
    print(f"  Model output: {model_path}")
    print(f"  TextGrid output: {output_dir}")
    print("=" * 60)

    result: subprocess.CompletedProcess[str] = run_mfa_command(
        args,
        mfa_binary=mfa_binary,
        timeout=timeout,
    )

    if result.returncode == 0:
        # Count output files
        textgrid_count: int = sum(
            1 for f in output_dir.rglob("*.TextGrid")
        )
        print(f"\nAlignment complete: {textgrid_count} TextGrid files generated")
        return True
    else:
        print(f"\nAlignment failed with return code {result.returncode}")
        return False


def align_with_model(
    corpus_dir: Path | None = None,
    dict_path: Path | None = None,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    mfa_binary: str | None = None,
    clean: bool = True,
    num_jobs: int | None = None,
    timeout: int = 86400,
) -> bool:
    """Align corpus using an existing acoustic model.

    Use this when you already have a trained model and want to align
    new data without retraining.

    Args:
        corpus_dir: Path to corpus directory.
        dict_path: Path to pronunciation dictionary.
        model_path: Path to trained acoustic model.
        output_dir: Output directory for TextGrid files.
        mfa_binary: Path to MFA binary.
        clean: Clean previous MFA temp files.
        num_jobs: Number of parallel jobs.
        timeout: Command timeout in seconds.

    Returns:
        True if successful.
    """
    if corpus_dir is None:
        corpus_dir = MFA_CORPUS_DIR
    if dict_path is None:
        dict_path = MFA_DICT_PATH
    if model_path is None:
        model_path = MFA_MODEL_PATH
    if output_dir is None:
        output_dir = MFA_OUTPUT_DIR

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_and_align first to create a model.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    args: list[str] = [
        "align",
        str(corpus_dir),
        str(dict_path),
        str(model_path),
        str(output_dir),
    ]

    if clean:
        args.append("--clean")

    if num_jobs is not None:
        args.extend(["--num_jobs", str(num_jobs)])

    print("=" * 60)
    print("MFA Alignment (using existing model)")
    print("=" * 60)
    print(f"  Corpus: {corpus_dir}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    result: subprocess.CompletedProcess[str] = run_mfa_command(
        args,
        mfa_binary=mfa_binary,
        timeout=timeout,
    )

    if result.returncode == 0:
        textgrid_count: int = sum(1 for f in output_dir.rglob("*.TextGrid"))
        print(f"\nAlignment complete: {textgrid_count} TextGrid files generated")
        return True
    else:
        print(f"\nAlignment failed with return code {result.returncode}")
        return False


def run_mfa_alignment(
    corpus_dir: Path | None = None,
    dict_path: Path | None = None,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    mfa_binary: str | None = None,
    num_jobs: int | None = None,
    retrain: bool = False,
) -> bool:
    """Run MFA alignment pipeline.

    If a model exists and retrain=False, uses existing model.
    Otherwise trains a new model.

    Args:
        corpus_dir: Path to corpus directory.
        dict_path: Path to pronunciation dictionary.
        model_path: Path to acoustic model.
        output_dir: Output directory for TextGrid files.
        mfa_binary: Path to MFA binary.
        num_jobs: Number of parallel jobs.
        retrain: Force retraining even if model exists.

    Returns:
        True if successful.
    """
    if model_path is None:
        model_path = MFA_MODEL_PATH

    if model_path.exists() and not retrain:
        print(f"Using existing model: {model_path}")
        return align_with_model(
            corpus_dir=corpus_dir,
            dict_path=dict_path,
            model_path=model_path,
            output_dir=output_dir,
            mfa_binary=mfa_binary,
            num_jobs=num_jobs,
        )
    else:
        return train_and_align(
            corpus_dir=corpus_dir,
            dict_path=dict_path,
            model_path=model_path,
            output_dir=output_dir,
            mfa_binary=mfa_binary,
            num_jobs=num_jobs,
        )
