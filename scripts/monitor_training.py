#!/usr/bin/env python3
"""Automated training monitoring: plot losses, generate test samples, detect plateaus."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# --- Test sentences covering diverse IPA inventory ---
TEST_SENTENCES: list[dict[str, str]] = [
    {
        "id": "german",
        "ipa": "ˈʃtʁaːsə ˈbɛɐ̯lɪn",
        "lang_style": "DEU",
        "description": "German: uvular R, long vowels, clusters",
    },
    {
        "id": "clicks",
        "ipa": "ǀʰõã ǃʼũ ǁʰa",
        "lang_style": "ZUL",
        "description": "Zulu: click consonants, nasalized vowels",
    },
    {
        "id": "thai",
        "ipa": "tʰaɪ̯ pʰə̀ʔ kʰǎːw",
        "lang_style": "THA",
        "description": "Thai: aspirates, tones, glottal stop",
    },
    {
        "id": "ejectives",
        "ipa": "kʼatʼɬʼi qʷʼəχʷ",
        "lang_style": "APW",
        "description": "Western Apache: ejectives, lateral fricatives",
    },
    {
        "id": "french",
        "ipa": "bɔ̃ʒuʁ mɔ̃d",
        "lang_style": "FRA",
        "description": "French: nasalized vowels, uvular R",
    },
    {
        "id": "english",
        "ipa": "həˈloʊ ˈwɜːld",
        "lang_style": "ENG",
        "description": "English: schwa, diphthongs, rhotics",
    },
]


def find_latest_run() -> Path | None:
    """Find the most recent training run directory."""
    output_dir = Path("data/vits_output")
    if not output_dir.exists():
        return None

    runs = sorted(output_dir.glob("ipavoice_vits-*/trainer_0_log.txt"))
    if not runs:
        return None

    return runs[-1].parent


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the highest-step checkpoint in a run directory."""
    checkpoints = list(run_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return max(checkpoints, key=get_step)


def get_checkpoint_step(checkpoint_path: Path) -> int:
    """Extract step number from checkpoint filename."""
    try:
        return int(checkpoint_path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


def parse_losses(log_path: Path) -> list[dict[str, float]]:
    """Parse training log for loss values."""
    content = log_path.read_text()

    # Remove ANSI codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    content = ansi_escape.sub("", content)

    records: list[dict[str, float]] = []
    current_record: dict[str, float] = {}

    step_pattern = re.compile(r"GLOBAL_STEP:\s*(\d+)")
    loss_pattern = re.compile(r"\| > (loss_\w+):\s*([\d.]+)")

    for line in content.split("\n"):
        step_match = step_pattern.search(line)
        if step_match:
            if current_record and "step" in current_record:
                records.append(current_record)
            current_record = {"step": int(step_match.group(1))}
            continue

        loss_match = loss_pattern.search(line)
        if loss_match:
            current_record[loss_match.group(1)] = float(loss_match.group(2))

    if current_record and "step" in current_record:
        records.append(current_record)

    return records


def detect_plateau(
    records: list[dict[str, float]],
    loss_name: str = "loss_mel",
    window: int = 1000,
    threshold: float = 0.01,
) -> bool:
    """Detect if a loss has plateaued.

    Returns True if the loss hasn't improved by more than threshold
    over the last `window` steps.
    """
    values = [r[loss_name] for r in records if loss_name in r]

    if len(values) < window:
        return False

    recent = values[-window:]
    older = values[-window * 2:-window] if len(values) >= window * 2 else values[:window]

    recent_mean = np.mean(recent)
    older_mean = np.mean(older)

    # Check if improvement is less than threshold
    improvement = (older_mean - recent_mean) / older_mean if older_mean > 0 else 0
    return improvement < threshold


def plot_losses(records: list[dict[str, float]], output_path: Path) -> None:
    """Plot loss curves and save to file."""
    import matplotlib.pyplot as plt

    key_losses = ["loss_mel", "loss_gen", "loss_disc", "loss_kl", "loss_feat", "loss_duration"]
    available = [k for k in key_losses if any(k in r for r in records)]

    if not available:
        return

    n_plots = len(available)
    cols = 2
    rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, loss_name in enumerate(available):
        ax = axes[idx]
        steps = [r["step"] for r in records if loss_name in r]
        values = [r[loss_name] for r in records if loss_name in r]

        if not steps:
            continue

        # Subsample
        if len(steps) > 1000:
            indices = np.linspace(0, len(steps) - 1, 1000, dtype=int)
            steps = [steps[i] for i in indices]
            values = [values[i] for i in indices]

        ax.plot(steps, values, linewidth=0.5, alpha=0.5, color="blue")

        # Smoothed line
        if len(values) > 20:
            window = min(50, len(values) // 5)
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smoothed, linewidth=2, color="red")

        ax.set_title(loss_name.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        if values:
            q1, q3 = np.percentile(values, [5, 95])
            margin = (q3 - q1) * 0.3
            ax.set_ylim(max(0, q1 - margin), q3 + margin)

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("VITS Training Losses", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_samples(
    checkpoint_path: Path,
    output_dir: Path,
    step: int,
) -> list[Path]:
    """Generate test samples using the synthesis module."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for sentence in TEST_SENTENCES:
        output_file = output_dir / f"step_{step:06d}_{sentence['id']}.wav"

        cmd = [
            "uv", "run", "python", "-m", "ipavoice.synthesize",
            sentence["ipa"],
            "--lang-style", sentence["lang_style"],
            "--checkpoint", str(checkpoint_path),
            "-o", str(output_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_file.exists():
            generated.append(output_file)
            print(f"  Generated: {output_file.name}")
        else:
            print(f"  Failed: {sentence['id']} - {result.stderr[:100]}")

    return generated


def save_report(
    output_dir: Path,
    step: int,
    records: list[dict[str, float]],
    plateau_detected: bool,
    samples: list[Path],
) -> Path:
    """Save monitoring report as JSON."""
    # Get latest losses
    latest = records[-1] if records else {}

    report = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "losses": {k: v for k, v in latest.items() if k.startswith("loss_")},
        "plateau_detected": plateau_detected,
        "samples": [str(p.name) for p in samples],
    }

    report_path = output_dir / f"report_step_{step:06d}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor VITS training: plot losses, generate samples, detect plateaus"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Training run directory (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/monitoring"),
        help="Output directory for reports and samples (default: data/monitoring)",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip generating audio samples",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating loss plot",
    )

    args = parser.parse_args()

    # Find run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run()
        if not run_dir:
            print("No training runs found!", file=sys.stderr)
            sys.exit(1)

    print(f"Monitoring: {run_dir.name}")

    # Find checkpoint
    checkpoint = find_latest_checkpoint(run_dir)
    if not checkpoint:
        print("No checkpoints found!", file=sys.stderr)
        sys.exit(1)

    step = get_checkpoint_step(checkpoint)
    print(f"Checkpoint: step {step:,}")

    # Create output directory for this step
    output_dir = args.output_dir / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse losses
    log_path = run_dir / "trainer_0_log.txt"
    records = parse_losses(log_path)
    print(f"Parsed {len(records):,} training steps")

    # Detect plateau
    plateau = detect_plateau(records)
    if plateau:
        print("⚠️  Plateau detected in mel loss!")

    # Print current losses
    if records:
        latest = records[-1]
        print("\nCurrent losses:")
        for k, v in sorted(latest.items()):
            if k.startswith("loss_"):
                print(f"  {k}: {v:.4f}")

    # Generate loss plot
    if not args.no_plot:
        plot_path = output_dir / f"losses_step_{step:06d}.png"
        plot_losses(records, plot_path)
        print(f"\nLoss plot: {plot_path}")

    # Generate samples
    samples: list[Path] = []
    if not args.no_samples:
        print("\nGenerating test samples...")
        samples = generate_samples(checkpoint, output_dir, step)

    # Save report
    report_path = save_report(output_dir, step, records, plateau, samples)
    print(f"\nReport: {report_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Step: {step:,}")
    print(f"Plateau: {'YES ⚠️' if plateau else 'No'}")
    print(f"Samples: {len(samples)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
