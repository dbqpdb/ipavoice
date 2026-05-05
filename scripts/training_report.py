#!/usr/bin/env python3
"""Parse VITS training logs and report/plot evaluation metrics."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_training_log(log_path: Path) -> list[dict[str, float]]:
    """Parse training log and extract loss values.

    Returns:
        List of dicts with step and loss values.
    """
    content = log_path.read_text()

    # Remove ANSI codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    content = ansi_escape.sub("", content)

    records: list[dict[str, float]] = []
    current_record: dict[str, float] = {}

    # Pattern to match step lines
    step_pattern = re.compile(r"GLOBAL_STEP:\s*(\d+)")

    # Pattern to match loss values
    loss_pattern = re.compile(r"\| > (loss_\w+|avg_loss_\w+):\s*([\d.]+)")

    for line in content.split("\n"):
        step_match = step_pattern.search(line)
        if step_match:
            if current_record and "step" in current_record:
                records.append(current_record)
            current_record = {"step": int(step_match.group(1))}
            continue

        loss_match = loss_pattern.search(line)
        if loss_match:
            loss_name = loss_match.group(1)
            loss_value = float(loss_match.group(2))
            current_record[loss_name] = loss_value

    if current_record and "step" in current_record:
        records.append(current_record)

    return records


def print_report(records: list[dict[str, float]]) -> None:
    """Print summary table of eval metrics."""
    # Filter to records with avg_loss (eval metrics)
    eval_records = [r for r in records if "avg_loss_mel" in r]

    if not eval_records:
        print("No evaluation metrics found.", file=sys.stderr)
        return

    print(
        f"{'Step':>12} | {'loss_gen':>10} | {'loss_kl':>10} | {'loss_feat':>10} | {'loss_mel':>10}"
    )
    print("-" * 68)
    for r in eval_records:
        print(
            f"{int(r['step']):>12,} | {r.get('avg_loss_gen', 0):>10.4f} | "
            f"{r.get('avg_loss_kl', 0):>10.4f} | {r.get('avg_loss_feat', 0):>10.4f} | "
            f"{r.get('avg_loss_mel', 0):>10.4f}"
        )

    if records:
        print(f"\nCurrent step: {int(records[-1]['step']):,}")

    if len(eval_records) >= 2:
        first_mel = eval_records[0].get("avg_loss_mel", 0)
        last_mel = eval_records[-1].get("avg_loss_mel", 0)
        if first_mel > 0:
            improvement = ((first_mel - last_mel) / first_mel) * 100
            print(f"Mel loss improvement: {first_mel:.4f} -> {last_mel:.4f} ({improvement:.1f}%)")


def plot_losses(
    records: list[dict[str, float]],
    output_path: Path | None = None,
) -> None:
    """Plot loss curves."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Key losses to plot
    key_losses = ["loss_mel", "loss_gen", "loss_disc", "loss_kl", "loss_feat", "loss_duration"]

    # Filter records that have training losses (not just eval)
    train_records = [r for r in records if "loss_mel" in r]

    if not train_records:
        print("No training loss data found!", file=sys.stderr)
        return

    # Find which losses are available
    available = [k for k in key_losses if any(k in r for r in train_records)]

    if not available:
        print("No plottable losses found!")
        return

    # Create figure
    n_plots = len(available)
    cols = 2
    rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, loss_name in enumerate(available):
        ax = axes[idx]

        # Extract data for this loss
        steps = [r["step"] for r in train_records if loss_name in r]
        values = [r[loss_name] for r in train_records if loss_name in r]

        if not steps:
            continue

        # Subsample if too many points
        if len(steps) > 1000:
            indices = np.linspace(0, len(steps) - 1, 1000, dtype=int)
            steps = [steps[i] for i in indices]
            values = [values[i] for i in indices]

        ax.plot(steps, values, linewidth=0.5, alpha=0.5, color="blue")

        # Add smoothed line
        if len(values) > 20:
            window = min(50, len(values) // 5)
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            smooth_steps = steps[window - 1:]
            ax.plot(smooth_steps, smoothed, linewidth=2, color="red", label="smoothed")

        ax.set_title(loss_name.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # Set reasonable y limits
        if values:
            q1, q3 = np.percentile(values, [5, 95])
            margin = (q3 - q1) * 0.3
            ax.set_ylim(max(0, q1 - margin), q3 + margin)

    # Hide unused subplots
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("VITS Training Losses", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse VITS training logs")
    parser.add_argument(
        "log_file",
        nargs="?",
        type=Path,
        help="Path to training log file (default: latest)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot loss curves",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Save plot to file (implies --plot)",
    )
    args = parser.parse_args()

    # Find log file
    if args.log_file:
        log_path = args.log_file
    else:
        output_dir = Path("data/vits_output")
        runs = sorted(output_dir.glob("ipavoice_vits-*/trainer_0_log.txt"))
        if not runs:
            print("No training logs found!", file=sys.stderr)
            sys.exit(1)
        log_path = runs[-1]
        print(f"Using: {log_path}")

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    records = parse_training_log(log_path)

    if not records:
        print("No data found in log!", file=sys.stderr)
        sys.exit(1)

    if args.plot or args.output:
        plot_losses(records, output_path=args.output)
    else:
        print_report(records)


if __name__ == "__main__":
    main()
