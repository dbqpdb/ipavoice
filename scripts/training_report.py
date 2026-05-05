#!/usr/bin/env python3
"""Parse VITS training logs and report evaluation metrics."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_training_log(log_path: Path) -> None:
    """Parse training log and print evaluation metrics table."""
    content = log_path.read_text()

    # Remove ANSI codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    content = ansi_escape.sub("", content)

    # Find eval blocks
    eval_pattern = re.compile(
        r"GLOBAL_STEP: (\d+).*?"
        r"--> EVAL PERFORMANCE.*?"
        r"avg_loss_gen: ([\d.]+).*?"
        r"avg_loss_kl: ([\d.]+).*?"
        r"avg_loss_feat: ([\d.]+).*?"
        r"avg_loss_mel: ([\d.]+)",
        re.DOTALL,
    )

    matches = eval_pattern.findall(content)

    if not matches:
        print("No evaluation metrics found in log file.", file=sys.stderr)
        sys.exit(1)

    print(
        f"{'Step':>12} | {'loss_gen':>10} | {'loss_kl':>10} | {'loss_feat':>10} | {'loss_mel':>10}"
    )
    print("-" * 68)
    for step, gen, kl, feat, mel in matches:
        print(
            f"{int(step):>12,} | {float(gen):>10.4f} | {float(kl):>10.4f} | {float(feat):>10.4f} | {float(mel):>10.4f}"
        )

    # Get current step
    step_pattern = re.compile(r"GLOBAL_STEP: (\d+)")
    all_steps = step_pattern.findall(content)
    if all_steps:
        print(f"\nCurrent step: {int(all_steps[-1]):,}")

    # Summary stats
    if len(matches) >= 2:
        first_mel = float(matches[0][4])
        last_mel = float(matches[-1][4])
        improvement = ((first_mel - last_mel) / first_mel) * 100
        print(f"Mel loss improvement: {first_mel:.4f} -> {last_mel:.4f} ({improvement:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse VITS training logs")
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to training log file",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    parse_training_log(args.log_file)


if __name__ == "__main__":
    main()
