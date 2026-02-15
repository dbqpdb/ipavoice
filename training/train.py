"""VITS training entry point.

Usage:
    python -m training.train                     # full training
    python -m training.train --test-run          # 1000 steps to validate pipeline
    python -m training.train --resume PATH       # resume from checkpoint
    python -m training.train --batch-size 16     # custom batch size
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from training.config import OUTPUT_DIR, build_config


def train(
    test_run: bool = False,
    resume_path: str | None = None,
    batch_size: int = 32,
    eval_batch_size: int = 16,
    mixed_precision: bool = False,
    num_loader_workers: int = 4,
) -> None:
    """Run VITS training.

    Args:
        test_run: If True, run only 1000 steps to validate the pipeline.
        resume_path: Path to checkpoint to resume from.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        mixed_precision: Enable mixed precision training.
        num_loader_workers: DataLoader workers.
    """
    # Build config
    epochs: int = 1 if test_run else 1000
    config = build_config(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        epochs=epochs,
        num_loader_workers=num_loader_workers,
        mixed_precision=mixed_precision,
    )

    if test_run:
        config.print_step = 10
        config.save_step = 500
        config.run_eval = True

    # Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Audio processor
    ap: AudioProcessor = AudioProcessor.init_from_config(config)

    # Load dataset samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Train samples: {len(train_samples):,}")
    print(f"Eval samples: {len(eval_samples):,}")

    # Speaker manager
    speaker_manager: SpeakerManager = SpeakerManager()
    speaker_manager.set_ids_from_data(
        train_samples + eval_samples, parse_key="speaker_name"
    )
    config.model_args.num_speakers = speaker_manager.num_speakers
    config.num_speakers = speaker_manager.num_speakers

    print(f"Speakers detected: {speaker_manager.num_speakers}")

    # Tokenizer
    tokenizer: TTSTokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Model
    model: Vits = Vits(config, ap, tokenizer, speaker_manager)

    # Trainer args
    trainer_args: TrainerArgs = TrainerArgs(
        restore_path=resume_path,
    )

    if test_run:
        trainer_args.total_steps = 1000

    # Trainer
    trainer: Trainer = Trainer(
        trainer_args,
        config,
        str(OUTPUT_DIR),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train VITS model on IPA-to-speech data"
    )
    parser.add_argument(
        "--test-run", action="store_true",
        help="Run 1000 steps to validate the training pipeline"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=16,
        help="Evaluation batch size (default: 16)"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Enable mixed precision training"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers (default: 4)"
    )

    args: argparse.Namespace = parser.parse_args()

    train(
        test_run=args.test_run,
        resume_path=args.resume,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        mixed_precision=args.mixed_precision,
        num_loader_workers=args.workers,
    )


if __name__ == "__main__":
    main()
