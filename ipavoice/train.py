"""VITS training entry point.

Usage:
    python -m ipavoice.train                     # full training
    python -m ipavoice.train --test-run          # 1000 steps to validate pipeline
    python -m ipavoice.train --resume PATH       # resume from checkpoint
    python -m ipavoice.train --batch-size 16     # custom batch size
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

from training.config import DATASET_UCLA, OUTPUT_DIR, VALID_DATASETS, build_config, get_training_dir
from training.vits_duration_supervised import VitsDurationSupervised


def train(
    test_run: bool = False,
    resume_path: str | None = None,
    batch_size: int = 32,
    eval_batch_size: int = 16,
    mixed_precision: bool = False,
    num_loader_workers: int = 4,
    dataset: str = DATASET_UCLA,
    duration_supervision: bool = False,
    duration_supervision_alpha: float = 1.0,
) -> None:
    """Run VITS training.

    Args:
        test_run: If True, run only 1000 steps to validate the pipeline.
        resume_path: Path to checkpoint to resume from.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        mixed_precision: Enable mixed precision training.
        num_loader_workers: DataLoader workers.
        dataset: Dataset source to use ("ucla", "cv", or "combined").
        duration_supervision: Enable MFA duration supervision.
        duration_supervision_alpha: Weight for duration supervision loss.
    """
    # Build config
    epochs: int = 1 if test_run else 1000
    config = build_config(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        epochs=epochs,
        num_loader_workers=num_loader_workers,
        mixed_precision=mixed_precision,
        dataset_source=dataset,
    )

    # Add duration supervision config if enabled
    if duration_supervision:
        training_dir = get_training_dir(dataset)
        config.duration_supervision_alpha = duration_supervision_alpha
        config.durations_dir = str(training_dir / "durations")
        print(f"Duration supervision enabled (alpha={duration_supervision_alpha})")
        print(f"  Durations dir: {config.durations_dir}")

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

    # Model - use duration supervised version if enabled
    if duration_supervision:
        model: Vits = VitsDurationSupervised(config, ap, tokenizer, speaker_manager)
        print(f"Using VitsDurationSupervised model")
    else:
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
    parser.add_argument(
        "--dataset", type=str, default=DATASET_UCLA, choices=VALID_DATASETS,
        help=f"Dataset source: {', '.join(VALID_DATASETS)} (default: {DATASET_UCLA})"
    )
    parser.add_argument(
        "--duration-supervision", action="store_true",
        help="Enable MFA duration supervision (requires mfa-extract to have been run)"
    )
    parser.add_argument(
        "--duration-alpha", type=float, default=1.0,
        help="Weight for duration supervision loss (default: 1.0)"
    )

    args: argparse.Namespace = parser.parse_args()

    train(
        test_run=args.test_run,
        resume_path=args.resume,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        mixed_precision=args.mixed_precision,
        num_loader_workers=args.workers,
        dataset=args.dataset,
        duration_supervision=args.duration_supervision,
        duration_supervision_alpha=args.duration_alpha,
    )


if __name__ == "__main__":
    main()
