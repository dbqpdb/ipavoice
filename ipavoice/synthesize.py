"""IPA-to-speech synthesis using trained VITS model.

Usage:
    # Synthesize IPA to WAV file
    python -m ipavoice.synthesize "həˈloʊ" -o hello.wav

    # Use specific language style (affects accent/phonetic realization)
    python -m ipavoice.synthesize "bɔ̃ʒuʁ" --lang-style FRA -o bonjour.wav

    # Postprocessing: pitch range, reverb, normalize
    python -m ipavoice.synthesize "həˈloʊ" --pitch-range female --reverb 0.3 --normalize

    # Custom pitch range in Hz
    python -m ipavoice.synthesize "həˈloʊ" --pitch-range 100-200

    # List available language styles
    python -m ipavoice.synthesize --list-styles

    # Use specific checkpoint
    python -m ipavoice.synthesize "test" --checkpoint path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor


# --- Pitch range presets (min_f0, max_f0) in Hz ---
PITCH_PRESETS: dict[str, tuple[float, float]] = {
    "male": (85.0, 180.0),      # Typical male F0 range
    "female": (165.0, 255.0),   # Typical female F0 range
    "child": (250.0, 400.0),    # Typical child F0 range
}


def estimate_f0_range(waveform: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Estimate fundamental frequency range of a waveform.

    Args:
        waveform: Audio waveform.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (min_f0, max_f0, median_f0) in Hz.
    """
    import librosa

    # Use librosa's pyin for robust F0 estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform.astype(np.float32),
        fmin=50,
        fmax=500,
        sr=sample_rate,
    )

    # Get voiced frames only
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        # Use percentiles to avoid outliers
        min_f0 = float(np.percentile(voiced_f0, 5))
        max_f0 = float(np.percentile(voiced_f0, 95))
        median_f0 = float(np.median(voiced_f0))
        return min_f0, max_f0, median_f0

    return 100.0, 200.0, 150.0  # Fallback


def scale_pitch_range(
    waveform: np.ndarray,
    sample_rate: int,
    target_range: tuple[float, float],
) -> np.ndarray:
    """Scale the pitch range of a waveform to a target range.

    Maps the pitch contour so that the current F0 center is shifted
    to match the target range center. This preserves the relative
    pitch movements (intonation contour shape).

    Args:
        waveform: Audio waveform.
        sample_rate: Sample rate in Hz.
        target_range: Tuple of (min_f0, max_f0) in Hz.

    Returns:
        Pitch-scaled waveform.
    """
    import librosa

    # Estimate current pitch range
    current_min, current_max, current_median = estimate_f0_range(waveform, sample_rate)
    target_min, target_max = target_range

    # Calculate the center points
    current_center = (current_min + current_max) / 2
    target_center = (target_min + target_max) / 2

    # Calculate the shift in semitones to move center to target center
    semitones = 12 * np.log2(target_center / current_center)

    print(f"  Pitch range: {current_min:.0f}-{current_max:.0f} Hz -> {target_min:.0f}-{target_max:.0f} Hz")
    print(f"  Shift: {semitones:+.1f} semitones")

    # Apply pitch shift
    shifted = librosa.effects.pitch_shift(
        waveform.astype(np.float32),
        sr=sample_rate,
        n_steps=semitones,
    )

    return shifted


def shift_pitch(
    waveform: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    """Shift pitch of waveform by a fixed amount.

    Args:
        waveform: Audio waveform.
        sample_rate: Sample rate in Hz.
        semitones: Shift amount in semitones (positive = higher).

    Returns:
        Pitch-shifted waveform.
    """
    import librosa

    if semitones == 0:
        return waveform

    shifted = librosa.effects.pitch_shift(
        waveform.astype(np.float32),
        sr=sample_rate,
        n_steps=semitones,
    )

    return shifted


def add_reverb(
    waveform: np.ndarray,
    sample_rate: int,
    amount: float = 0.3,
    room_size: float = 0.5,
) -> np.ndarray:
    """Add reverb effect to waveform using simple convolution.

    Args:
        waveform: Audio waveform.
        sample_rate: Sample rate in Hz.
        amount: Wet/dry mix (0.0 = dry, 1.0 = fully wet).
        room_size: Room size factor (0.0-1.0), affects decay time.

    Returns:
        Waveform with reverb applied.
    """
    from scipy import signal

    # Generate simple impulse response
    # Decay time based on room size (0.1s to 2s)
    decay_time = 0.1 + room_size * 1.9
    ir_length = int(decay_time * sample_rate)

    # Exponential decay impulse response
    t = np.linspace(0, decay_time, ir_length)
    decay_rate = 3.0 + room_size * 4.0  # Faster decay for smaller rooms
    ir = np.exp(-decay_rate * t)

    # Add some early reflections
    reflection_times = [0.01, 0.02, 0.035, 0.05]
    for rt in reflection_times:
        idx = int(rt * sample_rate)
        if idx < ir_length:
            ir[idx] += 0.3 * np.exp(-decay_rate * rt)

    # Normalize IR
    ir = ir / np.max(np.abs(ir))

    # Convolve
    wet = signal.convolve(waveform, ir, mode='full')[:len(waveform)]

    # Mix dry and wet
    output = (1 - amount) * waveform + amount * wet

    return output.astype(waveform.dtype)


def normalize_audio(
    waveform: np.ndarray,
    target_db: float = -3.0,
) -> np.ndarray:
    """Normalize audio to target dB level.

    Args:
        waveform: Audio waveform.
        target_db: Target peak level in dB (default: -3 dB).

    Returns:
        Normalized waveform.
    """
    # Calculate current peak
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform

    # Calculate target amplitude
    target_amp = 10 ** (target_db / 20)

    # Scale
    scale = target_amp / peak
    normalized = waveform * scale

    return normalized


def parse_pitch_range(value: str) -> tuple[float, float] | None:
    """Parse pitch range argument.

    Args:
        value: Either a preset name (male/female/child) or Hz range like "100-200".

    Returns:
        Tuple of (min_f0, max_f0) in Hz, or None if invalid.
    """
    # Check presets first
    if value.lower() in PITCH_PRESETS:
        return PITCH_PRESETS[value.lower()]

    # Try parsing as Hz range "min-max"
    if "-" in value:
        parts = value.split("-")
        if len(parts) == 2:
            try:
                min_hz = float(parts[0])
                max_hz = float(parts[1])
                if min_hz >= max_hz:
                    print(f"Error: Invalid range {min_hz}-{max_hz}, min must be less than max")
                    return None
                if not (20 <= min_hz <= 500 and 20 <= max_hz <= 500):
                    print(f"Warning: Range {min_hz}-{max_hz} Hz outside typical bounds (20-500 Hz)")
                return (min_hz, max_hz)
            except ValueError:
                pass

    print(f"Error: Invalid pitch range '{value}'")
    print("Use a preset (male, female, child) or Hz range (e.g., 100-200)")
    return None


def find_latest_checkpoint() -> tuple[Path, Path] | None:
    """Find the latest checkpoint across all training runs.

    Returns:
        Tuple of (checkpoint_path, config_path) or None if not found.
    """
    output_dir = Path("data/vits_output")
    if not output_dir.exists():
        return None

    checkpoints: list[tuple[int, Path, Path]] = []

    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("ipavoice_vits-"):
            continue

        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        for ckpt in run_dir.glob("checkpoint_*.pth"):
            # Extract step number from filename
            try:
                step = int(ckpt.stem.split("_")[1])
                checkpoints.append((step, ckpt, config_path))
            except (IndexError, ValueError):
                continue

        # Also check for best_model.pth
        best_model = run_dir / "best_model.pth"
        if best_model.exists():
            # Give it a high priority but after numbered checkpoints
            checkpoints.append((0, best_model, config_path))

    if not checkpoints:
        return None

    # Sort by step number (descending) and take the highest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    _, ckpt_path, config_path = checkpoints[0]
    return ckpt_path, config_path


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cuda",
) -> tuple[Vits, TTSTokenizer, SpeakerManager, AudioProcessor, VitsConfig]:
    """Load VITS model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to config.json.
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer, speaker_manager, audio_processor, config).
    """
    # Load config
    with open(config_path) as f:
        config_dict: dict[str, Any] = json.load(f)

    config = VitsConfig()
    config.from_dict(config_dict)

    # Audio processor
    ap = AudioProcessor.init_from_config(config)

    # Tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Speaker manager
    speaker_manager = SpeakerManager()
    speakers_file = config_dict.get("speakers_file") or config_dict.get("model_args", {}).get("speakers_file")
    if speakers_file and Path(speakers_file).exists():
        speaker_manager.load_ids_from_file(speakers_file)
    else:
        # Try to find speakers.pth in the same directory as checkpoint
        speakers_pth = checkpoint_path.parent / "speakers.pth"
        if speakers_pth.exists():
            speaker_manager.load_ids_from_file(str(speakers_pth))

    # Create model
    model = Vits(config, ap, tokenizer, speaker_manager)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Speakers: {speaker_manager.num_speakers}")

    return model, tokenizer, speaker_manager, ap, config


def synthesize(
    model: Vits,
    tokenizer: TTSTokenizer,
    speaker_manager: SpeakerManager,
    ap: AudioProcessor,
    config: VitsConfig,
    text: str,
    lang_style: str | None = None,
    device: str = "cuda",
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_scale_dp: float = 1.0,
) -> np.ndarray:
    """Synthesize audio from IPA text.

    Args:
        model: Loaded VITS model.
        tokenizer: Text tokenizer.
        speaker_manager: Speaker manager (stores language style embeddings).
        ap: Audio processor.
        config: Model config.
        text: IPA text to synthesize.
        lang_style: Language style code (e.g., "ENG", "FRA"). Affects phonetic
            realization and prosody based on patterns learned from that language.
        device: Device for inference.
        length_scale: Duration scaling (>1 = slower, <1 = faster).
        noise_scale: Noise scale for variational inference.
        noise_scale_dp: Noise scale for duration predictor.

    Returns:
        Audio waveform as numpy array.
    """
    # Get language style ID (internally stored as "speaker" in Coqui TTS)
    style_id: int | None = None
    if speaker_manager.num_speakers > 0:
        if lang_style and lang_style in speaker_manager.name_to_id:
            style_id = speaker_manager.name_to_id[lang_style]
        else:
            # Use first style as default
            style_id = 0
            if lang_style:
                print(f"Warning: Language style '{lang_style}' not found, using default")

    # Set inference parameters on model config
    model.length_scale = length_scale
    model.inference_noise_scale = noise_scale
    model.inference_noise_scale_dp = noise_scale_dp

    # Tokenize
    text_inputs = tokenizer.text_to_ids(text)
    text_inputs = torch.LongTensor(text_inputs).unsqueeze(0).to(device)

    # Language style embedding (stored as speaker_ids in model)
    speaker_ids = None
    if style_id is not None:
        speaker_ids = torch.LongTensor([style_id]).to(device)

    # Inference
    with torch.no_grad():
        outputs = model.inference(
            text_inputs,
            aux_input={
                "speaker_ids": speaker_ids,
                "d_vectors": None,
                "language_ids": None,
            },
        )

    # Extract waveform (stored as "model_outputs" in Coqui TTS VITS)
    waveform = outputs["model_outputs"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # Remove batch and channel dimensions: [1, 1, samples] -> [samples]
    waveform = waveform.squeeze()

    return waveform


def save_wav(waveform: np.ndarray, path: Path, sample_rate: int = 22050) -> None:
    """Save waveform to WAV file."""
    import scipy.io.wavfile as wavfile

    # Normalize to int16 range
    if waveform.dtype == np.float32 or waveform.dtype == np.float64:
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).astype(np.int16)

    wavfile.write(str(path), sample_rate, waveform)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize speech from IPA transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "həˈloʊ" -o hello.wav
  %(prog)s "bɔ̃ʒuʁ" --lang-style FRA -o bonjour.wav
  %(prog)s "həˈloʊ" --pitch-range female --reverb 0.2 --normalize
  %(prog)s "həˈloʊ" --pitch-range 100-200 -o male_voice.wav
  %(prog)s --list-styles
        """,
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="IPA text to synthesize",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output.wav"),
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "--lang-style", "-l",
        type=str,
        dest="lang_style",
        help="Language style code (e.g., ENG, FRA, DEU) - affects phonetic realization",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint (default: latest)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json (default: from checkpoint directory)",
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available language styles and exit",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Duration scale (>1 slower, <1 faster)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.667,
        help="Noise scale for variational inference",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )

    # Postprocessing options
    postproc = parser.add_argument_group("postprocessing")
    postproc.add_argument(
        "--pitch-range",
        type=str,
        help="Target pitch range: 'male', 'female', 'child', or Hz range like '100-200'",
    )
    postproc.add_argument(
        "--pitch-shift",
        type=float,
        help="Shift pitch by semitones (positive = higher, negative = lower)",
    )
    postproc.add_argument(
        "--reverb",
        type=float,
        metavar="AMOUNT",
        help="Add reverb (0.0-1.0, wet/dry mix)",
    )
    postproc.add_argument(
        "--normalize",
        nargs="?",
        const=-3.0,
        type=float,
        metavar="DB",
        help="Normalize audio to peak dB level (default: -3 dB)",
    )

    args = parser.parse_args()

    # Determine device
    device = "cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        config_path = args.config or checkpoint_path.parent / "config.json"
    else:
        result = find_latest_checkpoint()
        if result is None:
            print("Error: No checkpoints found in data/vits_output/", file=sys.stderr)
            print("Train a model first or specify --checkpoint", file=sys.stderr)
            sys.exit(1)
        checkpoint_path, config_path = result
        if args.config:
            config_path = args.config

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")

    # Load model
    model, tokenizer, speaker_manager, ap, config = load_model(
        checkpoint_path, config_path, device
    )

    # List language styles mode
    if args.list_styles:
        print("\nAvailable language styles:")
        for name, idx in sorted(speaker_manager.name_to_id.items(), key=lambda x: x[1]):
            print(f"  {idx:3d}: {name}")
        sys.exit(0)

    # Require text for synthesis
    if not args.text:
        parser.error("Text argument required for synthesis (or use --list-styles)")

    # Synthesize
    print(f"\nSynthesizing: {args.text!r}")
    if args.lang_style:
        print(f"Language style: {args.lang_style}")

    waveform = synthesize(
        model=model,
        tokenizer=tokenizer,
        speaker_manager=speaker_manager,
        ap=ap,
        config=config,
        text=args.text,
        lang_style=args.lang_style,
        device=device,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
    )

    sample_rate = config.audio.sample_rate

    # --- Postprocessing ---
    if args.pitch_range or args.pitch_shift or args.reverb is not None or args.normalize is not None:
        print("\nPostprocessing:")

    # Pitch range scaling (e.g., male/female/child)
    if args.pitch_range:
        target_range = parse_pitch_range(args.pitch_range)
        if target_range:
            waveform = scale_pitch_range(waveform, sample_rate, target_range)

    # Fixed pitch shift in semitones
    if args.pitch_shift:
        print(f"  Pitch shift: {args.pitch_shift:+.1f} semitones")
        waveform = shift_pitch(waveform, sample_rate, args.pitch_shift)

    # Reverb
    if args.reverb is not None:
        amount = max(0.0, min(1.0, args.reverb))
        print(f"  Reverb: {amount:.0%} wet")
        waveform = add_reverb(waveform, sample_rate, amount=amount)

    # Normalize
    if args.normalize is not None:
        print(f"  Normalize: {args.normalize:.1f} dB peak")
        waveform = normalize_audio(waveform, target_db=args.normalize)

    # Save
    save_wav(waveform, args.output, sample_rate)

    duration = len(waveform) / sample_rate
    print(f"Duration: {duration:.2f}s")


if __name__ == "__main__":
    main()
