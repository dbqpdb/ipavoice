"""VITS training configuration for IPA-to-speech model.

Builds Coqui TTS config objects from the preprocessed dataset, including
a custom character set derived from the IPA vocabulary and multi-speaker
setup with one speaker per language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import VitsArgs, VitsAudioConfig

from training.ipa_tokenizer import IPAVocabulary, normalize_ipa

# --- Paths ---

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR: Path = DATA_DIR / "training"
OUTPUT_DIR: Path = DATA_DIR / "vits_output"


def _build_character_set(vocab_path: Path) -> str:
    """Extract unique codepoints from the IPA vocabulary for VITS character config.

    VITS operates on character-level input internally, so we provide all
    unique Unicode codepoints found in the phoneme vocabulary. Our IPA
    normalizer ensures consistent NFD decomposition before tokenization,
    so combining diacritics appear as separate characters. The VITS encoder
    learns to compose base characters + diacritics from context.

    Args:
        vocab_path: Path to ipa_vocab.json.

    Returns:
        Sorted string of unique codepoints.
    """
    vocab: IPAVocabulary = IPAVocabulary.load(vocab_path)

    # Collect all unique codepoints from phoneme tokens
    codepoints: set[str] = set()
    for token in vocab.phoneme_tokens:
        for ch in token:
            codepoints.add(ch)

    # Sort by Unicode codepoint for reproducibility
    chars: str = "".join(sorted(codepoints))
    return chars


def build_config(
    batch_size: int = 32,
    eval_batch_size: int = 16,
    lr: float = 2e-4,
    epochs: int = 1000,
    num_loader_workers: int = 4,
    mixed_precision: bool = False,
    use_sdp: bool = True,
) -> VitsConfig:
    """Build the VITS training configuration.

    Requires preprocessing to have been run first (ipa_vocab.json,
    speakers.json, metadata CSVs must exist).

    Args:
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        lr: Learning rate.
        epochs: Total training epochs.
        num_loader_workers: DataLoader workers.
        mixed_precision: Enable mixed precision training.
        use_sdp: Use stochastic duration predictor.

    Returns:
        Configured VitsConfig ready for training.
    """
    vocab_path: Path = TRAINING_DIR / "ipa_vocab.json"
    speakers_path: Path = TRAINING_DIR / "speakers.json"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}. Run preprocessing first.")
    if not speakers_path.exists():
        raise FileNotFoundError(f"Speaker map not found: {speakers_path}. Run preprocessing first.")

    # Load speaker map for count
    with open(speakers_path, encoding="utf-8") as f:
        speaker_map: dict[str, int] = json.load(f)
    num_speakers: int = len(speaker_map)

    # Build character set from vocabulary
    ipa_chars: str = _build_character_set(vocab_path)
    print(f"IPA character set: {len(ipa_chars)} unique codepoints")

    # Audio config — VITS defaults already match our target format
    audio_config: VitsAudioConfig = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        fft_size=1024,
        mel_fmin=0,
        mel_fmax=None,
    )

    # Model args
    model_args: VitsArgs = VitsArgs(
        use_speaker_embedding=True,
        use_sdp=use_sdp,
    )

    # Characters config — all IPA codepoints as characters, no phonemizer
    characters_config: CharactersConfig = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=ipa_chars,
        punctuations="",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    )

    # Dataset config — coqui formatter: audio_file|text|speaker_name (with header)
    dataset_config: BaseDatasetConfig = BaseDatasetConfig(
        formatter="coqui",
        path=str(TRAINING_DIR),
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_val.csv",
    )

    # Main VITS config
    config: VitsConfig = VitsConfig(
        model_args=model_args,
        audio=audio_config,
        run_name="ipavoice_vits",
        use_speaker_embedding=True,
        num_speakers=num_speakers,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_loader_workers=num_loader_workers,
        num_eval_loader_workers=max(1, num_loader_workers // 2),
        run_eval=True,
        epochs=epochs,
        lr_gen=lr,
        lr_disc=lr,
        use_phonemes=False,
        text_cleaner=None,
        compute_input_seq_cache=False,
        print_step=50,
        print_eval=True,
        mixed_precision=mixed_precision,
        output_path=str(OUTPUT_DIR),
        datasets=[dataset_config],
        characters=characters_config,
    )

    # Force conversion of custom characters to config attribute
    config.from_dict(config.to_dict())

    print(f"VITS config built:")
    print(f"  Speakers: {num_speakers}")
    print(f"  Character set size: {len(ipa_chars)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  SDP: {use_sdp}")
    print(f"  Mixed precision: {mixed_precision}")
    print(f"  Output: {OUTPUT_DIR}")

    return config
