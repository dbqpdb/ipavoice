"""IPA Voice Gradio demo for Hugging Face Spaces."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# --- Model loading ---

MODEL = None
TOKENIZER = None
SPEAKER_MANAGER = None
AP = None
CONFIG = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model path - set via environment variable or default
MODEL_PATH = os.environ.get("IPAVOICE_MODEL_PATH", "model")


def load_model():
    """Load the VITS model (called once at startup)."""
    global MODEL, TOKENIZER, SPEAKER_MANAGER, AP, CONFIG

    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.utils.audio import AudioProcessor

    model_dir = Path(MODEL_PATH)
    config_path = model_dir / "config.json"
    checkpoint_path = list(model_dir.glob("*.pth"))
    checkpoint_path = [p for p in checkpoint_path if "speaker" not in p.name.lower()]

    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    checkpoint_path = checkpoint_path[0]

    print(f"Loading model from {checkpoint_path}")
    print(f"Device: {DEVICE}")

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    CONFIG = VitsConfig()
    CONFIG.from_dict(config_dict)

    # Audio processor
    AP = AudioProcessor.init_from_config(CONFIG)

    # Tokenizer
    TOKENIZER, CONFIG = TTSTokenizer.init_from_config(CONFIG)

    # Speaker manager
    SPEAKER_MANAGER = SpeakerManager()
    speakers_file = model_dir / "speakers.pth"
    if speakers_file.exists():
        SPEAKER_MANAGER.load_ids_from_file(str(speakers_file))

    # Model
    MODEL = Vits(CONFIG, AP, TOKENIZER, SPEAKER_MANAGER)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    MODEL.load_state_dict(checkpoint["model"], strict=False)
    MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Model loaded: {sum(p.numel() for p in MODEL.parameters()):,} parameters")
    print(f"Language styles: {SPEAKER_MANAGER.num_speakers}")


# --- Audio processing ---

def estimate_f0_range(waveform: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Estimate F0 range of waveform."""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        waveform.astype(np.float32),
        fmin=50,
        fmax=500,
        sr=sample_rate,
    )

    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        return (
            float(np.percentile(voiced_f0, 5)),
            float(np.percentile(voiced_f0, 95)),
            float(np.median(voiced_f0)),
        )
    return 100.0, 200.0, 150.0


def scale_pitch_range(
    waveform: np.ndarray,
    sample_rate: int,
    target_range: tuple[float, float],
) -> np.ndarray:
    """Scale pitch to target range."""
    import librosa

    current_min, current_max, _ = estimate_f0_range(waveform, sample_rate)
    target_min, target_max = target_range

    current_center = (current_min + current_max) / 2
    target_center = (target_min + target_max) / 2

    semitones = 12 * np.log2(target_center / current_center)

    return librosa.effects.pitch_shift(
        waveform.astype(np.float32),
        sr=sample_rate,
        n_steps=semitones,
    )


def add_reverb(
    waveform: np.ndarray,
    sample_rate: int,
    amount: float = 0.3,
) -> np.ndarray:
    """Add reverb effect."""
    from scipy import signal

    decay_time = 0.5
    ir_length = int(decay_time * sample_rate)

    t = np.linspace(0, decay_time, ir_length)
    ir = np.exp(-5.0 * t)

    for rt in [0.01, 0.02, 0.035, 0.05]:
        idx = int(rt * sample_rate)
        if idx < ir_length:
            ir[idx] += 0.3 * np.exp(-5.0 * rt)

    ir = ir / np.max(np.abs(ir))
    wet = signal.convolve(waveform, ir, mode="full")[: len(waveform)]

    return ((1 - amount) * waveform + amount * wet).astype(waveform.dtype)


def normalize_audio(waveform: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize to target dB."""
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform
    target_amp = 10 ** (target_db / 20)
    return waveform * (target_amp / peak)


# --- Pitch range presets ---

PITCH_PRESETS = {
    "Default": None,
    "Male (85-180 Hz)": (85.0, 180.0),
    "Female (165-255 Hz)": (165.0, 255.0),
    "Child (250-400 Hz)": (250.0, 400.0),
}


# --- Synthesis function ---

def synthesize(
    ipa_text: str,
    lang_style: str,
    pitch_preset: str,
    reverb_amount: float,
    normalize: bool,
    speed: float,
) -> tuple[int, np.ndarray] | None:
    """Synthesize IPA text to audio."""
    if not ipa_text.strip():
        return None

    if MODEL is None:
        load_model()

    # Get language style ID
    style_id = 0
    if lang_style and lang_style in SPEAKER_MANAGER.name_to_id:
        style_id = SPEAKER_MANAGER.name_to_id[lang_style]

    # Set inference parameters
    MODEL.length_scale = speed
    MODEL.inference_noise_scale = 0.667
    MODEL.inference_noise_scale_dp = 1.0

    # Tokenize
    text_inputs = TOKENIZER.text_to_ids(ipa_text)
    text_inputs = torch.LongTensor(text_inputs).unsqueeze(0).to(DEVICE)
    speaker_ids = torch.LongTensor([style_id]).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = MODEL.inference(
            text_inputs,
            aux_input={
                "speaker_ids": speaker_ids,
                "d_vectors": None,
                "language_ids": None,
            },
        )

    waveform = outputs["model_outputs"].cpu().numpy().squeeze()
    sample_rate = CONFIG.audio.sample_rate

    # Postprocessing
    if pitch_preset != "Default" and pitch_preset in PITCH_PRESETS:
        target_range = PITCH_PRESETS[pitch_preset]
        if target_range:
            waveform = scale_pitch_range(waveform, sample_rate, target_range)

    if reverb_amount > 0:
        waveform = add_reverb(waveform, sample_rate, reverb_amount)

    if normalize:
        waveform = normalize_audio(waveform)

    return (sample_rate, waveform)


# --- Gradio interface ---

def get_language_styles() -> list[str]:
    """Get available language styles."""
    if SPEAKER_MANAGER is None:
        return ["ENG"]
    return sorted(SPEAKER_MANAGER.name_to_id.keys())


# Example IPA inputs
EXAMPLES = [
    ["həˈloʊ ˈwɜːld", "ENG", "Default", 0.0, True, 1.0],
    ["bɔ̃ʒuʁ lə mɔ̃d", "FRA", "Default", 0.1, True, 1.0],
    ["ˈʃtʁaːsə ˈbɛɐ̯lɪn", "DEU", "Default", 0.0, True, 1.0],
    ["ǀʰõã ǃʼũ ǁʰa", "ZUL", "Default", 0.0, True, 1.0],
    ["kʼatʼɬʼi qʷʼəχʷ", "APW", "Default", 0.0, True, 1.0],
    ["həˈloʊ", "ENG", "Female (165-255 Hz)", 0.2, True, 1.0],
    ["həˈloʊ", "ENG", "Child (250-400 Hz)", 0.0, True, 1.0],
]


def create_app():
    """Create Gradio app."""
    # Load model at startup
    try:
        load_model()
        lang_styles = get_language_styles()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        lang_styles = ["ENG", "FRA", "DEU", "ZUL", "APW", "THA", "CMN"]

    with gr.Blocks(title="IPA Voice", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # IPA Voice

            Synthesize speech from International Phonetic Alphabet (IPA) transcriptions.
            Trained on the UCLA Phonetics Lab Archive covering 291 languages.

            **Usage:** Enter IPA text, select a language style, and click Synthesize.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                ipa_input = gr.Textbox(
                    label="IPA Text",
                    placeholder="Enter IPA transcription, e.g., həˈloʊ",
                    lines=2,
                )

                with gr.Row():
                    lang_style = gr.Dropdown(
                        choices=lang_styles,
                        value="ENG",
                        label="Language Style",
                        info="Affects phonetic realization based on language patterns",
                    )
                    pitch_preset = gr.Dropdown(
                        choices=list(PITCH_PRESETS.keys()),
                        value="Default",
                        label="Pitch Range",
                    )

                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed",
                    )
                    reverb = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        label="Reverb",
                    )
                    normalize = gr.Checkbox(
                        value=True,
                        label="Normalize",
                    )

                synthesize_btn = gr.Button("Synthesize", variant="primary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Output",
                    type="numpy",
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[ipa_input, lang_style, pitch_preset, reverb, normalize, speed],
            outputs=audio_output,
            fn=synthesize,
            cache_examples=False,
        )

        gr.Markdown(
            """
            ---
            **Notes:**
            - Language style uses the model's learned patterns for that language
            - Some rare IPA symbols may not be in the vocabulary
            - Model trained on UCLA Phonetics Lab Archive (CC BY-NC 2.0)

            [GitHub](https://github.com/your-username/ipavoice) |
            [Training Data Report](https://github.com/your-username/ipavoice/blob/main/docs/TRAINING_DATA_REPORT.md)
            """
        )

        synthesize_btn.click(
            fn=synthesize,
            inputs=[ipa_input, lang_style, pitch_preset, reverb, normalize, speed],
            outputs=audio_output,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
