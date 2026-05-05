---
title: IPA Voice
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc-by-nc-2.0
---

# IPA Voice

Synthesize speech from International Phonetic Alphabet (IPA) transcriptions.

## About

IPA Voice is a text-to-speech model trained on the UCLA Phonetics Lab Archive,
covering 291 languages and 205,000+ audio samples. It takes IPA transcriptions
as input and generates corresponding speech audio.

## Features

- **IPA input**: Directly synthesize from phonetic transcriptions
- **291 language styles**: Select a language to influence phonetic realization
- **Postprocessing**: Pitch range presets (male/female/child), reverb, normalization
- **Diverse phoneme coverage**: Clicks, ejectives, tones, nasalized vowels, and more

## Usage

1. Enter IPA text (e.g., `həˈloʊ` for "hello")
2. Select a language style (affects accent/realization)
3. Optionally adjust pitch, reverb, and speed
4. Click "Synthesize"

## Examples

| IPA | Language | Description |
|-----|----------|-------------|
| həˈloʊ ˈwɜːld | ENG | Hello world |
| bɔ̃ʒuʁ lə mɔ̃d | FRA | Bonjour le monde (French nasals) |
| ǀʰõã ǃʼũ ǁʰa | ZUL | Zulu clicks |
| kʼatʼɬʼi qʷʼəχʷ | APW | Western Apache ejectives |

## Limitations

- Tonal languages have limited training data (690 tone tokens)
- 345 rare IPA symbols appear only once in training
- 98 languages have <50 training samples

See the [Training Data Report](https://github.com/your-username/ipavoice/blob/main/docs/TRAINING_DATA_REPORT.md) for details.

## License

Model and training data: CC BY-NC 2.0 (matching UCLA Phonetics Lab Archive)

## Links

- [GitHub Repository](https://github.com/your-username/ipavoice)
- [UCLA Phonetics Lab Archive](https://archive.phonetics.ucla.edu)
