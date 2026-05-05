# Deploying IPA Voice to Hugging Face Spaces

## Prerequisites

1. Hugging Face account: https://huggingface.co/join
2. Hugging Face CLI: `pip install huggingface_hub`
3. Login: `huggingface-cli login`

## Step 1: Create the Space

```bash
# Create a new Space with GPU
huggingface-cli repo create ipavoice --type space --space_sdk gradio
```

Or create via web UI:
1. Go to https://huggingface.co/new-space
2. Name: `ipavoice`
3. SDK: Gradio
4. Hardware: GPU (T4 small recommended)
5. License: CC BY-NC 2.0

## Step 2: Prepare Model Files

The model needs these files in a `model/` directory:
- `config.json` - Model configuration
- `checkpoint_XXXXXX.pth` - Model weights (or `best_model.pth`)
- `speakers.pth` - Speaker/language embeddings

```bash
# Create model package from best checkpoint
mkdir -p demo/model

# Copy from your best training run (adjust path as needed)
BEST_RUN="data/vits_output/ipavoice_vits-March-02-2026_01+14PM-5505364"

cp "$BEST_RUN/config.json" demo/model/
cp "$BEST_RUN/checkpoint_140000.pth" demo/model/
cp "$BEST_RUN/speakers.pth" demo/model/
```

## Step 3: Upload to Hugging Face

Option A: Upload to the Space directly (model included):

```bash
cd demo
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/ipavoice
git add .
git commit -m "Initial commit"
git push -u origin main
```

Option B: Upload model separately to Hugging Face Hub (recommended for large models):

```bash
# Create a model repository
huggingface-cli repo create ipavoice-model --type model

# Upload model files
huggingface-cli upload YOUR_USERNAME/ipavoice-model demo/model/

# Then update app.py to load from Hub:
# MODEL_PATH = "YOUR_USERNAME/ipavoice-model"
```

## Step 4: Configure Hardware

In the Space settings (Settings tab):
1. Set Hardware to "T4 small" or "T4 medium"
2. Sleep timeout: 1 hour (to save credits)

## File Structure

Your Space should have:

```
ipavoice/
├── README.md          # Space metadata (YAML frontmatter)
├── app.py             # Gradio application
├── requirements.txt   # Python dependencies
└── model/             # Model files (if not using Hub)
    ├── config.json
    ├── checkpoint_140000.pth
    └── speakers.pth
```

## Testing Locally

```bash
cd demo
pip install -r requirements.txt

# Set model path
export IPAVOICE_MODEL_PATH="../data/vits_output/ipavoice_vits-March-02-2026_01+14PM-5505364"

# Run
python app.py
```

## Troubleshooting

**Out of memory**: Use a smaller batch or reduce model precision
**Missing speakers**: Ensure `speakers.pth` is in the model directory
**Slow startup**: First load downloads/compiles; subsequent loads are faster

## Cost

Hugging Face Spaces with GPU:
- T4 small: ~$0.60/hour (billed per second of usage)
- Free tier: Limited CPU-only spaces

The Space sleeps after inactivity, so costs are only incurred during use.
