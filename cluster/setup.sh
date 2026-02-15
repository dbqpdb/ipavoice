#!/usr/bin/env bash
# One-time environment setup for the HPC cluster.
#
# Usage (from login node):
#   cd /xdisk/dbrenner/dbrenner/ipavoice
#   bash cluster/setup.sh
#
# This script:
#   1. Loads CUDA 12 and micromamba modules
#   2. Creates a micromamba environment with Python 3.11
#   3. Installs PyTorch for CUDA 12.1 (P100 + driver 550 compatible)
#   4. Installs project dependencies via pip
#
# After setup, activate with:
#   module load cuda12/12.5 micromamba/2.0.2-2
#   micromamba activate ipavoice

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== IPA Voice HPC Setup ==="
echo "Project directory: ${PROJECT_DIR}"

# --- Load modules ---
module load cuda12/12.5
module load micromamba/2.0.2-2

# --- Check NVIDIA driver ---
echo ""
echo "=== GPU Info ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
else
    echo "(nvidia-smi not available on login node — GPUs are on compute nodes)"
fi

# --- Create micromamba environment ---
echo ""
echo "=== Creating micromamba environment ==="
if micromamba env list 2>/dev/null | grep -q ipavoice; then
    echo "Environment 'ipavoice' already exists. To recreate, run:"
    echo "  micromamba env remove -n ipavoice"
    echo "  bash cluster/setup.sh"
else
    micromamba create -n ipavoice python=3.11 -c conda-forge -y
fi

# --- Activate and install ---
eval "$(micromamba shell hook --shell=bash)"
micromamba activate ipavoice

echo ""
echo "=== Installing PyTorch for CUDA 12.1 ==="
# P100 (compute capability 6.0) with driver 550.90.07 supports CUDA 12.4.
# PyTorch cu121 build works with any driver that supports CUDA >= 12.1.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Installing project dependencies ==="
cd "${PROJECT_DIR}"
pip install coqui-tts librosa soundfile regex requests beautifulsoup4 lxml pydub tqdm

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('(CUDA not available on login node — this is expected)')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Transfer data: bash cluster/sync_data.sh  (from local machine)"
echo "  2. Submit job:     sbatch cluster/train.sbatch"
