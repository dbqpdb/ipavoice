#!/bin/bash
#SBATCH --job-name=ipavoice-vits
#SBATCH --account=ling696g-spring2026
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#
# VITS training on Tesla P100-PCIE-16GB.
#
# Usage:
#   cd /xdisk/dbrenner/dbrenner/ipavoice
#   mkdir -p logs
#   sbatch cluster/train.sbatch                            # full training
#   sbatch --export=TEST_RUN=1 cluster/train.sbatch        # test run (1000 steps)
#   sbatch --export=RESUME=/path/to/checkpoint.pth cluster/train.sbatch  # resume

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/xdisk/dbrenner/dbrenner/ipavoice"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TEST_RUN="${TEST_RUN:-0}"
RESUME="${RESUME:-}"

# --- Environment setup ---
module load cuda12/12.5
module load micromamba/2.0.2-2
eval "$(micromamba shell hook --shell=bash)"
micromamba activate ipavoice

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# --- GPU diagnostics ---
echo "=== Job ${SLURM_JOB_ID} started at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {props.name} — {props.total_mem / 1e9:.1f} GB')
"
echo ""

# --- Build training command ---
TRAIN_CMD="python -m training.train"
TRAIN_CMD+=" --batch-size ${BATCH_SIZE}"
TRAIN_CMD+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
TRAIN_CMD+=" --workers ${NUM_WORKERS}"

if [[ "${TEST_RUN}" == "1" ]]; then
    TRAIN_CMD+=" --test-run"
    echo ">>> TEST RUN (1000 steps) <<<"
fi

if [[ -n "${RESUME}" ]]; then
    TRAIN_CMD+=" --resume ${RESUME}"
    echo ">>> Resuming from: ${RESUME}"
fi

echo "Command: ${TRAIN_CMD}"
echo ""

# --- Run training ---
${TRAIN_CMD}

echo ""
echo "=== Job finished at $(date) ==="
