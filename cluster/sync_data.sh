#!/usr/bin/env bash
# Sync preprocessed training data to the HPC cluster.
#
# Usage (from local machine):
#   bash cluster/sync_data.sh
#
# Transfers:
#   data/training/  → cluster (5.2 GB wavs + manifests + vocab)
#   data/db/        → cluster (SQLite database)

set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="dbrenner@hpc.arizona.edu"
REMOTE_DIR="/xdisk/dbrenner/dbrenner/ipavoice"

echo "=== Syncing training data to HPC ==="
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${REMOTE}:${REMOTE_DIR}"
echo ""

# Sync preprocessed training data (wavs, manifests, vocab, speakers)
echo "--- Syncing data/training/ (5.2 GB) ---"
rsync -avhP --stats \
    "${LOCAL_DIR}/data/training/" \
    "${REMOTE}:${REMOTE_DIR}/data/training/"

echo ""

# Sync database
echo "--- Syncing data/db/ ---"
rsync -avhP --stats \
    "${LOCAL_DIR}/data/db/" \
    "${REMOTE}:${REMOTE_DIR}/data/db/"

echo ""
echo "=== Sync complete ==="
echo ""
echo "Next steps on the cluster:"
echo "  ssh ${REMOTE}"
echo "  cd ${REMOTE_DIR}"
echo "  sbatch cluster/train.sbatch"
