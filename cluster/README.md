# HPC Cluster Training (UArizona Ocelote)

## Hardware

- **GPU**: Tesla P100-PCIE-16GB (Pascal, compute capability 6.0)
- **Nodes**: 1-2 GPUs, 28 CPUs, 256 GB RAM
- **Partition**: `gpu_standard` (account: `ling696g-spring2026`)
- **Shared storage**: `/xdisk/dbrenner/dbrenner/ipavoice`

## Setup (one-time)

```bash
# Clone repo on the cluster
ssh hpc.arizona.edu
cd /xdisk/dbrenner/dbrenner
git clone <repo-url> ipavoice
cd ipavoice

# Create environment and install dependencies
bash cluster/setup.sh
```

This loads `cuda12/12.5` and `micromamba`, creates an `ipavoice` environment with Python 3.11, and installs PyTorch (CUDA 12.1) plus all project dependencies.

## Data transfer

From your local machine:

```bash
cd /home/db/Projects/ipavoice
bash cluster/sync_data.sh
```

This rsyncs `data/training/` (5.2 GB of preprocessed WAVs, manifests, vocab) and `data/db/` (SQLite database) to the cluster. If the SSH hostname differs from `hpc.arizona.edu`, edit the `REMOTE` variable in `sync_data.sh`.

## Training

```bash
ssh hpc.arizona.edu
cd /xdisk/dbrenner/dbrenner/ipavoice
mkdir -p logs

# Test run first (1000 steps, ~15 minutes on P100)
sbatch --export=TEST_RUN=1 cluster/train.sbatch

# Check status
squeue -u dbrenner
cat logs/train_*.out

# Full training (~1-2 days)
sbatch cluster/train.sbatch

# Resume from checkpoint
sbatch --export=RESUME=data/vits_output/ipavoice_vits-*/best_model.pth cluster/train.sbatch
```

## Monitoring

```bash
# Job status
squeue -u dbrenner

# Live log output
tail -f logs/train_<JOBID>.out

# TensorBoard (from the log output, look for the tensorboard command)
# You'll need to forward the port via SSH:
ssh -L 6006:localhost:6006 hpc.arizona.edu
tensorboard --logdir=/xdisk/dbrenner/dbrenner/ipavoice/data/vits_output
```

## Tuning

Environment variables can be passed via `--export`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_RUN` | `0` | Set to `1` for a 1000-step validation run |
| `RESUME` | (empty) | Path to checkpoint `.pth` file to resume from |
| `BATCH_SIZE` | `8` | Training batch size (P100 16 GB fits 8-12) |
| `EVAL_BATCH_SIZE` | `4` | Evaluation batch size |
| `NUM_WORKERS` | `4` | DataLoader workers |

Example with custom batch size:

```bash
sbatch --export=BATCH_SIZE=12 cluster/train.sbatch
```

## Output

Training outputs are written to `data/vits_output/ipavoice_vits-<timestamp>/`:

- `best_model.pth` — best checkpoint (by eval loss)
- `checkpoint_*.pth` — periodic checkpoints
- `events.out.tfevents.*` — TensorBoard logs
- `config.json` — full training config (for reproducibility)

## Troubleshooting

**Job fails immediately**: Check `logs/train_<JOBID>.err` for module or environment errors.

**Out of GPU memory**: Reduce batch size:
```bash
sbatch --export=BATCH_SIZE=4,EVAL_BATCH_SIZE=2 cluster/train.sbatch
```

**Preempted on windfall**: Use `gpu_standard` (the default) or resume with `--export=RESUME=...`.

**Re-creating the environment**:
```bash
module load micromamba/2.0.2-2
micromamba env remove -n ipavoice
bash cluster/setup.sh
```
