"""VITS model with MFA duration supervision.

Extends Coqui TTS VITS to accept ground truth phone durations from MFA
and adds supervision loss comparing predicted durations to MFA alignments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from coqpit import Coqpit
from torch.utils.data import DataLoader
from TTS.tts.models.vits import Vits, VitsDataset


class VitsDatasetWithDurations(VitsDataset):
    """VITS dataset that also loads MFA durations."""

    def __init__(self, *args, durations_dir: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.durations_dir = Path(durations_dir) if durations_dir else None

    def _load_durations(self, item: dict) -> list[int] | None:
        """Load durations for a sample."""
        # Try from sample dict first (embedded in CSV)
        if "durations" in item and item["durations"]:
            dur = item["durations"]
            if isinstance(dur, str):
                try:
                    return json.loads(dur)
                except json.JSONDecodeError:
                    pass
            elif isinstance(dur, list):
                return dur

        # Try from durations directory
        if self.durations_dir is not None:
            audio_file = item.get("audio_file", "")
            audio_id = Path(audio_file).stem
            duration_file = self.durations_dir / f"{audio_id}.json"
            if duration_file.exists():
                try:
                    with open(duration_file) as f:
                        data = json.load(f)
                        return data.get("durations")
                except (json.JSONDecodeError, IOError):
                    pass
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item with durations."""
        data = super().__getitem__(idx)
        item = self.samples[idx]
        data["durations"] = self._load_durations(item)
        return data

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate with duration padding."""
        # Check for durations
        has_durations = any(d.get("durations") is not None for d in batch)

        # Extract durations before parent collate modifies batch
        durations_list = []
        if has_durations:
            for sample in batch:
                dur = sample.get("durations")
                if dur is not None:
                    durations_list.append(torch.LongTensor(dur))
                else:
                    # Use token length as fallback
                    durations_list.append(torch.ones(sample["token_len"], dtype=torch.long))

        # Call parent collate
        batch_out = super().collate_fn(batch)

        # Add padded durations
        if durations_list:
            max_len = max(d.size(0) for d in durations_list)
            padded = torch.zeros(len(durations_list), max_len, dtype=torch.long)
            for i, d in enumerate(durations_list):
                padded[i, :d.size(0)] = d
            batch_out["gt_durations"] = padded

        return batch_out


class VitsDurationSupervised(Vits):
    """VITS with external duration supervision from MFA alignments.

    Adds a loss term that compares the Monotonic Alignment Search (MAS)
    durations to ground truth durations from Montreal Forced Aligner.

    This class minimally extends VITS by:
    1. Using a custom dataset that loads MFA durations
    2. Overriding train_step to compute MFA supervision loss after forward pass
    """

    def __init__(self, config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)

        # Duration supervision weight (configurable via config.duration_supervision_alpha)
        self.duration_supervision_alpha: float = getattr(
            config, 'duration_supervision_alpha', 1.0
        )

        # Durations directory (for loading MFA durations)
        self.durations_dir: Path | None = None
        if hasattr(config, 'durations_dir') and config.durations_dir:
            self.durations_dir = Path(config.durations_dir)

    def get_data_loader(
        self,
        config: Coqpit,
        assets: dict,
        is_eval: bool,
        samples: list[dict],
        verbose: bool,
        num_gpus: int,
        rank: int | None = None,
    ) -> DataLoader:
        """Create data loader with duration support."""
        if is_eval and not config.run_eval:
            return None

        # Use our custom dataset with durations
        dataset = VitsDatasetWithDurations(
            model_args=self.args,
            samples=samples,
            batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            phoneme_cache_path=config.phoneme_cache_path,
            precompute_num_workers=config.precompute_num_workers,
            tokenizer=self.tokenizer,
            start_by_longest=config.start_by_longest,
            durations_dir=self.durations_dir,
        )

        # Wait for DDP processes
        if num_gpus > 1:
            import torch.distributed as dist
            dist.barrier()

        # Preprocess samples
        dataset.preprocess_samples()

        # Get sampler
        sampler = self.get_sampler(config, dataset, num_gpus)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=False,
            sampler=sampler,
        )

        return loader

    def train_step(self, batch: dict, criterion: Any, optimizer_idx: int = 0) -> tuple[dict, dict]:
        """Training step with MFA duration supervision.

        Calls parent train_step, then adds MFA duration loss by comparing
        MAS-derived durations (from alignments) to ground truth MFA durations.
        """
        # Extract gt_durations before parent processes batch
        gt_durations = batch.get("gt_durations", None)

        # Call parent train_step (runs forward, computes all losses)
        outputs, loss_dict = super().train_step(batch, criterion, optimizer_idx)

        # Add MFA duration supervision loss
        if gt_durations is not None and self.duration_supervision_alpha > 0:
            loss_mfa = self._compute_mfa_duration_loss(outputs, gt_durations)
            if loss_mfa is not None:
                loss_dict["loss_duration_mfa"] = loss_mfa
                # Add to total loss
                loss_dict["loss"] = loss_dict["loss"] + loss_mfa

        return outputs, loss_dict

    def _compute_mfa_duration_loss(
        self,
        outputs: dict,
        gt_durations: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute MSE loss between MAS durations and MFA ground truth.

        Args:
            outputs: Model outputs containing 'alignments' [B, T_text, T_spec].
            gt_durations: MFA durations [B, T_text].

        Returns:
            Scalar loss tensor or None if alignments not available.
        """
        # Get alignments from outputs
        alignments = outputs.get("alignments")
        if alignments is None:
            return None

        # MAS durations: sum attention over spectrogram frames
        # alignments is [B, T_text, T_spec], sum over T_spec to get durations
        mas_durations = alignments.sum(dim=2)  # [B, T_text]

        # Get text mask
        x_mask = outputs.get("x_mask")
        if x_mask is not None:
            mask = x_mask.squeeze(1)  # [B, T_text]
        else:
            # Create mask from gt_durations shape
            mask = torch.ones_like(mas_durations)

        # Move gt to same device
        gt = gt_durations.float().to(mas_durations.device)

        # Handle length mismatches (truncate to shorter)
        min_len = min(mas_durations.size(1), gt.size(1), mask.size(1))
        mas_durations = mas_durations[:, :min_len]
        gt = gt[:, :min_len]
        mask = mask[:, :min_len]

        # MSE loss on log durations (more stable for varying magnitudes)
        mas_log = torch.log(mas_durations + 1e-6)
        gt_log = torch.log(gt + 1e-6)

        loss = (mas_log - gt_log) ** 2
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss * self.duration_supervision_alpha
