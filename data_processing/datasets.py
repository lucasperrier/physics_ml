import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import decimate

from .generate_data import read_trajectories_parquet_as_dicts, as_torch as sample_to_torch

class TrajectoryWindowDataset(Dataset):
    """
    Provides (states, future states) supervision windows.
    """

    @staticmethod
    def preprocess_series(sample: Dict, *, decimation: int = 1) -> torch.Tensor:
        """
        Apply the same decimation + dtype conversion used during training.

        Returns tensor shaped [time, states].
        """
        if "y" not in sample:
            raise KeyError("Sample missing 'y' key.")
        seq = sample["y"]
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)
        else:
            seq = seq.to(torch.float32)
        if seq.ndim != 2:
            raise ValueError("Expected 'y' tensors shaped [time, states].")

        decimation = int(decimation)
        if decimation > 1:
            arr = decimate(seq.cpu().numpy(), decimation, axis=0, zero_phase=True)
            seq = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        else:
            seq = seq.contiguous()
        return seq


    def __init__(
        self,
        samples: Sequence[Dict],
        input_length: int,
        target_length: int,
        *,
        step: int = 1,
        decimation: int = 1,
    ) -> None:
        if input_length <= 0 or target_length <= 0:
            raise ValueError("Window lengths must be positive.")
        if step <= 0:
            raise ValueError("Step must be positive.")
        
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.step = int(step)
        self.decimation = int(decimation)

        self.sequences: List[torch.Tensor] = []
        self.index: List[Tuple[int, int]] = []

        required = self.input_length + self.target_length

        for sample in samples:
            seq = self.preprocess_series(sample, decimation=self.decimation)
            total = seq.size(0)
            if total < required:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            max_start = total - required
            for start in range(0, max_start + 1, self.step):
                self.index.append((seq_idx, start))

        if not self.index:
            raise ValueError("No usable windows were generated; adjust lengths or data.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        past = seq[start : start + self.input_length]  # [T1, S]
        future = seq[
            start + self.input_length : start + self.input_length + self.target_length
        ]  # [T2, S]
        return past.transpose(0, 1).contiguous(), future.transpose(0, 1).contiguous()