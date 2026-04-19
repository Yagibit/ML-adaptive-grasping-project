from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset not found: {npz_path}")
        blob = np.load(npz_path)
        self.states = blob["states"].astype(np.float32)
        self.actions = blob["actions"].astype(np.float32)
        self.next_states = blob["next_states"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.from_numpy(self.states[idx])
        a = torch.from_numpy(self.actions[idx])
        s_next = torch.from_numpy(self.next_states[idx])
        return s, a, s_next
