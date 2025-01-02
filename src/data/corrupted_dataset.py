#src/data/corrupted_dataset.py

import torch
import numpy as np
from typing import Optional, Dict, Type
from torch.utils.data import Dataset
from .dataset import ModelNet40Dataset
from .corruptions import PointCloudCorruption

class CorruptedModelNet40Dataset(Dataset):
    def __init__(
        self,
        base_dataset: ModelNet40Dataset,
        corruption_type: Type[PointCloudCorruption],
        severity: int,
        seed: Optional[int] = None
    ):
        self.base_dataset = base_dataset
        self.corruption = corruption_type(seed=seed)
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        points, label = self.base_dataset[idx]
        
        # Convert to numpy for corruption
        points_np = points.numpy()
        
        # Apply corruption
        corrupted_points = self.corruption(points_np, self.severity)
        
        # Convert back to tensor
        corrupted_points = torch.from_numpy(corrupted_points).float()
        
        return corrupted_points, label