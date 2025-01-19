#src/data/corruptions.py

import torch
import numpy as np
from typing import Optional, Dict, Type, Tuple
from torch.utils.data import Dataset
from .dataset import ModelNet40Dataset

class PointCloudCorruption:
    """Base class for point cloud corruptions"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """Apply corruption to point cloud"""
        raise NotImplementedError

class OcclusionCorruption(PointCloudCorruption):
    """Occlusion corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        super().__init__(severity_levels, seed)
        # Percentage of points to remove for each severity level
        self.removal_ratios = {
            1: 0.1,  # 10% points removed
            2: 0.2,  # 20% points removed
            3: 0.3,  # 30% points removed
            4: 0.4,  # 40% points removed
            5: 0.5   # 50% points removed
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply occlusion corruption to point cloud.
        Args:
            points (np.ndarray): Point cloud of shape (N, C)
            severity (int): Severity level from 1 to 5
        Returns:
            np.ndarray: Corrupted point cloud
        """
        assert 1 <= severity <= self.severity_levels
        
        # Get number of points to remove
        num_points = len(points)
        num_remove = int(num_points * self.removal_ratios[severity])
        
        # Select a random center point for occlusion
        center_idx = self.rng.randint(0, num_points)
        center = points[center_idx, :3]  # Use only XYZ coordinates
        
        # Calculate distances from center
        distances = np.linalg.norm(points[:, :3] - center, axis=1)
        
        # Remove closest points to center
        keep_indices = distances.argsort()[num_remove:]
        corrupted_points = points[keep_indices]
        
        return corrupted_points

class CorruptedModelNet40Dataset(Dataset):
    """Dataset wrapper that applies corruptions to ModelNet40 point clouds"""
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