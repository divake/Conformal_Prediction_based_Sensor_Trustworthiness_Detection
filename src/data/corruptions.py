#src/data/corruptions.py

import torch
import numpy as np
from typing import Optional, Type
from torch.utils.data import Dataset
from .dataset import ModelNet40Dataset

class OcclusionCorruption:
    """Occlusion corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
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

class RainCorruption:
    """Rain corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
        # Parameters for different severity levels
        self.rain_params = {
            1: {'density': 0.02, 'noise_std': 0.01},  # Light rain
            2: {'density': 0.04, 'noise_std': 0.02},  # Moderate rain
            3: {'density': 0.06, 'noise_std': 0.03},  # Heavy rain
            4: {'density': 0.08, 'noise_std': 0.04},  # Very heavy rain
            5: {'density': 0.10, 'noise_std': 0.05}   # Extreme rain
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply rain corruption to point cloud.
        Args:
            points (np.ndarray): Point cloud of shape (N, C)
            severity (int): Severity level from 1 to 5
        Returns:
            np.ndarray: Corrupted point cloud
        """
        assert 1 <= severity <= self.severity_levels
        
        params = self.rain_params[severity]
        num_points = len(points)
        
        # Add random noise to simulate rain droplets
        noise = self.rng.normal(0, params['noise_std'], size=points[:, :3].shape)
        
        # Add random rain points
        num_rain = int(num_points * params['density'])
        
        # Generate rain points within the point cloud bounds
        mins = points[:, :3].min(axis=0)
        maxs = points[:, :3].max(axis=0)
        rain_points = self.rng.uniform(mins, maxs, size=(num_rain, 3))
        
        # Add small vertical streaks to simulate falling rain
        rain_points[:, 2] += self.rng.uniform(-0.1, 0, size=num_rain)  # Falling effect
        
        # Combine original points (with noise) and rain points
        noisy_points = points.copy()
        noisy_points[:, :3] += noise
        
        # Create rain point features (intensity and other features set to mean of original)
        rain_features = np.zeros((num_rain, points.shape[1]))
        rain_features[:, :3] = rain_points
        rain_features[:, 3:] = np.mean(points[:, 3:], axis=0)
        
        # Combine original and rain points
        corrupted_points = np.vstack([noisy_points, rain_features])
        
        return corrupted_points

class CorruptedModelNet40Dataset(Dataset):
    """Dataset wrapper that applies corruptions to ModelNet40 point clouds"""
    def __init__(
        self,
        base_dataset: ModelNet40Dataset,
        corruption_type: Type[OcclusionCorruption],
        severity: int,
        seed: Optional[int] = None
    ):
        """
        Args:
            base_dataset: Original ModelNet40 dataset
            corruption_type: Type of corruption to apply (e.g., OcclusionCorruption)
            severity: Severity level of corruption (1-5)
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.corruption = corruption_type(seed=seed)
        self.severity = severity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple:
        # Get original point cloud and label
        points, label = self.base_dataset[idx]
        
        # Convert to numpy for corruption
        points_np = points.numpy()
        
        # Apply corruption
        corrupted_points = self.corruption(points_np, self.severity)
        
        # Convert back to tensor
        corrupted_points = torch.from_numpy(corrupted_points).float()
        
        return corrupted_points, label