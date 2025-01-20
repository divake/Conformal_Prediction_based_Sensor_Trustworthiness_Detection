#src/data/corruptions.py

import torch
import numpy as np
from typing import Optional, Type
from torch.utils.data import Dataset
from .dataset import ModelNet40Dataset
from sklearn.neighbors import KDTree

class OcclusionCorruption:
    """Occlusion corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
        # Percentage of points to remove for each severity level
        self.removal_ratios = {
            1: 0.15,  # 15% points removed
            2: 0.25,  # 25% points removed
            3: 0.35,  # 35% points removed
            4: 0.45,  # 45% points removed
            5: 0.55   # 55% points removed
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply occlusion corruption to point cloud with optimized local region removal.
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
        
        # Select multiple center points for more natural occlusion
        num_centers = severity  # More occlusion centers for higher severity
        center_indices = self.rng.choice(num_points, num_centers, replace=False)
        centers = points[center_indices, :3]
        
        # Compute distances efficiently using broadcasting
        points_expanded = points[:, None, :3]  # Shape: (N, 1, 3)
        centers_expanded = centers[None, :, :]  # Shape: (1, C, 3)
        distances = np.min(np.linalg.norm(points_expanded - centers_expanded, axis=2), axis=1)
        
        # Remove closest points to centers
        keep_indices = distances.argsort()[num_remove:]
        corrupted_points = points[keep_indices]
        
        return corrupted_points

class RainCorruption:
    """Rain corruption for point clouds with optimized implementation"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
        # Parameters for different severity levels
        self.rain_params = {
            1: {'density': 0.05, 'noise_std': 0.02, 'streak_length': 0.1},  # Light rain
            2: {'density': 0.10, 'noise_std': 0.04, 'streak_length': 0.2},  # Moderate rain
            3: {'density': 0.20, 'noise_std': 0.06, 'streak_length': 0.3},  # Heavy rain
            4: {'density': 0.35, 'noise_std': 0.08, 'streak_length': 0.4},  # Very heavy rain
            5: {'density': 0.50, 'noise_std': 0.10, 'streak_length': 0.5}   # Extreme rain
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply rain corruption with optimized implementation.
        Args:
            points (np.ndarray): Point cloud of shape (N, C)
            severity (int): Severity level from 1 to 5
        Returns:
            np.ndarray: Corrupted point cloud
        """
        assert 1 <= severity <= self.severity_levels
        
        params = self.rain_params[severity]
        num_points = len(points)
        
        # Add noise to existing points (simulating rain interference)
        noise = self.rng.normal(0, params['noise_std'], size=points[:, :3].shape)
        noisy_points = points.copy()
        noisy_points[:, :3] += noise
        
        # Generate rain points more efficiently
        num_rain = int(num_points * params['density'])
        
        # Get point cloud bounds
        mins = points[:, :3].min(axis=0)
        maxs = points[:, :3].max(axis=0)
        
        # Generate rain streaks more efficiently (3 points per streak instead of 5)
        rain_starts = self.rng.uniform(mins, maxs, size=(num_rain, 3))
        rain_ends = rain_starts.copy()
        rain_ends[:, 2] -= params['streak_length'] * (maxs[2] - mins[2])
        
        # Generate points along streaks more efficiently
        t = np.linspace(0, 1, 3)[None, :, None]  # 3 points per streak, evenly spaced
        rain_points = rain_starts[:, None] * t + rain_ends[:, None] * (1 - t)
        rain_points = rain_points.reshape(-1, 3)
        
        # Add intensity variation more efficiently
        rain_intensity = self.rng.uniform(0.3, 1.0, size=(len(rain_points), 1))
        mean_features = np.mean(points[:, 3:], axis=0)
        
        # Create rain features more efficiently
        rain_features = np.zeros((len(rain_points), points.shape[1]))
        rain_features[:, :3] = rain_points
        rain_features[:, 3:] = mean_features * rain_intensity
        
        # Combine points
        corrupted_points = np.vstack([noisy_points, rain_features])
        
        return corrupted_points

class FogCorruption:
    """Fog corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
        # Parameters for different severity levels
        self.fog_params = {
            1: {'density': 0.05, 'noise_std': 0.02, 'attenuation': 0.1},  # Light fog
            2: {'density': 0.10, 'noise_std': 0.04, 'attenuation': 0.2},  # Moderate fog
            3: {'density': 0.20, 'noise_std': 0.06, 'attenuation': 0.3},  # Heavy fog
            4: {'density': 0.35, 'noise_std': 0.08, 'attenuation': 0.4},  # Very heavy fog
            5: {'density': 0.50, 'noise_std': 0.10, 'attenuation': 0.5}   # Extreme fog
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply fog corruption with distance-based attenuation.
        Args:
            points (np.ndarray): Point cloud of shape (N, C)
            severity (int): Severity level from 1 to 5
        Returns:
            np.ndarray: Corrupted point cloud
        """
        assert 1 <= severity <= self.severity_levels
        
        params = self.fog_params[severity]
        num_points = len(points)
        
        # Calculate distance from origin for each point
        distances = np.linalg.norm(points[:, :3], axis=1)
        max_distance = np.max(distances)
        
        # Add distance-based noise (more noise for distant points)
        distance_factor = distances / max_distance
        noise_scale = params['noise_std'] * (1 + distance_factor)[:, None]
        noise = self.rng.normal(0, noise_scale, size=points[:, :3].shape)
        
        # Add base noise to points
        noisy_points = points.copy()
        noisy_points[:, :3] += noise
        
        # Generate fog particles
        num_fog = int(num_points * params['density'])
        
        # Get point cloud bounds
        mins = points[:, :3].min(axis=0)
        maxs = points[:, :3].max(axis=0)
        
        # Generate fog points with distance-based density
        fog_points = self.rng.uniform(mins, maxs, size=(num_fog, 3))
        fog_distances = np.linalg.norm(fog_points, axis=1)
        
        # Apply distance-based attenuation to fog points
        attenuation = np.exp(-params['attenuation'] * fog_distances / max_distance)
        
        # Create fog features
        fog_features = np.zeros((num_fog, points.shape[1]))
        fog_features[:, :3] = fog_points
        fog_features[:, 3:] = np.mean(points[:, 3:], axis=0) * attenuation[:, None]
        
        # Combine original and fog points
        corrupted_points = np.vstack([noisy_points, fog_features])
        
        return corrupted_points

class BlurCorruption:
    """Blur corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
        # Parameters for different severity levels
        self.blur_params = {
            1: {'radius': 0.02, 'sigma': 0.01},  # Light blur
            2: {'radius': 0.04, 'sigma': 0.02},  # Moderate blur
            3: {'radius': 0.06, 'sigma': 0.03},  # Medium blur
            4: {'radius': 0.08, 'sigma': 0.04},  # Heavy blur
            5: {'radius': 0.10, 'sigma': 0.05}   # Extreme blur
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply gaussian blur corruption to point cloud.
        Args:
            points (np.ndarray): Point cloud of shape (N, C)
            severity (int): Severity level from 1 to 5
        Returns:
            np.ndarray: Corrupted point cloud
        """
        assert 1 <= severity <= self.severity_levels
        
        params = self.blur_params[severity]
        num_points = len(points)
        
        # Build KD-tree for efficient neighbor search
        tree = KDTree(points[:, :3])
        
        # Find neighbors within radius for each point
        neighbors = tree.query_radius(points[:, :3], r=params['radius'])
        
        # Apply gaussian weights and compute new positions
        blurred_points = points.copy()
        for i, neighbor_indices in enumerate(neighbors):
            if len(neighbor_indices) > 1:  # Only blur if there are neighbors
                distances = np.linalg.norm(
                    points[neighbor_indices, :3] - points[i, :3], 
                    axis=1
                )
                weights = np.exp(-(distances**2) / (2 * params['sigma']**2))
                weights /= weights.sum()
                
                # Apply weighted average to all features
                blurred_points[i] = np.average(
                    points[neighbor_indices], 
                    weights=weights,
                    axis=0
                )
        
        return blurred_points

class SnowCorruption:
    """Snow corruption for point clouds"""
    def __init__(self, severity_levels: int = 5, seed: Optional[int] = None):
        self.severity_levels = severity_levels
        self.rng = np.random.RandomState(seed)
        # Parameters for different severity levels
        self.snow_params = {
            1: {'density': 0.05, 'size': 0.02, 'noise_std': 0.01},  # Light snow
            2: {'density': 0.10, 'size': 0.03, 'noise_std': 0.02},  # Moderate snow
            3: {'density': 0.20, 'size': 0.04, 'noise_std': 0.03},  # Heavy snow
            4: {'density': 0.35, 'size': 0.05, 'noise_std': 0.04},  # Very heavy snow
            5: {'density': 0.50, 'size': 0.06, 'noise_std': 0.05}   # Extreme snow
        }

    def __call__(self, points: np.ndarray, severity: int) -> np.ndarray:
        """
        Apply snow corruption with accumulation effects.
        Args:
            points (np.ndarray): Point cloud of shape (N, C)
            severity (int): Severity level from 1 to 5
        Returns:
            np.ndarray: Corrupted point cloud
        """
        assert 1 <= severity <= self.severity_levels
        
        params = self.snow_params[severity]
        num_points = len(points)
        
        # Add noise to existing points (simulating snow interference)
        noise = self.rng.normal(0, params['noise_std'], size=points[:, :3].shape)
        noisy_points = points.copy()
        noisy_points[:, :3] += noise
        
        # Get point cloud bounds
        mins = points[:, :3].min(axis=0)
        maxs = points[:, :3].max(axis=0)
        
        # Generate snow particles
        num_snow = int(num_points * params['density'])
        snow_points = np.zeros((num_snow, 3))
        
        # Generate snow with accumulation on surfaces
        snow_points[:, 0] = self.rng.uniform(mins[0], maxs[0], num_snow)
        snow_points[:, 1] = self.rng.uniform(mins[1], maxs[1], num_snow)
        
        # Initialize z-coordinates with mean z value
        snow_points[:, 2] = (mins[2] + maxs[2]) / 2
        
        # Find nearest points in original cloud for z-coordinate (accumulation)
        tree = KDTree(points[:, :3])
        distances, indices = tree.query(snow_points, k=1)
        indices = indices.flatten()  # Ensure indices is 1D
        
        # Add snow on top of nearest surface points
        snow_points[:, 2] = points[indices, 2] + self.rng.uniform(0, params['size'], num_snow)
        
        # Create snow features
        snow_features = np.zeros((num_snow, points.shape[1]))
        snow_features[:, :3] = snow_points
        
        # Make snow particles bright white
        mean_features = np.mean(points[:, 3:], axis=0)
        brightness_factor = 1.2  # Make snow brighter than average
        snow_features[:, 3:] = mean_features * brightness_factor
        
        # Combine points
        corrupted_points = np.vstack([noisy_points, snow_features])
        
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