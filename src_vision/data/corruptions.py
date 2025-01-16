"""
#src_vision/data/corruptions.py
Image corruptions for CIFAR-100 dataset with different severity levels.
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

class FogCorruption:
    """
    Applies fog corruption to images with varying severity levels.
    Implementation follows the paper "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"
    """
    def __init__(self, severity=1):
        """Initialize fog corruption with given severity level (1-5)."""
        assert 1 <= severity <= 5, "Severity must be between 1 and 5"
        self.severity = severity
        
        # Fog intensity parameters for different severity levels
        self.fog_params = {
            1: {'max_val': 0.1, 'decay': 0.8},
            2: {'max_val': 0.2, 'decay': 0.7},
            3: {'max_val': 0.5, 'decay': 0.5},
            4: {'max_val': 0.6, 'decay': 0.3},
            5: {'max_val': 0.8, 'decay': 0.1}
        }

    def _create_fog_mask(self, shape):
        """Create a fog intensity mask based on severity."""
        params = self.fog_params[self.severity]
        
        # Create linear fade for fog effect
        x = np.arange(0, shape[0]) / shape[0]
        fade = np.exp(-x / params['decay'])
        fade = np.tile(fade[:, np.newaxis], (1, shape[1]))
        
        # Add some randomness to the fog pattern
        noise = np.random.rand(*shape) * 0.5
        mask = fade * params['max_val'] + noise * params['max_val']
        mask = np.clip(mask, 0, 1)
        
        return mask.astype(np.float32)

    def __call__(self, img):
        """
        Apply fog corruption to an image.
        Args:
            img: PIL Image or torch.Tensor [C,H,W] with values in [0,1]
        Returns:
            Corrupted image in the same format as input
        """
        if isinstance(img, Image.Image):
            # Convert PIL Image to tensor
            img = transforms.ToTensor()(img)
        
        # Ensure input is float tensor in [0,1]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
        # Get image dimensions
        _, H, W = img.shape
        
        # Create fog mask
        fog_mask = torch.from_numpy(self._create_fog_mask((H, W)))
        
        # Apply fog effect
        fog_mask = fog_mask.unsqueeze(0).repeat(3, 1, 1)
        corrupted = img * (1 - fog_mask) + fog_mask
        
        # Ensure output is in valid range
        corrupted = torch.clamp(corrupted, 0, 1)
        
        return corrupted

class CorruptedCIFAR100Dataset(torch.utils.data.Dataset):
    """
    Wrapper for CIFAR100 dataset that applies corruptions on-the-fly.
    """
    def __init__(self, base_dataset, corruption_type, severity=1):
        """
        Args:
            base_dataset: Original CIFAR100 dataset
            corruption_type: Corruption class to apply (e.g., FogCorruption)
            severity: Corruption severity level (1-5)
        """
        self.base_dataset = base_dataset
        self.corruption = corruption_type(severity=severity)
        self.severity = severity
        
        # Keep original transform for normalization
        self.base_transform = None
        if hasattr(base_dataset, 'transform'):
            self.base_transform = base_dataset.transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # If image is already normalized, denormalize before corruption
        if self.base_transform is not None:
            # Assuming the last transform is normalization
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
            
            # Denormalize
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
        
        # Apply corruption
        corrupted_img = self.corruption(img)
        
        # Renormalize if needed
        if self.base_transform is not None:
            for t, m, s in zip(corrupted_img, mean, std):
                t.sub_(m).div_(s)
        
        return corrupted_img, label