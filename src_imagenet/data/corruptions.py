import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import random
from typing import Tuple, Optional, List
import torch.nn.functional as F

class ImageCorruption:
    """Base class for image corruptions."""
    def __init__(self, severity: int = 1):
        self.severity = severity
        
    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

class FogCorruption(ImageCorruption):
    """Apply fog corruption to images."""
    def __call__(self, image: Image.Image) -> Image.Image:
        # Convert PIL to numpy array
        image = np.array(image).astype(np.float32) / 255.
        max_val = image.max()
        
        # Fog intensity based on severity
        fog_intensity = self.severity * 0.15
        
        # Create fog effect
        fog = np.random.uniform(max_val-fog_intensity, max_val, image.shape)
        fog = np.clip(fog, 0, 1)
        
        # Blend original image with fog
        blend_factor = 1 - (self.severity * 0.1)
        foggy = image * blend_factor + fog * (1 - blend_factor)
        
        return Image.fromarray((np.clip(foggy, 0, 1) * 255).astype(np.uint8))

class SnowCorruption(ImageCorruption):
    """Apply snow corruption to images."""
    def __call__(self, image: Image.Image) -> Image.Image:
        # Convert PIL to numpy array
        image = np.array(image).astype(np.float32) / 255.
        
        # Snow parameters based on severity
        snow_density = self.severity * 0.1
        snow_brightness = 1.0 + (self.severity * 0.1)
        
        # Create snow mask
        snow_mask = np.random.random(image.shape[:2]) < snow_density
        snow_mask = snow_mask[:, :, np.newaxis].astype(np.float32)
        
        # Apply snow effect
        snow = np.ones_like(image) * snow_brightness
        snowy = image * (1 - snow_mask) + snow * snow_mask
        
        return Image.fromarray((np.clip(snowy, 0, 1) * 255).astype(np.uint8))

class RainCorruption:
    """Optimized rain corruption for batch processing."""
    def __init__(self, severity: int = 1):
        self.severity = severity
        # Precompute rain pattern parameters
        self.num_drops = int(severity * 1000)
        self.drop_length = int(severity * 2)
        self.drop_width = 1
        self.drop_color = 1.0
        
        # Cache for rain patterns
        self.cached_patterns = {}
        
    def _create_rain_pattern(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a rain pattern for the given image shape."""
        cache_key = (shape[0], shape[1])
        if cache_key in self.cached_patterns:
            return self.cached_patterns[cache_key].copy()
        
        rain_mask = np.zeros(shape[:2], dtype=np.float32)
        
        # Generate all rain drop positions at once
        x_positions = np.random.randint(0, shape[1], size=self.num_drops)
        y_positions = np.random.randint(0, shape[0], size=self.num_drops)
        
        # Vectorized rain streak creation
        for i in range(self.drop_length):
            valid_drops = y_positions + i < shape[0]
            if not np.any(valid_drops):
                break
                
            valid_y = y_positions[valid_drops] + i
            valid_x = x_positions[valid_drops]
            rain_mask[valid_y, valid_x] = self.drop_color
        
        # Cache the pattern
        if len(self.cached_patterns) < 10:  # Limit cache size
            self.cached_patterns[cache_key] = rain_mask.copy()
        
        return rain_mask
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply rain corruption to a batch of images."""
        if isinstance(images, Image.Image):
            # Handle single PIL image case
            image_np = np.array(images).astype(np.float32) / 255.0
            rain_mask = self._create_rain_pattern(image_np.shape)
            rain_mask = rain_mask[..., np.newaxis] if len(rain_mask.shape) == 2 else rain_mask
            rainy = image_np * 0.8 + rain_mask * 0.2
            return Image.fromarray((np.clip(rainy, 0, 1) * 255).astype(np.uint8))
        
        # Handle batch of tensors
        device = images.device
        batch_size, channels, height, width = images.shape
        
        # Create rain pattern once and repeat for batch
        rain_pattern = self._create_rain_pattern((height, width))
        rain_pattern = torch.from_numpy(rain_pattern).to(device)
        
        # Expand rain pattern for the batch
        rain_pattern = rain_pattern.view(1, 1, height, width).expand(batch_size, channels, -1, -1)
        
        # Apply rain effect efficiently using tensor operations
        rainy = images * 0.8 + rain_pattern * 0.2
        return torch.clamp(rainy, 0, 1)

class MotionBlurCorruption(ImageCorruption):
    """Apply motion blur corruption to images."""
    def __call__(self, image: Image.Image) -> Image.Image:
        # Motion blur parameters based on severity
        kernel_size = self.severity * 3
        angle = random.randint(0, 360)
        
        # Apply motion blur using PIL
        blurred = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        
        # Rotate image to simulate motion direction
        blurred = blurred.rotate(angle, resample=Image.BILINEAR)
        blurred = blurred.rotate(-angle, resample=Image.BILINEAR)
        
        return blurred

class CorruptedImageNetDataset(Dataset):
    """Optimized dataset wrapper for corruptions."""
    def __init__(
        self,
        base_dataset: Dataset,
        corruption: RainCorruption,
        severity: int = 1,
        batch_size: int = 32
    ):
        self.base_dataset = base_dataset
        self.corruption = corruption(severity)
        self.batch_size = batch_size
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        
        # Apply corruption
        if isinstance(image, torch.Tensor):
            # Process as tensor directly
            corrupted = self.corruption(image.unsqueeze(0)).squeeze(0)
        else:
            # Handle PIL image case
            corrupted = self.corruption(image)
            if isinstance(corrupted, Image.Image):
                corrupted = transforms_to_tensor(corrupted)
        
        return corrupted, label
    
    def __len__(self) -> int:
        return len(self.base_dataset)

def transforms_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    # Convert to numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0
    # Change to CxHxW format
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to normalized tensor."""
    return transforms_to_tensor(image) 