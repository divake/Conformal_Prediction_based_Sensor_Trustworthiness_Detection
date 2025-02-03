import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import random
from typing import Tuple, Optional, List
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import gaussian_filter

class ImageCorruption:
    """Base class for image corruptions."""
    def __init__(self, severity: int = 1):
        self.severity = severity
        
    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

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

class SnowCorruption:
    """Optimized snow corruption with 5 severity levels."""
    def __init__(self, severity: int = 1):
        self.severity = severity
        # Snow parameters for different severity levels - reduced intensity
        self.params = {
            1: {'flake_density': 0.005, 'brightness': 1.05, 'flake_size': 1},   # Very light snow
            2: {'flake_density': 0.01, 'brightness': 1.1, 'flake_size': 2},    # Light snow
            3: {'flake_density': 0.015, 'brightness': 1.15, 'flake_size': 2},  # Moderate snow
            4: {'flake_density': 0.02, 'brightness': 1.2, 'flake_size': 3},    # Heavy snow
            5: {'flake_density': 0.03, 'brightness': 1.25, 'flake_size': 3}    # Very heavy snow
        }
        # Cache for snow patterns
        self.cached_patterns = {}
        
    def _create_snow_pattern(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a snow pattern for the given image shape."""
        cache_key = (shape[0], shape[1], self.severity)
        if cache_key in self.cached_patterns:
            return self.cached_patterns[cache_key].copy()
        
        params = self.params[self.severity]
        snow_mask = np.zeros(shape[:2], dtype=np.float32)
        
        # Generate snow flake positions
        num_flakes = int(shape[0] * shape[1] * params['flake_density'])
        y_positions = np.random.randint(0, shape[0], size=num_flakes)
        x_positions = np.random.randint(0, shape[1], size=num_flakes)
        
        # Create snow flakes with varying sizes
        for i in range(num_flakes):
            flake_size = params['flake_size']
            y, x = y_positions[i], x_positions[i]
            
            # Draw each snowflake as a small bright region
            y_start = max(0, y - flake_size)
            y_end = min(shape[0], y + flake_size + 1)
            x_start = max(0, x - flake_size)
            x_end = min(shape[1], x + flake_size + 1)
            
            # Create gaussian-like falloff for realistic snow flakes
            for dy in range(y_start, y_end):
                for dx in range(x_start, x_end):
                    distance = np.sqrt((dy - y)**2 + (dx - x)**2)
                    if distance <= flake_size:
                        intensity = np.exp(-distance / flake_size) * 0.8  # Reduced intensity by 20%
                        snow_mask[dy, dx] = max(snow_mask[dy, dx], intensity)
        
        # Add slight noise to simulate snow texture (reduced noise)
        noise = np.random.normal(0, 0.05, snow_mask.shape)  # Reduced noise variance
        snow_mask = np.clip(snow_mask + noise * 0.05, 0, 1)  # Reduced noise impact
        
        # Cache the pattern if cache isn't too full
        if len(self.cached_patterns) < 10:
            self.cached_patterns[cache_key] = snow_mask.copy()
        
        return snow_mask
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply snow corruption to a batch of images."""
        if isinstance(images, Image.Image):
            # Handle single PIL image case
            image_np = np.array(images).astype(np.float32) / 255.0
            snow_mask = self._create_snow_pattern(image_np.shape)
            snow_mask = snow_mask[..., np.newaxis] if len(snow_mask.shape) == 2 else snow_mask
            
            # Apply brightness increase and snow effect with reduced intensity
            params = self.params[self.severity]
            brightened = image_np * params['brightness']
            snowy = brightened * (1 - snow_mask * 0.3) + snow_mask * 0.7  # Reduced snow impact
            
            return Image.fromarray((np.clip(snowy, 0, 1) * 255).astype(np.uint8))
        
        # Handle batch of tensors
        device = images.device
        batch_size, channels, height, width = images.shape
        
        # Create snow pattern once and repeat for batch
        snow_pattern = self._create_snow_pattern((height, width))
        snow_pattern = torch.from_numpy(snow_pattern).to(device)
        
        # Expand snow pattern for the batch
        snow_pattern = snow_pattern.view(1, 1, height, width).expand(batch_size, channels, -1, -1)
        
        # Apply snow effect with reduced intensity
        params = self.params[self.severity]
        brightened = images * params['brightness']
        snowy = brightened * (1 - snow_pattern * 0.3) + snow_pattern * 0.7  # Reduced snow impact
        
        return torch.clamp(snowy, 0, 1)

class MotionBlurCorruption:
    """Optimized motion blur corruption with 5 severity levels."""
    def __init__(self, severity: int = 1):
        self.severity = severity
        # Motion blur parameters for different severity levels - significantly reduced intensity
        self.params = {
            1: {'kernel_size': 3, 'angle_range': 5, 'intensity': 0.1},    # Barely noticeable motion
            2: {'kernel_size': 3, 'angle_range': 10, 'intensity': 0.15},  # Very mild motion
            3: {'kernel_size': 5, 'angle_range': 15, 'intensity': 0.2},   # Mild motion
            4: {'kernel_size': 5, 'angle_range': 20, 'intensity': 0.25},  # Moderate motion
            5: {'kernel_size': 7, 'angle_range': 25, 'intensity': 0.3}    # Strong motion
        }
        # Cache for motion kernels
        self.kernel_cache = {}
        
    def _create_motion_kernel(self, kernel_size: int, angle: float, intensity: float) -> np.ndarray:
        """Create a motion blur kernel with given parameters."""
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # Create motion blur effect with gaussian falloff
        sigma = kernel_size / 6.0  # Gaussian spread parameter
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Calculate distance from center
                y = i - center
                x = j - center
                
                # Rotate coordinates
                rot_x = x * cos_val + y * sin_val
                rot_y = -x * sin_val + y * cos_val
                
                # Apply gaussian-weighted motion blur
                if abs(rot_x) <= kernel_size // 2:
                    # Use gaussian falloff for smoother blur
                    dist = np.sqrt(rot_x**2 + rot_y**2)
                    weight = np.exp(-0.5 * (dist/sigma)**2)
                    kernel[i, j] = weight * intensity
        
        # Ensure kernel sums to 1
        kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
        return kernel
    
    def _apply_motion_blur(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply motion blur using efficient tensor operations with reduced effect."""
        # Pad image for valid convolution
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        # Prepare kernel for convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(image.shape[0], 1, 1, 1)
        
        # Apply convolution with blend
        blurred = F.conv2d(padded.unsqueeze(0), kernel, padding=0, groups=image.shape[0])
        blurred = blurred.squeeze(0)
        
        # Blend original and blurred image based on severity
        blend_factor = self.severity * 0.15  # Reduce blending effect
        return (1 - blend_factor) * image + blend_factor * blurred
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply motion blur corruption to images."""
        if isinstance(images, Image.Image):
            # Handle single PIL image case
            image_np = np.array(images).astype(np.float32) / 255.0
            
            # Get blur parameters
            params = self.params[self.severity]
            angle = np.random.uniform(-params['angle_range'], params['angle_range'])
            
            # Create and apply kernel
            kernel = self._create_motion_kernel(
                params['kernel_size'], 
                angle, 
                params['intensity']
            )
            kernel_tensor = torch.from_numpy(kernel)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
            
            # Apply blur with reduced effect
            blurred = self._apply_motion_blur(image_tensor, kernel_tensor)
            blurred = blurred.permute(1, 2, 0).numpy()
            
            return Image.fromarray((np.clip(blurred, 0, 1) * 255).astype(np.uint8))
        
        # Handle batch of tensors
        device = images.device
        batch_size = images.shape[0]
        
        # Get blur parameters
        params = self.params[self.severity]
        angles = np.random.uniform(
            -params['angle_range'], 
            params['angle_range'], 
            size=batch_size
        )
        
        # Process each image with a different random angle
        blurred_images = []
        for idx in range(batch_size):
            # Create kernel for this image
            kernel = self._create_motion_kernel(
                params['kernel_size'],
                angles[idx],
                params['intensity']
            )
            kernel_tensor = torch.from_numpy(kernel).to(device)
            
            # Apply motion blur with reduced effect
            blurred = self._apply_motion_blur(images[idx], kernel_tensor)
            blurred_images.append(blurred)
        
        # Combine results
        return torch.stack(blurred_images)

class FogCorruption:
    """Optimized fog corruption with 5 severity levels."""
    def __init__(self, severity: int = 1):
        self.severity = severity
        # Fog parameters for different severity levels
        self.params = {
            1: {'max_intensity': 0.1, 'contrast_reduction': 0.05},  # Very light fog
            2: {'max_intensity': 0.15, 'contrast_reduction': 0.1},  # Light fog
            3: {'max_intensity': 0.2, 'contrast_reduction': 0.15},  # Moderate fog
            4: {'max_intensity': 0.25, 'contrast_reduction': 0.2},  # Heavy fog
            5: {'max_intensity': 0.3, 'contrast_reduction': 0.25}   # Very heavy fog
        }
        # Cache for fog patterns
        self.cached_patterns = {}
        
    def _create_fog_pattern(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a fog pattern for the given image shape."""
        cache_key = (shape[0], shape[1], self.severity)
        if cache_key in self.cached_patterns:
            return self.cached_patterns[cache_key].copy()
        
        params = self.params[self.severity]
        
        # Create base fog using perlin-like noise
        x = np.linspace(0, 4, shape[1])
        y = np.linspace(0, 4, shape[0])
        x_idx, y_idx = np.meshgrid(x, y)
        
        # Generate multiple octaves of noise
        fog_pattern = np.zeros(shape[:2])
        amplitude = 1.0
        frequency = 1.0
        persistence = 0.5
        octaves = 4
        
        for _ in range(octaves):
            phase_x = np.random.uniform(0, 4)
            phase_y = np.random.uniform(0, 4)
            noise = np.sin(2 * np.pi * frequency * (x_idx + phase_x)) * \
                   np.sin(2 * np.pi * frequency * (y_idx + phase_y))
            fog_pattern += amplitude * noise
            amplitude *= persistence
            frequency *= 2
        
        # Normalize and apply gaussian blur for smoothness
        fog_pattern = (fog_pattern - fog_pattern.min()) / (fog_pattern.max() - fog_pattern.min())
        fog_pattern = gaussian_filter(fog_pattern, sigma=2.0)
        
        # Scale pattern based on severity
        fog_pattern *= params['max_intensity']
        
        # Cache the pattern if cache isn't too full
        if len(self.cached_patterns) < 10:
            self.cached_patterns[cache_key] = fog_pattern.copy()
        
        return fog_pattern
    
    def _apply_fog(self, image: torch.Tensor, fog_pattern: torch.Tensor) -> torch.Tensor:
        """Apply fog effect using efficient tensor operations."""
        params = self.params[self.severity]
        device = image.device
        
        # Convert fog pattern to tensor and expand to match image dimensions
        if not isinstance(fog_pattern, torch.Tensor):
            fog_pattern = torch.from_numpy(fog_pattern).to(device)
        
        fog_pattern = fog_pattern.unsqueeze(0) if fog_pattern.dim() == 2 else fog_pattern
        fog_pattern = fog_pattern.expand_as(image)
        
        # Apply contrast reduction
        mean = image.mean(dim=(1, 2), keepdim=True)
        contrast_reduced = (1 - params['contrast_reduction']) * (image - mean) + mean
        
        # Add fog effect
        fogged = contrast_reduced * (1 - fog_pattern) + fog_pattern
        
        return torch.clamp(fogged, 0, 1)
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply fog corruption to images."""
        if isinstance(images, Image.Image):
            # Handle single PIL image case
            image_np = np.array(images).astype(np.float32) / 255.0
            fog_pattern = self._create_fog_pattern(image_np.shape)
            
            # Convert to tensor for processing
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
            fog_tensor = torch.from_numpy(fog_pattern)
            
            # Apply fog effect
            fogged = self._apply_fog(image_tensor, fog_tensor)
            fogged = fogged.permute(1, 2, 0).numpy()
            
            return Image.fromarray((np.clip(fogged, 0, 1) * 255).astype(np.uint8))
        
        # Handle batch of tensors
        device = images.device
        batch_size, channels, height, width = images.shape
        
        # Create fog pattern
        fog_pattern = self._create_fog_pattern((height, width))
        fog_pattern = torch.from_numpy(fog_pattern).to(device)
        
        # Process batch efficiently
        fogged_images = self._apply_fog(images, fog_pattern)
        
        return fogged_images

class CorruptedImageNetDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies corruption to images."""
    def __init__(self, base_dataset, corruption_type, severity=1, batch_size=None, transform=None):
        self.base_dataset = base_dataset
        self.corruption = corruption_type(severity=severity)
        self.batch_size = batch_size
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert tensor to PIL for corruption
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Apply corruption
        corrupted = self.corruption(image)
        
        # Convert back to tensor
        if not isinstance(corrupted, torch.Tensor):
            corrupted = transforms.ToTensor()(corrupted)
        
        # Apply additional transforms if specified
        if self.transform is not None:
            corrupted = self.transform(corrupted)
        
        return corrupted, label
    
    def __len__(self):
        return len(self.base_dataset)