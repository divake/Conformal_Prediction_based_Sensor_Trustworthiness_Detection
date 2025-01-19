#src_vision/data/corruptions.py

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2

class FogCorruption:
    def __init__(self, severity=1):
        assert 1 <= severity <= 5, "Severity must be between 1 and 5"
        self.severity = severity
        
        # Base parameters from severity 3
        base_max_val = 0.5
        base_decay = 0.5
        
        # Linear noise scale based on severity
        noise_scale = 0.1 + (severity - 1) * 0.1  # 0.1 to 0.5
        
        # For severities 1-2, gradually approach base parameters
        if severity <= 2:
            self.fog_params = {
                1: {'max_val': 0.1, 'decay': 0.8, 'noise_scale': noise_scale},
                2: {'max_val': 0.2, 'decay': 0.7, 'noise_scale': noise_scale}
            }[severity]
        else:
            # For severities 3-5, keep base params but increase noise
            self.fog_params = {
                'max_val': base_max_val,
                'decay': base_decay,
                'noise_scale': noise_scale
            }

    def _create_fog_mask(self, shape):
        """Create a fog intensity mask based on severity."""
        params = self.fog_params
        
        # Create linear fade for fog effect
        x = np.arange(0, shape[0]) / shape[0]
        fade = np.exp(-x / params['decay'])
        fade = np.tile(fade[:, np.newaxis], (1, shape[1]))
        
        # Add noise with severity-dependent scale
        noise = np.random.rand(*shape) * params['noise_scale']
        mask = fade * params['max_val'] + noise
        mask = np.clip(mask, 0, 1)
        
        return mask.astype(np.float32)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
        _, H, W = img.shape
        fog_mask = torch.from_numpy(self._create_fog_mask((H, W)))
        fog_mask = fog_mask.unsqueeze(0).repeat(3, 1, 1)
        corrupted = img * (1 - fog_mask) + fog_mask
        corrupted = torch.clamp(corrupted, 0, 1)
        
        return corrupted

class SnowCorruption:
    """Simulate snow effects on images with varying severity levels."""
    
    def __init__(self, severity=1):
        """Initialize snow corruption with given severity level (1-5)."""
        assert 1 <= severity <= 5, "Severity must be between 1 and 5"
        self.severity = severity
        
        # Snow effect parameters for different severity levels
        self.snow_params = {
            1: {'snow_scale': 0.1, 'brightness': 0.2, 'blur': 1.0},
            2: {'snow_scale': 0.2, 'brightness': 0.3, 'blur': 1.5},
            3: {'snow_scale': 0.3, 'brightness': 0.4, 'blur': 2.0},
            4: {'snow_scale': 0.4, 'brightness': 0.45, 'blur': 2.5},
            5: {'snow_scale': 0.5, 'brightness': 0.5, 'blur': 3.0}
        }

    def __call__(self, img):
        """
        Apply snow effect to image.
        Args:
            img: PIL Image or torch.Tensor [C,H,W] with values in [0,1]
        Returns:
            torch.Tensor with snow effect
        """
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
            
        # Convert tensor to numpy for processing
        if isinstance(img, torch.Tensor):
            img = img.numpy()
            
        # Ensure image is in [0,1] range
        if img.max() > 1.0:
            img = img / 255.0
            
        # Get image dimensions
        if len(img.shape) == 3:
            C, H, W = img.shape
            img = np.transpose(img, (1, 2, 0))  # Change to HWC format
        else:
            H, W = img.shape
            
        params = self.snow_params[self.severity]
        
        # Create snow particles
        snow_layer = np.random.normal(
            size=(H, W),
            loc=0.05,
            scale=0.3
        )
        
        # Scale snow effect based on severity
        snow_layer = np.clip(snow_layer * params['snow_scale'], 0, 1)
        snow_layer = np.expand_dims(snow_layer, axis=2)
        snow_layer = np.tile(snow_layer, [1, 1, 3])
        
        # Add blur to snow with severity-dependent sigma
        snow_layer = cv2.GaussianBlur(
            snow_layer.astype(np.float32), 
            (3, 3), 
            params['blur']
        )
        
        # Increase brightness based on severity
        brightness_increase = params['brightness']
        img = np.clip(img + brightness_increase, 0, 1)
        
        # Combine the snow layer with the image
        combined = np.clip(img + snow_layer, 0, 1)
        
        # Convert back to tensor in CHW format
        combined = np.transpose(combined, (2, 0, 1))
        return torch.from_numpy(combined).float()

class RainCorruption:
    """Simulate rain effects on images with varying severity levels."""
    
    def __init__(self, severity=1):
        """Initialize rain corruption with given severity level (1-5)."""
        assert 1 <= severity <= 5, "Severity must be between 1 and 5"
        self.severity = severity
        
        # Rain effect parameters for different severity levels
        self.rain_params = {
            1: {'density': 0.01, 'length': 0.2, 'width': 1.0, 'brightness': 0.1},
            2: {'density': 0.02, 'length': 0.3, 'width': 1.0, 'brightness': 0.15},
            3: {'density': 0.03, 'length': 0.4, 'width': 1.0, 'brightness': 0.2},
            4: {'density': 0.04, 'length': 0.5, 'width': 1.0, 'brightness': 0.25},
            5: {'density': 0.05, 'length': 0.6, 'width': 1.0, 'brightness': 0.3}
        }

    def _generate_rain_layer(self, shape):
        """Generate rain streaks layer."""
        params = self.rain_params[self.severity]
        H, W = shape
        
        # Create empty rain layer
        rain_layer = np.zeros((H, W), dtype=np.float32)
        
        # Number of rain drops based on density
        n_drops = int(H * W * params['density'])
        
        # Generate random rain streaks
        for _ in range(n_drops):
            # Random starting position
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            
            # Rain streak length and angle
            length = int(H * params['length'])
            angle = np.random.uniform(0.7, 1.0)  # Mostly vertical
            
            # Calculate end point
            x2 = int(x + length * np.sin(angle))
            y2 = int(y + length * np.cos(angle))
            
            # Draw rain streak
            cv2.line(
                rain_layer,
                (x, y),
                (x2, y2),
                1.0,
                int(params['width']),
                cv2.LINE_AA
            )
        
        # Apply Gaussian blur for more natural look
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        return rain_layer * params['brightness']

    def __call__(self, img):
        """
        Apply rain effect to image.
        Args:
            img: PIL Image or torch.Tensor [C,H,W] with values in [0,1]
        Returns:
            torch.Tensor with rain effect
        """
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
            
        # Convert tensor to numpy for processing
        if isinstance(img, torch.Tensor):
            img = img.numpy()
            
        # Ensure image is in [0,1] range
        if img.max() > 1.0:
            img = img / 255.0
            
        # Get image dimensions and transpose if needed
        if len(img.shape) == 3:
            C, H, W = img.shape
            img = np.transpose(img, (1, 2, 0))  # Change to HWC format
        else:
            H, W = img.shape
            
        # Generate rain layer
        rain_layer = self._generate_rain_layer((H, W))
        rain_layer = np.expand_dims(rain_layer, axis=2)
        rain_layer = np.tile(rain_layer, [1, 1, 3])
        
        # Add slight brightness increase to simulate wet surfaces
        brightness = self.rain_params[self.severity]['brightness']
        img = np.clip(img * (1 + brightness * 0.1), 0, 1)
        
        # Combine rain with image
        combined = np.clip(img + rain_layer, 0, 1)
        
        # Convert back to tensor in CHW format
        combined = np.transpose(combined, (2, 0, 1))
        return torch.from_numpy(combined).float()

class MotionBlurCorruption:
    def __init__(self, severity=1):
        self.severity = severity
        
        # Adjusted parameters for better visual effect
        self.motion_params = {
            1: {'kernel_size': 7, 'angle': 15},
            2: {'kernel_size': 11, 'angle': 15},
            3: {'kernel_size': 15, 'angle': 15},
            4: {'kernel_size': 19, 'angle': 15},
            5: {'kernel_size': 23, 'angle': 15}
        }

    def _create_motion_kernel(self, kernel_size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel with proper normalization."""
        # Create a horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Create line with proper intensity
        kernel[center, :] = np.ones(kernel_size)
        # Normalize kernel to preserve brightness
        kernel = kernel / kernel.sum()
        
        # Rotate the kernel
        angle_rad = np.radians(angle)
        rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Renormalize after rotation to ensure brightness preservation
        kernel = kernel / kernel.sum()
        
        return kernel

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
            
        # Convert tensor to numpy for processing
        if isinstance(img, torch.Tensor):
            img = img.numpy()
            
        # Ensure image is in [0,1] range
        if img.max() > 1.0:
            img = img / 255.0
            
        # Get image dimensions and transpose if needed
        if len(img.shape) == 3:
            C, H, W = img.shape
            img = np.transpose(img, (1, 2, 0))  # Change to HWC format
        else:
            H, W = img.shape
            
        params = self.motion_params[self.severity]
        
        # Create motion blur kernel
        kernel = self._create_motion_kernel(
            params['kernel_size'],
            params['angle']
        )
        
        # Apply motion blur to each channel
        blurred = np.zeros_like(img)
        for i in range(3):
            blurred[..., i] = cv2.filter2D(img[..., i], -1, kernel)
        
        # Ensure output is in valid range
        blurred = np.clip(blurred, 0, 1)
        
        # Convert back to tensor in CHW format
        blurred = np.transpose(blurred, (2, 0, 1))
        return torch.from_numpy(blurred).float()

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