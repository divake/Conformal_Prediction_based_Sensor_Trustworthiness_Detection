import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from corruptions import (
    FogCorruption,
    SnowCorruption,
    RainCorruption,
    MotionBlurCorruption,
    CorruptedImageNetDataset
)

def plot_corrupted_images(original_image: Image.Image, corrupted_images: dict, save_path: str):
    """Plot original and corrupted images side by side."""
    n_corruptions = len(corrupted_images)
    n_severities = len(next(iter(corrupted_images.values())))
    
    # Create a grid: original + (corruptions × severities)
    fig, axes = plt.subplots(n_corruptions + 1, n_severities, 
                            figsize=(4*n_severities, 4*(n_corruptions + 1)))
    
    # Plot original image in the first row
    for ax in axes[0]:
        ax.imshow(original_image)
        ax.axis('off')
        ax.set_title('Original')
    
    # Plot corrupted images
    for i, (corruption_name, images) in enumerate(corrupted_images.items(), 1):
        for j, (severity, img) in enumerate(images.items()):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'{corruption_name}\nSeverity {severity}')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def test_corruptions():
    """Test all corruption types on sample ImageNet images."""
    # Setup ImageNet data
    val_dir = '/ssd_4TB/divake/CP_trust_IJCNN/dataset/imagenet/val'
    
    # First get PIL images for testing corruptions
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    
    # Create a dataset that returns PIL images
    class PILImageFolder(ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = self.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            return image, target
    
    dataset = PILImageFolder(val_dir, transform=base_transform)
    
    # Get a few sample images directly
    n_samples = 5
    samples = []
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    for idx in indices:
        image, _ = dataset[idx]
        samples.append(image)  # Already a PIL image
    
    # Create output directory
    output_dir = 'corruption_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each corruption type
    corruptions = {
        'Fog': FogCorruption,
        'Snow': SnowCorruption,
        'Rain': RainCorruption,
        'Motion Blur': MotionBlurCorruption
    }
    
    for idx, original_image in enumerate(samples):
        corrupted_images = {}
        
        for corruption_name, corruption_class in corruptions.items():
            corrupted_images[corruption_name] = {}
            
            # Apply corruption at different severities
            for severity in range(1, 6):
                corruption = corruption_class(severity=severity)
                corrupted = corruption(original_image)
                corrupted_images[corruption_name][severity] = corrupted
        
        # Plot and save results
        save_path = os.path.join(output_dir, f'sample_{idx+1}_corruptions.png')
        plot_corrupted_images(original_image, corrupted_images, save_path)
        print(f"Saved corruption test results for sample {idx+1}")

def test_dataset_wrapper():
    """Test the CorruptedImageNetDataset wrapper."""
    # Setup ImageNet data
    val_dir = '/ssd_4TB/divake/CP_trust_IJCNN/dataset/imagenet/val'
    
    # First transform without normalization for corruption
    corruption_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Final transform for model input
    final_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    base_dataset = ImageFolder(val_dir, transform=corruption_transform)
    
    # Test dataset wrapper with each corruption
    corruptions = {
        'Fog': FogCorruption,
        'Snow': SnowCorruption,
        'Rain': RainCorruption,
        'Motion Blur': MotionBlurCorruption
    }
    
    output_dir = 'corruption_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    for corruption_name, corruption_class in corruptions.items():
        print(f"\nTesting {corruption_name} corruption dataset wrapper...")
        
        # Create corrupted dataset
        corrupted_dataset = CorruptedImageNetDataset(
            base_dataset,
            corruption_class,
            severity=3,  # Test with middle severity
            transform=final_transform  # Apply normalization after corruption
        )
        
        # Create data loader
        loader = DataLoader(corrupted_dataset, batch_size=4, shuffle=True)
        
        # Get a batch of images
        images, labels = next(iter(loader))
        
        # Convert and save sample images
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(4):
            # Denormalize
            img = images[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            
            # Convert to PIL
            img = transforms.ToPILImage()(img.clamp(0, 1))
            
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        plt.suptitle(f'{corruption_name} Corruption (Severity 3)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{corruption_name.lower()}_dataset_test.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved {corruption_name} dataset test results")

class PILImageFolder(ImageFolder):
    """Custom ImageFolder that returns PIL images directly."""
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        return image, target

def test_rain_corruption():
    """Test rain corruption at all severity levels."""
    # Create results directory
    save_dir = 'corruption_test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load a few sample images from ImageNet validation set
    val_dir = '/ssd_4TB/divake/CP_trust_IJCNN/dataset/imagenet/val'
    dataset = PILImageFolder(val_dir)
    
    # Select 3 random images
    np.random.seed(42)
    n_samples = 3
    sample_indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    # Create figure for all samples
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3*n_samples))
    fig.suptitle('Rain Corruption Test: Original and Severity Levels 1-5', fontsize=16)
    
    # Process each sample
    for idx, sample_idx in enumerate(sample_indices):
        image, _ = dataset[sample_idx]
        
        # Plot original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')
        
        # Apply and plot rain corruption at each severity level
        for severity in range(1, 6):
            rain = RainCorruption(severity=severity)
            corrupted_image = rain(image)
            
            axes[idx, severity].imshow(corrupted_image)
            axes[idx, severity].set_title(f'Severity {severity}')
            axes[idx, severity].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rain_corruption_test.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Rain corruption test completed. Results saved in corruption_test_results/rain_corruption_test.png")

if __name__ == '__main__':
    print("Testing individual corruptions...")
    test_corruptions()
    
    print("\nTesting dataset wrapper...")
    test_dataset_wrapper()
    
    test_rain_corruption() 