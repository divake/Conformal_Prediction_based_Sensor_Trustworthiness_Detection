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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    base_dataset = ImageFolder(val_dir, transform=transform)
    
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
            severity=3  # Test with middle severity
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

if __name__ == '__main__':
    print("Testing individual corruptions...")
    test_corruptions()
    
    print("\nTesting dataset wrapper...")
    test_dataset_wrapper() 