import torch
import matplotlib.pyplot as plt
import numpy as np
from corruptions import FogCorruption
from cifar100 import setup_cifar100

def visualize_fog_corruption(image_index=1000, base_path='/ssd_4TB/divake/CP_trust_IJCNN'):
    """
    Visualize fog corruption on a fixed image from CIFAR100.
    Args:
        image_index: Index of the image to use (default: 1000)
        base_path: Path to your project directory
    """
    # Load your preprocessed dataset
    _, _, _, _, _, test_dataset = setup_cifar100(base_path=base_path)
    
    # Use fixed image index
    img, label = test_dataset[image_index]
    
    # Denormalize the image
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    img = img * std + mean
    
    # Create a figure with 6 subplots (original + 5 severity levels)
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle(f'Fog Corruption Visualization (CIFAR100 Image #{image_index}, Class: {test_dataset.classes[label]})')
    
    # Plot original image
    axes[0].imshow(img.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Apply and plot fog corruptions for each severity level
    for severity in range(1, 6):
        fog = FogCorruption(severity=severity)
        corrupted_img = fog(img)
        
        # Plot corrupted image
        axes[severity].imshow(corrupted_img.permute(1, 2, 0).clamp(0, 1))
        axes[severity].set_title(f'Severity {severity}\nmax_val={fog.fog_params[severity]["max_val"]:.1f}\ndecay={fog.fog_params[severity]["decay"]:.1f}')
        axes[severity].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'fog_corruption_visualization_img{image_index}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print parameters for each severity level
    print(f"\nAnalyzing CIFAR100 Image #{image_index}")
    print(f"Class: {test_dataset.classes[label]}")
    print("\nFog Corruption Parameters:")
    print("------------------------")
    for severity in range(1, 6):
        fog = FogCorruption(severity=severity)
        params = fog.fog_params[severity]
        print(f"Severity {severity}:")
        print(f"  max_val: {params['max_val']:.1f}")
        print(f"  decay: {params['decay']:.1f}")
        print()

def analyze_multiple_images(image_indices=[1000, 2000, 3000, 4000, 5000]):
    """
    Analyze fog corruption on multiple fixed images.
    Args:
        image_indices: List of image indices to analyze
    """
    for idx in image_indices:
        visualize_fog_corruption(image_index=idx)
        print("-" * 50)

if __name__ == "__main__":
    # Analyze a single fixed image
    visualize_fog_corruption(image_index=1000)
    
    # Or analyze multiple fixed images
    # analyze_multiple_images()