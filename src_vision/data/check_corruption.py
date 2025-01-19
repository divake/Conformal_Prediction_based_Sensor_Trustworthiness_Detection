import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from corruptions import FogCorruption, SnowCorruption, RainCorruption, MotionBlurCorruption
from cifar100 import setup_cifar100

def visualize_corruptions(image_index=1000, base_path='/ssd_4TB/divake/CP_trust_IJCNN', save_dir='corruption_visualizations'):
    """Visualize all corruptions on a fixed image from CIFAR100."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    _, _, _, _, _, test_dataset = setup_cifar100(base_path=base_path)
    
    # Get image and convert to proper format
    img, label = test_dataset[image_index]
    
    # Denormalize if tensor
    if isinstance(img, torch.Tensor):
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        img = img * std + mean
        
        # Convert to numpy and transpose to HWC format for display
        img_display = img.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
    
    # Create a figure with 4 rows (one for each corruption) and 6 columns (original + 5 severity levels)
    fig, axes = plt.subplots(4, 6, figsize=(20, 16))
    fig.suptitle(f'Corruption Visualization\nCIFAR100 Image #{image_index}, Class: {test_dataset.classes[label]}', 
                fontsize=16, y=0.95)
    
    # List of corruptions and their parameters to display
    corruptions = [
        (FogCorruption, "Fog", lambda x: f'max_val={x.fog_params[severity]["max_val"]:.1f}\ndecay={x.fog_params[severity]["decay"]:.1f}'),
        (SnowCorruption, "Snow", lambda x: f'snow_scale={x.snow_params[severity]["snow_scale"]:.1f}\nbrightness={x.snow_params[severity]["brightness"]:.1f}\nblur={x.snow_params[severity]["blur"]:.1f}'),
        (RainCorruption, "Rain", lambda x: f'density={x.rain_params[severity]["density"]:.3f}\nlength={x.rain_params[severity]["length"]:.1f}\nbrightness={x.rain_params[severity]["brightness"]:.2f}'),
        (MotionBlurCorruption, "Motion Blur", lambda x: f'kernel_size={x.motion_params[severity]["kernel_size"]}\nangle={x.motion_params[severity]["angle"]}°')
    ]
    
    # Process each corruption type
    for row, (CorruptionClass, corruption_name, param_formatter) in enumerate(corruptions):
        # Plot original image in first column
        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f'Original\n{corruption_name}')
        axes[row, 0].axis('off')
        
        # Create corruption instance once and reuse
        corruption = CorruptionClass(severity=1)
        
        # Apply corruptions for each severity level
        for severity in range(1, 6):
            corruption.severity = severity
            corrupted = corruption(img)
            
            if isinstance(corrupted, torch.Tensor):
                corrupted_display = corrupted.permute(1, 2, 0).numpy()
                corrupted_display = np.clip(corrupted_display, 0, 1)
            
            # Plot with parameters
            axes[row, severity].imshow(corrupted_display)
            axes[row, severity].set_title(f'Severity {severity}\n{param_formatter(corruption)}')
            axes[row, severity].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'all_corruptions_img{image_index}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

    # Print parameters for each corruption type
    print(f"\nAnalyzing CIFAR100 Image #{image_index}")
    print(f"Class: {test_dataset.classes[label]}")
    
    for CorruptionClass, corruption_name, _ in corruptions:
        print(f"\n{corruption_name} Parameters:")
        print("-" * 20)
        corruption = CorruptionClass(severity=1)
        for severity in range(1, 6):
            corruption.severity = severity
            if hasattr(corruption, 'fog_params'):
                params = corruption.fog_params[severity]
            elif hasattr(corruption, 'snow_params'):
                params = corruption.snow_params[severity]
            elif hasattr(corruption, 'rain_params'):
                params = corruption.rain_params[severity]
            else:
                params = corruption.motion_params[severity]
            print(f"Severity {severity}:")
            for param_name, param_value in params.items():
                print(f"  {param_name}: {param_value}")
            print()

def analyze_multiple_images(image_indices=[1000, 2000, 3000, 4000, 5000], save_dir='corruption_visualizations'):
    """Analyze corruptions on multiple fixed images."""
    for idx in image_indices:
        visualize_corruptions(image_index=idx, save_dir=save_dir)
        print("-" * 50)

if __name__ == "__main__":
    try:
        # Analyze a single fixed image
        visualize_corruptions(image_index=1000)
        print("Visualization completed successfully!")
        
        # Or analyze multiple images
        # analyze_multiple_images()
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
