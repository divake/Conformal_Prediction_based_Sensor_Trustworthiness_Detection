import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.corruptions import OcclusionCorruption, RainCorruption, FogCorruption, BlurCorruption, SnowCorruption
from data.dataset import ModelNet40Dataset
from pathlib import Path

def visualize_point_cloud(ax, points, title, color='b', alpha=0.6):
    """Helper function to plot a single point cloud"""
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=1, alpha=alpha)
    ax.set_title(title)
    ax.set_axis_off()
    
    # Set same scale for all axes
    max_range = np.array([
        points[:,0].max()-points[:,0].min(),
        points[:,1].max()-points[:,1].min(),
        points[:,2].max()-points[:,2].min()
    ]).max()
    
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)

def check_corruptions(save_dir: str = 'corruption_checks'):
    """Visualize the effects of occlusion, rain, and fog corruptions"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset and get samples
    dataset = ModelNet40Dataset(root_dir='dataset/modelnet40_normal_resampled', split='test')
    
    # Set up corruptions
    occlusion = OcclusionCorruption(seed=42)
    rain = RainCorruption(seed=42)
    fog = FogCorruption(seed=42)
    blur = BlurCorruption(seed=42)
    snow = SnowCorruption(seed=42)
    
    # Sample multiple objects
    sample_indices = [0, 100, 200]  # Add more indices if needed
    corruptions = {
        'Occlusion': (occlusion, 'b'),  # Blue for occlusion
        'Rain': (rain, 'r'),            # Red for rain
        'Fog': (fog, 'g'),              # Green for fog
        'Blur': (blur, 'm'),            # Magenta for blur
        'Snow': (snow, 'c')             # Cyan for snow
    }
    
    for corruption_name, (corruption, color) in corruptions.items():
        # Create one large figure for all samples
        fig = plt.figure(figsize=(20, 12))  # Increased height for 3 rows
        plt.suptitle(f'{corruption_name} Corruption Effects Across Different Objects', y=0.95, fontsize=16)
        
        # For each sample
        for sample_idx, idx in enumerate(sample_indices):
            points, label = dataset[idx]
            points = points.numpy()
            
            # Plot original
            ax = fig.add_subplot(3, 6, sample_idx*6 + 1, projection='3d')
            visualize_point_cloud(ax, points, f'Original\nObject {idx}\n{dataset.classes[label]}\n{len(points)} points')
            
            # Plot severities 1-5
            for severity in range(1, 6):
                ax = fig.add_subplot(3, 6, sample_idx*6 + severity + 1, projection='3d')
                corrupted_points = corruption(points, severity)
                
                visualize_point_cloud(
                    ax, 
                    corrupted_points, 
                    f'Severity {severity}\n{len(corrupted_points)} points',
                    color=color
                )
        
        plt.tight_layout()
        plt.savefig(
            save_dir / f'{corruption_name}_all_samples.png', 
            bbox_inches='tight', 
            dpi=300
        )
        plt.close()
    
    print(f"Corruption visualizations saved to {save_dir}")

if __name__ == '__main__':
    check_corruptions()
