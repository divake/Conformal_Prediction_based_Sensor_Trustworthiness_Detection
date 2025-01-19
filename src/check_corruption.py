import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.corruptions import OcclusionCorruption
from data.dataset import ModelNet40Dataset
from pathlib import Path

def visualize_point_cloud(ax, points, title):
    """Helper function to plot a single point cloud"""
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_axis_off()
    # Set same scale for all axes
    max_range = np.array([points[:,0].max()-points[:,0].min(),
                         points[:,1].max()-points[:,1].min(),
                         points[:,2].max()-points[:,2].min()]).max()
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)

def check_corruption(save_dir: str = 'corruption_checks'):
    """
    Visualize the effects of occlusion corruption at different severity levels
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset and get a sample
    dataset = ModelNet40Dataset(root_dir='dataset/modelnet40_normal_resampled', split='test')
    
    # Set up the corruption
    corruption = OcclusionCorruption(seed=42)
    
    # Create figure with 6 subplots (original + 5 severity levels)
    fig = plt.figure(figsize=(20, 4))
    
    # Sample multiple objects to check
    sample_indices = [0, 100, 200, 300]  # Add more indices as needed
    
    for idx in sample_indices:
        fig = plt.figure(figsize=(20, 4))
        points, label = dataset[idx]
        points = points.numpy()
        
        # Plot original
        ax = fig.add_subplot(161, projection='3d')
        visualize_point_cloud(ax, points, f'Original\n{len(points)} points')
        
        # Plot corrupted versions
        for severity in range(1, 6):
            ax = fig.add_subplot(161 + severity, projection='3d')
            corrupted_points = corruption(points, severity)
            visualize_point_cloud(
                ax, 
                corrupted_points, 
                f'Severity {severity}\n{len(corrupted_points)} points'
            )
        
        plt.suptitle(f'Object {idx} (Class: {dataset.classes[label]})', y=1.05)
        plt.tight_layout()
        plt.savefig(save_dir / f'occlusion_corruption_sample_{idx}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Corruption visualization saved to {save_dir}")

if __name__ == '__main__':
    check_corruption()
