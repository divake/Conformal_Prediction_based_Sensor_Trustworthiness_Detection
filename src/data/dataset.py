# src/data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# src/data/dataset.py
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        
        # Read file list
        with open(self.root_dir / f'modelnet40_{split}.txt', 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
            
        # Read class names
        with open(self.root_dir / 'modelnet40_shape_names.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Find the matching class name from our classes list
        class_name = None
        for cls in self.classes:
            if filename.startswith(cls):
                class_name = cls
                break
        
        if class_name is None:
            raise ValueError(f"Could not determine class for filename: {filename}")
        
        # Load point cloud
        file_path = self.root_dir / class_name / f"{filename}.txt"
        
        # Use delimiter=',' for comma-separated values
        point_cloud = np.loadtxt(str(file_path), delimiter=',')
        
        # If the point cloud has more points than needed, randomly sample
        if len(point_cloud) > self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud = point_cloud[indices]
        # If it has fewer points, randomly repeat points
        elif len(point_cloud) < self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points - len(point_cloud))
            extra_points = point_cloud[indices]
            point_cloud = np.concatenate([point_cloud, extra_points], axis=0)
            
        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)
        label = self.class_to_idx[class_name]
        
        return point_cloud, label