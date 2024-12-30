# src/data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir)
        self.split = split
        
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
        
        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)
        label = self.class_to_idx[class_name]
        
        return point_cloud, label