# src/utils/data_utils.py
import torch
from torch.utils.data import DataLoader
from data.dataset import ModelNet40Dataset

def create_dataloaders(root_dir, batch_size=32, num_workers=4, num_points=1024):
    """
    Create train and test dataloaders for ModelNet40
    
    Args:
        root_dir: Path to dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        num_points: Number of points per point cloud
    """
    # Create training dataset
    train_dataset = ModelNet40Dataset(
        root_dir=root_dir,
        split='train'
    )
    
    # Create test dataset
    test_dataset = ModelNet40Dataset(
        root_dir=root_dir,
        split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    num_classes = len(train_dataset.classes)
    
    return train_loader, test_loader, num_classes