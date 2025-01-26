import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from collections import defaultdict
import os

class ImageNetSplitDataset:
    def __init__(self, val_dir, split_ratio=0.2, samples_per_class=None, random_state=42, batch_size=32, num_workers=4):
        """
        Initialize ImageNet dataset with fixed split functionality and balanced class sampling.
        
        Args:
            val_dir (str): Path to ImageNet validation directory
            split_ratio (float): Ratio of data to use for calibration (default: 0.2)
            samples_per_class (int, optional): Number of samples to use per class. If None, uses all samples
            random_state (int): Random seed for reproducibility (fixed split)
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for data loading
        """
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Define standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load full validation dataset
        self.full_dataset = ImageFolder(val_dir, transform=self.transform)
        
        # Create splits with balanced classes
        self.cal_indices, self.test_indices = self._create_balanced_split(
            split_ratio=split_ratio,
            samples_per_class=samples_per_class,
            random_state=random_state
        )
        
        # Create calibration and test datasets
        self.calibration_dataset = Subset(self.full_dataset, self.cal_indices)
        self.test_dataset = Subset(self.full_dataset, self.test_indices)
        
        # Store class names and mapping
        self.classes = self.full_dataset.classes
        self.class_to_idx = self.full_dataset.class_to_idx
    
    def _create_balanced_split(self, split_ratio, samples_per_class, random_state):
        """
        Create balanced split ensuring equal number of samples per class.
        
        Args:
            split_ratio (float): Ratio for calibration set
            samples_per_class (int): Number of samples per class
            random_state (int): Random seed
            
        Returns:
            tuple: (calibration indices, test indices)
        """
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.full_dataset.samples):
            class_indices[label].append(idx)
        
        # Determine number of samples per class if not specified
        if samples_per_class is None:
            # Find minimum number of samples across all classes
            min_samples = min(len(indices) for indices in class_indices.values())
            samples_per_class = min_samples
        
        cal_indices = []
        test_indices = []
        
        # For each class
        for class_label in sorted(class_indices.keys()):
            indices = np.array(class_indices[class_label])
            
            # Shuffle indices deterministically
            rng = np.random.RandomState(random_state + class_label)
            rng.shuffle(indices)
            
            # Take only the specified number of samples
            indices = indices[:samples_per_class]
            
            # Split into calibration and test
            n_cal = int(samples_per_class * split_ratio)
            
            cal_indices.extend(indices[:n_cal])
            test_indices.extend(indices[n_cal:])
        
        return cal_indices, test_indices
    
    def get_loaders(self, batch_size=None, num_workers=None):
        """
        Get DataLoaders for both calibration and test sets.
        
        Args:
            batch_size (int, optional): Override default batch size
            num_workers (int, optional): Override default number of workers
            
        Returns:
            tuple: (calibration loader, test loader)
        """
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
            
        calibration_loader = DataLoader(
            self.calibration_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling to maintain fixed order
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling to maintain fixed order
            num_workers=num_workers,
            pin_memory=True
        )
        
        return calibration_loader, test_loader
    
    def get_split_sizes(self):
        """
        Get sizes of calibration and test sets.
        
        Returns:
            tuple: (calibration size, test size)
        """
        return len(self.calibration_dataset), len(self.test_dataset)
    
    def get_class_distribution(self):
        """
        Get class distribution for both splits.
        
        Returns:
            tuple: (calibration distribution, test distribution)
        """
        cal_labels = [self.full_dataset.targets[i] for i in self.cal_indices]
        test_labels = [self.full_dataset.targets[i] for i in self.test_indices]
        
        cal_dist = np.bincount(cal_labels)
        test_dist = np.bincount(test_labels)
        
        return cal_dist, test_dist
    
    def get_class_names(self):
        """
        Get class names and their mapping.
        
        Returns:
            tuple: (list of class names, class to index mapping)
        """
        return self.classes, self.class_to_idx

def get_imagenet_dataset(val_dir='/ssd_4TB/divake/CP_trust_IJCNN/dataset/imagenet/val', 
                        split_ratio=0.2,
                        samples_per_class=None,
                        batch_size=32,
                        num_workers=4,
                        random_state=42):
    """
    Helper function to quickly get ImageNet dataset and loaders.
    
    Args:
        val_dir (str): Path to ImageNet validation directory
        split_ratio (float): Ratio for calibration set
        samples_per_class (int, optional): Number of samples per class
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        random_state (int): Random seed
        
    Returns:
        tuple: (ImageNetSplitDataset, calibration loader, test loader)
    """
    dataset = ImageNetSplitDataset(
        val_dir=val_dir,
        split_ratio=split_ratio,
        samples_per_class=samples_per_class,
        batch_size=batch_size,
        num_workers=num_workers,
        random_state=random_state
    )
    
    cal_loader, test_loader = dataset.get_loaders()
    return dataset, cal_loader, test_loader

# Usage example
if __name__ == "__main__":
    # Example usage with balanced sampling (50 samples per class)
    dataset, cal_loader, test_loader = get_imagenet_dataset(
        samples_per_class=50,  # Use 50 samples per class
        split_ratio=0.2        # 20% for calibration, 80% for test
    )