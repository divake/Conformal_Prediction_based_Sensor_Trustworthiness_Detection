# src_vision/utils/model_utils.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple

def get_model_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions and corresponding labels from a dataloader.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader containing the dataset
        device (torch.device): Device to run predictions on
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Softmax probabilities array of shape (n_samples, n_classes)
            - Labels array of shape (n_samples,)
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move images to device
            images = images.to(device)
            
            # Get model predictions
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store results
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Optional progress logging for large datasets
            if batch_idx % 20 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}", end='\r')
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_probs, all_labels