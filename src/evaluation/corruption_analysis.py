#src/evaluation/corruption_analysis.py

import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from data.corrupted_dataset import CorruptedModelNet40Dataset
from data.corruptions import OcclusionCorruption
from conformal_prediction import conformal_prediction, get_softmax_predictions
from data.dataset import ModelNet40Dataset
from typing import Optional, Dict, Type
import matplotlib.pyplot as plt
from data.corruptions import OcclusionCorruption, PointCloudCorruption
from pathlib import Path

def find_qhat_for_target_coverage(
    val_softmax: np.ndarray,
    val_labels: np.ndarray,
    target_coverage: float = 0.9,
    k_reg: int = 5,
    lam_reg: float = 0.01
) -> float:
    """
    Find the quantile (qhat) that achieves target coverage for the validation set
    """
    n_val = len(val_labels)
    n_classes = val_softmax.shape[1]
    
    # Create regularization vector
    reg_vec = np.array(k_reg*[0,] + (n_classes-k_reg)*[lam_reg,])[None,:]
    
    # Get scores for validation set
    val_pi = val_softmax.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_softmax, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    val_L = np.where(val_pi == val_labels[:,None])[1]
    
    # Calculate scores
    rand_vals = np.random.rand(n_val)
    val_scores = val_srt_reg.cumsum(axis=1)[np.arange(n_val),val_L] - rand_vals*val_srt_reg[np.arange(n_val),val_L]
    
    # Find qhat that gives target coverage
    qhat = np.quantile(val_scores, target_coverage, interpolation='higher')
    
    return qhat

def evaluate_corruption_robustness(
    model: torch.nn.Module,
    base_dataset: ModelNet40Dataset,
    cal_softmax: np.ndarray,
    cal_labels: np.ndarray,
    device: torch.device,
    corruption_types: List[Type[PointCloudCorruption]] = [OcclusionCorruption],
    severity_levels: List[int] = [1, 2, 3, 4, 5],
    batch_size: int = 32,
    alpha: float = 0.1,  # Fixed alpha for 90% coverage
    k_reg: int = 2,
    lam_reg: float = 0.05,
    seed: int = 42
) -> Dict:
    """Evaluate model robustness against corruptions using conformal prediction."""
    results = {}
    
    for corruption_type in corruption_types:
        corruption_results = {}
        
        for severity in severity_levels:
            # Create corrupted dataset
            corrupted_dataset = CorruptedModelNet40Dataset(
                base_dataset=base_dataset,
                corruption_type=corruption_type,
                severity=severity,
                seed=seed
            )
            
            # Create dataloader
            corrupted_loader = DataLoader(
                corrupted_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Get predictions
            val_softmax, val_labels = get_softmax_predictions(
                model, corrupted_loader, device
            )
            
            # Find qhat for target coverage
            qhat = find_qhat_for_target_coverage(
                val_softmax=val_softmax,
                val_labels=val_labels,
                target_coverage=1-alpha,  # 0.9 for alpha=0.1
                k_reg=k_reg,
                lam_reg=lam_reg
            )
            
            # Run conformal prediction with the found qhat
            n_val = len(val_labels)
            n_classes = val_softmax.shape[1]
            reg_vec = np.array(k_reg*[0,] + (n_classes-k_reg)*[lam_reg,])[None,:]
            
            val_pi = val_softmax.argsort(1)[:,::-1]
            val_srt = np.take_along_axis(val_softmax, val_pi, axis=1)
            val_srt_reg = val_srt + reg_vec
            
            rand_vals_val = np.random.rand(n_val,1)
            indicators = (val_srt_reg.cumsum(axis=1) - rand_vals_val*val_srt_reg) <= qhat
            indicators[:,0] = True
            
            prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
            
            # Calculate metrics
            coverage = prediction_sets[np.arange(n_val), val_labels].mean()
            avg_set_size = prediction_sets.sum(axis=1).mean()
            
            corruption_results[severity] = {
                'coverage': coverage,
                'avg_set_size': avg_set_size,
                'qhat': qhat,
                'prediction_sets': prediction_sets
            }
            
        results[corruption_type.__name__] = corruption_results
    
    return results


def plot_corruption_results(results, save_dir='plots'):
    """Plot coverage, set size, and qhat vs corruption severity"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for corruption_type, corruption_results in results.items():
        plt.figure(figsize=(15, 5))
        
        # Coverage plot
        plt.subplot(1, 3, 1)
        severities = list(corruption_results.keys())
        coverages = [corruption_results[s]['coverage'] for s in severities]
        plt.plot(severities, coverages, 'o-', label='Empirical Coverage')
        plt.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
        plt.xlabel('Corruption Severity')
        plt.ylabel('Coverage')
        plt.title(f'{corruption_type}: Coverage vs Severity')
        plt.grid(True)
        plt.legend()
        
        # Set size plot
        plt.subplot(1, 3, 2)
        set_sizes = [corruption_results[s]['avg_set_size'] for s in severities]
        plt.plot(severities, set_sizes, 'o-')
        plt.xlabel('Corruption Severity')
        plt.ylabel('Average Set Size')
        plt.title(f'{corruption_type}: Set Size vs Severity')
        plt.grid(True)
        
        # Qhat plot
        plt.subplot(1, 3, 3)
        qhats = [corruption_results[s]['qhat'] for s in severities]
        plt.plot(severities, qhats, 'o-')
        plt.xlabel('Corruption Severity')
        plt.ylabel('Quantile (qhat)')
        plt.title(f'{corruption_type}: qhat vs Severity')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{corruption_type.lower()}_analysis.png')
        plt.close()

        # Create set size distribution plot
        plt.figure(figsize=(10, 6))
        for severity in severities:
            set_sizes = corruption_results[severity]['prediction_sets'].sum(axis=1)
            unique_sizes, counts = np.unique(set_sizes, return_counts=True)
            percentages = (counts / len(set_sizes)) * 100
            plt.plot(unique_sizes, percentages, 'o-', label=f'Severity {severity}')
        
        plt.xlabel('Set Size')
        plt.ylabel('Percentage of Predictions (%)')
        plt.title(f'{corruption_type}: Set Size Distribution')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f'{corruption_type.lower()}_set_size_dist.png')
        plt.close()