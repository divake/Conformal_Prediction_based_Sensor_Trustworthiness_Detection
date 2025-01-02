#src/corruption_analysis_main.py

import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from data.dataset import ModelNet40Dataset
from models.point_transformer_v2 import PointTransformerV2
from config import Config
from main import setup_logging
from data.corruptions import OcclusionCorruption
from evaluation.corruption_analysis import evaluate_corruption_robustness, plot_corruption_results
from conformal_prediction import get_softmax_predictions, create_dataloader, conformal_prediction



def setup_logging(name='corruption_analysis'):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "corruption_analysis.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def main():
    # Initialize config and logging
    Config.initialize()
    logger = setup_logging('corruption_analysis')
    
    device = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load calibration data (clean)
    calibration_loader = create_dataloader(Config.ROOT_DIR, 'calibration')
    test_dataset = ModelNet40Dataset(root_dir=Config.ROOT_DIR, split='test')
    
    # Initialize and load model
    model = PointTransformerV2(
        num_classes=Config.MODEL['num_classes'],
        in_channels=Config.MODEL['in_channels']
    ).to(device)
    
    checkpoint = torch.load('/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded trained model")
    
    # Get calibration predictions
    logger.info("Getting calibration set predictions...")
    cal_softmax, cal_labels = get_softmax_predictions(model, calibration_loader, device)
    
    # Run corruption analysis
    logger.info("Running corruption analysis...")
    results = evaluate_corruption_robustness(
        model=model,
        base_dataset=test_dataset,
        cal_softmax=cal_softmax,
        cal_labels=cal_labels,
        device=device,
        corruption_types=[OcclusionCorruption]
    )
    
    # Plot and log results
    plot_corruption_results(results)
    
    logger.info("\nCorruption Analysis Results:")
    for corruption_type, corruption_results in results.items():
        logger.info(f"\n{corruption_type}:")
        
        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(f"{'Severity':<10} {'Coverage':<10} {'Avg Set Size':<15} {'qhat':<10}")
        logger.info("-" * 45)
        
        for severity, metrics in corruption_results.items():
            logger.info(f"{severity:<10} {metrics['coverage']:.4f}      {metrics['avg_set_size']:.2f}          {metrics['qhat']:.4f}")
        
        # Print set size distribution
        logger.info("\nSet Size Distribution:")
        for severity, metrics in corruption_results.items():
            set_sizes = metrics['prediction_sets'].sum(axis=1)
            unique_sizes, counts = np.unique(set_sizes, return_counts=True)
            percentages = (counts / len(set_sizes)) * 100
            
            logger.info(f"\nSeverity Level {severity}:")
            for size, percentage in zip(unique_sizes, percentages):
                logger.info(f"  Size {int(size)}: {percentage:.1f}%")

if __name__ == '__main__':
    main()