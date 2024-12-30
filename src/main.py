# src/main.py
import torch
import logging
import os
from pathlib import Path
from utils.data_utils import create_dataloaders
from utils.trainer import train_model
from models.point_transformer_v2 import PointTransformerV2
from config import Config


def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    # Initialize config and logging
    Config.initialize()
    logger = setup_logging()
    
    # Set device
    device = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, test_loader, num_classes = create_dataloaders(
        Config.ROOT_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        num_points=Config.NUM_POINTS
    )
    
    # Initialize model with 6 input channels
    model = PointTransformerV2(
        num_classes=Config.MODEL['num_classes'],
        in_channels=Config.MODEL['in_channels']
    ).to(device)
    
    logger.info(f"Training PointTransformerV2 from scratch")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=Config.TRAIN['num_epochs'],
        learning_rate=Config.TRAIN['learning_rate'],
        weight_decay=Config.TRAIN['weight_decay'],
        device=device,
        save_dir=Config.CHECKPOINT_DIR,
        grad_norm_clip=Config.TRAIN['grad_norm_clip'],
        warmup_epochs=Config.TRAIN['warmup_epochs']
    )
    
    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()