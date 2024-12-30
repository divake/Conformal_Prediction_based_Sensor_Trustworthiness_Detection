# src/main.py
import torch
import logging
import os
from data.dataset import ModelNet40Dataset
from torch.utils.data import DataLoader
import sys

from utils.data_utils import create_dataloaders  # Changed to absolute import
from utils.trainer import train_model           # Changed to absolute import
from models.point_transformer import PointTransformer  # Changed to absolute import
from config import Config  # Changed to absolute import


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Initialize config
    Config.initialize()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Specify GPU 1
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, test_loader, num_classes = create_dataloaders(
        Config.ROOT_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    
    # Initialize model
    model = PointTransformer(
        num_classes=num_classes,
        dim=Config.MODEL['dim'],
        depth=Config.MODEL['depth'],
        num_heads=Config.MODEL['num_heads']
    ).to(device)
    
    # Train model
    best_acc = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=Config.TRAIN['num_epochs'],
        learning_rate=Config.TRAIN['learning_rate'],
        device=device,
        save_dir=Config.CHECKPOINT_DIR
    )
    
    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()