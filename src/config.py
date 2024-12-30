# src/config.py
from pathlib import Path

class Config:
    # Paths
    ROOT_DIR = "/ssd_4TB/divake/CP_trust_IJCNN/dataset/modelnet40_normal_resampled"
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    
    # Dataset parameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Model parameters
    MODEL = {
        'dim': 256,          # Embedding dimension
        'depth': 6,          # Number of transformer blocks
        'num_heads': 8,      # Number of attention heads
        'dropout': 0.5       # Dropout rate
    }
    
    # Training parameters
    TRAIN = {
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'min_lr': 1e-5,
        'clip_grad': 1.0
    }
    
    # Optimizer parameters
    OPTIMIZER = {
        'type': 'AdamW',
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8
    }
    
    # Scheduler parameters
    SCHEDULER = {
        'type': 'CosineAnnealingLR',
        'T_max': 200,  # Should match num_epochs
        'eta_min': 1e-5
    }
    
    @classmethod
    def initialize(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)