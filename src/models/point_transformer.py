# src/models/point_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # MLP block
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        
        return x

class PointTransformer(nn.Module):
    def __init__(self, num_classes=40, dim=256, depth=6, num_heads=8):
        super().__init__()
        
        # Initial point embedding
        self.embedding = nn.Sequential(
            nn.Linear(6, 64),  # 3D points + normals (6 dimensions)
            nn.ReLU(),
            nn.Linear(64, dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        
        # Global pooling and classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_points, 6)
        
        # Embed points
        x = self.embedding(x)  # Shape: (batch_size, num_points, dim)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)  # Shape: (batch_size, dim)
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        return x