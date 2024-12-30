# src/models/point_transformer_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformerV2(nn.Module):
    def __init__(self, num_classes=40, in_channels=6):
        super().__init__()
        self.trans_dim = 384  # Hidden dimension
        self.depth = 12      # Number of blocks
        self.drop_path_rate = 0.1
        self.num_heads = 6   # Number of attention heads
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        
        # Point embedding layers
        self.encoder = PointEncoder(
            in_channels=in_channels,     # Now accepts 6 channels directly
            first_channels=128,
            second_channels=512,
            out_channels=256
        )
        
        # Dimension reduction layer
        self.reduce_dim = nn.Linear(256, self.trans_dim)
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        # Transformer blocks
        self.blocks = TransformerBlocks(
            dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=4
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(self.trans_dim)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [B, N, 6]
        B, N, _ = x.shape
        
        # Get positions for pos embedding (first 3 coordinates)
        pos = x[:, :, :3].clone()
        
        # Encode points
        x = self.encoder(x)  # [B, N, 256]
        x = self.reduce_dim(x)  # [B, N, 384]
        
        # Add position embeddings
        pos_embed = self.pos_embed(pos)
        x = x + pos_embed
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat((cls_pos, pos_embed), dim=1)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Extract features
        cls_feature = x[:, 0]
        global_feature = torch.mean(x[:, 1:], dim=1)
        
        # Concatenate features for classification
        concat_features = torch.cat([cls_feature, global_feature], dim=1)
        x = self.cls_head(concat_features)
        
        return x

class PointEncoder(nn.Module):
    def __init__(self, in_channels=6, first_channels=128, second_channels=512, out_channels=256):
        super().__init__()
        
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, first_channels, 1),
            nn.BatchNorm1d(first_channels),
            nn.GELU(),
            nn.Conv1d(first_channels, 256, 1)
        )
        
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, second_channels, 1),  # Changed input channels from 512 to 256
            nn.BatchNorm1d(second_channels),
            nn.GELU(),
            nn.Conv1d(second_channels, out_channels, 1)
        )
        
    def forward(self, x):
        # [B, N, 6] -> [B, 6, N]
        x = x.transpose(1, 2)
        
        # First conv block
        x = self.first_conv(x)
        
        # Second conv block
        x = self.second_conv(x)
        
        # [B, C, N] -> [B, N, C]
        x = x.transpose(1, 2)
        return x

class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x