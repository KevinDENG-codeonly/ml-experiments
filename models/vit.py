import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .base import BaseModel
from .registry import register_model

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Optimized for PyTorch 2.1.0 - use more efficient convolution
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # Input: B, C, H, W
        # Output: B, n_patches, embed_dim
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2)  # B, embed_dim, n_patches
        x = x.transpose(1, 2)  # B, n_patches, embed_dim
        return x


# 2. Adding Positional Embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        # Use nn.Parameter for learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        # Initialize with normal distribution
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # GELU is typically used in transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout, 
            batch_first=True  # PyTorch 2.1.0 optimization
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dim, dropout)
        
    def forward(self, x):
        # Pre-LN Transformer architecture (helps with training stability)
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
    
@register_model("vit")
class VisionTransformer(BaseModel):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3,
        num_classes=10, 
        embed_dim=768, 
        num_heads=8, 
        depth=6, 
        mlp_dim=1024,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        # Patch embedding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        
        # Dropout after pos encoding
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # MLP head
        self.mlp_head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Get batch size
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # B, n_patches, embed_dim
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, embed_dim
        x = torch.cat((cls_tokens, x), dim=1)  # B, n_patches+1, embed_dim
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.pos_dropout(x)
        
        # Pass through transformer encoder blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Use class token for classification
        x = x[:, 0]  # B, embed_dim
        
        # MLP head for classification
        x = self.mlp_head(x)  # B, num_classes
        
        return x