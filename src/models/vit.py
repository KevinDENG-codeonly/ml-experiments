"""
Vision Transformer (ViT) implementation.

Based on the paper:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
by Alexey Dosovitskiy et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them.
    
    Args:
        image_size: Size of the input image.
        patch_size: Size of each patch.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, in_channels, height, width).
            
        Returns:
            Patch embeddings with shape (batch_size, num_patches, embed_dim).
        """
        # x: (batch_size, in_channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, grid_height, grid_width)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        return x


class Attention(nn.Module):
    """Multi-head self-attention mechanism.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, dim).
            
        Returns:
            Output tensor with shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, dim = x.shape
        
        # QKV projections
        qkv = self.qkv(x)  # (batch_size, seq_len, dim*3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Output projection
        out = (attn @ v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, dim)  # (batch_size, seq_len, dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """MLP feed-forward network.
    
    Args:
        dim: Input dimension.
        hidden_dim: Hidden dimension.
        dropout: Dropout rate.
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        dropout: Dropout rate.
        attention_dropout: Attention dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        """Forward pass with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model.
    
    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        dropout: Dropout rate.
        attention_dropout: Attention dropout rate.
        representation_size: Dimension of the representation layer (None means no representation layer).
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size: int = None,
    ):
        super().__init__()
        # Image size must be divisible by patch size
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(depth)
        ])
        
        # Representation layer (if specified)
        self.representation = None
        if representation_size is not None:
            self.representation = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
            )
            classifier_input_size = representation_size
        else:
            classifier_input_size = embed_dim
        
        # Layer normalization and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(classifier_input_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize patch embedding
        nn.init.normal_(self.patch_embed.projection.weight, std=0.02)
        
        # Initialize class token and position embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize transformer blocks
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward_features(self, x):
        """Forward pass through patch embedding and transformer blocks."""
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Use class token as representation
        x = x[:, 0]
        
        return x
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, in_channels, height, width).
            
        Returns:
            Output tensor with shape (batch_size, num_classes).
        """
        x = self.forward_features(x)
        
        # Apply representation layer if specified
        if self.representation is not None:
            x = self.representation(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x


def vit_tiny_patch16(
    image_size: int = 224,
    num_classes: int = 1000,
    **kwargs
) -> VisionTransformer:
    """ViT-Tiny configuration."""
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_dim=768,
        **kwargs
    )


def vit_small_patch16(
    image_size: int = 224,
    num_classes: int = 1000,
    **kwargs
) -> VisionTransformer:
    """ViT-Small configuration."""
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


def vit_base_patch16(
    image_size: int = 224,
    num_classes: int = 1000,
    **kwargs
) -> VisionTransformer:
    """ViT-Base configuration."""
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def vit_large_patch16(
    image_size: int = 224,
    num_classes: int = 1000,
    **kwargs
) -> VisionTransformer:
    """ViT-Large configuration."""
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_dim=4096,
        **kwargs
    ) 