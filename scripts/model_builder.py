"""
Contains PyTorch model code to instantiate the Vision Transformer (ViT) architecture.
"""

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """Converts a 2D image into a sequence of flattened patches."""
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # Apply convolution to project patches
        x = self.projection(x)  # Shape: [B, E, H, W]
        x = x.flatten(2)  # Flatten H and W into one dimension -> Shape: [B, E, N]
        x = x.permute(0, 2, 1)  # Rearrange to [B, N, E] where N is the sequence length
        return x


class MultiheadSelfAttentionBlock(nn.Module):
    """Multi-Head Self Attention Block."""
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )

    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    """Multilayer Perceptron Block."""
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm(x)
        x_mlp = self.mlp(x_norm)
        return x_mlp

class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block."""
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads)
        self.mlp_block = MLPBlock(embedding_dim, mlp_size, dropout)

    def forward(self, x):
        x = x + self.msa_block(x)
        x = x + self.mlp_block(x)
        return x

class ViT(nn.Module):
    """Vision Transformer (ViT) Model."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_size=3072,
        dropout=0.1,
        attn_dropout=0
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim, num_heads, mlp_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        class_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x
