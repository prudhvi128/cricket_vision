"""
EfficientNet-B0 + Transformer cricket shot classifier.

Input:
    (B, 30, 3, 224, 224)

Architecture:
    EfficientNet-B0
        ↓
    Spatial Attention
        ↓
    1280 → 256 projection
        ↓
    Sinusoidal Positional Encoding
        ↓
    2-layer Transformer Encoder
        ↓
    Temporal Attention
        ↓
    Classifier
        ↓
    10 cricket shot classes
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class SpatialAttention(nn.Module):
    """
    Channel/spatial attention applied to EfficientNet feature maps.
    """

    def __init__(self, feature_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Stored as `pe` so that it matches the checkpoint key:
        positional_encoding.pe
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 30,
    ):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TemporalAttention(nn.Module):
    """
    Attention pooling across the 30 temporal frames.

    256 → 64 → 1
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                feature_dim,
                feature_dim // 4,
            ),
            nn.ReLU(),
            nn.Linear(
                feature_dim // 4,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        # x: (B, T, 256)

        scores = self.attention(x).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        attended = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return attended, weights


class ImprovedSOTAModel(nn.Module):
    """
    Deployment version of the trained
    EfficientNet-B0 + Transformer model.
    """

    def __init__(
        self,
        num_classes: int = 10,
        temporal_dim: int = 256,
        n_frames: int = 30,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.n_frames = n_frames
        self.feature_dim = 1280
        self.temporal_dim = temporal_dim

        # --------------------------------------------------
        # EfficientNet-B0
        # --------------------------------------------------

        self.backbone = efficientnet_b0(
            weights="IMAGENET1K_V1"
        )

        # Remove ImageNet classifier.
        self.backbone.classifier = nn.Identity()

        # --------------------------------------------------
        # Spatial attention
        # --------------------------------------------------

        self.spatial_attention = SpatialAttention(
            self.feature_dim
        )

        # --------------------------------------------------
        # 1280 → 256
        # --------------------------------------------------

        self.input_proj = nn.Linear(
            self.feature_dim,
            temporal_dim,
        )

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------

        self.positional_encoding = PositionalEncoding(
            d_model=temporal_dim,
            max_len=n_frames,
        )

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # --------------------------------------------------
        # Temporal attention
        # --------------------------------------------------

        self.temporal_attention = TemporalAttention(
            temporal_dim
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),

            nn.Linear(
                temporal_dim,
                512,
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                512,
                num_classes,
            ),
        )

        # Compatibility with the Lightning checkpoint.
        self.register_buffer(
            "class_weights",
            torch.ones(
                num_classes,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "Expected input shape "
                f"(B,T,C,H,W), got {tuple(x.shape)}"
            )

        batch_size, time_steps, channels, height, width = x.shape

        if time_steps != self.n_frames:
            raise ValueError(
                f"Expected {self.n_frames} frames, "
                f"got {time_steps}"
            )

        # --------------------------------------------------
        # Process all frames through EfficientNet
        # --------------------------------------------------

        frames = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        features = self.backbone.features(frames)

        # --------------------------------------------------
        # Spatial attention
        # --------------------------------------------------

        features = self.spatial_attention(features)

        # --------------------------------------------------
        # Global average pooling
        # --------------------------------------------------

        features = self.backbone.avgpool(features)

        features = torch.flatten(
            features,
            start_dim=1,
        )

        # --------------------------------------------------
        # Restore temporal dimension
        # --------------------------------------------------

        features = features.reshape(
            batch_size,
            time_steps,
            self.feature_dim,
        )

        # --------------------------------------------------
        # 1280 → 256
        # --------------------------------------------------

        features = self.input_proj(features)

        # --------------------------------------------------
        # Positional encoding
        # --------------------------------------------------

        features = self.positional_encoding(features)

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        features = self.temporal_encoder(features)

        # --------------------------------------------------
        # Temporal attention
        # --------------------------------------------------

        attended_features, _ = self.temporal_attention(
            features
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        logits = self.classifier(
            attended_features
        )

        return logits