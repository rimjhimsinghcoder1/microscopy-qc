"""Small PyTorch models: a CNN on raw pixels, and a fusion of CNN + QC features.

Design notes:
- Images are NOT normalised per image. Brightness itself is the signal for
  overexposure, so per-image normalisation would erase it. We only shift by a
  fixed 0.5 so inputs are centred around zero.
- No data augmentation that changes blur, brightness or noise, because those
  would change the true QC label.
"""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Three conv blocks -> global average pooling -> embedding -> class scores."""

    def __init__(self, n_classes: int, width: int = 16) -> None:
        super().__init__()
        w = width

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            )

        # No pooling in the first block: blur and noise live in fine detail,
        # so the first layers see the image at full 28x28 resolution.
        self.features = nn.Sequential(
            block(1, w),
            block(w, w),
            nn.MaxPool2d(2),  # 28 -> 14
            block(w, 2 * w),
            nn.MaxPool2d(2),  # 14 -> 7
            block(2 * w, 2 * w),
            nn.AdaptiveAvgPool2d(1),  # 7x7 -> 1x1
            nn.Flatten(),
        )
        self.embedding_dim = 2 * w
        self.head = nn.Linear(self.embedding_dim, n_classes)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """(N, H, W) images in [0, 1] -> (N, embedding_dim) vectors."""
        return self.features(images.unsqueeze(1) - 0.5)

    def forward(self, images: torch.Tensor, features: torch.Tensor | None = None):
        return self.head(self.embed(images))


class FusionNet(nn.Module):
    """Concatenate the CNN embedding with the standardised QC features.

    The CNN can learn patterns the hand-made features miss, while the features
    give the model reliable, physically meaningful measurements to lean on.
    """

    def __init__(self, n_classes: int, n_features: int, width: int = 16) -> None:
        super().__init__()
        self.cnn = SmallCNN(n_classes, width)
        self.head = nn.Sequential(
            nn.Linear(self.cnn.embedding_dim + n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, images: torch.Tensor, features: torch.Tensor | None = None):
        if features is None:
            raise ValueError("FusionNet needs QC features")
        return self.head(torch.cat([self.cnn.embed(images), features], dim=1))
