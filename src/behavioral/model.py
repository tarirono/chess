import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from src.behavioral.encoder import INPUT_CHANNELS, NUM_MOVES


class ResidualBlock(nn.Module):
    """Standard residual block with two conv layers and a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)   # skip connection


class ChessResNet(nn.Module):
    """
    Lightweight ResNet policy network for chess move prediction.

    Input  : (batch, 13, 8, 8) board tensor
    Output : (batch, 4096)     move logits
    """

    def __init__(self,
                 channels:    int = 128,
                 num_blocks:  int = 10,
                 dropout:     float = 0.3):
        super().__init__()

        # Initial conv to expand input channels
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, NUM_MOVES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.tower(x)
        return self.policy_head(x)

    def predict_move(self, board_tensor: torch.Tensor,
                     legal_mask: torch.Tensor = None) -> int:
        """
        Given a board tensor, return the predicted move index.
        Optionally mask illegal moves before argmax.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(board_tensor.unsqueeze(0))[0]
            if legal_mask is not None:
                logits[~legal_mask] = float("-inf")
            return logits.argmax().item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)