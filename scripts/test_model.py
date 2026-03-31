import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.behavioral.model import ChessResNet

model = ChessResNet()
x = torch.randn(4, 13, 8, 8)
y = model(x)

print(f"Input  shape: {x.shape}")
print(f"Output shape: {y.shape}  (expected: torch.Size([4, 4096]))")
print(f"Parameters  : {model.count_parameters():,}")
print("Model OK — ready to train")

