import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from datetime import datetime

from src.behavioral.model   import ChessResNet
from src.behavioral.dataset import ChessDataset

# ── Config ────────────────────────────────────────────────────────────
BRACKETS     = ["1200", "1400", "1600"]
MAX_GAMES    = 3000       # all games per bracket
EPOCHS       = 20
BATCH_SIZE   = 256
LR           = 1e-3
VAL_SPLIT    = 0.1
MODELS_DIR   = Path("data/models/behavioral")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE       = torch.device("cpu")


def train_bracket(bracket: str):
    print(f"\n{'='*60}")
    print(f"Training bracket: Elo {bracket}")
    print(f"{'='*60}")

    # ── Dataset ───────────────────────────────────────────────────────
    # Delete cache to rebuild with full games
    cache = Path(f"data/processed/cache/dataset_{bracket}.pt")
    if cache.exists():
        cache.unlink()

    dataset  = ChessDataset(bracket=bracket, max_games=MAX_GAMES)
    val_size = int(len(dataset) * VAL_SPLIT)
    trn_size = len(dataset) - val_size
    trn_ds, val_ds = random_split(dataset, [trn_size, val_size])

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    print(f"Train: {trn_size:,} positions  |  Val: {val_size:,} positions")

    # ── Model ─────────────────────────────────────────────────────────
    model = ChessResNet(channels=128, num_blocks=10, dropout=0.3).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    best_path     = MODELS_DIR / f"chess_bot_{bracket}.pt"
    history       = []

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        trn_loss, trn_correct, trn_total = 0.0, 0, 0

        for x, y in tqdm(trn_loader,
                         desc=f"  Epoch {epoch:02d}/{EPOCHS} [train]",
                         leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            trn_loss    += loss.item() * len(y)
            trn_correct += (logits.argmax(1) == y).sum().item()
            trn_total   += len(y)

        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss   = criterion(logits, y)
                val_loss    += loss.item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total   += len(y)

        trn_loss_avg = trn_loss / trn_total
        val_loss_avg = val_loss / val_total
        trn_acc      = trn_correct / trn_total * 100
        val_acc      = val_correct / val_total * 100

        history.append({
            "epoch":    epoch,
            "trn_loss": round(trn_loss_avg, 4),
            "val_loss": round(val_loss_avg, 4),
            "trn_acc":  round(trn_acc, 2),
            "val_acc":  round(val_acc, 2),
        })

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"trn_loss={trn_loss_avg:.4f}  trn_acc={trn_acc:.1f}%  "
              f"val_loss={val_loss_avg:.4f}  val_acc={val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "bracket":    bracket,
                "epoch":      epoch,
                "val_acc":    val_acc,
                "state_dict": model.state_dict(),
            }, best_path)
            print(f"  Saved best model → {best_path}  (val_acc={val_acc:.1f}%)")

    # Save training history
    hist_path = MODELS_DIR / f"history_{bracket}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val_acc: {best_val_acc:.1f}%")
    print(f"  Model saved : {best_path}")
    return best_val_acc


if __name__ == "__main__":
    print("Phase B — Behavioral Cloning Training")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")

    results = {}
    for bracket in BRACKETS:
        results[bracket] = train_bracket(bracket)

    print(f"\n{'='*60}")
    print("Training complete — all brackets")
    print(f"{'='*60}")
    for bracket, acc in results.items():
        print(f"  Elo {bracket}: best val_acc = {acc:.1f}%")
    print(f"\nModels saved to: {MODELS_DIR}")

