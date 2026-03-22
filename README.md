# Intelligent Chess Ecosystem — PFE 2025

An integrated hardware-software system that uses Edge AI to digitize 
physical chess games and match players against a human-like AI bot.

## Setup
```bash
conda create -n chess-env python=3.11 -y
conda activate chess-env
pip install -r requirements.txt
```

## Phase A — Vision System

Train the piece detector:
```bash
python scripts/train_chess.py
```

Run the live camera pipeline:
```bash
python scripts/run_vision.py
```

Test individual components:
```bash
python scripts/test_detector.py
python scripts/test_board_mapper.py
python scripts/test_motion.py
```

## Phase B — Behavioral AI (in progress)

Download Lichess data and train the bot:
```bash
python scripts/download_lichess.py
python scripts/train_behavioral.py
```

## Model weights

Download trained weights from [Releases](../../releases) or retrain using the scripts above.