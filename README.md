# Intelligent Chess Ecosystem — PFE 2025

An integrated hardware-software platform that uses Edge AI to digitize physical chess games, match players against a human-like AI bot adapted to their skill level, and track their tactical development through a knowledge graph.

---

## System Architecture
```
Camera Feed → Motion Detection → YOLOv8-Nano → FEN/PGN
                                                    ↓
                                         Behavioral AI Bot (ResNet)
                                                    ↓
                                      Neo4j Knowledge Graph + IRT
                                                    ↓
                                         Flask Dashboard
```

## Project Structure
```
chess-ecosystem/
├── src/
│   ├── vision/           # Phase A — Camera pipeline
│   │   ├── piece_detector.py     # YOLOv8-Nano inference
│   │   ├── board_mapper.py       # Pixel boxes → chess squares → FEN
│   │   ├── board_localizer.py    # Board corner detection
│   │   ├── motion_detector.py    # Motion-triggered inference
│   │   └── vision_loop.py        # Full camera pipeline
│   ├── behavioral/       # Phase B — AI Bot
│   │   ├── encoder.py            # Board → 13×8×8 tensor
│   │   ├── dataset.py            # Lichess game dataset
│   │   └── model.py              # ResNet policy network
│   ├── graph/            # Phase C — Knowledge Graph
│   │   ├── neo4j_client.py       # Neo4j CRUD operations
│   │   ├── skill_tagger.py       # Tactical pattern detection
│   │   ├── irt_model.py          # Item Response Theory
│   │   └── skill_tree.py         # ZPD recommendations
│   ├── api/              # FastAPI bot service
│   │   ├── move_service.py       # Loads trained models
│   │   └── app.py                # /move and /health endpoints
│   ├── integration/      # Phase A + B + C wiring
│   │   └── game_manager.py       # Central game orchestrator
│   └── dashboard/        # Flask web dashboard
│       ├── app.py                # Flask routes
│       └── templates/
│           └── index.html        # Live game UI
├── scripts/
│   ├── train_chess.py            # Train YOLOv8-Nano
│   ├── train_behavioral.py       # Train ResNet bot
│   ├── download_lichess.py       # Download training data
│   ├── run_vision.py             # Start live camera
│   ├── test_detector.py          # Test piece detection
│   ├── test_board_mapper.py      # Test FEN generation
│   ├── test_motion.py            # Test motion detection
│   ├── test_move_service.py      # Test bot API
│   └── test_skill_tree.py        # Test Neo4j + IRT
├── data/
│   ├── models/
│   │   ├── chess_nano_v1/        # YOLOv8 weights
│   │   └── behavioral/           # ResNet weights (3 Elo brackets)
│   └── processed/
│       └── lichess/              # Filtered game data
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create environment
```bash
conda create -n chess-env python=3.11 -y
conda activate chess-env
pip install -r requirements.txt
```

### 2. Set OpenMP fix (Windows)
```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda env config vars set OMP_NUM_THREADS=1
conda activate chess-env
```

### 3. Start Neo4j

- Open Neo4j Desktop
- Start the `chess-db` instance (password: `chess123`)
- Verify: `python scripts/test_neo4j.py`

---

## Phase A — Vision System

### Train the piece detector
```bash
python scripts/train_chess.py
```

Trains YOLOv8-Nano on 202 chess board images. Achieves **98.4% mAP@50** across 13 piece classes in ~2 hours on CPU.

### Test detection pipeline
```bash
python scripts/test_detector.py
python scripts/test_board_mapper.py
python scripts/test_motion.py
```

### Run live camera
```bash
python scripts/run_vision.py
```

Opens webcam, detects pieces via motion-triggered YOLO inference, outputs real-time FEN/PGN. Press `q` to quit, `s` to save FEN.

---

## Phase B — Behavioral AI Bot

### Download Lichess training data
```bash
python scripts/download_lichess.py
```

Downloads ~20MB from Lichess open database (Jan 2013). Extracts 3,000 games per Elo bracket (1200, 1400, 1600).

### Train on local machine (CPU)
```bash
python scripts/train_behavioral.py
```

~9 hours on CPU. Trains 3 ResNet models (9.27M parameters each) via behavioral cloning.

### Train on Google Colab (recommended — 35 min)

Upload `data/processed/lichess/games_*.jsonl` to Colab and run the provided notebook. Achieved results:

| Bracket | Val accuracy |
|---------|-------------|
| Elo 1200 | 26.0% |
| Elo 1400 | 26.4% |
| Elo 1600 | 27.5% |

### Start bot API
```bash
uvicorn src.api.app:app --reload --port 8087
```

Exposes `POST /move` — accepts FEN + Elo, returns UCI move.
Interactive docs: `http://localhost:8087/docs`

### Test bot
```bash
python scripts/test_move_service.py
```

---

## Phase C — Knowledge Graph

### Graph schema
```
(Player)-[:PLAYED]->(Game)-[:HAS_MOVE]->(Move)-[:INVOLVES]->(Skill)
(Player)-[:PERFORMANCE {attempts, successes, irt_ability}]->(Skill)

Skill nodes: Pin, Fork, Discovery, Skewer, Checkmate_pattern,
             Endgame, Opening, Pawn_structure, Piece_activity, Blunder
```

### Test skill tree
```bash
python scripts/test_skill_tree.py
```

Simulates a 5-move game, stores in Neo4j, and outputs ZPD recommendations.

### Item Response Theory

Each skill node has a difficulty parameter. The system estimates the player's ability on each skill using the Rasch model:
```
P(correct) = 1 / (1 + exp(-(ability - difficulty)))
```

Zone of Proximal Development = skills where P(correct) ≈ 0.60 (not too easy, not too hard).

---

## Running the Full System

Start all three services in separate terminals:

**Terminal 1 — Bot API:**
```bash
conda activate chess-env
uvicorn src.api.app:app --port 8087
```

**Terminal 2 — Dashboard:**
```bash
conda activate chess-env
python src/dashboard/app.py
```

**Terminal 3 — Camera (optional):**
```bash
conda activate chess-env
python scripts/run_vision.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Model Weights

Model weights are not included in this repository due to file size.

**To reproduce:**
- Phase A: run `python scripts/train_chess.py` (~2h on CPU)
- Phase B: run `python scripts/train_behavioral.py` or use the Colab notebook (~35min on GPU)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Piece detection | YOLOv8-Nano (Ultralytics) |
| Board mapping | OpenCV + python-chess |
| Behavioral bot | PyTorch ResNet (behavioral cloning) |
| Training data | Lichess open database |
| Knowledge graph | Neo4j 5.x |
| Psychometric model | Item Response Theory (Rasch) |
| Bot API | FastAPI + Uvicorn |
| Dashboard | Flask + Bootstrap 5 |
| Environment | Python 3.11, Anaconda |

---

## Results

| Metric | Value |
|--------|-------|
| Piece detection mAP@50 | 98.4% |
| Bot val accuracy (Elo 1600) | 27.5% |
| Motion detection CPU saving | 96% |
| Training time Phase A (CPU) | 2.1 hours |
| Training time Phase B (GPU) | ~35 minutes |

---

## Authors

Tarik Ouabrk 