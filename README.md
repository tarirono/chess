# Intelligent Chess Ecosystem - PFE 2026

An integrated hardware-software platform that uses Edge AI to digitize physical chess games, match players against a human-like AI bot adapted to their skill level, and track their tactical development through a knowledge graph.

---

## System Architecture
```text
Camera Feed -> Motion Detection -> YOLOv8-Nano -> FEN/PGN
                                                |
                                     Behavioral AI Bot (ResNet)
                                                |
                                  Neo4j Knowledge Graph + IRT
                                                |
                                     Flask Dashboard
```

## Project Structure
```text
chess-ecosystem/
|-- src/
|   |-- vision/           # Phase A - Camera pipeline
|   |   |-- piece_detector.py     # YOLOv8-Nano inference
|   |   |-- board_mapper.py       # Pixel boxes -> chess squares -> FEN
|   |   |-- board_localizer.py    # 4-point board calibration / warp helpers
|   |   |-- fen_extractor.py      # Shared image -> board -> FEN pipeline
|   |   |-- motion_detector.py    # Motion-triggered inference
|   |   `-- vision_loop.py        # Full camera pipeline
|   |-- behavioral/       # Phase B - AI Bot
|   |   |-- encoder.py            # Board -> 13x8x8 tensor
|   |   |-- dataset.py            # Lichess game dataset
|   |   `-- model.py              # ResNet policy network
|   |-- graph/            # Phase C - Knowledge Graph
|   |   |-- neo4j_client.py       # Neo4j CRUD operations
|   |   |-- skill_tagger.py       # Tactical pattern detection
|   |   |-- irt_model.py          # Item Response Theory
|   |   `-- skill_tree.py         # ZPD recommendations
|   |-- api/              # FastAPI bot service
|   |   |-- move_service.py       # Loads trained models
|   |   `-- app.py                # /move and /health endpoints
|   |-- integration/      # Phase A + B + C wiring
|   |   `-- game_manager.py       # Central game orchestrator
|   `-- dashboard/        # Flask web dashboard
|       |-- app.py                # Flask routes
|       `-- templates/
|           `-- index.html        # Live game UI
|-- scripts/
|   |-- train_chess.py            # Train YOLOv8-Nano
|   |-- train_behavioral.py       # Train ResNet bot
|   |-- download_lichess.py       # Download training data
|   |-- run_vision.py             # Start live camera
|   |-- image_to_fen.py           # Convert one image into FEN
|   |-- calibrate_board.py        # Save 4 board corners / test warp
|   |-- test_detector.py          # Test piece detection
|   |-- test_board_mapper.py      # Shared image -> FEN smoke test
|   |-- test_motion.py            # Test motion detection
|   |-- test_move_service.py      # Test bot API
|   `-- test_skill_tree.py        # Test Neo4j + IRT
|-- data/
|   |-- models/
|   |   |-- chess_nano_v1/        # YOLOv8 weights
|   |   `-- behavioral/           # ResNet weights (3 Elo brackets)
|   `-- processed/
|       `-- lichess/              # Filtered game data
|-- requirements.txt
`-- README.md
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
- Verify with `python scripts/test_neo4j.py`

---

## Phase A - Vision System

### Train the piece detector
```bash
python scripts/train_chess.py
```

Trains YOLOv8-Nano on the chess piece dataset. The current shared inference path uses:
- YOLO piece detection
- 4-point perspective correction when calibration points exist
- orientation-aware square mapping
- FEN placement generation

### Test detection pipeline
```bash
python scripts/test_detector.py
python scripts/test_board_mapper.py
python scripts/test_motion.py
```

### Calibrate the board once
```bash
python scripts/calibrate_board.py
```

Click the playable 8x8 grid corners in this order:
- top-left
- top-right
- bottom-right
- bottom-left

This saves `data/models/board_config.json` so the image/FEN pipeline and the live vision loop can reuse the same calibration.

### Convert one image into FEN
```bash
python scripts/image_to_fen.py path\to\board.jpg
```

By default this also saves a warped-board preview to `data/raw/_debug_warped_board.jpg`.

### Run live camera
```bash
python scripts/run_vision.py
```

Opens the webcam, detects pieces via motion-triggered YOLO inference, and converts stable frames into FEN/PGN using the shared `src/vision/fen_extractor.py` pipeline. Press `q` to quit and `s` to save the current FEN.

---

## Phase B - Behavioral AI Bot

### Download Lichess training data
```bash
python scripts/download_lichess.py
```

Downloads the January 2013 Lichess database sample and extracts 3,000 games per Elo bracket (1200, 1400, 1600).

### Train on local machine (CPU)
```bash
python scripts/train_behavioral.py
```

Trains 3 ResNet models (one per Elo bracket) via behavioral cloning.

### Start bot API
```bash
uvicorn src.api.app:app --reload --port 8087
```

Exposes `POST /move`, which accepts FEN + Elo and returns a UCI move. Interactive docs: `http://localhost:8087/docs`

### Test bot
```bash
python scripts/test_move_service.py
```

---

## Phase C - Knowledge Graph

### Graph schema
```text
(Player)-[:PLAYED]->(Game)-[:HAS_MOVE]->(Move)-[:INVOLVES]->(Skill)
(Player)-[:PERFORMANCE {attempts, successes, irt_ability}]->(Skill)

Skill nodes: Pin, Fork, Discovery, Skewer, Checkmate_pattern,
             Endgame, Opening, Pawn_structure, Piece_activity, Blunder
```

### Test skill tree
```bash
python scripts/test_skill_tree.py
```

Simulates a 5-move game, stores it in Neo4j, and outputs ZPD recommendations.

### Item Response Theory

Each skill node has a difficulty parameter. The system estimates the player's ability on each skill using the Rasch model:
```text
P(correct) = 1 / (1 + exp(-(ability - difficulty)))
```

Zone of Proximal Development = skills where `P(correct) ~= 0.60`.

---

## Running the Full System

Start all three services in separate terminals:

**Terminal 1 - Bot API**
```bash
conda activate chess-env
uvicorn src.api.app:app --port 8087
```

**Terminal 2 - Dashboard**
```bash
conda activate chess-env
python src/dashboard/app.py
```

**Terminal 3 - Camera (optional)**
```bash
conda activate chess-env
python scripts/run_vision.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Model Weights

This local project copy may already contain trained weights in `data/models/`.

If you start from a clean checkout without model files:
- Phase A: run `python scripts/train_chess.py`
- Phase B: run `python scripts/train_behavioral.py`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Piece detection | YOLOv8-Nano (Ultralytics) |
| Board mapping | OpenCV + python-chess |
| Behavioral bot | PyTorch ResNet (behavioral cloning) |
| Training data | Lichess open database |
| Knowledge graph | Neo4j |
| Psychometric model | Item Response Theory (Rasch) |
| Bot API | FastAPI + Uvicorn |
| Dashboard | Flask + Bootstrap 5 |
| Environment | Python 3.11, Anaconda |

---

## Authors

Tarik Ouabrk and Adam Hajjaji
