# Intelligent Chess Ecosystem — PFE 2026

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

---

## Project Structure

```text
chess-ecosystem/
|-- src/
|   |-- vision/                   # Phase A — Camera pipeline
|   |   |-- piece_detector.py     # YOLOv8-Nano inference
|   |   |-- board_mapper.py       # Pixel boxes -> chess squares -> FEN
|   |   |-- board_localizer.py    # 4-point board calibration / warp helpers
|   |   |-- fen_extractor.py      # Shared image -> board -> FEN pipeline
|   |   |-- motion_detector.py    # Motion-triggered inference
|   |   `-- vision_loop.py        # Full camera pipeline + live PGN writer
|   |-- behavioral/               # Phase B — AI Bot
|   |   |-- encoder.py            # Board -> 13x8x8 tensor
|   |   |-- dataset.py            # Lichess game dataset
|   |   `-- model.py              # ResNet policy network
|   |-- graph/                    # Phase C — Knowledge Graph
|   |   |-- neo4j_client.py       # Neo4j CRUD operations
|   |   |-- mock_neo4j_client.py  # In-memory fallback (demo / no-DB mode)
|   |   |-- skill_tagger.py       # Tactical pattern detection
|   |   |-- irt_model.py          # Item Response Theory (Rasch model)
|   |   `-- skill_tree.py         # ZPD recommendations + Neo4j/mock wiring
|   |-- api/                      # FastAPI bot service
|   |   |-- move_service.py       # Loads trained models
|   |   `-- app.py                # /move and /health endpoints
|   |-- integration/              # Phase A + B + C wiring
|   |   `-- game_manager.py       # Central game orchestrator
|   `-- dashboard/                # Flask web dashboard
|       |-- app.py                # Flask routes + /api/status health check
|       |-- static/
|       |   `-- custom.css        # Project-specific CSS overrides
|       `-- templates/
|           `-- index.html        # Live game UI with Neo4j status indicator
|-- scripts/
|   |-- train_chess.py            # Train YOLOv8-Nano
|   |-- train_behavioral.py       # Train ResNet bot
|   |-- download_lichess.py       # Download training data
|   |-- validate_elo.py           # Validate bot Elo vs Stockfish
|   |-- run_vision.py             # Start live camera
|   |-- image_to_fen.py           # Convert one image into FEN
|   |-- calibrate_board.py        # Save 4 board corners / test warp
|   |-- test_detector.py          # Test piece detection
|   |-- test_board_mapper.py      # Shared image -> FEN smoke test
|   |-- test_motion.py            # Test motion detection
|   |-- test_move_service.py      # Test bot API
|   |-- test_neo4j.py             # Verify Neo4j connection
|   `-- test_skill_tree.py        # Test Neo4j + IRT
|-- data/
|   |-- models/
|   |   |-- chess_nano_v1/        # YOLOv8 weights
|   |   |-- behavioral/           # ResNet weights (3 Elo brackets)
|   |   `-- board_config.json     # Saved 4-point board calibration
|   |-- pgn/                      # Live PGN files written during camera sessions
|   `-- processed/
|       `-- lichess/              # Filtered game data
|-- .gitignore
|-- env.example                   # Template for .env (never commit .env itself)
|-- requirements.txt
`-- README.md
```

---

## Setup

### 1. Configure environment variables

```bash
cp env.example .env
# Edit .env with your Neo4j password and Stockfish path
```

Your `.env` should look like:

```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
STOCKFISH_PATH=stockfish
```

> **Never commit `.env` to git.** It is listed in `.gitignore`. Only `env.example` is tracked.

### 2. Create Python environment

```bash
conda create -n chess-env python=3.11 -y
conda activate chess-env
pip install -r requirements.txt
```

### 3. Set OpenMP fix (Windows only)

```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda env config vars set OMP_NUM_THREADS=1
conda activate chess-env
```

### 4. Start Neo4j

- Open Neo4j Desktop
- Start the `chess-db` instance
- Verify the connection:

```bash
python scripts/test_neo4j.py
```

> **Neo4j is optional for demos.** If Neo4j is not running, the system automatically falls back to an in-memory mock database. All game flow and skill tracking work normally — data just does not persist between sessions. To force mock mode without running Neo4j, set `NEO4J_MOCK=true` in your `.env`.

---

## Phase A — Vision System

### Train the piece detector

```bash
python scripts/train_chess.py
```

Trains YOLOv8-Nano on the chess piece dataset. The inference pipeline uses:
- YOLO piece detection
- 4-point perspective correction when calibration points exist
- Orientation-aware square mapping
- FEN placement generation
- **Live PGN auto-writing** to `data/pgn/` on every detected move

### Test the detection pipeline

```bash
python scripts/test_detector.py
python scripts/test_board_mapper.py
python scripts/test_motion.py
```

### Calibrate the board once

```bash
python scripts/calibrate_board.py
```

Click the playable 8×8 grid corners in this order: top-left → top-right → bottom-right → bottom-left.

This saves `data/models/board_config.json` so the image/FEN pipeline and the live vision loop can reuse the same calibration across sessions.

### Convert one image into FEN

```bash
python scripts/image_to_fen.py path\to\board.jpg
```

Saves a warped-board debug preview to `data/raw/_debug_warped_board.jpg`.

### Run live camera

```bash
python scripts/run_vision.py
```

Opens the webcam, detects pieces via motion-triggered YOLO inference, and converts stable frames into FEN. Every detected move is immediately written to a timestamped PGN file in `data/pgn/`. Press `q` to quit, `s` to print the current FEN and PGN path.

---

## Phase B — Behavioral AI Bot

### Download Lichess training data

```bash
python scripts/download_lichess.py
```

Downloads the January 2013 Lichess database and extracts 3,000 games per Elo bracket (1200, 1400, 1600).

### Train on local machine (CPU)

```bash
python scripts/train_behavioral.py
```

Trains 3 ResNet models (one per Elo bracket) via behavioral cloning. Best weights are saved to `data/models/behavioral/`.

### Start the bot API

```bash
uvicorn src.api.app:app --reload --port 8087
```

Exposes `POST /move`, accepting FEN + Elo and returning a UCI move. Interactive docs: `http://localhost:8087/docs`

### Test the bot

```bash
python scripts/test_move_service.py
```

### Validate bot Elo against Stockfish

```bash
python scripts/validate_elo.py
python scripts/validate_elo.py --games 20 --depth 3
```

Plays N games per bracket against Stockfish at calibrated depths and reports Win/Draw/Loss rates, average centipawn loss, and estimated Elo. Stockfish path is read from your `.env` — no hardcoding required.

> **On model accuracy:** Validation accuracy of ~26–27% is expected for behavioral cloning on this task — chess has 4,096 possible move slots, and matching the exact human move is inherently hard. The meaningful metric is Elo performance against Stockfish, reported by `validate_elo.py`.

---

## Phase C — Knowledge Graph

### Graph schema

```text
(Player)-[:PLAYED]->(Game)-[:HAS_MOVE]->(Move)-[:INVOLVES]->(Skill)
(Player)-[:PERFORMANCE {attempts, successes, irt_ability}]->(Skill)

Skill nodes: Pin, Fork, Discovery, Skewer, Checkmate_pattern,
             Endgame, Opening, Pawn_structure, Piece_activity,
             Blunder, Mistake, Inaccuracy
```

### Test the skill tree

```bash
python scripts/test_skill_tree.py
```

Simulates a 5-move game, stores it in Neo4j (or mock), and outputs ZPD recommendations.

### Item Response Theory

Each skill node has a difficulty parameter. The system estimates the player's ability on each skill using the Rasch model:

```
P(correct) = 1 / (1 + exp(-(ability - difficulty)))
```

Zone of Proximal Development = skills where `P(correct) ≈ 0.60`.

Move quality (best / good / inaccuracy / mistake / blunder) is determined by Stockfish centipawn loss when available, with a pure python-chess heuristic fallback when Stockfish is not installed.

---

## Running the Full System

Start all three services in separate terminals:

**Terminal 1 — Bot API**
```bash
conda activate chess-env
uvicorn src.api.app:app --port 8087
```

**Terminal 2 — Dashboard**
```bash
conda activate chess-env
python src/dashboard/app.py
```

**Terminal 3 — Camera (optional)**
```bash
conda activate chess-env
python scripts/run_vision.py
```

Open `http://127.0.0.1:5000` in your browser. The dashboard shows a **Neo4j: connected / mock mode** status pill in the top bar so you always know which mode is active.

---

## Model Weights

Trained weights may already be present in `data/models/`. On a clean checkout without weight files:

- Phase A: `python scripts/train_chess.py`
- Phase B: `python scripts/train_behavioral.py`

Model files (`.pt`, `.onnx`) and the Lichess dataset (`.zst`) are excluded from git via `.gitignore`.

---

## Tech Stack

| Component | Technology |
|---|---|
| Piece detection | YOLOv8-Nano (Ultralytics) |
| Board mapping | OpenCV + python-chess |
| Behavioral bot | PyTorch ResNet (behavioral cloning) |
| Training data | Lichess open database |
| Knowledge graph | Neo4j (+ in-memory mock fallback) |
| Psychometric model | Item Response Theory (Rasch) |
| Engine analysis | Stockfish 16+ via python-chess |
| Bot API | FastAPI + Uvicorn |
| Dashboard | Flask + Bootstrap 5 |
| Environment | Python 3.11, Anaconda |

---

## Authors

Tarik Ouabrk and Adam Hajjaji