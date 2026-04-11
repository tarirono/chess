"""
bot_api.py — Phase B FastAPI service
=====================================
Exposes the behavioral cloning move engine over HTTP.

Start with:
    uvicorn src.api.bot_api:app --port 8087 --reload

Endpoints:
    POST /move    — given FEN + Elo, return a UCI move
    GET  /health  — liveness check
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from pydantic import BaseModel

from src.api.move_service import MoveService

app = FastAPI(title="Chess Bot API — Phase B", version="1.0.0")

# Load models once at startup
_service: MoveService | None = None


def _get_service() -> MoveService:
    global _service
    if _service is None:
        _service = MoveService()
    return _service


class MoveRequest(BaseModel):
    fen: str
    elo: int = 1400
    # temperature > 1.0 widens the probability distribution over legal moves,
    # making the bot pick slightly sub-optimal moves more often — this is
    # intentional: it simulates the natural variance of a human player at the
    # target Elo rather than always picking the single "most cloned" move.
    temperature: float = 1.1


class MoveResponse(BaseModel):
    uci: str | None
    bracket: str
    conf: float
    reason: str | None = None


@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    service = _get_service()
    result = service.get_move(req.fen, elo=req.elo, temperature=req.temperature)
    return MoveResponse(**result)


@app.get("/health")
def health():
    service = _get_service()
    return {
        "status": "ok",
        "loaded_brackets": list(service.models.keys()),
    }