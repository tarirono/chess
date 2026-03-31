import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.api.move_service import MoveService

app = FastAPI(
    title="Chess Bot Move Service",
    description="Phase B — Behavioral cloning bot API",
    version="1.0.0"
)

# Load models once at startup
service = MoveService()


class MoveRequest(BaseModel):
    fen:         str
    elo:         int   = 1400
    temperature: float = 1.0


class MoveResponse(BaseModel):
    uci:     str
    bracket: str
    conf:    float


@app.get("/health")
def health():
    return {
        "status":   "ok",
        "brackets": list(service.models.keys())
    }


@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    try:
        result = service.get_move(
            fen=req.fen,
            elo=req.elo,
            temperature=req.temperature
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.get("uci") is None:
        raise HTTPException(status_code=422, detail="Game is already over.")

    return MoveResponse(
        uci=result["uci"],
        bracket=result["bracket"],
        conf=result["conf"]
    )