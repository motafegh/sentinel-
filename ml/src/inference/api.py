"""
api.py — SENTINEL FastAPI Inference Endpoint (Cross-Attention + LoRA Upgrade)

WHAT CHANGED FROM TRACK 3 ORIGINAL:
    1. /health endpoint reports architecture type from loaded predictor
       No longer calls torch.load() on every health check (saves ~489MB I/O).

    2. Architecture is read from predictor.architecture (stored at load time).

    3. Default checkpoint matches actual filename: multilabel_crossattn_best.pt
       (no _v2_ suffix – this is correct).

WHAT DID NOT CHANGE:
    - PredictRequest / PredictResponse / VulnerabilityResult schemas
    - /predict endpoint logic
    - Lifespan pattern (Predictor loaded once at startup)
    - Error handling: 400/413/500/503/504

FIXES (2026-04-29):
    Bug 1 — import torch added. Was missing — caused NameError on every CUDA OOM
             instead of the intended HTTP 413 response.
    Bug 3 — v['class'] → v['vulnerability_class'] to match canonical key
             now emitted by predictor._score().
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch  # Bug 1 fix — was missing; needed for torch.cuda.OutOfMemoryError + empty_cache()
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from ml.src.inference.predictor import Predictor

CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    "ml/checkpoints/multilabel_crossattn_best.pt",
)


# ------------------------------------------------------------------
# Lifespan — Predictor loaded once at startup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"SENTINEL API starting — checkpoint: {CHECKPOINT}")

    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}. "
            "Check SENTINEL_CHECKPOINT env var or run: dvc pull"
        )

    app.state.predictor = Predictor(checkpoint=CHECKPOINT)
    logger.info("Predictor ready — API accepting requests")
    yield
    logger.info("SENTINEL API shutting down")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="SENTINEL Vulnerability API",
    description="GNN + CodeBERT (LoRA) + Cross-Attention multi-label vulnerability detector",
    version="3.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Schemas — unchanged from Track 3
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    source_code: str = Field(..., min_length=10)

    @field_validator("source_code")
    @classmethod
    def must_look_like_solidity(cls, v: str) -> str:
        if "pragma" not in v.lower() and "contract" not in v.lower():
            raise ValueError(
                "source_code does not appear to be Solidity "
                "(missing 'pragma' or 'contract' keyword)"
            )
        return v


class VulnerabilityResult(BaseModel):
    vulnerability_class: str = Field(..., description="Vulnerability type name")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    label: str = Field(..., description="'vulnerable' or 'safe'")
    vulnerabilities: list[VulnerabilityResult]
    threshold: float
    truncated: bool
    num_nodes: int
    num_edges: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> dict:
    """
    Liveness check — confirms predictor loaded and reports architecture.
    Does NOT read the checkpoint file (avoids expensive I/O on every call).
    """
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    predictor_loaded = predictor is not None

    architecture = predictor.architecture if predictor_loaded else "unknown"

    return {
        "status": "ok" if predictor_loaded else "degraded",
        "predictor_loaded": predictor_loaded,
        "checkpoint": CHECKPOINT,
        "architecture": architecture,   # "cross_attention_lora" confirms upgrade loaded
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Score a Solidity contract for multi-label vulnerability detection."""
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        logger.info(f"Inference request — {len(body.source_code)} chars")
        result: dict = await asyncio.wait_for(
            asyncio.to_thread(predictor.predict_source, body.source_code),
            timeout=60.0,
        )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout.")

    except ValueError as exc:
        logger.warning(f"Bad input: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))

    except torch.cuda.OutOfMemoryError:  # Bug 1 fix — now resolves correctly (torch is imported)
        torch.cuda.empty_cache()
        raise HTTPException(status_code=413, detail="Contract too large for GPU memory.")

    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail="Inference failed.")

    logger.info(
        f"Complete — label={result['label']} "
        f"detected={len(result['vulnerabilities'])} classes"
    )

    return PredictResponse(
        label=result["label"],
        vulnerabilities=[
            VulnerabilityResult(
                vulnerability_class=v["vulnerability_class"],  # Bug 3 fix — was v["class"]
                probability=v["probability"],
            )
            for v in result["vulnerabilities"]
        ],
        threshold=result["threshold"],
        truncated=result["truncated"],
        num_nodes=result["num_nodes"],
        num_edges=result["num_edges"],
    )
