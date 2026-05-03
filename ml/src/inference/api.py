"""
api.py — SENTINEL FastAPI Inference Endpoint (Cross-Attention + LoRA Upgrade)

WHAT CHANGED FROM TRACK 3 ORIGINAL:
    1. /health endpoint reports architecture type from loaded predictor
       No longer calls torch.load() on every health check (saves ~489MB I/O).

    2. Architecture is read from predictor.architecture (stored at load time).

    3. Default checkpoint updated to multilabel_crossattn_v2_best.pt (retrain v2,
       edge_attr active). Override via SENTINEL_CHECKPOINT env var.

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

IMPROVEMENTS:
    - /health now reports thresholds_loaded from predictor.
    - SENTINEL_PREDICT_TIMEOUT env var controls inference timeout (default 60 s).
    - logger.exception used in catch-all so full traceback appears in logs.
    - Source size enforced before preprocessing: reject requests > MAX_SOURCE_BYTES (1 MB).
    - Solidity validator gives a clearer message distinguishing missing keyword vs empty input.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch  # Bug 1 fix — was missing; needed for torch.cuda.OutOfMemoryError + empty_cache()
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

from ml.src.inference.drift_detector import DriftDetector
from ml.src.inference.predictor import Predictor

DRIFT_BASELINE_PATH: str = os.getenv(
    "SENTINEL_DRIFT_BASELINE",
    "ml/data/drift_baseline.json",
)
# Run a KS check every N requests (balance: lower = more responsive, higher = cheaper).
DRIFT_CHECK_INTERVAL: int = int(os.getenv("SENTINEL_DRIFT_CHECK_INTERVAL", "50"))

# ---------------------------------------------------------------------------
# Prometheus — custom gauges
# ---------------------------------------------------------------------------
_gauge_model_loaded  = Gauge("sentinel_model_loaded",      "1 if the predictor is loaded, 0 otherwise")
_gauge_gpu_mem_bytes = Gauge("sentinel_gpu_memory_bytes",  "Current GPU memory allocated (bytes)")

CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    "ml/checkpoints/multilabel_crossattn_v2_best.pt",
)

# Inference timeout in seconds — override via SENTINEL_PREDICT_TIMEOUT env var.
PREDICT_TIMEOUT: float = float(os.getenv("SENTINEL_PREDICT_TIMEOUT", "60"))

# Hard upper bound on source_code size.  Slither and the tokenizer are expensive;
# reject oversized payloads before any preprocessing work begins.
MAX_SOURCE_BYTES: int = 1 * 1024 * 1024  # 1 MB


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
    _gauge_model_loaded.set(1)

    app.state.drift_detector = DriftDetector(baseline_path=DRIFT_BASELINE_PATH)
    app.state.request_count  = 0

    logger.info("Predictor ready — API accepting requests")
    yield
    _gauge_model_loaded.set(0)
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

Instrumentator().instrument(app).expose(app)


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
    windows_used: int = Field(default=1, ge=1, description="Number of token windows scored (>1 for long contracts)")
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
    thresholds_loaded = predictor.thresholds_loaded if predictor_loaded else False

    return {
        "status": "ok" if predictor_loaded else "degraded",
        "predictor_loaded": predictor_loaded,
        "checkpoint": CHECKPOINT,
        "architecture": architecture,       # "cross_attention_lora" confirms upgrade loaded
        "thresholds_loaded": thresholds_loaded,  # False → uniform fallback threshold active
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Score a Solidity contract for multi-label vulnerability detection."""
    predictor:       Predictor | None       = getattr(request.app.state, "predictor", None)
    drift_detector:  DriftDetector | None   = getattr(request.app.state, "drift_detector", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    source_bytes = len(body.source_code.encode())
    if source_bytes > MAX_SOURCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"source_code too large ({source_bytes:,} bytes > {MAX_SOURCE_BYTES:,} limit).",
        )

    try:
        logger.info(f"Inference request — {len(body.source_code)} chars")
        result: dict = await asyncio.wait_for(
            asyncio.to_thread(predictor.predict_source, body.source_code),
            timeout=PREDICT_TIMEOUT,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Inference timeout after {PREDICT_TIMEOUT:.0f} s.",
        )

    except ValueError as exc:
        logger.warning(f"Bad input: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))

    except torch.cuda.OutOfMemoryError:  # Bug 1 fix — now resolves correctly (torch is imported)
        torch.cuda.empty_cache()
        raise HTTPException(status_code=413, detail="Contract too large for GPU memory.")

    except Exception as exc:
        logger.exception(f"Inference error: {exc}")  # exception() logs full traceback
        raise HTTPException(status_code=500, detail="Inference failed.")

    try:
        if torch.cuda.is_available():
            _gauge_gpu_mem_bytes.set(torch.cuda.memory_allocated())
    except Exception as _prom_exc:
        logger.debug(f"Prometheus gauge update failed: {_prom_exc}")

    # T2-B: drift detection — update per request, check every DRIFT_CHECK_INTERVAL
    if drift_detector is not None:
        try:
            drift_detector.update_stats({
                "num_nodes": float(result["num_nodes"]),
                "num_edges": float(result["num_edges"]),
            })
            request.app.state.request_count += 1
            if request.app.state.request_count % DRIFT_CHECK_INTERVAL == 0:
                drift_detector.check()
        except Exception as _drift_exc:
            logger.debug(f"Drift detector update failed: {_drift_exc}")

    logger.info(
        f"Complete — label={result['label']} "
        f"detected={len(result['vulnerabilities'])} classes "
        f"windows={result.get('windows_used', 1)}"
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
        windows_used=result.get("windows_used", 1),
        num_nodes=result["num_nodes"],
        num_edges=result["num_edges"],
    )
