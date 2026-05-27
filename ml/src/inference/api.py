"""
api.py — SENTINEL FastAPI Inference Endpoint

SCHEMA VERSION: Three-tier suspicion output (2026-05-27)
  PredictResponse now includes:
    label           "safe" | "suspicious" | "confirmed_vulnerable"
    probabilities   {class: float}  full 10-class vector, always present
    confirmed       [{vulnerability_class, probability, tier="CONFIRMED"}, ...]
    suspicious      [{vulnerability_class, probability, tier="SUSPICIOUS"}, ...]
    vulnerabilities legacy alias for confirmed (backward compat)
    tier_thresholds {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}

CHECKPOINT: GCB-P1-Run4-no-asl-pw_best.pt (epoch 32, F1=0.3362, all-time best)
  Pipeline verified FAIL=0 with compare_pipelines.py (2026-05-26).
  Override via SENTINEL_CHECKPOINT env var.

FIXES (2026-04-29):
    Bug 1 — import torch added.
    Bug 3 — v['class'] → v['vulnerability_class'].
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
from ml.src.inference.preprocess import ContractPreprocessor

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
    "ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt",
)

# Inference timeout in seconds — override via SENTINEL_PREDICT_TIMEOUT env var.
PREDICT_TIMEOUT: float = float(os.getenv("SENTINEL_PREDICT_TIMEOUT", "60"))

# Hard upper bound on source_code size — imported from ContractPreprocessor so both
# the HTTP boundary and the preprocessing layer share one definition.
MAX_SOURCE_BYTES: int = ContractPreprocessor.MAX_SOURCE_BYTES


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
    vulnerability_class: str   = Field(..., description="Vulnerability class name")
    probability:         float = Field(..., ge=0.0, le=1.0)
    tier:                str | None = Field(None, description="CONFIRMED | SUSPICIOUS (None in legacy vulnerabilities field)")


class PredictResponse(BaseModel):
    # Three-tier label: "safe" | "suspicious" | "confirmed_vulnerable"
    label: str = Field(..., description="Highest active tier: safe | suspicious | confirmed_vulnerable")

    # Full 10-class probability vector — always present, never filtered.
    # Enables agents to see all signal regardless of tier thresholds.
    probabilities: dict[str, float] = Field(..., description="Full per-class probability vector")

    # Tiered findings — sorted descending by probability within each tier.
    confirmed:  list[VulnerabilityResult] = Field(default_factory=list, description="prob >= tier_confirmed_threshold (default 0.55)")
    suspicious: list[VulnerabilityResult] = Field(default_factory=list, description="tier_suspicious_threshold <= prob < tier_confirmed_threshold")

    # Legacy field — backward compat alias for confirmed.
    # Old consumers reading vulnerabilities get CONFIRMED classes only.
    vulnerabilities: list[VulnerabilityResult] = Field(default_factory=list)

    # Tier boundaries used — allows agents to interpret tiers without hardcoding thresholds.
    tier_thresholds: dict[str, float] = Field(default_factory=dict)

    thresholds:   list[float] = Field(..., description="Per-class tuned decision thresholds")
    truncated:    bool
    windows_used: int = Field(default=1, ge=1, description="Token windows scored (>1 for long contracts)")
    num_nodes:    int
    num_edges:    int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> dict:
    """
    Liveness check — confirms predictor loaded and reports key model metadata.
    Does NOT read the checkpoint file (avoids expensive I/O on every call).
    """
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    predictor_loaded = predictor is not None

    if predictor_loaded:
        cfg = predictor._saved_cfg
        return {
            "status":            "ok",
            "predictor_loaded":  True,
            "checkpoint":        CHECKPOINT,
            "architecture":      predictor.architecture,
            "thresholds_loaded": predictor.thresholds_loaded,
            "tier_thresholds": {
                "confirmed":  predictor.tier_confirmed_threshold,
                "suspicious": predictor.tier_suspicious_threshold,
                "noteworthy": 0.10,
            },
            "model_epoch":  cfg.get("epoch",    "?"),
            "model_f1_val": cfg.get("best_f1", None),
        }

    return {
        "status":           "degraded",
        "predictor_loaded": False,
        "checkpoint":       CHECKPOINT,
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

    # T2-B: drift detection — update per request, check every DRIFT_CHECK_INTERVAL.
    # Track confirmed+suspicious counts as additional drift signal alongside graph features.
    if drift_detector is not None:
        try:
            drift_detector.update_stats({
                "num_nodes":        float(result["num_nodes"]),
                "num_edges":        float(result["num_edges"]),
                "confirmed_count":  float(len(result.get("confirmed",  []))),
                "suspicious_count": float(len(result.get("suspicious", []))),
            })
            request.app.state.request_count += 1
            if request.app.state.request_count % DRIFT_CHECK_INTERVAL == 0:
                drift_detector.check()
        except Exception as _drift_exc:
            logger.debug(f"Drift detector update failed: {_drift_exc}")

    logger.info(
        f"Complete — label={result['label']} "
        f"confirmed={len(result.get('confirmed', []))} "
        f"suspicious={len(result.get('suspicious', []))} "
        f"windows={result.get('windows_used', 1)}"
    )

    def _vuln_results(lst: list[dict]) -> list[VulnerabilityResult]:
        return [
            VulnerabilityResult(
                vulnerability_class=v["vulnerability_class"],
                probability=v["probability"],
                tier=v.get("tier"),
            )
            for v in lst
        ]

    return PredictResponse(
        label=result["label"],
        probabilities=result.get("probabilities", {}),
        confirmed=_vuln_results(result.get("confirmed",  [])),
        suspicious=_vuln_results(result.get("suspicious", [])),
        vulnerabilities=_vuln_results(result.get("vulnerabilities", [])),
        tier_thresholds=result.get("tier_thresholds", {}),
        thresholds=result["thresholds"],
        truncated=result["truncated"],
        windows_used=result.get("windows_used", 1),
        num_nodes=result["num_nodes"],
        num_edges=result["num_edges"],
    )
