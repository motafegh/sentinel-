"""
api.py — SENTINEL FastAPI Inference Endpoint

SCHEMA VERSION: Three-tier suspicion output (2026-05-27)
  PredictResponse now includes:
    label           "safe" | "suspicious" | "confirmed_vulnerable"
    probabilities   {class: float}  full NUM_CLASSES-class vector (10 in Run 12, 9 in Run 13)
    confirmed       [{vulnerability_class, probability, tier="CONFIRMED"}, ...]
    suspicious      [{vulnerability_class, probability, tier="SUSPICIOUS"}, ...]
    vulnerabilities legacy alias for confirmed (backward compat)
    tier_thresholds {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}

CHECKPOINT: read from mlops_config.json (`checkpoint` field) or SENTINEL_CHECKPOINT
            env var. Defaults to Run 12 FINAL for forward compat.
  Pipeline verified FAIL=0 with compare_pipelines.py (2026-05-26).
  Override via SENTINEL_CHECKPOINT env var.

FIXES (2026-04-29):
    Bug 1 — import torch added.
    Bug 3 — v['class'] → v['vulnerability_class'].

FIXES (2026-06-15, Q4 MLOps Phase A.2):
    Doc only — replaced schema-agnostic phrasing to avoid hardcoding class count.
"""

from __future__ import annotations

import asyncio
import json
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


def _load_mlops_config() -> dict:
    """Load mlops_config.json if present. Env vars take precedence over file values."""
    config_path = os.getenv("SENTINEL_CONFIG", "ml/mlops_config.json")
    cp = Path(config_path)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return {}


_CONFIG = _load_mlops_config()

DRIFT_BASELINE_PATH: str = os.getenv(
    "SENTINEL_DRIFT_BASELINE",
    _CONFIG.get("drift_baseline", "ml/data/drift_baseline.json"),
)
# Run a KS check every N requests (balance: lower = more responsive, higher = cheaper).
DRIFT_CHECK_INTERVAL: int = int(os.getenv(
    "SENTINEL_DRIFT_CHECK_INTERVAL",
    str(_CONFIG.get("drift_check_interval", 50)),
))

# ---------------------------------------------------------------------------
# Prometheus — custom gauges
# ---------------------------------------------------------------------------
_gauge_model_loaded  = Gauge("sentinel_model_loaded",      "1 if the predictor is loaded, 0 otherwise")
_gauge_gpu_mem_bytes = Gauge("sentinel_gpu_memory_bytes",  "Current GPU memory allocated (bytes)")

CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    _CONFIG.get(
        "checkpoint",
        "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
    ),
)

# Inference timeout in seconds — override via SENTINEL_PREDICT_TIMEOUT env var.
PREDICT_TIMEOUT: float = float(os.getenv(
    "SENTINEL_PREDICT_TIMEOUT",
    str(_CONFIG.get("predict_timeout", 60)),
))

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

    # P5 (2026-06-26): Enable deterministic mode if requested
    # This ensures reproducible inference across runs (for ZK proof generation)
    _deterministic_mode = os.getenv("SENTINEL_DETERMINISTIC", "").lower() in ("1", "true", "yes")
    if _deterministic_mode:
        logger.info("SENTINEL_DETERMINISTIC mode enabled — setting torch deterministic algorithms")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

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

class HotspotsRequest(BaseModel):
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


class FunctionHotspot(BaseModel):
    fn_name:   str         = Field(..., description="Canonical function name from Slither AST")
    node_id:   int         = Field(..., description="PyG node index (0-based, stable per contract)")
    score:     float       = Field(..., ge=0.0, le=1.0, description="Normalised GNN embedding norm [0,1]")
    lines:     list[int]   = Field(default_factory=list, description="Source line numbers")
    node_type: str         = Field(..., description="FUNCTION | MODIFIER | FALLBACK | RECEIVE | CONSTRUCTOR")


class HotspotsResponse(BaseModel):
    hotspots:     list[FunctionHotspot] = Field(..., description="Top-20 function nodes by GNN attention score")
    hotspot_stats: dict                 = Field(..., description="total_function_nodes, num_nodes, attention_source")
    # Also include the full prediction so callers get ML + hotspots in one round-trip
    label:         str
    probabilities: dict[str, float]
    confirmed:     list[VulnerabilityResult] = Field(default_factory=list)
    suspicious:    list[VulnerabilityResult] = Field(default_factory=list)


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

    # Full NUM_CLASSES-class probability vector — always present, never filtered.
    # NUM_CLASSES is read from the loaded checkpoint config (10 in Run 12, 9 in Run 13).
    # Enables agents to see all signal regardless of tier thresholds.
    probabilities: dict[str, float] = Field(..., description="Full per-class probability vector")

    # Tiered findings — sorted descending by probability within each tier.
    confirmed:  list[VulnerabilityResult] = Field(default_factory=list, description="prob >= tier_confirmed_threshold (default 0.55)")
    suspicious: list[VulnerabilityResult] = Field(default_factory=list, description="tier_suspicious_threshold <= prob < tier_confirmed_threshold")

    # Legacy field — backward compat alias for confirmed.
    # Old consumers reading vulnerabilities get CONFIRMED classes only.
    vulnerabilities: list[VulnerabilityResult] = Field(default_factory=list)

    # Tier boundaries used — allows agents to interpret tiers without hardcoding thresholds.
    # "confirmed" is a per-class list when per-class thresholds are loaded; "suspicious"
    # and "noteworthy" are scalar defaults.
    tier_thresholds: dict[str, float | list[float]] = Field(default_factory=dict)

    thresholds:   list[float] = Field(..., description="Per-class tuned decision thresholds")
    truncated:    bool
    windows_used: int = Field(default=1, ge=1, description="Token windows scored (>1 for long contracts)")
    num_nodes:    int
    num_edges:    int

    # D4 (WS3, 2026-06-22): per-eye auxiliary predictions as discountable CLUES.
    # Each eye: {class_name: probability}. Only present for four_eye architectures.
    eye_predictions: dict[str, dict[str, float]] | None = None

    # P5 (2026-06-26): model hash for reproducibility tracking.
    # SHA-256 of the checkpoint file — stable across restarts unless checkpoint is replaced.
    model_hash: str = Field(..., description="SHA-256 hash of the checkpoint file (64 hex chars)")


class FusionEmbeddingResponse(BaseModel):
    fusion_embedding: list[float] = Field(..., min_length=128, max_length=128,
                                          description="128-dim CrossAttentionFusion output — ZKML circuit input")
    num_nodes: int
    num_edges: int
    model_hash: str = Field(..., description="SHA-256 hash of the teacher checkpoint file (64 hex chars)")
    windows_used: int = Field(default=1, ge=1)


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
            "model_hash":   predictor.model_hash,
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
        # Rule 5C (CLAUDE.md, 2026-06-25): surface the actual failure reason
        # to the caller, not a generic "Inference failed." that hides whether
        # the cause was a solc compile error, a model OOM, a Slither AST
        # bug, or a network timeout. The exception type + message is the
        # minimum information the audit pipeline needs to surface the
        # failure correctly (without it, 22 specific contracts silently
        # returned "unknown" labels).
        logger.exception(f"Inference error: {exc}")  # exception() logs full traceback
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {str(exc)[:500]}",
        )

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
        eye_predictions=result.get("eye_predictions"),
        model_hash=predictor.model_hash,
    )


@app.post("/hotspots", response_model=HotspotsResponse)
async def hotspots(request: Request, body: HotspotsRequest) -> HotspotsResponse:
    """
    GNN attention hotspots + ML prediction in one round-trip.

    Returns per-function hotspot scores derived from GNN node embedding norms —
    the real model signal, not Slither-proxy scoring.  Higher score = the GNN
    concentrated more structural attention on that function.

    Intended for graph_inspector_server Phase 2: replaces the Slither-proxy
    hotspot scoring with ground-truth model signal, enabling CPG-sliced LLM
    reasoning on the functions the model actually flagged as interesting.

    Response includes:
      hotspots       — top-20 functions, sorted by score desc
      hotspot_stats  — total_function_nodes, num_nodes, attention_source
      label/probabilities/confirmed/suspicious — full ML result (no extra round-trip needed)
    """
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    source_bytes = len(body.source_code.encode())
    if source_bytes > MAX_SOURCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"source_code too large ({source_bytes:,} bytes > {MAX_SOURCE_BYTES:,} limit).",
        )

    try:
        logger.info(f"Hotspots request — {len(body.source_code)} chars")
        result: dict = await asyncio.wait_for(
            asyncio.to_thread(predictor.predict_with_hotspots, body.source_code),
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
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=413, detail="Contract too large for GPU memory.")
    except Exception as exc:
        logger.exception(f"Hotspot extraction error: {exc}")
        raise HTTPException(status_code=500, detail="Hotspot extraction failed.")

    def _vuln_results(lst: list[dict]) -> list[VulnerabilityResult]:
        return [
            VulnerabilityResult(
                vulnerability_class=v["vulnerability_class"],
                probability=v["probability"],
                tier=v.get("tier"),
            )
            for v in lst
        ]

    logger.info(
        f"Hotspots complete — label={result['label']} "
        f"hotspots={len(result.get('hotspots', []))} "
        f"function_nodes={result.get('hotspot_stats', {}).get('total_function_nodes', 0)}"
    )

    return HotspotsResponse(
        hotspots=[
            FunctionHotspot(
                fn_name=h["fn_name"],
                node_id=h["node_id"],
                score=h["score"],
                lines=h.get("lines", []),
                node_type=h.get("node_type", "FUNCTION"),
            )
            for h in result.get("hotspots", [])
        ],
        hotspot_stats=result.get("hotspot_stats", {}),
        label=result["label"],
        probabilities=result.get("probabilities", {}),
        confirmed=_vuln_results(result.get("confirmed", [])),
        suspicious=_vuln_results(result.get("suspicious", [])),
    )


@app.post("/fusion-embedding", response_model=FusionEmbeddingResponse)
async def fusion_embedding(request: Request, body: PredictRequest) -> FusionEmbeddingResponse:
    """
    Return the 128-dim CrossAttentionFusion embedding for ZKML proof generation.

    This is the ZK boundary: the proxy model maps this 128-dim vector to
    10 class scores, and the EZKL circuit proves proxy(fusion_128) = scores[10].
    Does NOT return probabilities or verdicts — just the raw fusion vector.
    """
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    source_bytes = len(body.source_code.encode())
    if source_bytes > MAX_SOURCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"source_code too large ({source_bytes:,} bytes > {MAX_SOURCE_BYTES:,} limit).",
        )

    try:
        result: dict = await asyncio.wait_for(
            asyncio.to_thread(predictor.predict_fusion_embedding, body.source_code),
            timeout=PREDICT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Inference timeout after {PREDICT_TIMEOUT:.0f} s.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=413, detail="Contract too large for GPU memory.")
    except Exception as exc:
        logger.exception(f"Fusion embedding error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Fusion embedding failed: {type(exc).__name__}: {str(exc)[:500]}",
        )

    return FusionEmbeddingResponse(
        fusion_embedding=result["fusion_embedding"],
        num_nodes=result["num_nodes"],
        num_edges=result["num_edges"],
        model_hash=result["model_hash"],
        windows_used=result["windows_used"],
    )
