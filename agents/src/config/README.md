# Config — Externalized Decision Numbers

Pydantic-validated configuration that externalizes every decision number in the agent
pipeline (P1, 2026-06-24). Per Rule 5B: no threshold, weight, or confidence value may
be a constant buried in `.py` — all are versioned YAML, measurable, and logged.

## Files

| File | Purpose |
|------|---------|
| `schema.py` | `SentinelConfig` Pydantic model — full schema for all config fields |
| `loader.py` | `get_config()` singleton — loads `configs/verdicts_default.yaml` |

## `schema.py` — Config Schema

The top-level `SentinelConfig` model groups settings by concern:

```python
class SentinelConfig(BaseModel):
    routing:     RoutingConfig      # DEEP_THRESHOLDS, ROUTING_RULES, verdict cutoffs
    verdicts:    VerdictConfig      # CONFIRMED/LIKELY/DISPUTED/SAFE thresholds
    consensus:   ConsensusConfig    # per-class tool weights, ML_WEIGHT_SCALE
    confidence:  ConfidenceConfig   # Bayesian update factors (Slither/RAG boost/shrink)
    attribution: AttributionConfig  # LIME-style evidence floor percentages
    metrics:     MetricsConfig      # Fbeta β value, gate thresholds
```

All fields have explicit defaults so the schema is always valid even with a partial YAML.

## `loader.py` — `get_config()`

```python
from src.config import get_config

cfg = get_config()
threshold = cfg.routing.deep_thresholds["Reentrancy"]  # e.g. 0.35
```

`get_config()` is a module-level singleton — the YAML is parsed once at first call and
cached. It reads `configs/verdicts_default.yaml` relative to the `agents/` root.

**Thread-safe read:** The config object is immutable after construction (Pydantic
`model_config = ConfigDict(frozen=True)`). Nodes read from it concurrently without
locking.

## YAML Files

```
agents/configs/
  verdicts_default.yaml     L1 — hand-set baselines (current policy)
  reliability_v1.yaml       L2 — first measurement pass (historical)
  reliability_v2.yaml       L2 — second measurement pass (historical)
  reliability_v3.yaml       L3 — Bayesian-fitted per-tool reliability weights (active)
  reliability_justifications.yaml  Human-readable justifications for each L3 value
```

`reliability_v3.yaml` is the active L3 config. It was fitted via Bayesian shrinkage
(α=5) over the full eval corpus. The `verdict/reliability.py` module reads it at
runtime and falls back to L1 (`verdicts_default.yaml`) when the file is missing,
malformed, or has a wrong `schema_version`.

## Consumers

Every decision number in the following modules is loaded from config, not hardcoded:

| Module | Config section used |
|--------|-------------------|
| `routing.py` | `routing.*` — per-class thresholds, routing rules, verdict cutoffs |
| `consensus.py` | `consensus.*` — tool weights, `ML_WEIGHT_SCALE` |
| `confidence.py` | `confidence.*` — Bayesian update factors |
| `attribution.py` | `attribution.*` — evidence floor percentages |
| `eval/pipeline_metrics.py` | `metrics.*` — Fbeta β, gate thresholds |

## Maturity Levels (Rule 5B)

| Level | Description | Status |
|-------|-------------|--------|
| L0 | Hand-set constant buried in `.py` | Eliminated |
| L1 | Externalized in `verdicts_default.yaml` | All fields |
| L2 | Measured against baseline before change | Per `reliability_v1/v2.yaml` |
| L3 | Learned from data (Bayesian shrinkage) | `reliability_v3.yaml` active |

A decision number may only be changed when a before/after eval measurement justifies it
(Rule 5B). "It feels right" is not sufficient.
