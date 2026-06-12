"""sentinel-data — SENTINEL data engineering module.

This package is the **data engineering layer** of the SENTINEL smart-contract
security oracle. It is a fully self-contained, reproducible pipeline that
turns raw Solidity source code from multiple public corpora (SolidiFI, DIVE,
DeFiHackLabs, SmartBugs Curated, BCCC) into model-ready training artifacts.

The package is organized as a five-stage directed pipeline plus three
cross-cutting support systems. Each stage reads what the previous stage
wrote, applies one transformation, and writes its outputs to a well-known
location under ``data/``::

    raw ──► [ingest] ──► preprocessed ──► [preprocess] ──► labeled ──► [label] ──►
    verified ──► [represent] ──► representations ──► [split] ──► splits

The five stage subpackages are:

* :mod:`sentinel_data.ingestion`     — pull and stage raw contract corpora
* :mod:`sentinel_data.preprocessing` — clean, normalize, deduplicate, segment
* :mod:`sentinel_data.labeling`      — map source labels to canonical taxonomy
* :mod:`sentinel_data.representation` — build PyG graphs and CodeBERT tokens
* :mod:`sentinel_data.splitting`    — train/val/test splits with leakage audit

The three cross-cutting subpackages are:

* :mod:`sentinel_data.verification`  — Go/No-Go gate that catches label noise
* :mod:`sentinel_data.registry`      — provenance, lineage, and dataset diffs
* :mod:`sentinel_data.analysis`      — post-hoc diagnostics and visualizations

**Architectural contract**: ``sentinel-data`` has a one-way dependency on
nothing in the ML training code (``sentinel-ml``). Training consumes
artifacts produced here; the training code never feeds back. This is what
makes the data pipeline independently testable, versionable, and reusable
across training experiments.

**Reproducibility contract**: every artifact is content-addressed by SHA-256
and stamped with ``schema_version`` + ``extractor_version`` in a JSON
sidecar. Two runs with the same config produce byte-identical outputs.

**Thin-adapter contract**: the representation subpackage re-exports the
PyG graph and CodeBERT token logic from the existing ``ml/`` package via
~10-line re-export files. This means bug fixes in the graph or token
extraction logic apply once (in ``ml/``) and automatically propagate to
the new path. Stage 7 of the data pipeline deletes the wrappers and
re-binds ``ml/`` to import from this package directly.

See :mod:`sentinel_data.cli` for the user-facing entry point.

Versioning
----------
This is ``sentinel-data`` v0.1.0. The version is intentionally separate
from the version of the artifacts it produces (``schema_version="v9"``,
``extractor_version="v2.0-thin-adapter"``); changing the package
implementation does NOT automatically invalidate cached representations
unless the explicit version stamps in the sidecar change.
"""
__version__ = "0.1.0"
