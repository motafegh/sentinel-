"""Stage 6 — probe_dataset: re-export from verification (D-6.6).

The probe dataset is a hand-curated set of contracts used by the model
interpretability suite. The single source of truth is
`sentinel_data.verification.probe_dataset` (D-4.7); this module is a
re-export so callers can `from sentinel_data.analysis.probe_dataset import ...`
without coupling to the verification module.
"""
from sentinel_data.verification.probe_dataset import (  # noqa: F401
    ClassProbeBucket,
    ProbeDataset,
    ProbeEntry,
    build_probe_dataset,
)


__all__ = [
    "ClassProbeBucket",
    "ProbeDataset",
    "ProbeEntry",
    "build_probe_dataset",
]
