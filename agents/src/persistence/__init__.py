"""Report persistence package — job-scoped, contained, atomic writes.

Replaces the old pattern of using ``contract_address`` as a filename
component (D2-AGT-002 / R0-REPORT-CONTAINMENT).
"""

from .legacy_adapter import find_legacy_hotspot, find_legacy_report
from .paths import (
    assert_contained,
    is_valid_job_id,
    job_report_dir,
    job_report_path,
    validate_address,
    validate_job_id,
)
from .report_writer import (
    HOTSPOT_PERSISTENCE_TOOL_KEY,
    PERSISTENCE_TOOL_KEY,
    REPORT_PERSISTENCE_TOOL_KEY,
    persist_hotspot,
    persist_report,
)

__all__ = [
    "PERSISTENCE_TOOL_KEY",
    "HOTSPOT_PERSISTENCE_TOOL_KEY",
    "REPORT_PERSISTENCE_TOOL_KEY",
    "assert_contained",
    "find_legacy_hotspot",
    "find_legacy_report",
    "is_valid_job_id",
    "job_report_dir",
    "job_report_path",
    "persist_hotspot",
    "persist_report",
    "validate_address",
    "validate_job_id",
]
