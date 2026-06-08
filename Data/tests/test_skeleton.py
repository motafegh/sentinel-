"""Stage 0 smoke tests — verify the skeleton installs and the CLI works.

These tests must pass from a clean `poetry install` with no ML deps.
They are the gate for 'Stage 0 complete' and run in every subsequent CI.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = DATA_DIR / "config.yaml"


# ── 1. All 9 subpackages are importable ───────────────────────────────────────

def test_subpackages_importable():
    subpackages = [
        "sentinel_data",
        "sentinel_data.ingestion",
        "sentinel_data.preprocessing",
        "sentinel_data.representation",
        "sentinel_data.labeling",
        "sentinel_data.verification",
        "sentinel_data.splitting",
        "sentinel_data.registry",
        "sentinel_data.analysis",
        "sentinel_data.export",
    ]
    for pkg in subpackages:
        __import__(pkg)


# ── 2. Representation stub exposes correct v9 constants ───────────────────────

def test_schema_version_is_v9():
    from sentinel_data.representation import FEATURE_SCHEMA_VERSION
    assert FEATURE_SCHEMA_VERSION == "v9", (
        f"Expected 'v9', got '{FEATURE_SCHEMA_VERSION}'. "
        "The proposal §2 says v8 but the live schema is v9 (verified 2026-06-08)."
    )


def test_schema_dimensions_v9():
    from sentinel_data.representation import NODE_FEATURE_DIM, NUM_EDGE_TYPES, NUM_NODE_TYPES, _MAX_TYPE_ID
    assert NODE_FEATURE_DIM == 12
    assert NUM_NODE_TYPES == 14
    assert NUM_EDGE_TYPES == 12
    assert _MAX_TYPE_ID == 13.0


def test_stub_flag_is_true():
    from sentinel_data.representation import STUB
    assert STUB is True, "STUB must be True in Stage 0 — removed only after Stage 7 seam swap"


def test_graph_extractor_raises_not_implemented():
    from sentinel_data.representation.graph_extractor import extract_contract_graph
    with pytest.raises(NotImplementedError):
        extract_contract_graph("dummy.sol")


def test_graph_extraction_error_importable():
    from sentinel_data.representation.graph_extractor import GraphExtractionError
    assert issubclass(GraphExtractionError, Exception)


# ── 3. config.yaml is valid and has expected structure ────────────────────────

def test_config_yaml_is_valid():
    assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_config_has_scabench_enabled():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sources = cfg.get("sources", {})
    assert "scabench" in sources, "scabench must be in sources"
    assert sources["scabench"]["enabled"] is True
    # The scabench entry must point at scabench-org/scabench, NOT SWC-registry
    url = sources["scabench"].get("url", "")
    assert "scabench-org/scabench" in url, (
        f"scabench URL must be scabench-org/scabench, got {url!r}. "
        "The original skeleton pointed at SmartContractSecurity/SWC-registry which is wrong."
    )


def test_config_mlflow_uri():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    uri = cfg["pipeline"]["mlflow"]["uri"]
    assert uri == "sqlite:///mlruns.db", (
        f"MLflow URI must be 'sqlite:///mlruns.db', got '{uri}'. "
        "The file:/// backend is corrupt (experiments 1/2/3)."
    )


def test_config_solc_baseline_versions():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    versions = cfg["pipeline"]["solc"]["baseline_versions"]
    assert len(versions) == 6, f"Expected 6 baseline solc versions, got {len(versions)}"


def test_config_deferred_bccc_exists():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    deferred = cfg.get("deferred_sources", {})
    assert "bccc" in deferred, "BCCC must be in deferred_sources (not regular sources)"


def test_config_has_all_tier1_sources():
    """The 8 Tier-1 (gold) sources from the binding proposal §6 must all be in config."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sources = cfg.get("sources", {})
    expected_tier1 = [
        "scabench", "smartbugs_curated", "solidifi", "dive",
        "forge", "web3bugs", "defihacklabs", "solidity_defi_vulns",
        # Bastet is also Tier-1 (binding proposal has 8 gold sources; we have 9 here
        # because the friend suggested DIVE+FORGE+SolidiFI added on top of the 5 original)
    ]
    # Note: bastet is also Tier-1
    expected_tier1.append("bastet")
    for source in expected_tier1:
        assert source in sources, f"Tier-1 source {source!r} must be in sources"
        assert sources[source].get("tier") == 1, (
            f"{source} must have tier=1, got tier={sources[source].get('tier')!r}"
        )


def test_config_forge_has_real_url():
    """FORGE URL must be filled in (friend provided https://github.com/shenyimings/FORGE-Artifacts)."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    forge = cfg.get("sources", {}).get("forge", {})
    url = forge.get("url", "")
    assert "shenyimings/FORGE-Artifacts" in url, (
        f"FORGE URL must be https://github.com/shenyimings/FORGE-Artifacts, got {url!r}"
    )


def test_config_source_count():
    """We expect 20 enabled sources across all tiers + BCCC deferred + 2 v1 extras = 23 entries."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sources = cfg.get("sources", {})
    assert len(sources) >= 18, (
        f"Expected at least 18 sources in config, got {len(sources)}. "
        "The v2 build has 8 Tier-1 + 2 Tier-2 clean + 3 Tier-2 silver/gold + 3 Tier-3 + 2 Tier-4 = 18+."
    )


def test_config_sentinel_ml_not_referenced():
    with open(CONFIG_PATH) as f:
        content = f.read()
    assert "sentinel-ml" not in content.lower(), (
        "config.yaml must not reference sentinel-ml (one-way dependency)"
    )


# ── 4. CLI smoke tests ────────────────────────────────────────────────────────

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sentinel_data.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(DATA_DIR),
    )


def test_cli_help_exits_zero():
    result = _run_cli("--help")
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"


def test_cli_help_lists_all_stages():
    result = _run_cli("--help")
    expected_stages = [
        "ingest", "preprocess", "represent", "label",
        "verify", "split", "register", "analyze", "export",
    ]
    for stage in expected_stages:
        assert stage in result.stdout, f"Stage '{stage}' not found in --help output"


def test_cli_run_dry_run_lists_all_stages():
    result = _run_cli("run", "--dry-run")
    assert result.returncode == 0
    for stage in ["ingest", "preprocess", "represent", "label", "verify", "split", "register", "analyze", "export"]:
        assert stage in result.stdout


def test_cli_run_from_stage_verify():
    result = _run_cli("run", "--dry-run", "--from-stage", "verify")
    assert result.returncode == 0
    output = result.stdout
    # stages before verify must NOT appear
    for early_stage in ["ingest", "preprocess", "represent", "label"]:
        assert early_stage not in output, f"Stage '{early_stage}' should be skipped when --from-stage=verify"
    # verify and later must appear
    for late_stage in ["verify", "split", "register", "analyze", "export"]:
        assert late_stage in output


def test_cli_ingest_help():
    result = _run_cli("ingest", "--help")
    assert result.returncode == 0
    assert "--source" in result.stdout


# ── 5. Package boundary: sentinel-ml must not appear in pyproject.toml ────────

def test_no_sentinel_ml_dependency():
    pyproject = DATA_DIR / "pyproject.toml"
    content = pyproject.read_text()
    assert "sentinel-ml" not in content, (
        "sentinel-data must not depend on sentinel-ml. One-way dependency rule violated."
    )


# ── 6. Directory structure checks ─────────────────────────────────────────────

def test_data_subdirs_exist():
    for subdir in ["raw", "preprocessed", "representations", "labels", "verification",
                   "splits", "registry", "exports", "analysis"]:
        p = DATA_DIR / "data" / subdir
        assert p.exists(), f"data/{subdir}/ is missing"


def test_legacy_bccc_deep_dive_exists():
    legacy = DATA_DIR / "docs" / "legacy" / "bccc_deep_dive"
    assert legacy.exists(), "docs/legacy/bccc_deep_dive/ must exist (moved from Data/Deep_Dive/)"


def test_v14_csv_reachable():
    csv = DATA_DIR / "docs" / "legacy" / "bccc_deep_dive" / \
          "Phase5_LabelVerification_2026-06-08" / "outputs" / "contracts_clean_v1.4.csv"
    assert csv.exists(), f"contracts_clean_v1.4.csv not found at {csv}"


def test_schema_constants_md_exists():
    md = DATA_DIR / "sentinel_data" / "representation" / "_schema_constants.md"
    assert md.exists()


def test_schema_version_registry_json_exists():
    j = DATA_DIR / "sentinel_data" / "representation" / "_schema_version_registry.json"
    assert j.exists()
    import json
    with open(j) as f:
        reg = json.load(f)
    assert reg["active"] == "v9"


def test_dockerfile_exists_and_uses_bookworm():
    df = DATA_DIR / "docker" / "Dockerfile.data"
    assert df.exists()
    content = df.read_text()
    assert "python:3.12.1-bookworm" in content, (
        "Dockerfile must use python:3.12.1-bookworm (not slim — slither needs build-essential)"
    )
    # Reject python:3.12.1-slim or python:3.12-slim-* as the FROM line.
    # (The word "slim" may appear in comments saying "NOT slim" — that is fine.)
    import re
    from_lines = [ln for ln in content.splitlines() if ln.strip().upper().startswith("FROM ")]
    assert from_lines, "Dockerfile has no FROM line"
    for line in from_lines:
        assert "slim" not in line, f"Dockerfile FROM uses slim image: {line!r}"
