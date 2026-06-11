# Audit Report — Ingestion Module (Stage 1)

**Scope:** `ingestion/__init__.py`, `ingest.py`, `manifest.py`, `freshness.py`, `label_folderize.py`, `connectors/*`
**Plan Reference:** `02_stage_1_ingest_preprocess.md` (D-1.1 through D-1.3)

---

## Executive Summary

| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| D-1.1 Connector-per-family | ✅ | | |
| D-1.2 Manifest append-only | | ⚠️ | |
| D-1.3 Pin enforcement | | | ❌ |
| Security | | ⚠️ | |
| Code quality | | ⚠️ | |
| Test coverage | | ⚠️ | |
| **TOTAL** | **5** | **7** | **1** |

---

## 1. `ingestion/__init__.py`

| Check | Status | Detail |
|-------|--------|--------|
| Package marker exists | PASS | |
| Public API re-exports | **WARN** | No `__all__`; consumers must import from submodules directly |

**Finding:** Should re-export `IngestionManifest`, `ingest_source`, `ingest_all` for clean public API.

---

## 2. `ingest.py` (107 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Source merging (critical_path + additive) | PASS | 15-21 | Correctly merges 3 dicts |
| Dry-run support | PASS | 69-73 | |
| Connector instantiation | PASS | 75 | Factory pattern works |
| Manifest construction | PASS | 78-89 | All D-1.2 fields present |
| Error on disabled source | PASS | 61-62 | |
| Error on unknown source | PASS | 63 | |

### FAIL: D-1.3 Pin enforcement missing

`ingest_source()` at line 78 creates the manifest without checking `source_cfg.pin` is non-empty.

**Plan requirement:** "Every connector requires a pin in config.yaml. Wildcard versions ('HEAD', 'latest', 'main') are rejected at config-load time."

**Current behavior:** Config has `pin: ""` for 20+ sources. The code happily ingests them with empty pins.

**Impact:** HIGH — breaks reproducibility guarantee.

**Fix:**
```python
# In ingest.py, before line 78:
if not source_cfg.pin and source_cfg.connector == "git":
    raise ConnectorError(
        f"Source '{name}' has no pin. D-1.3 requires a pinned version for reproducibility."
    )
```

### WARN: Manifest overwrite (D-1.2)

`manifest.save(raw_dir / "ingestion_manifest.json")` at line 89 overwrites the previous manifest.

**Plan requirement:** "The manifest is append-only — past ingestions are never deleted, they are versioned."

**Current behavior:** A re-ingest replaces the previous manifest silently. The audit trail is lost.

**Fix options:**
1. Write to `ingestion_manifest_v<N>.json` with incrementing version
2. Maintain a `manifest_history.jsonl` append-only log
3. Use git-tracked manifest directory

---

## 3. `manifest.py` (98 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| FileRecord fields | PASS | 19-22 | path, sha256, size_bytes |
| IngestionManifest fields | PASS | 26-36 | All D-1.2 fields present |
| SHA-256 computation | PASS | 73-78 | Chunked (64KB), correct |
| save/load roundtrip | PASS | 40-50 | |
| verify() re-checks hashes | PASS | 54-68 | |
| Missing file detection | PASS | 62-63 | |
| `load_manifest` path | PASS | 92-93 | Correct path construction |
| Type hints | PASS | | Complete |
| Docstrings | PASS | 1-6 | Good module docstring |

**No significant issues.** Clean, correct implementation.

---

## 4. `freshness.py` (119 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Git upstream check | PASS | 65-76 | `git ls-remote` correct |
| Slither version check | PASS | 79-93 | PyPI + importlib.metadata |
| Report generation | PASS | 17-62 | |

### WARN: `datetime.utcnow()` deprecated

Line 24 uses `datetime.utcnow()`. Deprecated since Python 3.12; should use `datetime.now(timezone.utc)`.

### WARN: Pin staleness comparison bug

```python
# Line 40
if upstream and pin and not upstream.startswith(pin):
```

This checks if the upstream HEAD SHA *starts with* the pinned value. For full 40-char SHA pins, this works. But:
- If `pin` is a branch name (e.g., `main`), `upstream.startswith("main")` is always False — false positive STALE.
- The comparison is asymmetric — should be `pin.startswith(upstream[:len(pin)])` or exact match.

**Severity:** MEDIUM — false STALE alerts for branch-name pins.

### WARN: HF/Zenodo freshness unchecked

Lines 47 return "UNCHECKED" for HF and Zenodo sources. The plan requires checking these too.

---

## 5. `label_folderize.py` (172 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Symlink creation | PASS | 166 | Relative symlinks |
| Flat-to-`__source__` move | PASS | 103-123 | Idempotent |
| Multi-label handling | PASS | 147-148 | Correctly counts |
| Missing source file | PASS | 156-158 | Silent skip (correct) |
| Idempotency | PASS | 164 | Checks `link_path.exists()` |
| Type hints | PASS | | Complete |
| Docstring | PASS | 1-38 | Excellent, detailed |

**No significant issues.** Well-implemented utility module.

---

## 6. `connectors/__init__.py` (34 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Registry has 5+2 types | PASS | git, hf, zenodo, etherscan, manual + audit_report, rekt_scraper |
| Factory returns instance | PASS | `return cls()` |
| Unknown type error | PASS | Clear error message |
| D-1.1 compliance | PASS | One connector per family |

**Clean implementation.**

---

## 7. `connectors/base.py` (94 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| ABC interface | PASS | 42-61 | `@abstractmethod _pull` |
| SourceConfig fields | PASS | 13-24 | Complete |
| PullResult fields | PASS | 28-35 | Complete |
| find_sol_files include/exclude | PASS | 64-94 | Correctly handles both |

### WARN: `datetime.utcnow()` deprecated

Line 56: `datetime.datetime.utcnow().isoformat() + "Z"`. Should use `datetime.datetime.now(datetime.timezone.utc).isoformat()`.

### WARN: `find_sol_files` path traversal

Line 87: `rglob("*.sol")` — no restriction on symlink targets. If a symlink points outside the repo, files outside the repo could be collected. Low risk since the repo directory is controlled.

---

## 8. `connectors/git_connector.py` (84 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Shallow clone (no pin) | PASS | 65 | `--depth 1` |
| Full clone (pinned) | PASS | 60 | Full history for audit |
| Checkout pin | PASS | 61 | |
| `_current_commit` | PASS | 68-74 | Uses `check=True` |
| Error reporting | PASS | 77-83 | Includes stdout/stderr |
| Empty URL check | PASS | 25-26 | |

### WARN: `post_clone_cmd.split()` fragility

Line 38: `_run(post_cmd.split(), cwd=repo_dir)` — `str.split()` doesn't handle quoted arguments correctly. Should use `shlex.split()`.

### WARN: No subprocess timeout

Line 78: `subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)` — no `timeout` parameter. If `git clone` hangs (network issue, auth prompt), the pipeline hangs forever.

**Fix:** Add `timeout=300` (5 min).

---

## 9. `connectors/manual_connector.py` (187 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Zip extraction | PASS | 159-187 | |
| Zip-slip protection | PASS | 176-178 | `str(target.resolve()).startswith(str(dest.resolve()))` |
| macOS metadata stripping | PASS | 171 | `__MACOSX/`, `.DS_Store` |
| Symlink materialization | PASS | 149-150 | |
| Copy materialization | PASS | 151-152 | `dirs_exist_ok=True` |
| Glob resolution | PASS | 113-120 | |
| Staging path validation | PASS | 122-125 | |
| Materialize mode validation | PASS | 128-131 | |
| Idempotent re-runs | PASS | 63-66 | |
| Empty sol_files check | PASS | 75-80 | |

### WARN: Unused imports

Lines 34, 38: `fnmatch` and `subprocess` imported but never used.

---

## 10–12. Stub Connectors (HF, Etherscan, Zenodo)

All three raise `NotImplementedError` with helpful messages. Correct for Stage 1 — they're registered in the factory (D-1.1) but non-functional.

---

## Security Audit

| Check | Status | File:Line | Notes |
|-------|--------|-----------|-------|
| Zip-slip (path traversal in zip) | PASS | `manual_connector.py:176` | Properly defended |
| Command injection via `post_clone_cmd` | WARN | `git_connector.py:38` | `split()` on untrusted string; config is trusted but no sandboxing |
| Path traversal in manifest verify | WARN | `manifest.py:61` | `raw_dir / rec.path` — if manifest is tampered, path could escape |
| Symlink attacks in label_folderize | WARN | `label_folderize.py:166` | Relative symlinks are safe, but no validation target stays within repo_dir |
| Subprocess injection (git) | PASS | `git_connector.py:60,65` | Uses list args, not shell=True |
| Empty pin (reproducibility) | FAIL | D-1.3 | See above |

---

## Test Coverage Gaps

| Gap | Severity | Plan Ref |
|-----|----------|----------|
| No test for `ingest_source()` | HIGH | Task 1.8 |
| No test for `ingest_all()` | MEDIUM | Task 1.8 |
| No test for `freshness.py` | MEDIUM | Task 1.7 |
| No test for pin validation (D-1.3) | HIGH | Task 1.8 |
| No test for manifest versioning/append-only | MEDIUM | Task 1.1 |
| No test for `git_connector._clone()` | MEDIUM | Task 1.8 |
| No integration test for full ingest pipeline | HIGH | Task 1.8 |

---

## Actionable Fix Priority

| Priority | Finding | File:Line | Effort |
|----------|---------|-----------|--------|
| **P0** | Add pin validation (D-1.3) | `ingest.py:78` | 5 lines |
| **P1** | Manifest versioning (D-1.2) | `manifest.py:40-43` | 30 lines |
| **P1** | Add subprocess timeout | `git_connector.py:78` | 1 line |
| **P2** | Fix `freshness.py` pin comparison | `freshness.py:40` | 5 lines |
| **P2** | Replace `datetime.utcnow()` | `base.py:56`, `freshness.py:24` | 4 lines |
| **P2** | Use `shlex.split()` for post_clone_cmd | `git_connector.py:38` | 2 lines |
| **P3** | Remove unused imports | `manual_connector.py:34,38` | 2 lines |
| **P3** | Add missing tests | `tests/test_ingestion/` | 100+ lines |
