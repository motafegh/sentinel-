# Change Log: 2026-04-30 — Project READMEs + ML Module Hardening

**Session date:** 2026-04-30  
**Commit message prefix:** `docs(readme)` + `feat(ml)` + `fix(ml)`  
**Status:** Committed to `claude/review-project-progress-X8YuG`

---

## Summary

Three independent workstreams completed in this session:
1. **Phase A** — Project-wide documentation: 5 README files created/expanded
2. **Phase B** — ML module hardening: predictor, API, and preprocessor improved against improvement ledger §5.6–5.8 and §6.4
3. **Phase C** — Truncation analysis tooling: `ml/scripts/analyse_truncation.py` written

---

## Phase A — Project-wide README Documentation

### Problem

The project had detailed spec files (SENTINEL-SPEC.md, improvement ledger) but no developer-facing README files explaining how to run, build, or understand any module. New contributors (and returning collaborators) had no starting point.

### What Was Done

Five README files were created or substantially expanded:

| File | Action | Contents |
|---|---|---|
| `README.md` (root) | Created | Architecture overview, data-flow ASCII diagram, module table with status, quick start, port map, critical constraints, test commands |
| `ml/README.md` | Created | Model architecture (GNN+CodeBERT+CrossAttention), node feature vector table, 10-class output table, dataset facts, checkpoint details, inference API contract, training instructions, DVC workflow, known limitations |
| `agents/README.md` | Created | LangGraph AuditState fields, risk routing logic, final report schema, usage examples, all 3 MCP server tool tables, RAG pipeline (FAISS+BM25+RRF), ingestion/feedback loop, env vars, test matrix |
| `zkml/README.md` | Created | ProxyMLP architecture and justification, full 7-step pipeline, step-by-step run commands, critical encoding details, artifact table, EZKL version notes, deployed addresses, Do Not Change warnings |
| `contracts/README.md` | Expanded | Contract architecture, AuditResult struct, prerequisites, build instructions, ZKMLVerifier dual-solc handling, forge test commands, Sepolia deploy flow, cast query examples, critical constraints, known limitations |

### gitignore Exception Required

The repo's `.gitignore` contained `*.md` (source-only design — excludes session notes, spec files, etc.). New READMEs were silently excluded. Fix: added three `!` exception lines so module-level READMEs are tracked:

```gitignore
!README.md
!*/README.md
!*/*/README.md
```

### Verification

```bash
git status | grep README   # should show 5 new README files staged/tracked
git log --oneline -3       # confirm commit
```

---

## Phase B — ML Module Hardening

### Scope

Improvement ledger §5.6, §5.7, §5.8, and §6.4 items implemented. No retraining required — all changes are inference-path improvements only.

### `ml/src/inference/api.py` (§5.6)

| Item | Change |
|---|---|
| `PREDICT_TIMEOUT` | `float(os.getenv("SENTINEL_PREDICT_TIMEOUT", "60"))` — env-configurable |
| `MAX_SOURCE_BYTES` | `1 * 1024 * 1024` — source size guard before preprocessing |
| `/health` response | Now includes `"thresholds_loaded": predictor.thresholds_loaded` |
| `/predict` size check | `len(body.source_code.encode()) > MAX_SOURCE_BYTES` → HTTP 413 |
| `logger.exception()` | Full traceback in catch-all (was `logger.error()`) |
| Timeout message | Includes actual value: `f"Inference timeout after {PREDICT_TIMEOUT:.0f} s."` |

### `ml/src/inference/predictor.py` (§5.7, §6.4)

| Item | Change |
|---|---|
| `thresholds_loaded` | `bool` flag set at `__init__`; exposed for `/health` |
| Per-class threshold warning | Logs `missing_classes` list when threshold JSON is incomplete |
| Metadata cross-check | Validates `fusion_output_dim` and `class_names` against checkpoint config dict at load time |
| `_warmup()` | Dummy forward pass at `__init__` end — minimal PyG graph + zero token tensors — catches CUDA/shape issues before first live request |
| Legacy mode warning | `legacy_binary` mode logs explicit production warning |
| Redundant import removed | `from torch_geometric.data import Batch as PyGBatch` removed; uses `Batch` directly |

Warmup implementation:

```python
def _warmup(self) -> None:
    dummy_x = torch.zeros(1, 8, dtype=torch.float32, device=self.device)
    dummy_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
    dummy_graph = Data(x=dummy_x, edge_index=dummy_edge_index)
    dummy_batch = Batch.from_data_list([dummy_graph]).to(self.device)
    dummy_ids = torch.zeros(1, 512, dtype=torch.long, device=self.device)
    dummy_mask = torch.zeros(1, 512, dtype=torch.long, device=self.device)
    dummy_mask[0, 0] = 1  # at least one real token to avoid empty masked mean
    with torch.no_grad():
        _ = self.model(dummy_batch, dummy_ids, dummy_mask)
```

### `ml/src/inference/preprocess.py` (§5.8, §6.4)

| Item | Change |
|---|---|
| Docstring path | `ast_extractor_v4_production.py` → `ml/data_extraction/ast_extractor.py` |
| `MAX_SOURCE_BYTES` | Class attribute `1 * 1024 * 1024`; size guard in `process_source()` |
| Temp file prefix | `safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name[:32])` |
| Edge type tracking | `edge_types: list[int]` tracked alongside `edges`; `_add_edge()` takes `etype: int` arg |
| `edge_attr` | `torch.tensor(edge_types, dtype=torch.long)` added to `Data()` object for offline/online parity |
| `assert` → `RuntimeError` | Explicit `RuntimeError` for tokenizer shape mismatches (python -O safe) |
| Error differentiation | `ImportError` → `RuntimeError` ("Slither not installed"); Slither compilation keywords → `ValueError`; other Slither/OS failures → `RuntimeError` |

### Breaking Changes

None. All changes are additive or fix silent failures. The `/predict` API schema and model architecture are unchanged.

### Verification

```bash
# predictor thresholds flag
cd ml
poetry run python -c "
from ml.src.inference.predictor import Predictor
p = Predictor(checkpoint='checkpoints/multilabel_crossattn_best.pt')
print('thresholds_loaded:', p.thresholds_loaded)  # True if thresholds JSON present
print('architecture:', p.architecture)             # cross_attention_lora
"

# API health response
curl http://localhost:8001/health
# expect: {"status":"ok","predictor_loaded":true,"architecture":"cross_attention_lora","thresholds_loaded":true,...}

# Size guard (pipe 1.1MB payload — expect 413)
python -c "print('pragma solidity ^0.8.0; contract X {}' + ' ' * 1100000)" | \
  curl -s -X POST http://localhost:8001/predict -H 'Content-Type: application/json' \
  -d '{"source_code": "'"$(python -c "print('pragma solidity ^0.8.0; contract X {}' + 'x' * 1100000)")"'"}'
```

---

## Phase C — Truncation Analysis Tooling

### Problem

No tooling existed to measure how much of the 68K-contract corpus is affected by CodeBERT's 512-token ceiling, or whether vulnerable functions fall after token 512 (making them invisible to the model).

### What Was Done

Created `ml/scripts/analyse_truncation.py`:

- Loads `multilabel_index.csv` for per-class label lookup
- Reads token `.pt` files: `sum(attention_mask)` = real token count; if count == MAX_TOKENS, truncation occurred
- Computes: overall truncation rate, per-class truncation rates, token count percentiles (p50/p90/p95/p99)
- Recommendation logic:
  - `< 5%` → `ACCEPT` — document the limitation, do not retrain
  - `5-25%` → `SLIDING-WINDOW` — implement sliding-window CodeBERT (§5.11 Option B)
  - `> 25%` → `LONG-CONTEXT MODEL` — StarCoder2-3B or DeepSeek-Coder-1.3B (§5.11 Option A)
- CLI: `--tokens-dir`, `--label-csv`, `--sample N`, `--output-json`, `--seed`
- Outputs JSON report or prints to stdout

### Usage

```bash
cd ml

# Full corpus analysis (slow — ~68K files):
poetry run python scripts/analyse_truncation.py \
    --tokens-dir  data/tokens/ \
    --label-csv   data/processed/multilabel_index.csv \
    --output-json scripts/truncation_report.json

# Quick sample (5000 contracts):
poetry run python scripts/analyse_truncation.py --sample 5000
```

### Decision Required After Running

The truncation rate will determine the next training milestone:
- **Accept:** proceed directly to ZKML pipeline execution
- **Sliding-window:** implement Option B in `preprocess.py` + `dual_path_dataset.py` before ZKML
- **Long-context:** architecture change + full retrain (major milestone)

---

## Files Changed This Session

| File | Action |
|---|---|
| `README.md` | Created |
| `ml/README.md` | Created |
| `agents/README.md` | Created |
| `zkml/README.md` | Created |
| `contracts/README.md` | Expanded |
| `.gitignore` | Added `!README.md`, `!*/README.md`, `!*/*/README.md` exceptions |
| `ml/src/inference/api.py` | Hardened (timeout, size limit, thresholds_loaded, logger.exception) |
| `ml/src/inference/predictor.py` | Hardened (thresholds_loaded, metadata validation, warmup, legacy warning) |
| `ml/src/inference/preprocess.py` | Hardened (edge_attr, safe prefix, assert→RuntimeError, error types) |
| `ml/scripts/analyse_truncation.py` | Created |
| `SENTINEL_ACTIVE_IMPROVEMENT_LEDGER_UPDATED_2026-04-28.md` | Updated §1, §2, §4, §5.6-5.8, §6.4, §9 |
| `docs/changes/2026-04-30-*.md` | Created (this file) |

---

## Spec References

- SENTINEL-SPEC §5.11 — Tokenization & context window improvements (3 solution options)
- SENTINEL-SPEC §6 — Critical constraints (fusion_output_dim=128 locked, MAX_TOKEN_LENGTH=512)
- SENTINEL-SPEC §3 — ML inference API contract (Track 3 PredictResponse schema)
- ADR-019 — `assert` → `RuntimeError`
- ADR-025 — CrossAttentionFusion output_dim=128
- Improvement ledger §5.6 — api.py improvements
- Improvement ledger §5.7 — predictor.py improvements
- Improvement ledger §5.8 — preprocess.py improvements
- Improvement ledger §6.4 — ML inference production hardening
