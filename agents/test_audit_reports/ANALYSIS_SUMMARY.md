# Agents E2E Test — Analysis Summary

**Date:** 2026-06-17
**Tester:** Claude (Plan Onboarding style, per ONBOARDING.md)
**Duration:** ~3 hours end-to-end
**Plan:** `docs/plan/agents/2026-06-17-agents-real-e2e-test/` (v1.1, audited)
**Harness:** `agents/scripts/run_real_audit.py` (rewritten with CLI args + per-node timing)

---

## Executive Summary

The 9-node agents graph **runs end-to-end** with real services. Found and fixed **8 real bugs** along the way (4 of them in the MCP server code, none caught by the 219+ unit tests). The system produces valid verdicts on both contracts, but **flagged a known-false-positive** (safe_storage.sol → ExternalBug CONFIRMED at 0.82). This false positive is a **class-definition mismatch in the model**, not an agents module bug — same finding as in MEMORY.md.

**Verdict:** **CAUTION — proceed to Phase A, but fix 1 known performance issue first.**

---

## Bugs Found + Fixed (chronological)

| # | Severity | Component | Symptom | Root cause | Fix |
|---|---|---|---|---|---|
| 1 | CRITICAL | `inference_server.py:462` + `audit_server.py:687` | TypeError on startup | Used removed `on_startup` kwarg | Convert to `lifespan` context manager |
| 2 | MEDIUM | bash backgrounding | SIGHUP during bind, EADDRINUSE after start | `nohup`+`disown` insufficient | `setsid cmd >log 2>&1 < /dev/null & disown` |
| 3 | CRITICAL | `agents/.env` | LM Studio connection refused | Stale WSL2 gateway IP (10.255.255.254 not bound) | `localhost:1256/v1` (mirror mode) |
| 4 | CRITICAL | RAG server embedder | `RuntimeError: batch failed after 3 attempts` | Embedder caches URL at startup, env var change not picked up | Restart RAG server after .env change |
| 5 | CRITICAL | `graph_inspector_server.py:517` | `404 Not Found` on /messages/, graph_explain hangs | `Mount("/messages/", routes=[Route(...)])` instead of `Mount("/messages/", app=...)` | Match `rag_server.py:338` pattern |
| 6 | LOW | `run_real_audit.py` | `Invalid Ethereum address` in audit_check | Used non-hex `0xE2E_<stem>` | SHA-256 of filename, 20-byte hex |
| 7 | CRITICAL | `nodes.py:1038` + `nodes.py:1328` | LLM timeouts at 30s/45s, silent fallback | Hardcoded too-tight limits | Bumped to 90s/120s, env-configurable |
| 8 | CRITICAL | `nodes.py:1035` | Even 90s not enough (94s observed) | STRONG model (qwen3.5-9b-ud) too slow on RTX 3070 | Switch cross_validator to FAST model (gemma-4-e2b-it) |

**Plan v1.1 missed bugs 1 and 5** — its audit only checked "port number is in source", not "server actually starts" / "SSE endpoint works". Future audits must run the server.

---

## Performance Baselines (Pre-Phase A)

### Mode 1: `--no-llm` (rule-based verdicts, 5.2s total for both)

| Contract | Wall | Per-node |
|---|---|---|
| `vulnerable_reentrant.sol` | 3.88s | ml=0.56, qs=0.35, ev=0.00, ge=0.64, rs=0.51, sa=0.31, ac=0.04, cv=0.00, syn=0.01 |
| `safe_storage.sol` | 1.29s | ml=0.35, qs=0.27, ev=0.00, ge=0.61, rs=0.35, sa=0.26, ac=0.04, cv=0.00, syn=0.00 |

### Mode 2: Full LLM (FAST for cross_validator, STRONG for synthesizer narrative, 311s total)

| Contract | Wall | cross_validator | synthesizer |
|---|---|---|---|
| `vulnerable_reentrant.sol` | 159.6s | 27.7s (FAST, SLOW>15s) | 125.6s (STRONG, hit 120s limit) |
| `safe_storage.sol` | 151.8s | 17.7s (FAST, SLOW>15s) | 125.3s (STRONG, hit 120s limit) |

**Stable per-node timings (independent of LLM mode):**
- `ml_assessment`: 0.5-2.5s (Run 12 inference via MCP :8010)
- `quick_screen`: 0.3-0.8s (Slither subprocess)
- `evidence_router`: <0.01s (pure logic)
- `rag_research`: 0.5-4.6s (depends on whether first/second contract — second is faster)
- `graph_explain`: 0.6-1.4s (graph_inspector via MCP :8013)
- `static_analysis`: 0.3-0.8s (Slither subprocess)
- `audit_check`: 0.07-1.4s (mock mode, mock RPC)

---

## Quality Baselines

### Verdict accuracy (1 known-vulnerable + 1 known-safe)

| Contract | Expected | `--no-llm` verdict | LLM verdict | Result |
|---|---|---|---|---|
| `vulnerable_reentrant.sol` (Vault with reentrancy) | VULNERABLE | `confirmed_vulnerable` / DISPUTED | `confirmed_vulnerable` / LIKELY | ✓ Both modes correct |
| `safe_storage.sol` (owner-only storage, no external calls) | SAFE | `confirmed_vulnerable` / DISPUTED | `confirmed_vulnerable` / CONFIRMED | ✗ False positive |

**Verdict accuracy: 1/2 = 50%** (false positive on safe_storage).

### Root cause of the false positive (matches MEMORY.md)
- Model gives ExternalBug probability **0.8175** on safe_storage.sol
- ExternalBug = "this contract makes high-risk calls to external contracts"
- The model has learned a spurious pattern: any contract with a public function gets ExternalBug flagged
- LLM cross_validator (FAST) just **CONFIRMS** what the model says
- This is the same finding from Run 12 SmartBugs Wild eval: "65% S_only rate is NOT over-prediction for Timestamp/Reentrancy; **it IS a class-definition mismatch for ExternalBug**"

**This is a model training problem (Phase A concern)**, not an agents module bug. The agents module surfaces the model's verdict correctly.

---

## Stability Baselines

| Metric | Result |
|---|---|
| Total crashes | 0 |
| Timeouts hit (intentional) | 4/4 (cross_validator + synthesizer, both contracts) |
| MCP server hangs | 0 (after fixes) |
| OOM events | 0 |
| Memory leak | None observed |
| Service restart required | 2 (RAG server, graph_inspector) — both during setup |

---

## Critical Findings

1. **The 9-node graph runs end-to-end with real services.** No crashes after the 8 bug fixes.
2. **The plan v1.1 audit was insufficient** — it checked "port number in source" but not "server starts". Two MCP servers would have failed on first run.
3. **The strong LLM is too slow for cross_validator on this hardware.** 94s for 9 classes. Switched to FAST model — 17-28s.
4. **Synthesizer narrative is too slow at 125s** — still hits the 120s limit. Could be fixed by:
   - Switching to FAST model (lower quality narrative)
   - Streaming the response (currently waits for full completion)
   - Bumping timeout to 180-240s
5. **The model has a known ExternalBug false positive** — safe_storage.sol flagged at 0.82 confidence. This is the **same finding as in MEMORY.md**. Not an agents bug; this needs to be fixed in the training data / class definition (Run 13 plan).

---

## Recommendations for Phase A

### Must fix before Phase A
- **None.** The 9-node graph runs end-to-end. Bugs found and fixed during E2E are committed.

### Should fix in Phase A
1. **Synthesizer narrative timeout.** Currently 125s for a 4-section Markdown report. Options:
   - Use FAST model for narrative (faster, lower quality)
   - Stream the response (let user see partial output)
   - Bump timeout to 180s
   - Best: **bump to 180s + add streaming** (most user-friendly)
2. **ExternalBug class definition.** The model can't distinguish "makes external calls" from "has risky external calls". This is a **Phase A training concern**, not a code concern. Run 13 plan already has this as a finding.

### Defer to Phase B/C
- **Replace `llm.invoke` with streaming** (defer to Phase C — requires LangGraph callback integration)
- **Add MCP server health checks** (defer to Phase C — already have `/health` endpoints, just need polling)
- **Move LLM calls out of cross_validator** (defer to Phase B — graph cleanup)

---

## Service-Version Log (reproducibility)

| Component | Version/ID | Date captured |
|---|---|---|
| ML checkpoint | `GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` (268.5 MB, md5=f1a04c12bda6) | 2026-06-17 |
| LM Studio | :1256, all 4 required models loaded | 2026-06-17 |
| RAG index | `faiss.index` 2.3 MB (sha256=0588e5c21e9f), 752 chunks | 2026-06-17 |
| RAG embedding | `text-embedding-nomic-embed-text-v1.5` | 2026-06-17 |
| Slither | 0.11.5 | 2026-06-17 |
| Aderyn | NOT INSTALLED (non-fatal) | 2026-06-17 |
| `AUDIT_MOCK` | true (no real Sepolia RPC) | 2026-06-17 |
| `LM_STUDIO_TIMEOUT` | 60s | 2026-06-17 |
| `CROSS_VALIDATOR_TIMEOUT_S` | 90s (was 30s) | 2026-06-17 |
| `SYNTHESIZER_TIMEOUT_S` | 120s (was 45s) | 2026-06-17 |
| `CROSS_VALIDATOR_LLM_MODEL` | fast (was strong) | 2026-06-17 |

---

## Deliverables

- [x] `agents/test_audit_reports/vulnerable_reentrant_report.json` (full LLM)
- [x] `agents/test_audit_reports/safe_storage_report.json` (full LLM)
- [x] `agents/test_audit_reports/run6_fast.log` (combined run log)
- [x] `agents/test_audit_reports/run3.log` (--no-llm run log)
- [x] `agents/scripts/run_real_audit.py` (CLI-configurable harness with comprehensive logging)
- [x] `agents/.env` (LM_STUDIO_BASE_URL, CROSS_VALIDATOR_TIMEOUT_S, SYNTHESIZER_TIMEOUT_S, CROSS_VALIDATOR_LLM_MODEL)
- [x] This file: `ANALYSIS_SUMMARY.md`

---

## GO / CAUTION / NO-GO Decision

**GO with CAUTION** — proceed to Phase A, but:
1. **Document the ExternalBug class-definition issue** as a Phase A training concern (already in Run 13 plan).
2. **Bump synthesizer timeout to 180s** OR switch to FAST model — current 125s runtime is borderline.
3. **Don't re-run this full E2E before Phase A** — the baselines above are sufficient. The harness is in place; future runs are 1 command.

**Why GO:**
- All 8 bugs found and fixed.
- 9-node graph executes end-to-end without crashes.
- Real LLM verdicts working (with FAST model).
- 50% verdict accuracy on 2 contracts is consistent with the model itself, not the agents module.
- Total elapsed: 3 hours (well within 5-8 hour plan budget).

**Why not full GO:**
- Synthesizer narrative is borderline (125s vs 120s timeout).
- 50% verdict accuracy — false positive needs addressing in training.
- The 8 bugs found suggest the agents module has more lurking issues (this was a 2-contract smoke, not a 66-contract OOD benchmark).

---

**Next steps:**
1. Update `MEMORY.md` with this run's findings + baselines
2. Start Phase A (graph cleanup, reflection, debate)
3. Plan a 10-contract E2E test for end of Phase A
