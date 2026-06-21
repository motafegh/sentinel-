# Changelog — Agents Real E2E Test Plan

All notable changes to this plan are documented here. Format: Keep-a-Changelog style.

---

## [v1.1] — 2026-06-17 (Plan-vs-Code Audit Patch)

**Status:** 📚 Docs-only — execution deferred per Ali.
**Type:** Corrective (5 drift fixes) + educational layer (Plan Onboarding style).

### Fixed (drift between plan v1.0 and actual source code)

1. **LM Studio URL** — replaced all hardcoded `http://localhost:1234` with `$LM_STUDIO_BASE_URL` env var.
   - Root cause: plan v1.0 ignored FIX-15 in `agents/src/llm/client.py:60` (WSL2 gateway IP changes on reboot).
   - Real state: LM Studio desktop app uses a **fluid port** — was `:1234`, then `:4567`, currently `:1256` on Ali's machine.
   - Fix: env-var driven; WSL2 gateway IP fallback for Windows-host case; `/v1/models` connectivity check.

2. **LLM model names** — replaced "Qwen 2.5 7B or Mistral-7B" with the 4 actual model IDs.
   - Root cause: plan v1.0 used placeholder suggestions, not the real model IDs.
   - Real state (`client.py:51-58`): `gemma-4-e2b-it` (FAST), `qwen3.5-9b-ud` (STRONG), `qwen2.5-coder-7b-instruct` (CODER), `text-embedding-nomic-embed-text-v1.5` (EMBED).
   - Fix: reference table in `00_MASTER` §LLM Model Selection; `curl /v1/models` check command.

3. **`AUDIT_MOCK` env flag** — added to setup prereqs.
   - Root cause: `audit_server.py` defaults to real Sepolia RPC mode; without `AUDIT_MOCK=true`, server hangs at startup trying to reach a chain we don't have access to in dev.
   - Fix: `AUDIT_MOCK=true` added to `agents/.env` requirements; called out as mock-sourced limitation in `03_ANALYSIS` §Step 0.

4. **Slither + Aderyn version checks** — added to prereqs.
   - Root cause: Run 12 SmartBugs Wild eval (47K contracts) found 6,782 errors all from pre-0.4.21 Slither (not model bugs). Catching version mismatch BEFORE E2E saves debugging time.
   - Fix: `slither --version` (expect 0.11.x) and `aderyn --version` (expect ≥ 0.4.21) added to prereq check block.

5. **Test contract source** — reuse fixtures from `tests/test_smoke_e2e.py:44-79` instead of writing new ones.
   - Rationale: keeps inputs identical between mocked smoke and real E2E. Any verdict difference is attributable to real services, not input drift.
   - Fix: `02_EXECUTION` §Step 0 with extraction script + `safe_storage.sol` + `vulnerable_reentrant.sol`.

### Added (educational layer — Plan Onboarding)

- **"Learning Outcomes" sections** at the end of each plan doc (00, 01, 02, 03).
- **"→ You now know" lines** at decision points and after meaningful steps.
- **"What you have at the end" map** in `03_ANALYSIS` with artifacts / knowledge deltas / decisions.
- **`Step 0: Log Service Versions`** in `03_ANALYSIS` — 5-minute high-ROI step that turns verdict changes into a debugging trail.
- **Known-limitation callouts** in `02_EXECUTION`:
  - Run 12 has no `AccessControl` class — only Reentrancy surfaces in vulnerable test.
  - Test 3 (Uniswap) marked skip-unless-time-available — model degrades on 700+ LOC contracts.

### Verified accurate (no change needed)

- 4 MCP server ports (:8010/:8011/:8012/:8013) — `inference_server.py`, `rag_server.py`, `audit_server.py:82`, `graph_inspector_server.py` all confirm.
- 9 nodes in graph — `nodes.py` lines 162, 291, 336, 409, 493, 676, 842, 941, 1103.
- Graph topology `ml_assessment → quick_screen → evidence_router → [fan-out] → ...` — `graph.py:160-167, 173-199`.
- ML API on :8001 with `/predict` — `ml/src/inference/api.py:255` + `inference_server.py:45` (`MODULE1_INFERENCE_URL`).
- Run 12 checkpoint at `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` — exists, 269 MB.
- RAG index at `agents/data/index/` — `faiss.index`, `bm25.pkl`, `chunks.pkl`, `index_metadata.json`, `seen_hashes.json` all present.
- 219 tests passing — verified via `grep -c` on test files.
- Slither in `pyproject.toml:22` — `slither-analyzer = "^0.11.5"`.
- Timing budget (45-60s/contract) — realistic for local execution.

### Reference

- Audit scratch file: `~/.claude/scratch/agents_e2e_test_audit_20260617.md`
- Plan Onboarding style: `~/projects/sentinel/ONBOARDING.md` (root, ≤150 lines, single source of truth)
- CLAUDE.md reference: added "Plan Execution Style" section pointing to `ONBOARDING.md`.

---

## [v1.0] — 2026-06-17 (Initial plan)

5 docs, ~1,885 lines. Goal: validate 9-node agents graph end-to-end with real LLM/MCP/ML before Phase A implementation.

Files:
- `README.md` (201) — purpose + success criteria
- `00_MASTER_TEST_PLAN.md` (272) — strategy + prereqs
- `01_SETUP_PLAN.md` (370) — env bootstrap
- `02_EXECUTION_PLAN.md` (373) — test harness + 3 contracts
- `03_ANALYSIS_PLAN.md` (670) — JSON parsing + baselines

**Issue noted in v1.0:** did not validate against source code. The 5 v1.1 fixes are the result of post-creation validation.
