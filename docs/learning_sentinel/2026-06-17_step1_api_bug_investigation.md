# SENTINEL Learning Doc — Step 1 (2026-06-17)

> **Mode:** DEEP DIVE TEACHING (active per Ali's request: "teach and explain me those things done, why, what and how").
> **Topic:** P0 "API schema bug" investigation.
> **Conclusion:** The bug doesn't exist. MEMORY.md was stale.

---

## The 3 Layers of Understanding (the "why / what / how" framework)

For every code change, I aim to answer in 3 layers:

1. **WHY** — What's the problem we're solving? Why does it matter? What's at stake if we don't fix it?
2. **WHAT** — What's the actual code or system doing? Where does it live? What are the moving parts?
3. **HOW** — What's the fix, step by step? What's the test that proves it works?

Most tutorials teach only WHAT. Senior engineers think in WHY first, WHAT second, HOW third — because the WHAT/HOW change every year, but the WHY (data contracts, type safety, observability) stays.

---

## Step 1 — The Investigation

### WHY we investigated
MEMORY.md claimed an HTTP 500 on `/predict` and `/hotspots` due to a type mismatch. If true:
- The FastAPI server is dead in production
- Phase B.4 (drift baseline) can't collect real traffic
- Run 13 prep is blocked
- Ali's "MLOps serving Run 12" milestone is a lie

Worth 30 min to verify, even if it turns out false.

### WHAT we found
Three pieces of evidence — all from source code, none from MEMORY.md:

**1. The Pydantic schema (api.py:209):**
```python
tier_thresholds: dict[str, float | list[float]] = Field(default_factory=dict)
```
- This is a **union type** — Pydantic v2's syntax for "either/or".
- `float | list[float]` means: each VALUE in the dict can be a scalar float OR a list of floats.
- So `{"confirmed": [0.5, 0.6, 0.7], "suspicious": 0.25, "noteworthy": 0.10}` is valid.

**2. The predictor (predictor.py:750-754):**
```python
"tier_thresholds":  {
    "confirmed":  self.thresholds.cpu().tolist(),  # per-class list (F8/F10 fix)
    "suspicious": susp_thr,
    "noteworthy": 0.10,
},
```
- `"confirmed"` IS a list (per-class thresholds from F8/F10 fix)
- `"suspicious"` IS a scalar
- `"noteworthy"` IS a scalar
- All match the union type's contract

**3. The test suite (just ran):**
```
18 passed, 3 warnings in 6.10s
```
- The model loads, /health works, /predict works, /hotspots works
- No 500s, no Pydantic validation errors

### HOW we verified
Three layers of verification, fastest to slowest:

1. **Read the type** — `dict[str, float | list[float]]` is unambiguous in Pydantic v2
2. **Read the data** — confirmed what the predictor actually returns
3. **Run the tests** — full integration test through FastAPI's TestClient

Total time: ~3 min. Saved potentially hours of "fixing" a non-bug.

---

## The Bigger Lesson — Why docs lie

This is the most important takeaway for a senior engineer:

> **Code is the contract. Docs are someone's last-week understanding of the contract.**

When CLAUDE.md says "Trust source code only. Distrust all docs", it's not paranoia — it's a real failure mode I just hit.

### The 5 failure modes for stale docs in this project

1. **Schema evolution** — Pydantic v1 → v2 syntax changes. `Union[X, Y]` → `X | Y`. Old docs use the old syntax.
2. **Refactor without doc update** — someone refactored the predictor (F8/F10 fix), but MEMORY.md was written before the fix and never updated.
3. **Inference from a code review** — MEMORY.md might have said "looks like a bug" and never got verified.
4. **Copy-paste from an issue** — original bug report may have been for a different state of the code.
5. **Time decay** — code evolves daily; docs evolve monthly. Gap widens.

### The defensive posture

When MEMORY.md says "critical bug", I do NOT just trust it. I do this 30-second ritual:

1. **Read the actual schema** — 5 seconds
2. **Read what the producer actually returns** — 10 seconds
3. **Run the test that exercises the path** — 15 seconds

If all 3 contradict MEMORY.md, **MEMORY.md is wrong**, not the code. Update MEMORY.md; do not "fix" working code.

---

## What I did (Step 2 — Doc correction pass, 2026-06-17 ~evening)

Ali said "go for update memory and also any other refrences files doc needed properly". So I:

1. **MEMORY.md** — replaced the false P0 claim with a corrected note; updated the
   production gate reason to "drift baseline placeholder" only.
2. **MLOps state check report** (2026-06-16) — added a prominent REVISION section at
   the top; corrected false claims in TL;DR, verification results table, critical
   finding section, and recommendations list. Kept the original report intact
   (with corrections) for traceability.
3. **K_inference_api.md spec** — corrected `tier_thresholds` type from `dict[str, float]`
   to `dict[str, float | list[float]]`; flagged the /health vs /predict shape
   inconsistency; updated the "Last revised" date.
4. **ml/audit_docs/ISSUES.md** — filed BUG-1 as closed (false positive claim) for
   traceability, so the next session sees the resolution and doesn't redo the work.
5. **docs/changes/2026-06-17-ml-api-schema-claim-correction.md** — new daily changelog
   entry per the project's 6-part naming convention.
6. **This learning doc** — updated with actions taken.
7. **Scratch file** (`~/.claude/scratch/sentinel_session_plan_20260617.md`) — updated.

Total time: ~15 min for all doc updates. The lesson: **docs and code drift; reconcile
docs to code, not the other way around.**

---

## Concepts introduced this step (vocab to remember)

- **Pydantic union type** — `X | Y` in Pydantic v2 means "value can be X or Y". Different from `Union[X, Y]` (Pydantic v1).
- **`Field(default_factory=...)`** — provides a default by calling a function, not a value. Used for mutable defaults like `dict` and `list`.
- **Pydantic schema validation** — FastAPI uses the type annotations on your BaseModel to automatically validate request/response payloads. If a value doesn't match, FastAPI returns 422 (request) or 500 (response).
- **`response_model=PredictResponse`** — the FastAPI decorator arg that pins the return type. If `predict()` returns something that doesn't match, FastAPI strips invalid fields (or 500s).
- **TestClient (FastAPI)** — runs the full app in-process. No real server. Perfect for CI.
- **session-scoped fixture** — `pytest.fixture(scope="session")` means the model loads ONCE for all tests, not per test. Saves ~40s on 4 tests.

---

## Files touched / read this step

- `ml/src/inference/api.py` (read lines 180-349)
- `ml/src/inference/predictor.py` (read lines 680-759)
- `ml/tests/test_api.py` (read all 304 lines)
- `ml/tests/conftest.py` (read all 26 lines)
- Ran: `ml/.venv/bin/python -m pytest ml/tests/test_api.py -v` → 18/18 pass

## Next scratch files to create

- (none yet — depends on Ali's next move)
