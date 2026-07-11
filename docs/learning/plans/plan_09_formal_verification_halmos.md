# Plan: Doc 09 — Formal Verification (Halmos): Symbolic Execution as Evidence

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/09_formal_verification_halmos.md`
**Session:** 5 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 02 (Evidence/Fuse)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that the deep path fans out to 4 parallel nodes: `rag_research`, `static_analysis`, `graph_explain`, and `formal_verification`. All converge at `audit_check`. The `formal_verification` node was added in P8a.

**From Doc 02 (Evidence/Fuse):** You learned that `Evidence` has a `Kind` enum with `FORMAL` as one of the values. Formal evidence is the strongest type — it's a mathematical proof, not a statistical guess. `Evidence.formal()` constructor creates this type with `deterministic=True` and `reliability≈0.95`. Adding Halmos didn't require any changes to `fuse()` — it just added a new emitter (`emit_halmos_evidence()`).

**Connection to this doc:** This doc explains how Halmos symbolic execution works, how it's integrated into the pipeline, and what kind of evidence it produces. It's the newest evidence channel (P8a) and demonstrates the "plug-in" nature of the Evidence model.

**Key concepts carried forward:** `formal_verification` node in graph deep path, `Evidence.formal()` constructor, `Kind.FORMAL`, `emit_halmos_evidence()`, fail-soft principle.

---

## Step 1: Read source files

- [ ] `agents/src/orchestration/nodes/formal_verification.py` (~270 lines) — `formal_verification()` node, `_run_halmos()` async wrapper, `_generate_test_harness()`, `_parse_halmos_output()`, `_INVARIANT_TO_CLASS` mapping
- [ ] `agents/src/orchestration/verdict/evidence.py` (lines 158-180) — `Evidence.formal()` constructor
- [ ] `agents/src/orchestration/verdict/emit.py` (lines 205-240) — `emit_halmos_evidence()` function
- [ ] `agents/src/orchestration/graph.py` (lines 135, 175, 203) — `formal_verification` in node registration, deep path fan-out, fan-in to `audit_check`

## Step 2: Read tests

- [ ] `agents/tests/test_formal_verification.py` — 15 tests:
  - `TestEmitHalmosEvidence`: violation_emits_supports, proven_emits_refutes, non_halmos_ignored, empty_findings, multiple_findings
  - `TestEvidenceFormalConstructor`: supports_violation, refutes_safety
  - `TestParseHalmosOutput`: parse_pass, parse_fail, parse_empty, parse_invalid_json, parse_unknown_invariant_skipped
  - `TestFormalVerificationNode`: skip_deterministic_mode, skip_no_contract_code, fail_soft_on_missing_tools

## Step 3: Read toolchain docs

- [ ] `~/tools/TOOLCHAIN_ENV.md` — Halmos install (via `agents/.venv/bin/pip install halmos`), forge requirement, invocation path

## Step 4: Write sections

- [ ] **TL;DR:** Halmos symbolic execution node in the deep path. Creates temp Foundry project, generates test harness with invariant checks, runs `forge build` + `halmos --json-output`, parses results. Emits `Evidence(kind=FORMAL, deterministic=True)`. Fail-soft on missing tools/compile errors/timeout. 5 invariants mapped to vulnerability classes
- [ ] **The Problem:** ML guesses, static tools pattern-match, but neither proves. Need formal verification that mathematically proves or refutes invariants (reentrancy, arithmetic, access control). "Formal evidence is the strongest type — it asserts what ML can only guess" (Principle 7)
- [ ] **How We Arrived at This Design:** invariant (formal evidence is strongest, Principle 7) → constraint (Halmos needs a full Foundry project, not just a .sol file) → simplest integration (temp directory + generated test harness) → stress-test (large contracts timeout at 120s) → measure (5 invariants mapped, 15 tests)
- [ ] **The Solution:** Node flow diagram:
  ```
  contract_code → extract contract name
    → create temp Foundry project (src/Target.sol, test/FormalVerify.t.sol, foundry.toml)
    → symlink forge-std
    → forge build (compile)
    → halmos --json-output (symbolic execution)
    → parse JSON: {results: [{name: "check_reentrancy()", status: "pass"|"fail"}]}
    → map invariant → vulnerability class
    → emit Evidence.formal(source="halmos", polarity=SUPPORTS|REFUTES, ...)
    → fail-soft: empty findings + tool_status["halmos"]["ran": False] on any error
  ```
  The 5 invariants and their mappings:
  | Halmos test function | Invariant | Vulnerability class |
  |---------------------|-----------|-------------------|
  | `check_reentrancy()` | reentrancy | Reentrancy |
  | `check_arithmetic()` | arithmetic | IntegerUO |
  | `check_access_control()` | access_control | AccessControl |
  | `check_unchecked_return()` | unchecked_return | UnusedReturn |
  | `check_denial_of_service()` | denial_of_service | DenialOfService |
- [ ] **Key Code:**
  - `formal_verification()` function (formal_verification.py:85-135) — main node, checks `SENTINEL_DETERMINISTIC`, checks halmos/forge installed, calls `_run_halmos()`, emits evidence
  - `_run_halmos()` (formal_verification.py:145-220) — creates temp Foundry project, writes contract + test harness, runs `forge build` + `halmos`, reads JSON output
  - `_generate_test_harness()` (formal_verification.py:60-80) — generates Foundry test file with 5 invariant check functions
  - `_parse_halmos_output()` (formal_verification.py:230-270) — parses Halmos JSON, maps test name → invariant → vulnerability class
  - `Evidence.formal()` (evidence.py:158-180) — `source`, `vuln_class`, `polarity`, `invariant`, `proven`, `counterexample`. Strength 1.0 for SUPPORTS (formal proof of violation), 0.9 for REFUTES (formal proof of safety, but limited by tool coverage)
  - `emit_halmos_evidence()` (emit.py:205-240) — converts findings list to Evidence list, filters for `tool=="halmos"`
- [ ] **Design Decision:** Halmos vs Certora vs Foundry invariant testing (tradeoff table: cost, license, coverage, Python integration, maturity)
- [ ] **Technology Choice:** Halmos (5-question framework: category, alternatives, why Halmos, when Certora is better, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ Run Halmos on every contract — "formal proof is always better." Breaks: path explosion on large contracts, 120s timeout, wasted compute on safe contracts. Right: only in deep path (when ML/static tools flagged something)
  - ❌ Halmos as sole verdict source — "formal proof is the strongest." Breaks: only covers 5 invariants, can't detect logic bugs or economic exploits. Right: Halmos is one evidence channel among 6
- [ ] **Mistakes & Fixes:**
  - Invariant-to-class mapping is fragile (test function name → class name). If the test function name doesn't match `_INVARIANT_TO_CLASS`, it's silently skipped. Fix: explicit dict, skip unknown invariants (log warning). Future: more robust mapping
  - Halmos needs a full Foundry project (not just a .sol file). Needs `foundry.toml`, `src/`, `test/`, `lib/forge-std` symlink. Fix: `_run_halmos()` creates temp directory with all required structure
  - `forge build` can fail (missing imports, syntax errors in contract, incompatible Solidity version). Fix: fail-soft — return empty findings + `tool_status["halmos"]["ran": False, "reason": "forge_build_failed"]`
  - `SENTINEL_DETERMINISTIC=1` skips Halmos — Halmos is deterministic, but the Foundry project setup + forge build can have side effects. Fix: skip in deterministic mode (conservative choice)
- [ ] **What Would Break Without This:** Remove `formal_verification` → no formal evidence channel. ML + static tools can guess but not prove. `verdict_provable` loses a `deterministic=True` source. The "defense-in-depth has standalone value" principle (Principle 7) is weakened
- [ ] **At Scale:** 61 contracts (current, most complete in <30s) / 610 (some large contracts timeout at 120s) / 6,100 (need per-contract timeout tuning) / 61,000 (need parallel Halmos workers)
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  python -m pytest tests/test_formal_verification.py -v
  which halmos && halmos --version   # check if installed
  which forge && forge --version     # check if installed
  ```
- [ ] **Limitations:** Only 5 invariants (reentrancy, arithmetic, access control, unchecked return, DoS). Can't detect logic bugs, economic exploits, or front-running. 120s timeout (large contracts may timeout). Needs forge + halmos installed. Test harness is generic (not contract-specific). `SENTINEL_DETERMINISTIC=1` skips it (conservative). No incremental verification (runs all invariants every time)
- [ ] **Transferable Patterns:** (1) Formal evidence as the strongest tier — mathematical proof > statistical guess (2) Fail-soft for external tools — timeout/compile error → empty findings, not crash (3) Temp project pattern for tool integration — create isolated environment, run tool, clean up. Each with interview story + when wrong.

## Step 5: Verify

- [ ] Open `formal_verification.py` and verify `_INVARIANT_TO_CLASS` has 5 entries
- [ ] Confirm `Evidence.formal()` exists in `evidence.py` with strength 1.0 for SUPPORTS, 0.9 for REFUTES
- [ ] Confirm `emit_halmos_evidence()` exists in `emit.py` and filters for `tool=="halmos"`
- [ ] Confirm `formal_verification` is registered in `graph.py` and wired to deep path + `audit_check` fan-in
- [ ] Confirm test count: 15 tests in `test_formal_verification.py`
