# SolidiFI Analysis — Preprocessing & Inference Issues (2026-06-10)

Discovered during step-by-step wrong-contract analysis on SolidiFI benchmark.
These are **not blocking** — the 350-contract benchmark results are valid — but must
be fixed before Run 10 inference is used in production or v2 evaluation.

---

## Issue 1 — Thresholds JSON in wrong location (Predictor silently uses uniform 0.5)

**Severity:** High — affects all production `Predictor.predict()` calls for Run 9  
**Affects benchmark results:** No — raw probabilities and rankings are unaffected  

### What happens
`Predictor.__init__` looks for the per-class thresholds at:
```
ml/checkpoints/GCB-P1-Run9-v11-20260606_best_thresholds.json
```
That file does not exist. The actual file is at:
```
ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json
```
Predictor falls back to **uniform 0.5 threshold for all classes** and logs a WARNING
that is easy to miss. Any code using `result["confirmed"]` or `result["suspicious"]`
gets wrong bucket assignments.

### Impact
With calibrated thresholds (0.300–0.375 per class) vs uniform 0.5, the confirmed/
suspicious split on `buggy_4.sol` changes from:
- Uniform 0.5:    5 confirmed, 5 suspicious
- Calibrated:     9 confirmed, 0 suspicious, 1 below threshold (DenialOfService)

The calibrated thresholds are very low (optimised for recall on the test set), so
this contract triggers near-universal "confirmed" under calibration — which itself
is a sign that the thresholds need revisiting in v2.

### Fix
```bash
cp ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json \
   ml/checkpoints/GCB-P1-Run9-v11-20260606_best_thresholds.json
```
Long-term: update `Predictor.__init__` to also search `ml/calibration/` as a
fallback, or accept an explicit `thresholds_path` constructor argument.

---

## Issue 2 — `process_source()` silently truncates to 512 tokens (not usable for inference)

**Severity:** Medium — only matters if someone uses this method directly for inference  
**Affects benchmark results:** No — benchmark used `process_source_windowed()` throughout  

### What happens
`ContractPreprocessor.process_source()` tokenizes the full source but returns only
a single 512-token tensor (truncates and discards the rest). For `buggy_4.sol`
(6921 chars, 2864 tokens) this loses 2352/2864 tokens = 82% of the source text.

The model was trained with 4-window sliding input via `process_source_windowed()`.
Using `process_source()` output directly in a forward pass gives different (degraded)
predictions.

### Fix
Add a docstring warning to `process_source()`:
```python
# WARNING: returns a single truncated 512-token tensor.
# For model inference always use process_source_windowed() instead.
```
Or deprecate `process_source()` for inference use entirely and only expose
`process_source_windowed()`.

---

## Issue 3 — RETURN_TO edges (type 9) absent despite CALL_ENTRY edges (type 8) present

**Severity:** Medium — weakens ICFG cross-function reasoning in GNN Phase 2  
**Affects benchmark results:** No — all 350 contracts processed identically  

### What happens
`buggy_4.sol` graph has:
- 10 `CALL_ENTRY` edges (type 8) — calling statement → callee ENTRYPOINT  
-  0 `RETURN_TO` edges (type 9) — callee exit → caller successor  

CALL_ENTRY and RETURN_TO are supposed to be paired to form complete ICFG-Lite
cross-function paths. Without RETURN_TO, GNN Phase 2 can traverse into a called
function but cannot route information back to the call site. Control flow is
one-directional across function boundaries.

### Fix
Already listed in the v2 data module plan as one of the 3 still-open bugs to fix
in Stage 7 (`pipeline.fix_return_to_edges`). Fix is in
`ml/src/preprocessing/graph_extractor.py` — the `_build_icfg_edges()` function.

---

## Issue 4 — `call_target_typed` (feat[8]) = 1.0 for all nodes on `.transfer()`-only contracts

**Severity:** Low — correct behaviour, but zero discriminative signal  
**Affects benchmark results:** No  

### What happens
`_node_call_target_typed()` returns 1.0 (default "typed / not applicable") for any
node that does not contain a `LowLevelCall` or raw-address `HighLevelCall`.
Solidity `.transfer()` is a built-in, not a `LowLevelCall`, so it passes through as
typed. For ERC-20 contracts that only use `.transfer()` (like all SolidiFI base
contracts), feat[8] = 1.0 everywhere and carries zero information.

The feature is working correctly — it's designed to detect raw `.call()` patterns —
but it means the model cannot distinguish "uses transfer" from "uses typed interface"
for this class of contracts.

### Fix
No code fix needed. Document as a known limitation for SolidiFI Unchecked-Send
category analysis (SolidiFI uses `.transfer()` which always reverts; SENTINEL's
CallToUnknown detects raw `.call()` — fundamentally different concepts).
Consider adding a `uses_transfer` feature in v2 schema to complement this.

---

## Issue 5 — `in_unchecked` (feat[11]) = 1.0 universally on pre-0.8 Solidity contracts

**Severity:** Low — known v9 schema limitation, already documented  
**Affects benchmark results:** No  

### What happens
Slither 0.10 sets `scope.is_checked = False` for all nodes in pre-0.8 Solidity
contracts (which have no `unchecked{}` syntax). `_node_in_unchecked()` therefore
returns 1.0 for every node in a Solidity 0.5 contract. The feature becomes a
**Solidity era detector** (0 = 0.8+, 1 = pre-0.8) rather than an unchecked-block
detector as intended.

For the SolidiFI benchmark specifically: all 350 contracts are `>=0.4.22 <0.6.0`,
so feat[11] = 1.0 for ~83% of nodes in every contract — no discriminative signal
across the benchmark.

### Fix
Already documented in v9 schema notes. In v2, consider splitting into:
- `is_pre_08_era` (1.0 if compiled with <0.8) — contract-level
- `in_unchecked_block` (1.0 only inside explicit `unchecked{}`) — statement-level

---

## Summary Table

| # | Issue | Blocks analysis? | Fix effort | When |
|---|---|---|---|---|
| 1 | Thresholds JSON wrong path | No | 1 `cp` command | Now |
| 2 | `process_source()` truncates | No | Docstring warning | Next PR |
| 3 | RETURN_TO edges absent | No | Graph extractor fix | v2 Stage 7 |
| 4 | `call_target_typed` zero signal on `.transfer()` | No | Optional v2 schema feat | v2 schema |
| 5 | `in_unchecked` era proxy on pre-0.8 | No | v2 schema redesign | v2 schema |
