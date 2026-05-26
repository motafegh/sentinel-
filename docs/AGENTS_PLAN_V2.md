# SENTINEL Agents Module — Updated Architecture & Implementation Plan

**Date:** 2026-05-27  
**Status:** Supersedes `docs/AGENTS_PLAN.md` (2026-05-20)  
**Trigger:** Run 4 complete (F1=0.3362); three-tier ML output designed; pipeline fully verified

---

## What Has Changed Since V1

The V1 plan was written on 2026-05-20, before Run 4 was killed or evaluated.
Three findings invalidate some V1 assumptions:

| Finding | Impact on agents |
|---------|-----------------|
| **Run 4 capacity ceiling** — model F1 locked at 0.31–0.34 for 12 epochs. No more training gains without data quality fixes. | The agents layer is now load-bearing. ML triage quality is fixed for now — agents must compensate. |
| **Three-tier suspicion output** — 8 of 12 missed classes had raw probability just below threshold. Full probability vector exposes this signal. | All routing logic that reads `ml_result["vulnerabilities"]` (binary filtered) is **stale**. Must be updated to read from the new schema. |
| **ExternalBug structural impossibility** — GNN sees `call_target_typed=1.00` (typed interface calls) as safer than raw `.call()`, which is opposite of what oracle manipulation detection needs. This cannot be fixed at ML layer without inter-contract analysis. | ExternalBug needs dedicated agent-layer handling: Slither external call summary + LLM reasoning. |

---

## 1. Phase 0 Status — COMPLETE

All Phase 0 items from V1 are implemented and tested (46/46 tests pass):

| Item | Status | Location |
|------|--------|----------|
| P0-1 routing.py: per-class thresholds + tool matrix | ✅ done | `agents/src/orchestration/routing.py` |
| P0-2 scoped Slither detectors per class | ✅ done | `CLASS_TO_DETECTORS` in routing.py; static_analysis node |
| P0-3 SqliteSaver checkpointer | ✅ done | `graph.py` |
| P0-4 routing_decisions append reducer | ✅ done | `state.py` |
| P0-5 verdicts/confirmations/contradictions in state | ✅ done | `state.py` |
| P0-6 rule-based verdict in synthesizer | ✅ done | `nodes.py:synthesizer` |
| P0-7 routing tests | ✅ done | `tests/test_routing_phase0.py` |

---

## 2. The Most Important Change: Three-Tier ML Output Integration

### Why This Comes First

The ML model currently outputs only classes above a hard per-class threshold.
The new three-tier output (designed in `docs/proposal/2026-05-27-three-tier-inference-output.md`)
always returns the full 10-class probability vector.

Until this integration is done, the agents are operating on **less information than the model has**.
8 vulnerability classes that the model detects at probability 0.25–0.54 are currently silently
discarded before the agents see them. Those are exactly the borderline cases agents should verify.

### New `ml_result` Schema (after predictor.py update)

```python
{
    "label": "suspicious",          # "safe" | "suspicious" | "confirmed_vulnerable"

    # Full 10-class vector — ALWAYS PRESENT — no filtering
    "probabilities": {
        "CallToUnknown": 0.638, "DenialOfService": 0.312, "ExternalBug": 0.261,
        "GasException": 0.302, "IntegerUO": 0.314, "MishandledException": 0.292,
        "Reentrancy": 0.620, "Timestamp": 0.197, "TransactionOrderDependence": 0.281,
        "UnusedReturn": 0.249,
    },

    # CONFIRMED (prob ≥ 0.55) — high-confidence findings
    "confirmed": [
        {"vulnerability_class": "CallToUnknown", "probability": 0.638, "tier": "CONFIRMED"},
        {"vulnerability_class": "Reentrancy",    "probability": 0.620, "tier": "CONFIRMED"},
    ],

    # SUSPICIOUS (prob ≥ 0.25) — non-trivial signal, not yet confirmed
    "suspicious": [
        {"vulnerability_class": "DenialOfService", "probability": 0.312, "tier": "SUSPICIOUS"},
    ],

    # Legacy field — preserved, contains CONFIRMED only
    "vulnerabilities": [
        {"vulnerability_class": "CallToUnknown", "probability": 0.638},
        {"vulnerability_class": "Reentrancy",    "probability": 0.620},
    ],

    # Metadata
    "thresholds": [0.40, 0.45, 0.35, 0.40, 0.50, 0.40, 0.40, 0.30, 0.35, 0.35],
    "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
    "truncated": false,
    "windows_used": 2,
    "num_nodes": 47,
    "num_edges": 89,
}
```

### Files to Change (in order)

#### Step 1 — predictor.py (ML side — do this first)

File: `ml/src/inference/predictor.py`, method `_format_result()`

Changes:
1. Always emit `probabilities` dict with all 10 classes (no threshold filter)
2. Compute `confirmed` list: classes with prob ≥ 0.55
3. Compute `suspicious` list: classes with 0.25 ≤ prob < 0.55
4. Keep `vulnerabilities` as legacy alias for `confirmed` content (backward compat)
5. Emit `label` as `"safe"` / `"suspicious"` / `"confirmed_vulnerable"`
6. Emit `tier_thresholds` dict in output

The tier thresholds (0.55 / 0.25) must be configurable — pass them into `Predictor.__init__`
or load from a companion JSON file. Not hardcoded.

#### Step 2 — routing.py (agents side)

**Replace `compute_active_tools()` and `build_routing_decisions()`** to read from the full
`probabilities` dict instead of the filtered `vulnerabilities` list:

```python
def compute_active_tools(ml_result: dict) -> list[str]:
    active: set[str] = set()
    probs = ml_result.get("probabilities", {})

    if not probs:
        # Legacy fallback: old binary schema
        for vuln in ml_result.get("vulnerabilities", []):
            cls  = vuln.get("vulnerability_class", "")
            prob = vuln.get("probability", 0.0)
            if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
                active.update(ROUTING_RULES.get(cls, []))
        return sorted(active)

    for cls, prob in probs.items():
        if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
            active.update(ROUTING_RULES.get(cls, []))

    return sorted(active)
```

`DEEP_THRESHOLDS` remains meaningful: they are **agent investigation thresholds**, separate from
ML tier thresholds. The ML says "SUSPICIOUS at 0.25" but the agent may decide "I only investigate
at 0.35 or above for this class because below that Slither almost never confirms." These are tuned
independently.

**Update `build_routing_decisions()`** to annotate tier in the decision string:

```python
tier = ml_result.get("tier_thresholds", {})
conf_thr = tier.get("confirmed", 0.55)
susp_thr = tier.get("suspicious", 0.25)

# derive tier label for log
if prob >= conf_thr:
    tier_label = "[CONFIRMED]"
elif prob >= susp_thr:
    tier_label = "[SUSPICIOUS]"
else:
    tier_label = "[NOTEWORTHY]"
```

#### Step 3 — nodes.py (four spots)

**ml_assessment:**
- Update log line to use `ml_result.get("confirmed", [])` for count
- Log count of suspicious classes separately
- Sanity check: if `probabilities` absent but old schema present, log a warning

**static_analysis:**
- Current: reads flagged classes from `vulnerabilities` at prob ≥ 0.35
- New: read from `probabilities` dict using DEEP_THRESHOLDS (same logic as routing)
- This is consistent and doesn't need a second code path

**rag_research:**
- Current: single query on the highest-prob class from `vulnerabilities`
- New: build queries for all CONFIRMED classes, plus top SUSPICIOUS class if no CONFIRMED exist
- Rank them: if only SUSPICIOUS classes exist, prefix the query with "possible"
- Implementation: loop over confirmed + top_suspicious, fire parallel RAG calls

```python
# Build query targets
targets = []
for entry in ml_result.get("confirmed", []):
    targets.append((entry["vulnerability_class"], entry["probability"], "confirmed"))
if not targets:
    susp = ml_result.get("suspicious", [])
    if susp:
        top_s = max(susp, key=lambda x: x["probability"])
        targets.append((top_s["vulnerability_class"], top_s["probability"], "suspicious"))
```

**synthesizer:**
- Current: derives risk_probability + top_vulnerability from `vulnerabilities` list
- New: if `confirmed` present, use max probability from `confirmed`; else use max from `suspicious`
- Verdict computation for SUSPICIOUS classes: emit verdict "WATCH" (new tier):
  - `WATCH` = "ML model detected non-trivial signal; insufficient evidence to confirm"
  - Route: static_analysis ran → report any scoped findings; otherwise note that agent investigation was below CONFIRMED tier
- Update `vuln_verdicts` to include both confirmed and suspicious classes

#### Step 4 — state.py

Update the `ml_result` field comment to document the new three-tier schema.

#### Step 5 — inference_server.py

Update `_mock_prediction()` to return the three-tier schema. The mock is used in all development
testing — if it produces the old schema, tests will miss the routing regression.

```python
def _mock_prediction(contract_code: str) -> dict:
    code_lower = contract_code.lower()
    has_reentrancy = "call.value" in code_lower or ".call{" in code_lower
    probs = {
        "Reentrancy": 0.72 if has_reentrancy else 0.28,
        "CallToUnknown": 0.65 if has_reentrancy else 0.22,
        "IntegerUO": 0.31, "GasException": 0.29, "DenialOfService": 0.18,
        "ExternalBug": 0.24, "Timestamp": 0.17, "TransactionOrderDependence": 0.21,
        "MishandledException": 0.19, "UnusedReturn": 0.15,
    }
    confirmed  = [{"vulnerability_class": c, "probability": p, "tier": "CONFIRMED"}
                  for c, p in probs.items() if p >= 0.55]
    suspicious = [{"vulnerability_class": c, "probability": p, "tier": "SUSPICIOUS"}
                  for c, p in probs.items() if 0.25 <= p < 0.55]
    label = "confirmed_vulnerable" if confirmed else ("suspicious" if suspicious else "safe")
    return {
        "label": label,
        "probabilities": probs,
        "confirmed": confirmed,
        "suspicious": suspicious,
        "vulnerabilities": confirmed,   # legacy compat
        "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
        "thresholds": [0.40, 0.45, 0.35, 0.40, 0.50, 0.40, 0.40, 0.30, 0.35, 0.35],
        "truncated": False, "windows_used": 2, "num_nodes": 42, "num_edges": 58,
    }
```

#### Step 6 — tests

- `test_routing_phase0.py`: `_ml()` helper needs to produce three-tier schema
- Add `test_routing_phase0.py::test_suspicious_class_routes_to_static()` — SUSPICIOUS above DEEP_THRESHOLD routes even though not in `vulnerabilities`
- Add `test_routing_phase0.py::test_full_probabilities_dict_used()` — routing reads from `probabilities` not just `vulnerabilities`
- Add mock-schema validation test to `test_inference_server.py`

---

## 3. ExternalBug — Agent-Layer Solution

### Why ML Cannot Detect It

`_select_contract()` in graph_extractor.py filters interfaces out:
`candidates = [c for c in sl.contracts if not c.is_from_dependency()]`
then prefers non-interface contracts. This is correct behavior — but it means
the GNN has no visibility into what `IPriceOracle.getPrice()` or `IVault.deposit()`
might do.

ExternalBug contracts show `call_target_typed=1.00` — all calls via typed interfaces.
The GNN interprets this as *safer* than raw `.call()`, because in the training corpus,
typed calls to known interfaces (ERC20, Ownable) are overwhelmingly safe. An oracle
manipulation attack looks identical at the graph level to a legitimate price oracle call.

### Solution: Slither External Call Summary in static_analysis

The `static_analysis` node already runs Slither on the contract source. Slither's
`contract.called_contracts` and `function.high_level_calls` give us exactly what we need:
which external functions are called, on what interfaces, in what value-sensitive context.

Add a helper in `nodes.py`:

```python
def _extract_external_call_summary(sl_obj, flagged_classes: set) -> dict | None:
    """
    If ExternalBug is in flagged_classes and called_contracts is non-empty,
    return a structured summary of external calls for agent reasoning.
    """
    if "ExternalBug" not in flagged_classes:
        return None

    summary = []
    for contract in sl_obj.contracts:
        if contract.is_from_dependency() or contract.is_interface:
            continue
        for fn in contract.functions:
            for called_contract, called_fn in fn.high_level_calls:
                summary.append({
                    "caller_function":   fn.name,
                    "called_contract":   called_contract.name,
                    "called_function":   called_fn.name if called_fn else "unknown",
                    "is_interface":      called_contract.is_interface,
                    "is_in_value_path":  any(
                        v.name in ("value", "amount", "price", "balance")
                        for v in fn.local_variables
                    ),
                })
    return {"external_calls": summary} if summary else None
```

Return this as `external_call_summary` field in `static_findings` metadata.
The synthesizer then includes this in the LLM context when ExternalBug is SUSPICIOUS.

**RAG enhancement for ExternalBug:** When ExternalBug is detected (any tier), the RAG query
should specifically include the called interface names:

```
"oracle manipulation ExternalBug: contract calls IPriceOracle.getPrice() in value-sensitive context"
```

This is a much more targeted query than the generic "ExternalBug vulnerability" query currently used.

---

## 4. Updated Verdict Model

Add a fifth verdict tier to handle SUSPICIOUS classes that have no corroboration:

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| `CONFIRMED` | ML CONFIRMED tier + Slither or RAG corroboration | Multiple tools agree — high confidence |
| `LIKELY` | ML CONFIRMED tier + partial corroboration (rag score 0.5–0.79) | Strong signal, one tool confirms |
| `DISPUTED` | ML CONFIRMED tier + no corroboration | ML flagged but nothing confirms — worth noting |
| `WATCH` | ML SUSPICIOUS tier + anything or nothing | Non-trivial signal below commit threshold |
| `SAFE` | Below DEEP_THRESHOLDS for all classes | Nothing worth investigating |

`WATCH` is new and important: it captures exactly the "DoS=0.36" cases that the old system
called SAFE but the new system knows are worth watching. Agents can route WATCH cases to
lightweight analysis (static only, no LLM call) while CONFIRMED/LIKELY trigger full deep analysis.

**Update `prob_to_severity()`:**
```python
def prob_to_severity(prob: float, tier: str = "") -> str:
    if tier == "CONFIRMED" and prob >= 0.70: return "HIGH"
    if tier == "CONFIRMED":                   return "MEDIUM"
    if tier == "SUSPICIOUS" and prob >= 0.40: return "LOW"
    if tier == "SUSPICIOUS":                  return "INFO"
    return "INFO"
```

**Update `compute_overall_verdict()`:**
```python
OVERALL_VERDICT_RANK = {"CONFIRMED": 5, "LIKELY": 4, "DISPUTED": 3, "WATCH": 2, "SAFE": 1}
```

---

## 5. Updated Graph Topology (Phase 1)

```
START
  │
  ▼
ml_assessment              ← now reads three-tier ml_result
  │
  ▼
evidence_router            ← reads from probabilities dict; annotates tier in decisions
  │
  ├─ (CONFIRMED classes) → rag_research  ← per-class parallel queries
  │                      → static_analysis ← scoped detectors + ext_call_summary if ExternalBug
  │
  ├─ (SUSPICIOUS only)   → static_analysis ← scoped detectors only; rag skipped unless
  │                                           class in HIGH_VALUE_RAG_CLASSES
  │
  └─ (all below DEEP_THRESHOLDS)
        → synthesizer      ← fast path; WATCH verdicts for SUSPICIOUS, SAFE otherwise
  │
  ▼ (fan-in)
audit_check
  │
  ▼
synthesizer                ← WATCH/CONFIRMED/LIKELY/DISPUTED verdicts; ext_call_summary
  │                          included in LLM context when ExternalBug present
  ▼
END
```

The key routing distinction vs V1: SUSPICIOUS classes can route to static_analysis alone (not rag)
depending on class value. HIGH_VALUE_RAG_CLASSES (rag worth calling for SUSPICIOUS):
`{"Reentrancy", "IntegerUO", "DenialOfService", "ExternalBug", "TOD"}`

---

## 6. Updated AuditState

```python
class AuditState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    contract_code:    str
    contract_address: str

    # ── ML evidence ────────────────────────────────────────────────────────────
    ml_result: dict[str, Any]
    # Three-tier schema (after predictor.py update):
    #   label:              str   — "safe" | "suspicious" | "confirmed_vulnerable"
    #   probabilities:      dict  — {class: prob} for all 10 classes, always present
    #   confirmed:          list  — [{vulnerability_class, probability, tier="CONFIRMED"}, ...]
    #   suspicious:         list  — [{vulnerability_class, probability, tier="SUSPICIOUS"}, ...]
    #   vulnerabilities:    list  — legacy alias for confirmed (backward compat)
    #   thresholds:         list  — per-class decision boundaries (10 values)
    #   tier_thresholds:    dict  — {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}
    #   truncated:          bool  — True if contract exceeded max windows
    #   windows_used:       int   — number of 512-token windows processed
    #   num_nodes:          int   — graph node count
    #   num_edges:          int   — graph edge count

    ml_hotspots: list[dict[str, Any]]
    # Phase 1 — {class, fn_name, lines, node_ids, score}

    # ── Routing trace ──────────────────────────────────────────────────────────
    routing_decisions: Annotated[list[str], operator.add]
    # Format: "{Class} prob={p:.3f} [{CONFIRMED|SUSPICIOUS|NOTEWORTHY}] >= threshold={t} → tools"

    # ── Graph explanation (Phase 1) ────────────────────────────────────────────
    graph_explanations: dict[str, Any]

    # ── Static analysis ────────────────────────────────────────────────────────
    static_findings: list[dict[str, Any]]
    # Each: {tool, detector, impact, confidence, description, lines, function_names}
    # Optional extra key: external_call_summary (ExternalBug only)

    # ── RAG evidence ───────────────────────────────────────────────────────────
    rag_results: list[dict[str, Any]]

    # ── Economic simulation (Phase 3) ──────────────────────────────────────────
    econ_scenarios: list[dict[str, Any]]

    # ── Cross-validation ───────────────────────────────────────────────────────
    verdicts: dict[str, str]
    # {class: "CONFIRMED" | "LIKELY" | "DISPUTED" | "WATCH" | "SAFE"}  ← WATCH is new

    confirmations:  dict[str, list[str]]
    contradictions: dict[str, list[str]]

    # ── On-chain history ───────────────────────────────────────────────────────
    audit_history: list[dict[str, Any]]

    # ── Final output ───────────────────────────────────────────────────────────
    final_report: dict[str, Any]
    narrative:    str | None
    error:        str | None
```

---

## 7. Implementation Sequence

### Step A — Three-Tier Output (no retraining, unblocked today)

**Estimated effort:** 3–4 hours  
**Files:** `ml/src/inference/predictor.py`

- Add `tier_confirmed_threshold` and `tier_suspicious_threshold` params to `Predictor.__init__`
- Default: confirmed=0.55, suspicious=0.25
- Update `_format_result()` to compute and return full schema

**Test:** Run `Predictor.predict_source()` on `ml/scripts/test_contracts/01_reentrancy_classic.sol`.
Verify `probabilities` dict has 10 entries, `suspicious` has DoS≈0.31 and others, `confirmed` has Reentrancy≈0.62.

---

### Step B — Agent Schema Integration

**Estimated effort:** 4–6 hours  
**Files:** `agents/src/orchestration/routing.py`, `nodes.py`, `state.py`, `mcp/servers/inference_server.py`, `tests/`

1. `routing.py` — update `compute_active_tools()` + `build_routing_decisions()` to read from `probabilities`
2. `routing.py` — add `WATCH` to verdict logic: SUSPICIOUS tier with no corroboration → WATCH
3. `routing.py` — add `prob_to_severity()` tier param  
4. `nodes.py::static_analysis` — update flagged class detection to use `probabilities`
5. `nodes.py::static_analysis` — add `_extract_external_call_summary()` for ExternalBug
6. `nodes.py::rag_research` — add per-class queries from `confirmed` + top `suspicious`
7. `nodes.py::synthesizer` — include `suspicious` classes in verdicts as WATCH; include ext_call_summary in LLM context
8. `state.py` — update ml_result comment
9. `inference_server.py` — update `_mock_prediction()` to three-tier schema
10. Tests — update `_ml()` helper; add three-tier routing tests

---

### Step C — ExternalBug Slither Enhancement

**Estimated effort:** 2–3 hours  
**Files:** `agents/src/orchestration/nodes.py`

Add `_extract_external_call_summary()` inside `static_analysis` node.  
Add ExternalBug-specific RAG query construction in `rag_research` node.  
Add ext_call_summary rendering in `synthesizer` LLM prompt.

---

### Step D — Phase 1: Graph Inspector (now unblocked — Run 4 checkpoint is stable)

**Estimated effort:** 1–2 days  
**Files:** new `agents/src/mcp/servers/graph_inspector_server.py`, new `agents/src/orchestration/nodes.py::graph_explain`

The V1 plan's P1-1 through P1-5 apply here. Run 4 checkpoint is the stable v8 checkpoint
the plan was waiting for. The checkpoint is at `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`.

Key difference from V1: GNN attention hotspots need to be extracted at inference time via
`predictor.py` hooks, not post-hoc. This requires a `/hotspots` endpoint addition to the
inference FastAPI server (Module 1) alongside `/predict`.

---

### Step E — Phase 2: cross_validator + Send API

**Estimated effort:** 1 day  
**Files:** new `cross_validator` node in `nodes.py`, graph update in `graph.py`

V1 §9 design is still valid. No changes needed there.

The LLM adjudication path for DISPUTED cases becomes more important now that WATCH cases
exist — there's a natural upgrade path: WATCH cases that have any static analysis finding
should be adjudicated by LLM (low-cost single call) rather than left as WATCH.

---

## 8. MCP Servers — No Changes Needed for A/B/C

The three-tier integration is entirely in the agents orchestration layer and in predictor.py.
The MCP server at :8010 passes the predictor output through unchanged — it's already a JSON
passthrough. No protocol changes needed.

The `_mock_prediction()` update in inference_server.py (Step B item 9) is the only server-side
change, and it's local to the mock code path.

---

## 9. Escalation Tiers (Updated)

| Tier | Condition | Path | LLM calls |
|------|-----------|------|-----------|
| **Tier 0 — Fast pass** | all probs < DEEP_THRESHOLDS | → synthesizer (WATCH verdicts for SUSPICIOUS, SAFE for rest) | 0 |
| **Tier 1 — Static only** | SUSPICIOUS classes above DEEP_THRESHOLDS | static_analysis → synthesizer | 0 |
| **Tier 2 — Standard** | CONFIRMED classes → static + rag; rule-based cross_validator | static + rag → audit_check → synthesizer | 0–1 |
| **Tier 3 — Deep** | DISPUTED verdict exists | LLM adjudication in cross_validator (Phase 2) | 1 |
| **Tier 4 — Narrative** | CONFIRMED or LIKELY verdict | synthesizer LLM narrative | 1 |

Tier 0 is new: contracts where all probabilities are below DEEP_THRESHOLDS but some are
SUSPICIOUS (0.25–0.35). These go fast path but get WATCH verdicts, not SAFE. The report
notes the SUSPICIOUS signals without triggering full analysis.

---

## 10. What NOT to Build

Unchanged from V1:
- Triage Agent as separate process
- Debate Agent pair
- Foundry fuzz tests in the agent loop
- Full Mythril symbolic execution (scoped to hotspot functions only in Phase 3)
- Cross-contract dependency graph before Phase 3

New additions:
- **Do NOT add hardcoded ExternalBug triggers** (e.g., "if any external call exists → flag").
  60–70% of real DeFi contracts have external calls. The SUSPICIOUS tier + Slither summary
  is the right signal. Manual rules make FP rates unacceptable at scale.
- **Do NOT tune tier thresholds per-class**. The 0.55/0.25 tiers are global starting points.
  If per-class tuning is needed, it belongs in the predictor config file, not in routing.py.

---

## Appendix: Gap Analysis — Current Code vs This Plan

| Component | Current state | Gap |
|-----------|--------------|-----|
| `predictor._format_result()` | binary threshold, returns `vulnerabilities` only | needs three-tier output (Step A) |
| `routing.compute_active_tools()` | reads `ml_result["vulnerabilities"]` | needs to read from `probabilities` (Step B) |
| `nodes.static_analysis` | reads flagged classes from `vulnerabilities` at prob≥0.35 | needs `probabilities` dict (Step B) |
| `nodes.rag_research` | single query on top vuln | needs per-class queries from confirmed+suspicious (Step B) |
| `nodes.synthesizer` | verdicts from `vulnerabilities` only | needs WATCH tier for `suspicious` (Step B) |
| `routing.compute_verdict()` | 4 verdicts: CONFIRMED/LIKELY/DISPUTED/SAFE | needs 5th: WATCH (Step B) |
| `inference_server._mock_prediction()` | old binary schema | needs three-tier mock (Step B) |
| ExternalBug handling | relies on Slither `arbitrary-send-eth` detector (usually misses oracle manip) | needs `_extract_external_call_summary()` (Step C) |
| graph_inspector_server :8013 | not built | Phase 1 now unblocked — Run 4 checkpoint exists (Step D) |
| cross_validator node | not built | Phase 2 (Step E) |
