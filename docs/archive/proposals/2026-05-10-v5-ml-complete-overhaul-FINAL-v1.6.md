# SENTINEL v5 — Complete ML Module Overhaul
# Comprehensive Technical Plan — FINAL (Revision 1.6)

| Field        | Value                                                                          |
|--------------|--------------------------------------------------------------------------------|
| Date         | 2026-05-10                                                                     |
| Revision     | 1.6 — third critic round + independent hostile analysis (10 issues applied)   |
| Status       | ACTIVE — Implementation Guide                                                  |
| Supersedes   | v4 exp1 (`multilabel-v4-finetune-lr1e4_best.pt`)                              |
| Author       | Post-audit synthesis (code + results + three full review rounds)               |
| Priority     | Critical — production model is broken                                          |

---

## Revision History

### Revision 1.1–1.3

See original document. Six external review issues applied; three-eye architecture adopted;
auxiliary loss added; edge-type numbering fixed.

### Revision 1.4 (critic round 1 + hostile analysis)

Nine external issues and seven internal issues applied. Key changes: two-phase GNN
architecture; CFG node subtypes 8–12; `return_ignored` sentinel -1.0; `gas_intensity`
removed; DoS gate raised to 0.55; F1-macro gate raised to 0.58; `NotImplementedError`
moved to `TrainConfig.__post_init__()`; CFG ordering pinned; mutation verification added.

### Revision 1.5 (critic round 2 + hostile analysis)

Seven issues from the second critic review were assessed. All seven were accepted and
applied. Two additional issues found in independent analysis were also applied.

| Source | # | Issue | Decision |
|---|---|---|---|
| Critic | 1 | **Preflight test uses `global_mean_pool`** — dilutes signal across 15+ shared nodes | **Applied** — test now compares `withdraw` function-level node embedding directly. |
| Critic | 2 | **`_cfg_node_type()` priority collisions unspecified** | **Applied** — CALL > WRITE > READ > CHECK > OTHER with justification. |
| Critic | 3 | **Phase 2 signal cannot reach function node** | **Applied — critical fix.** Three-phase, 4-layer GNNEncoder. |
| Critic | 4 | **`pos_weight` hard cap of 50 is unjustified** | **Applied** — replaced with sqrt scaling. |
| Critic | 5 | **Slither minimum version never stated** | **Applied** — hard dependency `>=0.9.3` with import-time assertion. |
| Critic | 6 | **`call_target_typed` fallback closed-world assumption** | **Applied** — sentinel -1.0 when source unavailable. |
| Critic | 7 | **Behavioral test suite inventory undefined** | **Applied** — explicit 20-contract inventory in §8.2. |
| Hostile | A | **1 CONTROL_FLOW hop insufficient for diameter-4+ CFGs** | **Documented** — explicit limitation in §4.3 and §7.4. |
| Hostile | B | **`_find_function_node()` helper needs spec** | **Applied** — spec added to §Pre-flight. |

### Revision 1.6 Changes (critic round 3 + independent hostile analysis — THIS REVISION)

Ten issues assessed. All ten accepted and applied. Two were pre-existing critical
blockers that would prevent any code from running. Two were introduced by Rev 1.5 itself.

| Source | # | Issue | Severity | Decision |
|---|---|---|---|---|
| Critic | P1 | **Phase 3 reversed CONTAINS uses same type-5 embedding** — GNN cannot distinguish forward CONTAINS (function→CFG) from reversed CONTAINS (CFG→function); asymmetric semantics ("parent sends context" vs "child sends signal") are lost | Medium | **Documented** — explicit limitation added to §2.3, §7.2, §7.4, and §9 risks table. New edge type `REVERSE_CONTAINS = 7` deferred to v5.1 (non-breaking addition). |
| Critic | P2 | **`node_metadata` never built or stored during extraction** — `_find_function_node()` calls `graph.node_metadata[i].get("name")` but §2.2 never specifies when this is created or stored; pre-flight test fails with `AttributeError` before reaching the cosine similarity assertion | **Critical blocker** | **Applied** — §2.2 now specifies `node_metadata` list construction in `_add_node()` and `_build_control_flow_edges()`; `data.node_metadata` stored on the `Data` object. See §2.2B. |
| Critic | P3 | **Phase 3 zero-message behavior on FUNCTION nodes without CFG children is undocumented** — a developer may add `add_self_loops=True` to conv4 to "fix" this, breaking Phase 3 | Low | **Applied** — explicit documentation added to §2.3 forward pass and §7.4. |
| Critic | P4 | **`graph_idx = len(node_index_map) + ...` is an unresolved placeholder** — `...` is literal Python ellipsis, not completed assignment logic; `x_list` is not in function signature; two implementers produce different index schemes | **Critical blocker** | **Applied** — `_build_control_flow_edges()` signature updated (added `x_list` parameter); `graph_idx = len(x_list)` is the explicit assignment; nodes appended to shared `x_list` in-place. See §2.2C. |
| Critic | P5 | **Single-seed cosine threshold 0.95 unreliable** — randomly initialized GNN can produce similarity above or below threshold by chance; false negatives possible | Medium | **Applied** — fixed seed `torch.manual_seed(42)` added before model initialization; threshold tightened from 0.95 to **0.85** for stronger separation signal. See §Pre-flight. |
| Critic | P6 | **No expected graph size estimate; batch_size=16 unjustifiable without data** | Low | **Applied** — smoke run now logs `max(nodes_per_graph)` and `p95(nodes_per_graph)`; batch size decision documented as data-driven. See §5.3. |
| Hostile | A | **`_build_cfg_node_features()` called but never defined or specced** — called at one site in `_build_control_flow_edges()`; zero definitions or documentation anywhere in the document; two implementers will produce different 12-dim vectors for CFG nodes; `in_unchecked` inheritance from parent function is an explicit risk of this gap | **High** | **Applied** — complete `_build_cfg_node_features()` spec added to §2.2C, including explicit `in_unchecked = 0.0` (not inherited from parent), loc computation, and synthetic node handling. |
| Hostile | B | **`_find_function_node()` has a Python syntax error** — the `raise ValueError(...)` f-string splits a list comprehension across two f-string literals (`[... for i in range(...) "` + `"if int(...)...]}`); Python evaluates each f-string independently; the unclosed `[` inside the first f-string is a `SyntaxError` at parse time — the test file will not import | **High** | **Applied** — list comprehension extracted to a local variable before f-string interpolation. See §Pre-flight. |
| Hostile | C | **`in_unchecked` regex fallback misses whitespace variants** — scanning for `"unchecked {"` and `"unchecked{"` misses valid Solidity syntax `unchecked\n{` (keyword and brace on separate lines) | Minor | **Applied** — regex changed to `re.search(r'\bunchecked\s*\{', content)`. See §2.2B. |
| Hostile | D | **Slither synthetic nodes not documented in `_build_control_flow_edges()`** — `func.nodes` includes synthetic ENTRY_POINT, EXPRESSION, and other synthetic nodes that have no source_mapping and no IRS; these silently get type CFG_NODE_OTHER (12) and loc=0.0; a developer may add special-case filtering that changes graph topology | Minor | **Applied** — documentation note added to `_build_control_flow_edges()`. See §2.2C. |

---

## 0. The Problem, Stated Plainly

v4 cleared a validation gate of F1-macro=0.5422. Then we tested it on 20 hand-crafted
contracts and it collapsed:

- **Detection rate: 15%** — 3 of 19 expected vulnerabilities found.
- **Specificity: 33%** — 2 of 3 safe contracts flagged as vulnerable.
- The model fires `Reentrancy` and `CallToUnknown` on nearly every contract with an
  external call, regardless of whether protections exist.
- `DenialOfService`, `IntegerUO` on Solidity 0.8+, `TransactionOrderDependence`,
  and `GasException` are essentially invisible.

Validation metrics were lying. The model learned shallow shortcuts. **This is a clean rebuild.**

---

## 1. Why the Current Model Is Broken — Root Cause Audit

*(Unchanged from Rev 1.4; all root causes confirmed.)*

### 1.1 The Graph Has No Sense of Time or Order

`func.nodes` and `.sons` (successors) are never accessed. A contract that calls before
zeroing a balance (vulnerable) and one that zeros before calling (safe CEI) produce
**identical graphs**. The GNN cannot learn the difference.

### 1.2 Node Features Carry No Semantic Vulnerability Signal

Current 8-dim feature vector includes `reentrant` (Slither's pre-computed detection —
a shortcut) and lacks `return_ignored`, `call_target_typed`, `in_unchecked`, `has_loop`,
`external_call_count`.

### 1.3 GNN Architecture Deficiencies

`in_channels=8` hardcoded. 3-layer GAT insufficient for complex CFG propagation.

### 1.4 Cross-Attention Query Vectors Are Structurally Blind

Garbage-in from the GNN means cross-attention queries cannot ask order-dependent questions.
Fix the GNN; the cross-attention works as designed.

### 1.5 Data Distribution Is Severely Imbalanced

`DenialOfService: 137 samples` vs `IntegerUO: 5,343 samples` (39× ratio). Data is the
only fix for DoS.

### 1.6 Label Co-occurrence Teaches the Wrong Correlations

Too few DoS samples to isolate the signal. The model learns "multi-vulnerable contracts
have external calls" and fires every call-adjacent label on any contract with a call.

---

## 2. What Changes — Complete List

### 2.1 `ml/src/preprocessing/graph_schema.py`

```python
FEATURE_SCHEMA_VERSION = "v2"
NODE_FEATURE_DIM       = 12    # was 8; removed reentrant and gas_intensity, added 5 new

FEATURE_NAMES = (
    "type_id",              # 0  — NODE_TYPES int (range 0–12)
    "visibility",           # 1  — VISIBILITY_MAP ordinal 0-2
    "pure",                 # 2  — bool
    "view",                 # 3  — bool
    "payable",              # 4  — bool
    # "reentrant" REMOVED — Slither shortcut
    "complexity",           # 5  — CFG block count
    "loc",                  # 6  — lines of code
    "return_ignored",       # 7  — NEW: 0.0 / 1.0 / -1.0 (IR unavailable sentinel)
    "call_target_typed",    # 8  — NEW: 0.0 / 1.0 / -1.0 (source unavailable sentinel)
    "in_unchecked",         # 9  — NEW: bool
    "has_loop",             # 10 — NEW: bool
    "external_call_count",  # 11 — NEW: float, log-normalized
    # "gas_intensity" REMOVED — circular heuristic over features already in vector
)

EDGE_TYPES = {
    "CALLS":        0,   # function → internal function call
    "READS":        1,   # function → state variable read
    "WRITES":       2,   # function → state variable write
    "EMITS":        3,   # function → event
    "INHERITS":     4,   # contract → parent contract
    "CONTAINS":     5,   # function → CFG_NODE child (NEW)
    "CONTROL_FLOW": 6,   # CFG_NODE → successor CFG_NODE (NEW, DIRECTED)
}
NUM_EDGE_TYPES = 7

NODE_TYPES = {
    "STATE_VAR":       0,
    "FUNCTION":        1,
    "MODIFIER":        2,
    "EVENT":           3,
    "FALLBACK":        4,
    "RECEIVE":         5,
    "CONSTRUCTOR":     6,
    "CONTRACT":        7,
    # CFG subtypes — distinct initial embeddings for different statement roles
    "CFG_NODE_CALL":   8,   # statement containing an external call
    "CFG_NODE_WRITE":  9,   # statement writing a state variable
    "CFG_NODE_READ":   10,  # statement reading a state variable
    "CFG_NODE_CHECK":  11,  # require / assert / if condition
    "CFG_NODE_OTHER":  12,  # all other statement types
}
# Total node types: 13 (ids 0–12)
```

**Why CFG subtypes matter:** With a single `CFG_NODE=8`, a CALL-statement node and
a WRITE-statement node have **identical initial embeddings** before any message passing.
All discrimination must come from float features alone — fragile. Distinct `type_id`
values give the GNN strong categorical initial signal from Layer 1. A CALL node (8) and
WRITE node (9) start from different representations; the order (which precedes which)
is then encoded by directed CONTROL_FLOW edges.

**Why `gas_intensity` was removed:** It was `f(complexity, has_loop, external_call_count)` — a
handcrafted linear combination of three features already present at indices 5, 10, 11. Keeping
it adds noise (same information encoded twice at different scales). The GNN learns the
combination itself and learns it better. `NODE_FEATURE_DIM` drops from 13 to 12.

### 2.2 `ml/src/preprocessing/graph_extractor.py`

**Hard dependency — add version check at extractor startup:**
```python
import slither as _slither_pkg
_version = tuple(int(x) for x in _slither_pkg.__version__.split('.')[:3])
if _version < (0, 9, 3):
    raise RuntimeError(
        f"slither-analyzer {_slither_pkg.__version__} is too old. "
        "v5 requires >=0.9.3 for NodeType.STARTUNCHECKED support. "
        "Pin in ml/pyproject.toml: slither-analyzer>=0.9.3,<0.11"
    )
```

This is a hard failure at import, not a warning. An old Slither silently produces wrong
`in_unchecked` features on every contract in the 68K dataset.

**A. Remove `reentrant` from `_build_node_features()`.Both `_build_node_features()` and `_build_cfg_node_features()` must return
`list[float]` of exactly `NODE_FEATURE_DIM` (12) elements, not a tensor.
`torch.tensor(x_list, dtype=torch.float)` requires all entries to be lists
of equal length. Returning a tensor from either function causes a silent
type mismatch that crashes final assembly.**
Delete: `reentrant = 1.0 if getattr(obj, "is_reentrant", False) else 0.0`

**B. Implement 5 new features + `node_metadata` storage:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
node_metadata  (parallel list, not a feature index)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
During graph construction, maintain a `node_metadata` list in parallel with `x_list`.
Every call to `_add_node()` must append a corresponding entry to `node_metadata`.

Format: list of dicts, one entry per node, indexed identically to x_list.
Required keys for ALL nodes:
    {
        "name":         str,   # canonical_name for Function/StateVar/etc.,
                               # or str(slither_node) for CFG nodes
        "type":         str,   # NODE_TYPES key string, e.g. "FUNCTION", "CFG_NODE_CALL"
        "source_lines": list,  # node.source_mapping.lines if available, else []
    }

After building all nodes:
    data = Data(x=..., edge_index=..., edge_attr=...)
    data.node_metadata = node_metadata   # ← REQUIRED; test infrastructure depends on this

`_build_control_flow_edges()` also appends to `node_metadata` for each CFG node it
creates (see §2.2C). The caller must pass `node_metadata` alongside `x_list`.

Why this is needed: `_find_function_node()` in the pre-flight test calls
`graph.node_metadata[i].get("name")` to map node indices to function names. Without
this attribute on the Data object, the test fails with AttributeError before reaching
the cosine similarity assertion — wasting a run and potentially causing incorrect
rejection of a working architecture.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
return_ignored  (index 7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLITHER IR ONLY — no text/regex fallback:

    try:
        for op in func.slithir_operations:
            if isinstance(op, (LowLevelCall, HighLevelCall)):
                if op.lvalue is None:
                    return 1.0   # return value is discarded
        return 0.0               # all calls capture return value
    except AttributeError:
        logger.warning(f"Slither IR unavailable for {func.canonical_name}")
        return -1.0   # SENTINEL — "unknown", not assumed safe

Why no regex fallback: the regex r'(?<![=,(])\s*\.call[\({]' was eliminated because:
  (1) Multi-line calls like `(bool ok,) =\n    addr.call(...)` — the `=` is on the
      preceding line, so lookbehind fails; function is incorrectly labelled "ignored".
  (2) `address(this).call{value: 0}("")` — self-call, ignored return; regex misses it.
  (3) Regex over multi-line source content is inherently fragile.
The sentinel -1.0 gives the GNN a distinct embedding for "unknown" vs "safe" vs "ignored".
Silently returning 0.0 on failure is a systematic false negative for MishandledException.

For non-Function nodes: 0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
call_target_typed  (index 8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY (Slither type analysis):
    Iterate func.high_level_calls and func.low_level_calls.
    For each: check receiver type.
    If ContractType  → typed (safe).
    If AddressType   → raw address → return 0.0 immediately.
    If all typed or no external calls → return 1.0.

FALLBACK (used only when Slither type resolution raises or returns None):
    if func.source_mapping is None or not func.source_mapping.content:
        # Source unavailable — cannot determine
        logger.warning(f"source_mapping unavailable for {func.canonical_name}")
        return -1.0   # SENTINEL — "unknown"

    # Source available — scan for raw external address patterns
    content = func.source_mapping.content
    # Exclude address(this) — self-calls are not external unknown-target calls
    raw_addr_pattern = re.compile(
        r'address\s*\(\s*(?!this\b)[^)]+\)\s*\.call'
    )
    if raw_addr_pattern.search(content):
        return 0.0   # raw external address call found
    return 1.0       # no raw address calls detected in available source

Sentinel policy: -1.0 when the answer cannot be determined. Returning 1.0 (safe) when
source_mapping is unavailable is a closed-world assumption error: "no evidence of danger"
≠ "confirmed safe". Use the sentinel to distinguish "confirmed typed" from "unknown".

For non-Function nodes: 1.0 (not applicable, default safe — these nodes don't make calls)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
in_unchecked  (index 9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY (Slither ≥0.9.3 — version enforced at import):
    Iterate func.nodes. If any node has type NodeType.STARTUNCHECKED → return 1.0.
    Note: do NOT use NodeType.ASSEMBLY. Inline assembly and `unchecked {}` are distinct
    Solidity constructs. ASSEMBLY blocks may appear in gas-efficient code with no
    arithmetic overflow risk — using ASSEMBLY as a proxy produces false positives.

FALLBACK (regex, only when primary raises AttributeError):
    Scan func.source_mapping.content for the `unchecked` keyword followed by `{`.
    Use: re.search(r'\bunchecked\s*\{', func.source_mapping.content)
    This pattern covers all valid Solidity 0.8+ `unchecked {}` syntax variants:
      - `unchecked {`   (space between keyword and brace — common style)
      - `unchecked{`    (no space — valid Solidity)
      - `unchecked\n{`  (keyword and brace on separate lines — valid Solidity)
    Do NOT use a plain string search for "unchecked {" or "unchecked{" — these
    miss the newline variant and produce false negatives on valid Solidity.

For non-Function nodes: 0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
has_loop  (index 10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For Function nodes:
    Check func.nodes for any node with type in
    {NodeType.IFLOOP, NodeType.STARTLOOP, NodeType.ENDLOOP}.
For non-Function nodes: 0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
external_call_count  (index 11)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For Function nodes:
    count = len(func.high_level_calls) + len(func.low_level_calls)
    log1p(count) / log1p(20), clamped [0,1]
    (20 calls → 1.0;  1 call → 0.23;  5 calls → 0.60)
For non-Function nodes: 0.0
```

**C. Add `_cfg_node_type()`, `_build_cfg_node_features()`, and `_build_control_flow_edges()`:**

```python
def _cfg_node_type(slither_node) -> int:
    """
    Map a Slither CFG node to a NODE_TYPES CFG subtype id.

    PRIORITY ORDER (highest to lowest):
      1. CFG_NODE_CALL  — any node containing an external call
      2. CFG_NODE_WRITE — any node writing a state variable
      3. CFG_NODE_READ  — any node reading a state variable
      4. CFG_NODE_CHECK — require / assert / if condition
      5. CFG_NODE_OTHER — everything else

    Justification for priority order:
      External calls are the root cause of reentrancy, CallToUnknown, and
      MishandledException. When Slither generates a merged IR node that contains
      both an external call AND another operation (e.g., `(bool ok,) = addr.call(...)`
      produces a call node that also assigns to a local variable), the external call
      is the most vulnerability-relevant operation and must not be hidden behind a
      WRITE or CHECK label. WRITE takes second priority because state modifications
      are the other half of the call-before-write pattern. If a node has both a
      state write AND a require (common in compound assignment patterns), the write
      is more specific to the vulnerability pattern.

    On Slither IR merging: Slither occasionally emits one IR node for statements like
      `(bool ok,) = addr.call(...); require(ok);`
    when both are trivially sequential. Such a merged node has both has_ext_call=True
    and has_check=True. It is assigned CFG_NODE_CALL (priority 1). The require is
    represented in the node's feature vector via `return_ignored=0.0` (return value
    captured) rather than by the node type. This is an acceptable approximation.
    """
    from slither.core.cfg.node import NodeType as SNT
    from slither.slithir.operations import LowLevelCall, HighLevelCall
    from slither.core.variables.state_variable import StateVariable

    check_types = {SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.ENDLOOP,
                   SNT.THROW}

    # Priority 1: external call present in this node's IR
    has_ext_call = any(
        isinstance(op, (LowLevelCall, HighLevelCall))
        for op in slither_node.irs
    )
    if has_ext_call:
        return NODE_TYPES["CFG_NODE_CALL"]    # 8

    # Priority 2: state variable write present
    has_sv_write = any(
        hasattr(op, 'lvalue') and isinstance(op.lvalue, StateVariable)
        for op in slither_node.irs
    )
    if has_sv_write:
        return NODE_TYPES["CFG_NODE_WRITE"]   # 9

    # Priority 3: state variable read present
    has_sv_read = any(
        isinstance(v, StateVariable)
        for op in slither_node.irs
        for v in getattr(op, 'read', [])
    )
    if has_sv_read:
        return NODE_TYPES["CFG_NODE_READ"]    # 10

    # Priority 4: control-flow check (require / assert / if / loop)
    if slither_node.type in check_types:
        return NODE_TYPES["CFG_NODE_CHECK"]   # 11

    return NODE_TYPES["CFG_NODE_OTHER"]       # 12


def _build_cfg_node_features(slither_node, func, cfg_type: int) -> list:
    """
    Build the 12-dim feature vector for a CFG (statement-level) node.

    CFG nodes are statement-level. Most function-level features are not applicable
    at the statement level and are set to 0.0. See per-index notes below.

    Index  Name               Value for CFG nodes
    ─────  ─────────────────  ──────────────────────────────────────────────────
      0    type_id            cfg_type (8–12, from _cfg_node_type())
      1    visibility         0.0 — not applicable
      2    pure               0.0 — not applicable
      3    view               0.0 — not applicable
      4    payable            0.0 — not applicable
      5    complexity         0.0 — function-level metric, not per-statement
      6    loc                len(source_mapping.lines) if available, else 0.0
      7    return_ignored     0.0 — not per-statement in v5.0 (function node carries
                                    this signal; v5.1 target: per-CFG_NODE_CALL node)
      8    call_target_typed  1.0 — default safe (not applicable at statement level)
      9    in_unchecked       0.0 — NEVER inherited from parent func's flag.
                                    Reason: if a function has any unchecked block, ALL
                                    its child CFG nodes would get 1.0 — including
                                    statements outside the unchecked scope. This creates
                                    false positives for IntegerUO on safe statements.
                                    The function-level node already carries this signal
                                    and the GNN propagates it via Phase 1 CONTAINS edges.
                                    v5.1 target: per-node scope analysis using source
                                    line ranges to determine true containment.
     10    has_loop           0.0 — not applicable at statement level
     11    external_call_count 0.0 — not applicable at statement level

    IMPORTANT — Slither synthetic nodes:
      func.nodes includes synthetic nodes: ENTRY_POINT, EXPRESSION, BEGIN_LOOP, END_LOOP,
      and others that Slither generates internally. These nodes may have:
        - slither_node.irs == []  (no IR operations)
        - slither_node.source_mapping == None  (no source location)
      They are handled correctly by this function:
        - _cfg_node_type() falls through to CFG_NODE_OTHER (12) for them
        - loc defaults to 0.0 when source_mapping is absent
      Do NOT add filtering to skip synthetic nodes — they are part of the CFG structure
      and removing them changes the graph topology used for CONTROL_FLOW edges.
    """
    loc = 0.0
    if slither_node.source_mapping and slither_node.source_mapping.lines:
        loc = float(len(slither_node.source_mapping.lines))

    return [
        float(cfg_type),  # 0: type_id
        0.0,              # 1: visibility
        0.0,              # 2: pure
        0.0,              # 3: view
        0.0,              # 4: payable
        0.0,              # 5: complexity
        loc,              # 6: loc
        0.0,              # 7: return_ignored — not per-node in v5.0
        1.0,              # 8: call_target_typed — default safe
        0.0,              # 9: in_unchecked — NEVER inherited from parent function
        0.0,              # 10: has_loop
        0.0,              # 11: external_call_count
    ]


def _build_control_flow_edges(
    func,
    func_node_idx: int,
    node_index_map: dict,
    x_list: list,
    node_metadata: list,
) -> tuple:
    """
    For a given function, build CFG_NODE children and their edges.

    Appends new node feature vectors to x_list and entries to node_metadata in-place.
    Populates node_index_map with slither_node → graph_idx mappings for CFG nodes.

    Args:
        func:           Slither Function object.
        func_node_idx:  Graph node index of the parent FUNCTION node (already in x_list).
        node_index_map: Dict mapping slither objects → graph node indices. Mutated.
        x_list:         Shared list of all node feature vectors. Mutated.
        node_metadata:  Shared list of node metadata dicts. Mutated. Must be same length
                        as x_list at entry and at exit.

    Returns:
        (contains_edges, control_flow_edges):
            contains_edges:     list of (func_node_idx, cfg_graph_idx) pairs — edge type 5
            control_flow_edges: list of (cfg_src_idx, cfg_dst_idx) pairs — edge type 6

    INDEX ASSIGNMENT — CRITICAL:
      graph_idx = len(x_list)
      This is the correct index for the next node because x_list is the single shared
      list across ALL node types (CONTRACT, STATE_VAR, FUNCTION, CFG_NODE). Its length
      before appending is exactly the next available graph index.

      Do NOT use len(node_index_map): node_index_map only tracks Slither CFG objects,
      not the full set of graph nodes (which includes CONTRACT, STATE_VAR, FUNCTION
      nodes added by other code paths). Using node_index_map length produces incorrect
      indices and broken edge_index tensors.

    CFG NODE ORDERING — DETERMINISTIC SORT (mandatory):
      Slither's func.nodes list order is an internal traversal order that can change
      between Slither versions. Always sort by (source_line, node_id) to guarantee
      identical node ordering across extraction runs, Slither versions, and inference.
      Without this, training graphs and inference graphs may get different topologies
      from the same source code.

      Synthetic nodes (ENTRY_POINT etc.) with no source_mapping get line 0 as fallback.
      node_id is a unique integer per Slither node and provides a stable tiebreak.

    SLITHER SYNTHETIC NODES:
      func.nodes includes synthetic nodes (ENTRY_POINT, EXPRESSION, BEGIN_LOOP, etc.)
      with no source_mapping and empty IRS. These are included in the CFG structure —
      do NOT filter them out. They are typed as CFG_NODE_OTHER (12) by _cfg_node_type()
      and get loc=0.0 from _build_cfg_node_features(). Removing them would break
      CONTROL_FLOW edges that pass through them (e.g., loop entry/exit nodes).
    """
    cfg_nodes = sorted(
        func.nodes,
        key=lambda n: (
            n.source_mapping.lines[0]
            if n.source_mapping and n.source_mapping.lines else 0,
            n.node_id,  # tiebreak: same-line nodes (ternary expressions, etc.)
        )
    )

    contains_edges     = []   # (func_node_idx, cfg_graph_idx) — type 5
    control_flow_edges = []   # (cfg_src_idx, cfg_dst_idx) — type 6

    # Pass 1: assign indices, build features, populate x_list and node_metadata
    for slither_node in cfg_nodes:
        cfg_type  = _cfg_node_type(slither_node)
        graph_idx = len(x_list)                     # CORRECT: next available graph index
        node_index_map[slither_node] = graph_idx

        cfg_features = _build_cfg_node_features(slither_node, func, cfg_type)
        x_list.append(cfg_features)

        # Build matching metadata entry (must stay index-aligned with x_list)
        cfg_type_name = {v: k for k, v in NODE_TYPES.items()}.get(cfg_type, "CFG_NODE_OTHER")
        node_metadata.append({
            "name":         str(slither_node),
            "type":         cfg_type_name,
            "source_lines": list(slither_node.source_mapping.lines)
                            if slither_node.source_mapping and slither_node.source_mapping.lines
                            else [],
        })

        contains_edges.append((func_node_idx, graph_idx))

    # Pass 2: build CONTROL_FLOW edges (requires all CFG nodes to be indexed first)
    for slither_node in cfg_nodes:
        src_idx = node_index_map[slither_node]
        for successor in slither_node.sons:
            if successor in node_index_map:
                dst_idx = node_index_map[successor]
                control_flow_edges.append((src_idx, dst_idx))

    return contains_edges, control_flow_edges
```

**Graph construction order** (deterministic):
```
CONTRACT node (index 0)   → append to x_list AND node_metadata
STATE_VAR nodes            → append to x_list AND node_metadata
FUNCTION nodes             → append to x_list AND node_metadata (in source-line order)
  For each function:
    _build_control_flow_edges(func, func_node_idx, node_index_map, x_list, node_metadata)
    → appends CFG_NODE children (sorted by source_line, node_id)
MODIFIER nodes             → append to x_list AND node_metadata
EVENT nodes                → append to x_list AND node_metadata

Final assembly:
    data = Data(
        x          = torch.tensor(x_list, dtype=torch.float),
        edge_index = ...,
        edge_attr  = ...,
    )
    data.node_metadata = node_metadata   # list of dicts, same length as data.x
```

### 2.3 `ml/src/models/gnn_encoder.py`

**Three-phase, four-layer architecture:**

The critical fix for Phase 2 signal propagation (Critic Problem 3, Rev 1.5) is adding Phase 3:
a reverse-CONTAINS pass that aggregates Phase-2-enriched CFG_NODE embeddings back up
into the function nodes. Without Phase 3, CONTAINS edges run in Phase 1 with pre-Phase-2
CFG embeddings, and Phase 2's directional enrichment is trapped in the CFG subgraph —
the function node's final embedding contains no execution-order information.

```
Phase 1 (Layers 1+2): Structural aggregation
  Edges: types 0–5 (CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS)
  add_self_loops=True
  Propagates function-level properties DOWN into CFG_NODE children via CONTAINS.

Phase 2 (Layer 3): CFG-directed aggregation
  Edges: type 6 (CONTROL_FLOW) only
  add_self_loops=False  ← CRITICAL: self-loops destroy directional signal
  Enriches CFG_NODE embeddings with execution-order information.
  Note: 1 message-passing hop — sufficient for diameter-2 CFGs (basic reentrancy:
  require→call→write). Insufficient for diameter-4+ CFGs. v5.1 target: increase
  to 2 hops with gnn_layers=5.

Phase 3 (Layer 4): Reverse-CONTAINS aggregation
  Edges: type 5 CONTAINS, REVERSED (i.e., CFG_NODE → function direction)
  add_self_loops=False
  Aggregates Phase-2-enriched CFG_NODE embeddings UP into function nodes.
  This is the path by which execution-order information reaches the FUNCTION nodes
  that the classifier ultimately operates on. Without Phase 3, function-node embeddings
  are order-blind regardless of how well Phase 2 enriches the CFG nodes.

  KNOWN LIMITATION — Phase 3 edge embedding symmetry:
  The reversed CONTAINS edges (CFG_NODE → FUNCTION) use the same type-5 edge embedding
  as forward CONTAINS edges (FUNCTION → CFG_NODE). The GNN has no way to distinguish
  "I am a CONTAINS edge going down (parent→child)" from "I am a CONTAINS edge going up
  (child→parent)" via the edge attribute alone — both produce embedding(5).
  Impact: during training, if the model attempts to learn asymmetric semantics through
  Phase 1 (parent sends context down) and Phase 3 (child sends order signal up), it
  cannot encode this asymmetry in the edge representation. The positional asymmetry
  (which side of edge_index the node appears on) still provides some signal via the
  GATConv attention mechanism, but this is weaker than a dedicated edge type.
  v5.1 target: add REVERSE_CONTAINS = 7 with its own embedding. This is a non-breaking
  schema addition (NUM_EDGE_TYPES: 7 → 8). For v5.0 the limitation is accepted.
```

```python
class GNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim:    int   = 128,
        heads:         int   = 8,
        dropout:       float = 0.2,
        use_edge_attr: bool  = True,
        edge_emb_dim:  int   = 32,
        num_layers:    int   = 4,
    ) -> None:
        super().__init__()
        # num_layers is accepted here for interface completeness.
        # Validation is in TrainConfig.__post_init__(), not here,
        # so the guard fires before data loading / GPU allocation.
        self.num_layers  = num_layers
        self.hidden_dim  = hidden_dim
        self.dropout_p   = dropout
        _head_dim        = hidden_dim // heads   # 16 per head when hidden=128, heads=8

        self.edge_embedding = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)  # 7 × 32

        # ── Phase 1 — structural + CONTAINS forward ─────────────────────────────
        # add_self_loops=True: self-loops are harmless for non-directional aggregation.
        # out_channels = hidden_dim // heads = 16 per head; with 8 heads + concat=True
        # → total output 128 dim. Note: out_channels in PyG GATConv is PER HEAD, not total.
        # Passing out_channels=128 with heads=8, concat=True gives 1024-dim output — wrong.
        self.conv1 = GATConv(
            in_channels=NODE_FEATURE_DIM,  # 12
            out_channels=_head_dim,         # 16 per head
            heads=heads,                    # 8
            concat=True,                    # total out = 128
            add_self_loops=True,
            edge_dim=edge_emb_dim,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim,         # 128
            out_channels=_head_dim,         # 16
            heads=heads,
            concat=True,
            add_self_loops=True,
            edge_dim=edge_emb_dim,
        )

        # ── Phase 2 — CONTROL_FLOW directed ─────────────────────────────────────
        # add_self_loops=False: CRITICAL. Self-loops add an edge from each node to
        # itself. During attention for CONTROL_FLOW edges, each CFG_NODE would then
        # attend to both its genuine predecessor (execution order signal) and to itself
        # (no order information). The self-loop term partially cancels the directional
        # signal we are trying to capture. With add_self_loops=False, only genuine
        # directed CONTROL_FLOW edges participate in attention.
        # heads=1: multi-head attention is designed for diverse relationship types.
        # CONTROL_FLOW edges are a single relationship type (execution order); one
        # attention head with more capacity per head is preferable here.
        self.conv3 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,        # 128 total (1 head, concat=False)
            heads=1,
            concat=False,
            add_self_loops=False,           # CRITICAL
            edge_dim=edge_emb_dim,
        )

        # ── Phase 3 — reverse-CONTAINS (CFG_NODE → function) ────────────────────
        # Uses flipped CONTAINS edges. CFG_NODE nodes send their Phase-2-enriched
        # embeddings to their parent FUNCTION nodes.
        # add_self_loops=False: we only want CFG → function aggregation, not
        # self-attention at function nodes.
        self.conv4 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=False,
            edge_dim=edge_emb_dim,
        )

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_attr=None, return_intermediates=False):
        e = self.edge_embedding(edge_attr) if edge_attr is not None else None

        # ── Edge masks ──────────────────────────────────────────────────────────
        struct_mask   = edge_attr <= 5               # types 0–5 (structural + CONTAINS)
        cfg_mask      = edge_attr == 6               # type 6 (CONTROL_FLOW)
        contains_mask = edge_attr == 5               # type 5 (CONTAINS only, for reversal)

        struct_ei = edge_index[:, struct_mask]
        struct_ea = e[struct_mask] if e is not None else None

        cfg_ei    = edge_index[:, cfg_mask]
        cfg_ea    = e[cfg_mask]   if e is not None else None

        # Reverse-CONTAINS: flip source↔target so CFG_NODE → function.
        # Note: both forward and reversed CONTAINS edges use the same edge_attr value (5),
        # producing the same embedding(5). The GNN cannot distinguish direction purely from
        # the edge embedding — it relies on GATConv's positional asymmetry (which side of
        # the adjacency matrix the node appears on). This is a known v5.0 limitation.
        # v5.1: add REVERSE_CONTAINS = 7 with its own embedding for full asymmetry.
        rev_contains_ei = edge_index[:, contains_mask].flip(0)  # [2, E_contains]
        rev_contains_ea = e[contains_mask] if e is not None else None

        # Initialise intermediates dict early — populated at each phase boundary.
        # Only materialised into the return value when return_intermediates=True.
        _intermediates = {}

        # ── Phase 1 — structural aggregation ────────────────────────────────────
        # Layer 1: in_channels=12, out=128. No residual here — dimensions differ (12 ≠ 128).
        x  = self.conv1(x, struct_ei, struct_ea)   # [N, 12] → [N, 128]
        x  = self.relu(x)
        x  = self.dropout(x)
        # Layer 2: residual from Layer 1 output (both 128).
        x2 = self.conv2(x, struct_ei, struct_ea)   # [N, 128] → [N, 128]
        x2 = self.relu(x2)
        x  = self.dropout(x2 + x)                  # residual ✓ (same dim)

        _intermediates["after_phase1"] = x.detach().clone()

        # ── Phase 2 — CONTROL_FLOW directed ─────────────────────────────────────
        # Non-CFG_NODE nodes have no CONTROL_FLOW edges (cfg_ei has no entries
        # pointing to/from them). GATConv returns zero for nodes with no incoming edges.
        # They receive no Phase 2 messages and carry their Phase 1 embeddings forward.
        x2 = self.conv3(x, cfg_ei, cfg_ea)
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                  # residual

        _intermediates["after_phase2"] = x.detach().clone()

        # ── Phase 3 — reverse-CONTAINS ──────────────────────────────────────────
        # CFG_NODE nodes (enriched by Phase 2) send to FUNCTION nodes.
        # Non-CFG nodes and non-FUNCTION nodes are uninvolved (no entries in rev_contains_ei).
        #
        # ZERO-MESSAGE BEHAVIOUR (correct by design — do not "fix"):
        # FUNCTION nodes with no CFG children (e.g., view functions whose body Slither
        # does not generate CFG nodes for) appear as targets in rev_contains_ei only
        # if they have at least one CFG_NODE child. Functions with no CFG children
        # receive no Phase 3 messages, so conv4 returns zero for them, and the residual
        # x = x + dropout(0) is a no-op. These functions retain their Phase 2 embedding.
        # This is correct behaviour, not a bug.
        #
        # DO NOT add add_self_loops=True to conv4 to "fix" zero-message nodes. Self-loops
        # on conv4 would mix self-attention into the reverse aggregation for FUNCTION nodes
        # that DO have CFG children, diluting the Phase 2 order signal they receive.
        x2 = self.conv4(x, rev_contains_ei, rev_contains_ea)
        x2 = self.relu(x2)
        x  = x + self.dropout(x2)                  # residual

        _intermediates["after_phase3"] = x.detach().clone()

        if return_intermediates:
            return x, batch, _intermediates
        return x, batch
```

**`num_layers` validation — in `TrainConfig.__post_init__()` only:**
```python
# GNNEncoder.__init__() accepts num_layers, stores it, does NOT raise.
# Validation fires at startup, before data loading or GPU allocation.
self.num_layers = num_layers   # stored for serialisation only
```

**Update `gnn_hidden_dim` default to 128.**

### 2.4 `ml/src/models/fusion_layer.py`

**No structural changes.** The CrossAttentionFusion is correctly implemented.

**Documented v5.0 limitation:** The fused eye's token side uses masked mean pooling
(order-blind). The CLS token in the Transformer eye provides order-aware coverage for
that path. The fused eye adds cross-modal co-location signal even without order-awareness
on the token side. This limitation is known and documented — not a silent gap.
V5.1 target: replace mean pooling in fusion token aggregation with attention pooling
with positional bias.

### 2.5 `ml/src/models/sentinel_model.py`

**A. Add two new projection layers:**
```python
from torch_geometric.nn import global_max_pool, global_mean_pool

# GNN eye: max+mean pool → project
self.gnn_eye_proj = nn.Sequential(
    nn.Linear(gnn_hidden_dim * 2, fusion_output_dim),  # 256 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
)

# Transformer eye: CLS token → project
self.transformer_eye_proj = nn.Sequential(
    nn.Linear(768, fusion_output_dim),  # 768 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
)
# Activation is consistently ReLU+Dropout for both projections.
```

**B. Classifier (384→10):**
```python
self.classifier = nn.Linear(fusion_output_dim * 3, num_classes)  # 384 → 10
```

**C. Forward pass (unchanged from Rev 1.4 — three-eye + aux heads).**

**D. Auxiliary heads (3 × Linear 128→10) — training only, `return_aux=False` default.**

**E. Docstring** — updated to document three-phase GNN, three-eye architecture, v5.0
limitations (1-hop CONTROL_FLOW, mean-pool fused eye, Phase 3 edge embedding symmetry).

### 2.6 `ml/src/training/trainer.py`

**A. `TrainConfig.__post_init__()` — validation at startup:**
```python
@dataclass
class TrainConfig:
    gnn_hidden_dim:   int   = 128
    gnn_layers:       int   = 4
    checkpoint_name:  str   = "multilabel-v5-fresh_best.pt"
    epochs:           int   = 60
    lr:               float = 2e-4
    aux_loss_weight:  float = 0.1

    def __post_init__(self):
        if self.gnn_layers != 4:
            raise ValueError(
                f"gnn_layers={self.gnn_layers} is not supported in v5.0. "
                "Only gnn_layers=4 is implemented (3-phase: 2 structural, "
                "1 CONTROL_FLOW directed, 1 reverse-CONTAINS). "
                "v5.1 target: gnn_layers=5 for 2 CONTROL_FLOW hops."
            )
```

**B. Training loop — auxiliary loss (unchanged from Rev 1.4).**

**C. Per-eye gradient norm logging (unchanged from Rev 1.4).**

**D. Per-class calibration logging (unchanged from Rev 1.4).**

**E. `pos_weight` — replace hard cap with sqrt scaling:**
```python
def _compute_pos_weight(train_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class pos_weight for BCEWithLogitsLoss.

    Formula: pos_weight[i] = sqrt((N - n_pos[i]) / n_pos[i])

    Rationale for sqrt vs raw ratio:
      Raw ratio for DoS (437 samples in 68K): (68300-437)/437 ≈ 155.
      Raw ratio for IntegerUO (5343 samples): (68300-5343)/5343 ≈ 11.7.
      A 13× ratio between the rarest and most common class causes training
      instability — the optimizer spends most steps recovering from large
      DoS gradients.

      sqrt scaling: sqrt(155) ≈ 12.4 vs sqrt(11.7) ≈ 3.4 — a 3.6× ratio.
      This preserves the ordering (DoS gets proportionally more weight than
      IntegerUO) without extreme values that destabilise training.
      No arbitrary cap constant needed.

    IMPORTANT: recompute from scratch after augmentation. The distribution
    changes when 300 DoS + 500 safe contracts are added. Pre-augmentation
    values must not be reused.
    """
    N = train_labels.shape[0]
    n_pos = train_labels.sum(dim=0).clamp(min=1.0)  # avoid division by zero
    raw_weight = (N - n_pos) / n_pos
    return torch.sqrt(raw_weight)

pos_weight = _compute_pos_weight(train_labels)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 2.7 `ml/src/training/focalloss.py`

Add `MultiLabelFocalLoss(alpha: List[float], gamma: float)` for use at epoch 15
if over-prediction persists. Do **not** use global `α=0.25` default — documented
to severely hurt rare-class recall.

### 2.8 `ml/src/inference/preprocess.py`

Verify imports from `graph_extractor.py`. Update `predictor.py` for new architecture.

### 2.9 `ml/scripts/validate_graph_dataset.py`

```python
assert graph.x.shape[1] == NODE_FEATURE_DIM   # 12
assert graph.edge_attr.max() < NUM_EDGE_TYPES  # 7
# Existence checks (catch silent extraction failures):
assert any(g.edge_attr == 5 for g in graphs), "No CONTAINS edges found"
assert any(g.edge_attr == 6 for g in graphs), "No CONTROL_FLOW edges found"
assert any(g.x[:, 0] == 8 for g in graphs), "No CFG_NODE_CALL nodes found"
assert any(g.x[:, 0] == 9 for g in graphs), "No CFG_NODE_WRITE nodes found"
# node_metadata alignment check:
for g in graphs:
    assert hasattr(g, 'node_metadata'), "graph.node_metadata missing — extraction bug"
    assert len(g.node_metadata) == g.x.shape[0], (
        f"node_metadata length {len(g.node_metadata)} ≠ x.shape[0] {g.x.shape[0]}"
    )
# Sentinel tracking:
sentinel_count = sum((g.x[:, 7] == -1.0).sum().item() for g in graphs)
total_nodes    = sum(g.x.shape[0] for g in graphs)
if sentinel_count / total_nodes > 0.05:
    logger.warning(
        f"return_ignored sentinel rate {sentinel_count/total_nodes:.1%} exceeds 5%. "
        "Investigate Slither IR availability for affected contracts."
    )
```

---

## 3. Data Pipeline — Complete Plan

### 3.1 Current Dataset Inventory

| Class | Training Samples | Problem Level |
|---|---|---|
| IntegerUO | ~5,343 | None — strongest class |
| GasException | ~2,589 | Moderate — over-predicts |
| MishandledException | ~2,207 | Moderate — over-predicts |
| Reentrancy | ~2,501 | Moderate — false positives on safe CEI |
| TransactionOrderDependence | ~1,800 | High — over-predicts 2.3× |
| ExternalBug | ~1,622 | High — over-predicts 2.3× |
| UnusedReturn | ~1,716 | Moderate — over-predicts |
| Timestamp | ~1,077 | Moderate |
| CallToUnknown | ~1,266 | High — fires on all calls |
| DenialOfService | ~137 | Critical — data starvation |
| **Safe contracts** | **Unknown — likely very few** | **Critical — false positive source** |

### 3.2 Required Data Augmentation

**Priority 1 — Safe Contracts (500+ target):**
CEI-safe reentrancy, typed interfaces, transfer/send, pull-payment, ReentrancyGuard,
checked return values, bounded loops, safe `unchecked {}` (gas optimisation only).

**Generation strategy (mutation-based):**
```
Reentrancy → CEI-safe: swap call/write order. Compile check → Slither verify.
MishandledException → safe: wrap bare call() with (bool ok,) = ...; require(ok).
CallToUnknown → typed: replace raw address call with ITarget(addr).method().
DoS → bounded loop: add require(arr.length <= 100) guard.
```

**Priority 2 — DenialOfService (300+ target):** SmartBugs, SWC #128.
**Priority 3 — IntegerUO with `unchecked` (200+ target)**
**Priority 4 — CallToUnknown disambiguation (200+ target)**
**Priority 5 — TransactionOrderDependence / MishandledException (100+ each)**

### 3.3 Data Pipeline Steps

1. Source/generate contracts. 2. Compile. 3. Extract graphs (v2 schema, 12-dim, 7 types).
4. Tokenise. 5. Label. 6. Re-extract all 68K with `--force`. 7. Split with `--freeze-val-test`.
8. Validate.

**Split design:** Original v4 validation set preserved intact. Augmented contracts go
to training only. 10% of augmented data → secondary behavioural val (not used for
F1-macro gate).

### 3.4 Labeling Protocol

Slither automated first pass + manual review. For safe contracts: zero Slither findings
AND manually confirmed protective patterns.

### 3.5 `generate_safe_variants.py` — Two-Step Verification Protocol

```python
def generate_cei_safe(vulnerable_contract_path: Path) -> Optional[Path]:
    safe_path = _swap_call_and_write(vulnerable_contract_path)

    # Step 1: Compilation check — MUST precede Slither.
    # A syntactically invalid swap will fail here. Without this gate, Slither
    # may fail cryptically (mishandled as "zero findings") or produce incomplete
    # analysis on a malformed contract, generating a false "safe" verdict.
    compile_result = _compile_solidity(safe_path)
    if compile_result.returncode != 0:
        logger.error(f"CEI swap did not compile: {safe_path}\n{compile_result.stderr}")
        safe_path.unlink()
        return None

    # Step 2: Slither verification — catches semantically invalid swaps.
    # A syntactically correct swap may still be vulnerable due to:
    #   - A missed second write path
    #   - A modifier that re-enters before the write
    #   - A cross-function reentrancy the swap didn't address
    slither_findings = _run_slither(safe_path)
    reentrancy_findings = [f for f in slither_findings if "reentrancy" in f["check"]]
    if reentrancy_findings:
        logger.error(f"CEI swap still vulnerable: {safe_path}\n{reentrancy_findings}")
        safe_path.unlink()
        return None

    return safe_path
```

---

## 4. Model Architecture Choices — Decision Log

### 4.1 Why Remove `reentrant` Slither Flag

Removing it forces the model to detect reentrancy from structural patterns (call order vs.
state write order), not from Slither's pre-computed label.

### 4.2 Why Statement-Level CFG Nodes

"Call before write" requires separate nodes for the call statement and the write statement.
Adding CONTROL_FLOW edges between function-level nodes provides no intra-function ordering.

### 4.3 Why Three-Phase GNN (Phase 3 is the critical addition)

The fundamental flaw with a two-phase design is signal propagation:

```
Two-phase (broken):
  Phase 1: CONTAINS (function → CFG_NODE) — runs on pre-Phase-2 embeddings
  Phase 2: CONTROL_FLOW — enriches CFG_NODE embeddings
  ← Function node never receives Phase-2-enriched CFG embeddings.
     Order information is trapped in the CFG subgraph.

Three-phase (correct):
  Phase 1: structural + CONTAINS forward — function properties flow DOWN
  Phase 2: CONTROL_FLOW directed — execution order encoded in CFG nodes
  Phase 3: reverse-CONTAINS (CFG_NODE → function) — order signal flows UP
```

**Phase 3 is the path by which execution-order information reaches the FUNCTION nodes
that the classifier ultimately operates on.** Without it, the GNN eye produces
order-blind function embeddings despite all the CFG machinery below it.

**Known depth limitation:** Phase 2 (Layer 3) provides 1 CONTROL_FLOW message-passing hop.
For a basic reentrancy function (require → call → write, CFG diameter=2), 1 hop is
sufficient — the write node sees its predecessor (the call node) and the function node
aggregates both via Phase 3. For functions with complex branching or exception handling
(CFG diameter 4–6), 1 hop may not fully propagate ordering through the chain.
V5.1 target: `gnn_layers=5` (2 CONTROL_FLOW hops in Phase 2).

**Known Phase 3 edge embedding limitation:** Reversed CONTAINS edges use the same type-5
embedding as forward CONTAINS edges. The GNN cannot encode the directional asymmetry
("context flowing down" vs "order signal flowing up") in the edge representation.
This limits the model's ability to learn maximally asymmetric semantics across Phases 1
and 3. For v5.0, the GATConv positional asymmetry provides partial compensation.
V5.1 target: REVERSE_CONTAINS = 7 with a dedicated embedding.

**Why 4 layers total (not 5):**
For v5.0, 1 CONTROL_FLOW hop covers the primary reentrancy detection case and limits
architectural complexity while the rest of the system (features, data, training) is
being validated. Increasing to 2 hops is a conservative, isolated change for v5.1.

### 4.4 GATv2 vs GAT

Keep GAT (GATConv) for v5.0 stability. GATv2 is a v5.1 consideration.

### 4.5 Three-Eye Classifier Architecture

**Each eye answers a different question:**

| Eye | What it answers | Inductive bias | v5.0 limitation |
|---|---|---|---|
| GNN eye | "What structural patterns exist?" | Max+mean pool over CFG-enriched nodes | 1 CONTROL_FLOW hop; Phase 3 edge embedding symmetric |
| Transformer eye | "What does source code say?" | CLS token: 12-layer order-aware summary | None |
| Fused eye | "How do structure and tokens co-locate?" | Cross-attention: joint evidence | Token side uses mean pool (order-blind) |

**Eye dominance prevention:** Auxiliary loss (`main + 0.1 × (aux_gnn + aux_tf + aux_fused)`).
Effective auxiliary weight ≈ 23% of total loss. Per-eye gradient norm logged every
`log_interval` batches.

### 4.6 Frozen vs Trained

Train everything from scratch. Schema change makes v4 weights incompatible.

---

## 5. Training Strategy

### 5.1 Hyperparameters

```
epochs:              60 (early stopping patience=10)
batch_size:          16 (reduced from 32 — larger graphs with CFG nodes;
                         data-driven: smoke run logs max and p95 nodes/graph
                         to verify this is sufficient before full training)
lr:                  2e-4
weight_decay:        1e-2
warmup_pct:          0.10
grad_clip:           1.0
loss_fn:             bce (assess at epoch 10 for over-prediction)
early_stop_patience: 10
lora_r:              16
lora_alpha:          32
gnn_hidden_dim:      128
gnn_heads:           8
gnn_dropout:         0.2
gnn_edge_emb_dim:    32
fusion_attn_dim:     256
fusion_num_heads:    8
fusion_output_dim:   128
aux_loss_weight:     0.1
```

**`pos_weight` — sqrt scaling (compute after augmentation, do not reuse old values):**
```python
pos_weight[i] = sqrt((N - n_pos[i]) / n_pos[i])

# Post-augmentation reference (approximate):
#   DenialOfService (~437 samples in 68K): sqrt(155) ≈ 12.4
#   IntegerUO (~5343 samples):             sqrt(11.7) ≈ 3.4
#   Reentrancy (~2501 samples):            sqrt(26.3) ≈ 5.1
# DoS gets ~3.6× more weight than IntegerUO — proportional, not capped.
```

### 5.2 Weighted Sampler

Enable `use_weighted_sampler="all-rare"` from the start.

### 5.3 Training Phases

**Phase A — Smoke run (1-2 epochs, 10% subsample):** Verify shapes, forward pass, OOM.

During the smoke run, log graph size statistics before training begins:
```python
nodes_per_graph = [g.x.shape[0] for g in dataset]
logger.info(f"Graph size: max={max(nodes_per_graph)}, "
            f"p95={sorted(nodes_per_graph)[int(0.95 * len(nodes_per_graph))]}, "
            f"mean={sum(nodes_per_graph)/len(nodes_per_graph):.1f}")
```
Use these numbers to verify batch_size=16 is safe. If max(nodes_per_graph) × 16
approaches GPU VRAM limits, reduce batch size before full training.
Expected: pre-CFG ~25 nodes/graph → post-CFG ~75 nodes/graph for typical contracts;
DeFi contracts with large functions may reach 250+ nodes. OOM threshold on 8GB GPU
is approximately 4000 nodes/batch, so batch_size=16 is safe for p95 but may OOM on
max for DeFi-heavy datasets. Smoke run data is the authoritative decision input.

**Phase B — Short run (15 epochs, full data):** Check per-class F1 improvement and
over-prediction. At epoch 15, check precision/recall ratio — if still >2× for any class,
switch to `MultiLabelFocalLoss` with per-class alpha.

**Phase C — Full run (60 epochs).**

### 5.4 Threshold Tuning

After training, run `tune_threshold.py`. Gate:
- F1-macro > **0.58**
- Manual detection rate > 70%
- False positive rate on safe contracts < 20%

---

## 6. Implementation Order — Phase-by-Phase

### Pre-flight — Mandatory Embedding Separation Test

Write and run `ml/tests/test_cfg_embedding_separation.py` **before any production
code changes.** This is non-negotiable.

```python
def _find_function_node(graph, func_name: str) -> int:
    """
    Return the graph node index of the named function node.

    Implementation: iterate graph.x. For each node where x[node_idx, 0] equals
    NODE_TYPES["FUNCTION"] (type_id=1), look up the corresponding function name
    from the metadata dict stored alongside the graph (graph.node_metadata[node_idx]
    must contain a 'name' key set during extraction). Return the index of the
    first match.

    If no match: raise ValueError with a list of available function names.
    This helper is used only in tests — performance is not a concern.

    Requires: graph.node_metadata is a list of dicts, set during extraction in §2.2.
    If graph.node_metadata is missing, this function raises AttributeError — which
    means the extraction code does not yet comply with the §2.2 spec. Fix extraction
    before diagnosing the GNN architecture.
    """
    for i, type_id in enumerate(graph.x[:, 0].tolist()):
        if int(type_id) == NODE_TYPES["FUNCTION"]:
            meta_name = graph.node_metadata[i].get("name", "")
            if meta_name == func_name or meta_name.split(".")[-1] == func_name:
                return i

    # Build the error message before raising to avoid f-string syntax issues.
    # (Splitting a list comprehension across f-string literals is a SyntaxError
    # in Python — the list comprehension must be fully evaluated before interpolation.)
    available_names = [
        graph.node_metadata[i]['name']
        for i in range(graph.x.shape[0])
        if int(graph.x[i, 0]) == NODE_TYPES['FUNCTION']
    ]
    raise ValueError(
        f"Function '{func_name}' not found in graph. "
        f"Available: {available_names}"
    )


def test_reentrancy_embedding_separation():
    """
    Contract A: call BEFORE write  (vulnerable — classic reentrancy).
    Contract B: write BEFORE call  (safe — CEI pattern).
    These are the exact contracts from the original root cause analysis.

    With the correct three-phase GNN architecture:
      - CFG_NODE_CALL (type 8) and CFG_NODE_WRITE (type 9) have different type_id
        → different initial embeddings before any message passing.
      - Phase 2 CONTROL_FLOW edges encode "call precedes write" vs "write precedes call".
      - Phase 3 reverse-CONTAINS aggregates this order signal into the function node.
    → The `withdraw` function node must have meaningfully different embeddings.

    WHY FUNCTION NODE, NOT MEAN POOL:
    Both contracts share ~13 identical nodes (CONTRACT, STATE_VAR, require node, etc.).
    After global_mean_pool over all nodes, the 2-node difference is diluted to ~0.92
    cosine similarity — easily passing a 0.95 threshold even with useless Phase 2 layers.
    Comparing the function-level node directly tests exactly what Phase 3 was designed
    to do: propagate order information INTO the function node.

    WHY FIXED SEED + THRESHOLD 0.85:
    On a randomly-initialized model, cosine similarity between two 128-dim random vectors
    is ~0.0 ± 0.1 for independent vectors. However, the two graphs share structural
    similarity (same CONTRACT, STATE_VAR, and FUNCTION node count), so embedded
    representations can correlate somewhat by chance. A threshold of 0.95 is too
    permissive — a broken architecture (order-blind) could pass it by chance.
    A threshold of 0.85 requires meaningful structural separation.
    torch.manual_seed(42) is set before model initialization to make the test
    deterministic and reproducible. If the test fails with seed 42, the architecture
    is broken — do not change the seed; fix the architecture.
    """
    contract_a = '''
    pragma solidity ^0.8.0;
    contract A {
        mapping(address => uint) public balances;
        function withdraw(uint amount) external {
            require(balances[msg.sender] >= amount);
            (bool ok,) = msg.sender.call{value: amount}("");
            balances[msg.sender] -= amount;  // WRITE AFTER CALL — vulnerable
        }
    }
    '''
    contract_b = '''
    pragma solidity ^0.8.0;
    contract B {
        mapping(address => uint) public balances;
        function withdraw(uint amount) external {
            require(balances[msg.sender] >= amount);
            balances[msg.sender] -= amount;  // WRITE BEFORE CALL — safe CEI
            (bool ok,) = msg.sender.call{value: amount}("");
        }
    }
    '''

    graph_a = extract_contract_graph(contract_a, compile_and_run_slither(contract_a))
    graph_b = extract_contract_graph(contract_b, compile_and_run_slither(contract_b))

    torch.manual_seed(42)   # deterministic init — do not change; fix architecture if test fails
    gnn = GNNEncoder()      # randomly initialised — no trained weights needed
    node_embs_a, _ = gnn(graph_a.x, graph_a.edge_index, graph_a.batch, graph_a.edge_attr)
    node_embs_b, _ = gnn(graph_b.x, graph_b.edge_index, graph_b.batch, graph_b.edge_attr)

    # Compare the withdraw function node embedding ONLY — not the whole-graph pool.
    withdraw_a_idx = _find_function_node(graph_a, "withdraw")
    withdraw_b_idx = _find_function_node(graph_b, "withdraw")

    emb_a = node_embs_a[withdraw_a_idx].unsqueeze(0)   # [1, 128]
    emb_b = node_embs_b[withdraw_b_idx].unsqueeze(0)   # [1, 128]

    cosine_sim = F.cosine_similarity(emb_a, emb_b).item()
    assert cosine_sim < 0.85, (
        f"GNN cannot distinguish call-before-write from write-before-call at the "
        f"function-node level. Cosine similarity={cosine_sim:.4f} (threshold 0.85). "
        "Diagnose in this order: "
        "(1) Verify CFG_NODE_CALL (type 8) and CFG_NODE_WRITE (type 9) are correctly "
        "assigned in _cfg_node_type(). "
        "(2) Verify Phase 2 CONTROL_FLOW edges (type 6) exist in both graphs and "
        "conv3 uses add_self_loops=False. "
        "(3) Verify Phase 3 reverse-CONTAINS edges exist and conv4 receives them correctly. "
        "(4) Do NOT change the seed — fix the architecture."
    )
```

**If this test fails on a randomly-initialised model:** do not proceed. The architecture
is not structurally capable of the distinction it needs to learn. Fix the extractor or
the GNN before touching data or training.

### Phase 0 — Quick Code Quality Fixes (½ day)

1. Fix `in_channels=8` hardcode in `gnn_encoder.py` → `NODE_FEATURE_DIM`
2. Add `TrainConfig.__post_init__()` with `gnn_layers` validation
3. Add Slither version assertion to `graph_extractor.py`
4. Verify `preprocess.py` imports from `graph_extractor.py`
5. Add `NODE_FEATURE_DIM` import to `validate_graph_dataset.py`
6. Add CLI flags to `train.py`: `--lora-r`, `--lora-alpha`, `--smoke-subsample-fraction`
7. Add `--force` flag to `ast_extractor.py`
8. Add `--freeze-val-test` flag to `create_splits.py`

**Commit before Phase 1.**

### Phase 1 — Schema & Extractor Overhaul (1–2 days)

1. Update `graph_schema.py` (all changes in §2.1).
2. Update `graph_extractor.py` (all changes in §2.2).
3. Write unit tests in `ml/tests/test_preprocessing.py`:
   - Reentrancy contract: assert CFG_NODE_CALL (type 8) exists; CONTROL_FLOW edges exist;
     `return_ignored=1.0` on the call node; `return_ignored=0.0` on the write node.
   - CEI-safe contract: assert CFG_NODE_WRITE (type 9) precedes CFG_NODE_CALL (type 8)
     in CONTROL_FLOW ordering (i.e., CONTROL_FLOW edge from write-node to call-node exists).
   - `unchecked {}` contract: `in_unchecked=1.0` (NOT triggered by inline assembly).
   - Loop contract: `has_loop=1.0`.
   - Typed-interface contract: `call_target_typed=1.0`.
   - Merged-IR node: verify `_cfg_node_type()` assigns CFG_NODE_CALL when both
     external call and state write are present in one IR node.
   - `node_metadata` alignment: for every test graph, assert
     `len(graph.node_metadata) == graph.x.shape[0]` and that function nodes have
     a 'name' key matching their canonical name.
   - CFG node `in_unchecked`: assert all CFG nodes have `x[:, 9] == 0.0` even when
     the parent function has `in_unchecked=1.0`. This explicitly tests that the
     feature is NOT inherited from the parent function flag.
4. **Run pre-flight test.** Must pass before Phase 2.
5. Smoke-test extraction on 10 hand-crafted contracts. Verify features look sensible.

### Phase 2 — Model Architecture Update (½ day)

1. Update `gnn_encoder.py`: three-phase, four-layer; `in_channels=NODE_FEATURE_DIM=12`;
   Phase 2 and Phase 3 `add_self_loops=False`; residuals; `hidden_dim=128`.
2. Update `sentinel_model.py`: three-eye; aux heads; `return_aux` interface.
3. Update `trainer.py`: `TrainConfig.__post_init__()`; aux loss loop; gradient norm logging;
   calibration logging; `pos_weight` with sqrt scaling.
4. Run `test_model.py` with 12-dim fixtures — confirm no shape errors.

### Phase 3 — Data Augmentation (parallel with Phase 1 & 2, 3–5 days)

Write/source all contract types per §3.2 priority list. Run two-step verification (§3.5).
Label using protocol in §3.4.

### Phase 4 — Full Re-Extraction (1 day + compute)

```bash
python ml/src/data_extraction/ast_extractor.py --force

python ml/scripts/validate_graph_dataset.py \
  --graphs-dir ml/data/graphs \
  --check-dim 12 \
  --check-edge-types 7 \
  --check-contains-edges \
  --check-control-flow \
  --check-cfg-subtypes

python ml/scripts/create_splits.py \
  --freeze-val-test \
  --multilabel-csv ml/data/processed/multilabel_index.csv \
  --splits-dir ml/data/splits
```

**Do NOT run `create_splits.py` without `--freeze-val-test` after augmented data
has been added.**

### Phase 5 — Smoke Run Then Full Training (3–5 days)

```bash
python ml/scripts/train.py --run-name v5-smoke \
  --smoke-subsample-fraction 0.10 --epochs 2 --batch-size 16

# After smoke run: check logged graph size stats. Verify p95(nodes/graph) × 16
# fits in GPU VRAM. Reduce batch-size if needed; increase grad-accumulation steps.

python ml/scripts/train.py --run-name v5-check-15ep \
  --epochs 15 --batch-size 16 --lr 2e-4 \
  --weighted-sampler all-rare --lora-r 16 --lora-alpha 32

python ml/scripts/train.py --run-name v5-full \
  --epochs 60 --batch-size 16 --lr 2e-4 \
  --weighted-sampler all-rare --lora-r 16 --lora-alpha 32 \
  --early-stop-patience 10
```

### Phase 6 — Evaluation & Threshold Tuning (1 day)

1. `tune_threshold.py` on validation set.
2. Run all 20 behavioral-test contracts (see §8.2 inventory).
3. Compare per-class results to v4 baseline.
4. Document in `docs/changes/2026-05-XX-v5-results.md`.
5. If acceptance criteria met → promote. If not → identify gaps, plan v5.1.

---

## 7. Technical Specifications Reference

### 7.1 New Node Feature Vector (12 dims)

```
Index  Name                Type   Function Nodes                 Non-Function Nodes
─────  ─────────────────── ────── ────────────────────────────── ─────────────────────
0      type_id             float  NODE_TYPES[kind] (range 0–12)  NODE_TYPES[kind]
1      visibility          float  VISIBILITY_MAP ordinal 0-2      VISIBILITY_MAP ordinal
2      pure                float  1.0 if func.pure                0.0
3      view                float  1.0 if func.view                0.0
4      payable             float  1.0 if func.payable             0.0
5      complexity          float  len(func.nodes)                 0.0
6      loc                 float  len(source_mapping.lines)       len(source_mapping.lines)
7      return_ignored      float  0.0 / 1.0 / -1.0(sentinel)     0.0
8      call_target_typed   float  0.0 / 1.0 / -1.0(sentinel)     1.0 (not applicable)
9      in_unchecked        float  see §2.2B                       0.0 (never from parent)
10     has_loop            float  see §2.2B                       0.0
11     external_call_count float  log-normalized count            0.0
```

**Sentinel values (-1.0) for `return_ignored` and `call_target_typed`:**
Both use -1.0 as "IR or source unavailable — cannot determine." The GNN receives a
distinct embedding for "unknown" vs "confirmed safe (0.0)" vs "confirmed unsafe (1.0)".
Defaulting to 0.0 (safe) on failure produces systematic false negatives.

**`in_unchecked` for CFG nodes is always 0.0.** It is never inherited from the parent
function's flag. A function with any `unchecked` block would otherwise mark all its
CFG child nodes as `in_unchecked=1.0`, including statements outside the unchecked scope —
producing false positives for IntegerUO. The function-level node carries this signal;
the GNN propagates it via Phase 1 CONTAINS edges to CFG children as needed.

### 7.2 Edge Type Vocabulary (7 types)

```
ID  Name           Direction              Semantics
──  ─────────────  ─────────────────────  ─────────────────────────────────────
0   CALLS          function → function    internal function call
1   READS          function → state_var   state variable read
2   WRITES         function → state_var   state variable write
3   EMITS          function → event       event emission
4   INHERITS       contract → contract    inheritance (MRO order)
5   CONTAINS       function → cfg_node    function owns this basic block
6   CONTROL_FLOW   cfg_node → cfg_node    sequential execution order (DIRECTED)
```

CONTAINS edges (type 5) are used in TWO phases with opposite directions:
- Phase 1 (forward, type 5): function → CFG_NODE — propagates function properties down.
- Phase 3 (reversed, flipped type-5): CFG_NODE → function — propagates order signal up.
No new edge type is used in v5.0; the edge_attr embedding value (5) is identical in
both phases, and the direction flip is done via `edge_index.flip(0)` in `GNNEncoder.forward()`.

**v5.0 Limitation — Phase 3 edge embedding symmetry:**
Because forward and reversed CONTAINS edges share the same type-5 embedding, the GNN
cannot represent the semantic asymmetry between "parent sends context down" (Phase 1)
and "child sends order signal up" (Phase 3) in the edge attribute alone. The GATConv
attention mechanism's positional asymmetry (source vs. target node roles) provides
partial compensation but is weaker than a dedicated edge type.
v5.1 target: add `REVERSE_CONTAINS = 7` with its own embedding. This is a non-breaking
schema addition (NUM_EDGE_TYPES: 7 → 8). Extract reversed CONTAINS edges as type 7
during extraction and use `edge_attr == 7` in the Phase 3 mask.

### 7.3 Node Type Vocabulary (13 types)

```
ID   Name             What it represents
───  ───────────────  ──────────────────────────────────────────────────────
0    STATE_VAR        Contract state variable
1    FUNCTION         Regular function
2    MODIFIER         Solidity modifier
3    EVENT            Solidity event
4    FALLBACK         Fallback function
5    RECEIVE          Receive function
6    CONSTRUCTOR      Constructor
7    CONTRACT         Contract node (root)
8    CFG_NODE_CALL    Statement containing an external call
9    CFG_NODE_WRITE   Statement writing a state variable
10   CFG_NODE_READ    Statement reading a state variable
11   CFG_NODE_CHECK   require / assert / if / loop condition
12   CFG_NODE_OTHER   All other statement types (includes synthetic Slither nodes)
```

### 7.4 GNN Architecture (v5 — Three-Phase, Four-Layer)

```
Input:  x [N, 12], edge_index [2, E], edge_attr [E] (int64, 0–6), batch [N]

Edge embedding: nn.Embedding(7, 32) → 224 params

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — STRUCTURAL AGGREGATION  (edge types 0–5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1 (GATConv): in_channels=12, out_channels=16 (per head), heads=8,
                   concat=True → 128 total, add_self_loops=True, edge_dim=32
  ReLU + Dropout(0.2)
  No residual: in_dim (12) ≠ out_dim (128).

Layer 2 (GATConv): in_channels=128, out_channels=16, heads=8,
                   concat=True → 128 total, add_self_loops=True, edge_dim=32
  ReLU + Dropout(0.2) + residual from Layer 1 output (128 + 128 → dropout → 128)

  ─ Function properties now flow DOWN into CFG_NODE children via CONTAINS (type 5). ─

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — CFG-DIRECTED AGGREGATION  (edge type 6 only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 3 (GATConv): in_channels=128, out_channels=128, heads=1,
                   concat=False, add_self_loops=False ← CRITICAL, edge_dim=32
  ReLU + Dropout(0.2) + residual from Layer 2

  add_self_loops=False rationale: self-loops add an edge from each CFG node to itself.
  During attention, the node would attend to itself (no order info) and to its genuine
  predecessor (order info). The self-loop term dilutes the directional signal.

  Depth note: 1 message-passing hop. Sufficient for basic reentrancy (require→call→write,
  diameter=2). Insufficient for diameter-4+ CFGs. v5.1 target: 2 hops (gnn_layers=5).

  Non-CFG nodes: no CONTROL_FLOW edges → no Phase 2 messages → unchanged from Phase 1.

  ─ CFG_NODE embeddings now encode execution-order context. ─

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — REVERSE-CONTAINS AGGREGATION  (type-5 edges, REVERSED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 4 (GATConv): in_channels=128, out_channels=128, heads=1,
                   concat=False, add_self_loops=False, edge_dim=32
  Uses contains_ei.flip(0) — sources and targets of CONTAINS edges are swapped
  so messages flow CFG_NODE → FUNCTION (the reverse of the original direction).
  ReLU + Dropout(0.2) + residual from Layer 3

  ZERO-MESSAGE BEHAVIOUR (correct by design):
  FUNCTION nodes with no CFG children receive no Phase 3 messages. conv4 returns
  zero for them. The residual x = x + dropout(0) is a no-op — they retain their
  Phase 2 (= Phase 1) embedding. This is correct behaviour, not a bug.
  DO NOT add add_self_loops=True to conv4 — this would mix self-attention into
  reverse aggregation for FUNCTION nodes that DO have CFG children.

  v5.0 EDGE EMBEDDING LIMITATION:
  Reversed CONTAINS edges use the same type-5 embedding as forward CONTAINS edges.
  The GNN cannot represent "parent sends context down" vs "child sends signal up"
  purely from edge attributes. GATConv positional asymmetry provides partial
  compensation. v5.1 target: REVERSE_CONTAINS = 7 with dedicated embedding.

  ─ FUNCTION nodes now contain execution-order information from their CFG children. ─
  ─ Non-CFG nodes: no CONTROL_FLOW edges → unchanged by Phase 2.                  ─
  ─ Non-FUNCTION nodes: no reverse-CONTAINS edges → unchanged by Phase 3.         ─

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output: node_embeddings [N, 128], batch [N]
  GNN Eye pooling (in SentinelModel):
    global_max_pool  → [B, 128]
    global_mean_pool → [B, 128]
    cat → [B, 256] → gnn_eye_proj (Linear 256→128 + ReLU + Dropout) → [B, 128]

API note: GATConv out_channels is PER HEAD in PyG.
  out_channels=16, heads=8, concat=True → 128 total output.
  out_channels=128, heads=8, concat=True → 1024 total output — WRONG.
```

### 7.5 Full Model Architecture (v5) — Three-Eye Classifier

```
Input: (PyG graph batch, input_ids [B, 512], attention_mask [B, 512])

GNNEncoder(in=12, hidden=128, phases=3, layers=4, edge_types=7)
  → node_embs [N, 128] (function nodes contain execution-order signal via Phase 3)

  ┌─────────────────────────────────────────────────────────┐
  │  GNN Eye — max+mean pool → Linear(256,128)+ReLU+Dropout │
  │  → gnn_eye [B, 128]                                     │
  └─────────────────────────────────────────────────────────┘

CodeBERT(frozen, 125M) + LoRA(r=16, alpha=32, q+v)
  → token_embs [B, 512, 768]

  ┌─────────────────────────────────────────────────────────┐
  │  Transformer Eye — CLS token [:, 0, :]                 │
  │  → Linear(768,128)+ReLU+Dropout → [B, 128]             │
  └─────────────────────────────────────────────────────────┘

CrossAttentionFusion(node_dim=128, token_dim=768, attn_dim=256, num_heads=8,
                     output_dim=128)
  ┌─────────────────────────────────────────────────────────┐
  │  Fused Eye [B, 128]                                     │
  │  v5.0 note: token side uses mean pool (order-blind).    │
  │  Fix deferred to v5.1. Not a silent gap — documented.   │
  └─────────────────────────────────────────────────────────┘

cat([gnn_eye, transformer_eye, fused_eye]) → [B, 384]
Linear(384, 10) → logits [B, 10]

AUXILIARY HEADS (training only):
  aux_gnn, aux_transformer, aux_fused — each Linear(128, 10)
  Training loss = main + 0.1 × (aux_gnn + aux_tf + aux_fused)
  forward(..., return_aux=False) at inference — unchanged interface

TRAINABLE PARAMETER ESTIMATE:
  GNNEncoder (3-phase, in=12, edge_emb 7×32):  ~87K
  LoRA adapters (r=16, 12L × Q+V):             ~590K
  CrossAttentionFusion:                         ~600K
  gnn_eye_proj (256→128 + act):                 ~33K
  transformer_eye_proj (768→128 + act):         ~98K
  Classifier (384→10):                          ~3.9K
  Auxiliary heads (3 × 128→10):                 ~3.9K
  ─────────────────────────────────────────────────────
  Total trainable:                              ~1.42M
  Frozen (CodeBERT backbone):                   ~125M
```

---

## 8. Acceptance Criteria

### 8.1 Validation Set Metrics

- **F1-macro (tuned thresholds) > 0.58** — raised from 0.5422 (v4) + 4 points.
  Rationale: major architectural fix (three-phase GNN), improved features, 2× data.
  Staying near 0.54 would indicate the fixes had near-zero effect.
- No per-class F1 more than **0.10** below its v4 value (floor rule).
- **DenialOfService tuned F1 > 0.55** — raised from v4 gate of 0.40.
  Adding 300+ DoS samples specifically for this class must produce a meaningful improvement.
  Shipping at 0.41 with a gate of 0.40 is not success.

### 8.2 Behavioral Test Suite — Explicit Contract Inventory (20 contracts)

The behavioral suite must be compiled before Phase 5 begins. Results must be documented
per contract, not just as class-level fractions. The following pass/fail criteria apply
to specific, named contracts.

**Mandatory non-negotiable tests (must pass — not part of any fraction gate):**
- **Contract A / Contract B pair:** The exact two contracts from §Pre-flight. Contract A
  (call-before-write) must be detected as Reentrancy. Contract B (write-before-call) must
  be classified clean. If either fails, v5 is rejected regardless of other metrics.
  Rationale: these are the root cause contracts that exposed v4's blindness. A v5 that
  cannot distinguish them has not fixed the fundamental problem.

**Reentrancy (5 contracts):**
- 1× classic single-function reentrancy (call-before-write, no guard) → Reentrancy
- 1× cross-function reentrancy (state write in separate function) → Reentrancy
- 1× Contract A above (call-before-write) → Reentrancy [MANDATORY]
- 1× Contract B above (write-before-call, CEI) → Clean [MANDATORY]
- 1× ReentrancyGuard protected → Clean
Gate: 3/5 detected, 2/2 safe contracts clean (includes mandatory pair)

**MishandledException / UnusedReturn (4 contracts):**
- 2× bare call() with no return capture → MishandledException
- 1× (bool ok,) = call(); require(ok) → Clean
- 1× (bool ok,) = call(); [ok unused] → UnusedReturn
Gate: 3/4 correctly classified

**CallToUnknown (3 contracts):**
- 2× raw address.call{} → CallToUnknown
- 1× typed interface IToken(addr).transfer() → Clean
Gate: 2/3 detected; typed interface call NOT flagged

**IntegerUO with `unchecked` (3 contracts):**
- 2× unchecked { counter++ } with overflow risk → IntegerUO
- 1× unchecked { i++ } in loop (gas opt, no overflow risk) → Clean
Gate: 2/3 detected; gas-optimisation contract NOT flagged

**DenialOfService (2 contracts):**
- 1× unbounded loop over array → DoS
- 1× bounded loop (require(arr.length <= 100)) → Clean
Gate: 1/2 detected; bounded loop NOT flagged

**Safe contracts — no vulnerability (3 contracts):**
- 1× full CEI + typed interface + checked returns → All labels clean
- 1× pull-payment pattern → All labels clean
- 1× ERC-20 standard implementation → All labels clean
Gate: all 3 classified clean (0 false positives)

**Overall behavioral gates:**
- Detection rate (true positives / 19 expected positives): > 70% (was 15%)
- Safe-contract specificity: > 66% (was 33%)
- Mandatory Contract A/B pair: both correct (non-negotiable)

### 8.3 Per-Vulnerability Behavioral Minimums

| Vulnerability | Minimum Pass Rate | Mandatory Test |
|---|---|---|
| Reentrancy (classic + cross-function) | 3/5 | Contract A detected; Contract B clean |
| MishandledException / UnusedReturn | 3/4 | — |
| CallToUnknown (raw address) | 2/3; typed interface NOT flagged | — |
| `unchecked` IntegerUO | 2/3; gas-opt NOT flagged | — |
| DenialOfService | 1/2 | — |
| Safe contracts (clean classification) | 3/3 | — |

---

## 9. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Slither version below 0.9.3** | Medium | High | **Hard failure at extractor import** (version assertion added). Not a warning. Pin `slither-analyzer>=0.9.3,<0.11` in `ml/pyproject.toml`. |
| Pre-flight test fails on random init (seed 42) | Medium | Critical | Blocking. Do not proceed. Fix CFG_NODE subtype assignment or Phase 2/3 GNN layers. Do not change the seed. Root cause is in extractor or architecture — not training data. |
| `node_metadata` missing from Data object | Low (if §2.2B spec followed) | Critical | Pre-flight test raises AttributeError before any assertion. validate_graph_dataset.py checks alignment. If this fires in production, extraction code was not updated to §2.2B spec. |
| `return_ignored` sentinel -1.0 rate exceeds 5% | Low | Medium | Validate reports sentinel rate. Investigate Slither IR availability for affected contract types. |
| `call_target_typed` sentinel -1.0 rate exceeds 2% | Low | Medium | Investigate `source_mapping` availability. Consider pre-filtering contracts with no source mapping. |
| 1 CONTROL_FLOW hop insufficient for complex CFGs | Medium | Medium | Accepted for v5.0 (covers diameter-2 reentrancy). v5.1: gnn_layers=5 with 2 CONTROL_FLOW hops. |
| Phase 3 edge embedding symmetry limits asymmetric learning | Medium | Medium | Documented as v5.0 limitation. GATConv positional asymmetry provides partial compensation. v5.1: REVERSE_CONTAINS = 7. If behavioral tests show Phase 3 is ineffective, fast-path to v5.1 edge type addition. |
| Larger graphs (CFG nodes) cause CUDA OOM at batch_size=16 | Medium | High | Smoke run logs max and p95 nodes/graph. Reduce to 8 → gradient accumulation (steps=4, bs=4 → effective 16) if p95 × 16 approaches VRAM limit. |
| `generate_safe_variants.py` produces mislabelled contracts | Medium | High | Two-step gate (compile + Slither). Manual review of 10% sample. |
| Augmented data over-fitting | Low | High | Original v4 val set preserved. Never augmented. |
| DoS F1 still poor despite 300 new samples | Medium | Medium | Gate is now 0.55 — not trivially passable. If missed, document and plan v5.1 targeted intervention. |
| Reverse-CONTAINS edge reversal implementation error | Medium | High | Unit test: verify that after Phase 3, a FUNCTION node's embedding differs from its Phase-2 value in a graph with CFG children. Add assertion: `(node_embs_after_phase3[func_idx] != node_embs_after_phase2[func_idx]).any()` |
| `in_unchecked` incorrectly set on CFG nodes from parent function flag | Low (if §2.2C spec followed) | Medium | Unit test in test_preprocessing.py explicitly asserts all CFG node `in_unchecked` values are 0.0 even when the parent function has `in_unchecked=1.0`. |

---

## 10. Files Changed Summary

| File | Change Type | Phase |
|---|---|---|
| `ml/src/preprocessing/graph_schema.py` | dim=12; 13 node types (5 CFG subtypes); 7 edge types; version v2 | 1 |
| `ml/src/preprocessing/graph_extractor.py` | Slither version assertion; remove reentrant + gas_intensity; 5 new features with sentinel -1.0 for unknowns; `node_metadata` list built alongside x_list and stored on Data object; `_cfg_node_type()` with priority order; `_build_cfg_node_features()` (NEW); `_build_control_flow_edges()` with corrected signature (x_list + node_metadata params), correct `graph_idx = len(x_list)` assignment, and synthetic node documentation; `in_unchecked` regex fixed to `r'\bunchecked\s*\{'` | 1 |
| `ml/src/models/gnn_encoder.py` | Three-phase, 4-layer architecture; Phase 2 + Phase 3 `add_self_loops=False`; Phase 3 reverse-CONTAINS flip; `in_channels=NODE_FEATURE_DIM=12`; `hidden_dim=128`; Phase 1 residual corrected (no residual Layer 1, residual Layer 2); Phase 3 zero-message behavior documented; Phase 3 edge embedding symmetry limitation documented | 2 |
| `ml/src/models/sentinel_model.py` | Three-eye; both projections consistently `+ReLU+Dropout`; aux heads; `return_aux`; updated docstring with all v5.0 limitations | 2 |
| `ml/src/training/trainer.py` | `TrainConfig.__post_init__()`; aux loss loop; gradient norm logging; calibration logging; sqrt `pos_weight` scaling | 2 |
| `ml/src/training/focalloss.py` | Add `MultiLabelFocalLoss(alpha: List[float])` | 2 |
| `ml/scripts/validate_graph_dataset.py` | dim=12 check; CFG subtype existence checks; sentinel rate tracking; **node_metadata presence and alignment check (NEW)** | 0 |
| `ml/src/inference/preprocess.py` | Verify graph_extractor import; no stale logic | 0 |
| `ml/scripts/train.py` | Add `--lora-r`, `--lora-alpha`, `--smoke-subsample-fraction`; **graph size logging in smoke run (NEW)** | 0 |
| `ml/src/data_extraction/ast_extractor.py` | Add `--force` flag | 0 |
| `ml/scripts/create_splits.py` | Add `--freeze-val-test` flag | 0 |
| `ml/scripts/generate_safe_variants.py` | NEW: mutation generator with two-step verification (compile → Slither) | 3 |
| `ml/tests/test_cfg_embedding_separation.py` | NEW: pre-flight test; `_find_function_node()` with f-string syntax fix; Contract A/B pair; `torch.manual_seed(42)`; threshold **0.85** (tightened from 0.95) | Pre-flight |
| `ml/tests/test_preprocessing.py` | New unit tests: 5 new features; CFG subtypes; priority order; deterministic ordering; reverse-CONTAINS edge existence; **node_metadata alignment and function name lookup (NEW)**; **CFG node in_unchecked=0.0 regardless of parent (NEW)** | 1 |
| `ml/tests/test_model.py` | Update fixtures for 12-dim; three-phase GNN shape tests; Phase 3 residual update check | 2 |
| `multilabel_index.csv` | Add augmented contracts | 3 |
| `ml/data/splits/` | Update with `--freeze-val-test` | 4 |
| `ml/data/graphs/*.pt` | Full re-extraction: v2 schema, 12-dim, 13 node types, 7 edge types, node_metadata on every Data object | 4 |
| `ml/data/tokens/*.pt` | Re-pair with new graph files | 4 |

---

## 11. What This Does NOT Change

- `ml/src/models/fusion_layer.py` — CrossAttentionFusion is correctly implemented.
  Mean-pool limitation in fused eye token aggregation is documented as a v5.1 item.
- `ml/src/data_extraction/tokenizer.py` — independent of graph schema.
- `ml/src/datasets/dual_path_dataset.py` — schema-agnostic.
- `agents/` module.
- `zkml/` module — will need proxy model retraining after v5, architecture independent.
- `contracts/` — Solidity contracts unchanged.

---

## 12. v5.1 Targets — Deferred Items

These are accepted limitations of v5.0, explicitly documented rather than silently omitted.
All are non-breaking when addressed.

| Item | What it fixes | Effort |
|---|---|---|
| `REVERSE_CONTAINS = 7` edge type | Phase 3 edge embedding symmetry — GNN can distinguish forward vs reversed CONTAINS direction | Low: schema addition (NUM_EDGE_TYPES 7→8), new embedding row, extraction update |
| `gnn_layers=5` (2 CONTROL_FLOW hops) | Phase 2 depth limit — covers CFG diameter 4+ (complex branching, exception handling) | Low: add one GATConv layer, update TrainConfig validation |
| Per-CFG_NODE `return_ignored` | Currently 0.0 for all CFG nodes; CFG_NODE_CALL nodes could carry per-call return-ignore status more precisely than the function-level aggregate | Medium: update `_build_cfg_node_features()` |
| Per-CFG_NODE `in_unchecked` scope analysis | Currently 0.0 for all CFG nodes; proper scope containment using source line ranges | Medium: requires source line range comparison |
| Attention pooling in fused eye token side | Currently mean pool (order-blind); replace with positional-bias attention pool | Medium: fusion_layer.py change |
| GATv2 in place of GAT | More expressive attention; deferred for v5.0 stability | Low: swap GATConv → GATv2Conv |

---

## 13. Immediately After This Document — What to Do First

1. **Checkout the dev branch** (`claude/debug-model-overprediction-Q5aAC`).

2. **Write `test_cfg_embedding_separation.py` first.** This is the single most important
   thing in this document. It tests whether the three-phase GNN architecture is structurally
   capable of distinguishing call-before-write from write-before-call **at the function-node
   level** using a randomly-initialised model with fixed seed. If this test fails, fix the
   architecture before touching anything else. If it passes, you have structural confidence
   before spending days on extraction and training.

3. **Phase 0:** Fix hardcodes, add Slither version assertion, update validate script.

4. **Phase 1:** Schema + extractor. Unit-test every new feature including `node_metadata`
   alignment and CFG node `in_unchecked=0.0`. Run pre-flight test. Do not proceed until
   both pass.

5. **Phase 3** (augmentation) can start in parallel on a separate branch.

6. **Do not begin full re-extraction (Phase 4)** until Phase 1 unit tests pass.

7. **Do not begin training (Phase 5)** until Phase 4 extraction completes and
   `validate_graph_dataset.py` reports zero errors.

**The four biggest risks in this plan, in order:**
1. Pre-flight test fails (seed 42) — architecture cannot distinguish the root-cause contracts.
2. Phase 3 reverse-CONTAINS not implemented correctly — order signal trapped in CFG subgraph.
3. `node_metadata` missing from Data objects — pre-flight test fails at AttributeError.
4. Full re-extraction before feature logic is verified on hand-crafted test cases.

All four are caught by the unit tests and pre-flight test. Do not skip them.
