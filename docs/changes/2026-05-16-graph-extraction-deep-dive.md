# SENTINEL Graph Extraction — Deep-Dive Analysis

**Date:** 2026-05-16  
**Method:** Read full extractor source, ran live Slither extraction, inspected 5 real contracts  
**Contracts examined:** FiatContract (Reentrancy), Crowdsale (Reentrancy), DMarketNFTToken (Reentrancy), CryptoMinerToken (IntegerUO), Minaxis (Safe)

---

## How `graph_extractor.py` Works (Verified)

### Step-by-step pipeline

```
.sol file  →  solc (compile)  →  Slither (parse IR)  →  _select_contract()  →  graph
```

1. **Slither instantiation** — `Slither(sol_path, solc=binary, detectors_to_run=[])`. Runs full compilation and IR analysis. Detectors disabled (we use Slither as an AST/IR parser only, not a static analyser).

2. **Contract selection** — `_select_contract()`:
   - Filter out imported dependencies (`is_from_dependency()`)
   - Skip interfaces (`.is_interface`)
   - Among remaining: pick the one with the most functions
   - Reason: many BCCC files have a `SafeMath` library and several interfaces defined before the main contract. Without this filter, the extractor would pick the library (which has few nodes) → ghost graph.

3. **Node insertion order** (fixed, deterministic):
   ```
   CONTRACT → STATE_VARs → [FUNCTION → its CFG_NODEs (sorted by source_line)] × N → MODIFIERs → EVENTs
   ```
   Node 0 is always CONTRACT. State vars follow immediately. Then for each function: the function node is added, then all its CFG children sorted by source line. Modifiers and events come last.

4. **Node feature vector** (12-dim, `_build_node_features`):
   - `type_id / 12.0` — normalized NODE_TYPES value
   - `visibility` — VISIBILITY_MAP ordinal (0=priv/internal, 1=pub, 2=ext, 3=pub+ext... actually {private/internal=0, public=1, external=2})
   - `pure`, `view`, `payable` — bool flags on Function object
   - `complexity` — `len(func.nodes)` (CFG block count from Slither)
   - `loc` — `len(source_mapping.lines)` (line span of the declaration)
   - `return_ignored` — Slither IR scan: 1.0 if any external call discards return value
   - `call_target_typed` — 0.0 if any `LowLevelCall` or raw address call; 1.0 if all typed
   - `in_unchecked` — 1.0 if function body contains `unchecked {}` (Slither NodeType check + regex fallback)
   - `has_loop` — 1.0 if function contains `IFLOOP`/`STARTLOOP`/`ENDLOOP` CFG nodes
   - `external_call_count` — `log1p(high_level + low_level calls) / log1p(20)`, clamped [0,1]

5. **CFG node features** (`_build_cfg_node_features`):
   - Same 12-dim layout but only `type_id` and `loc` carry real values
   - All functional features (`payable`, `complexity`, `has_loop`, etc.) = 0.0
   - `call_target_typed` = 1.0 (default safe — "not applicable at statement level")
   - `in_unchecked` is ALWAYS 0.0 for CFG nodes (even inside an unchecked block) — design choice to avoid false positives

6. **CFG node type classification** (`_cfg_node_type`, priority order):
   ```
   CALL  (8) — any IR op is LowLevelCall or HighLevelCall   ← external call site
   WRITE (9) — node writes a state variable
   READ  (10) — node reads a state variable
   CHECK (11) — require/assert/if/loop condition
   OTHER (12) — everything else (entry points, returns, arithmetic)
   ```

7. **Edge types added** (from Slither's IR):
   - `CALLS(0)` — func.internal_calls: one function calling another
   - `READS(1)` — func.state_variables_read → STATE_VAR node
   - `WRITES(2)` — func.state_variables_written → STATE_VAR node
   - `EMITS(3)` — func.events_emitted → EVENT node
   - `INHERITS(4)` — contract.inheritance → parent contracts (declaration-level only)
   - `CONTAINS(5)` — function → each of its CFG_NODE children
   - `CONTROL_FLOW(6)` — CFG_NODE → successor CFG_NODE (from `node.sons`)
   - `REVERSE_CONTAINS(7)` — **NOT stored in .pt files**, injected at runtime

---

## Side-by-Side: .sol Source vs. Graph (3 Contracts)

### Contract 1: FiatContract — Reentrancy (Solidity 0.4.15)

**Vulnerability:** `execute()` function at line 112:
```solidity
function execute(address _to, uint _value, bytes _data) external returns (bytes32 _r) {
    require(msg.sender==creator);
    require(_to.call.value(_value)(_data));   // ← raw low-level call
    return 0;
}
```

**Graph captures it:**
- Function node (idx 44): `call_target_typed=0.0` (raw address call detected), `ext_calls=0.228` (1 call)
- CFG sub-graph of `execute()`:
  ```
  ENTRY_POINT (OTHER)
       ↓ CONTROL_FLOW
  require(msg.sender==creator) (CFG_NODE_READ)
       ↓ CONTROL_FLOW
  require(_to.call.value) (CFG_NODE_CALL)  ← type_id=8, the external call site
       ↓ CONTROL_FLOW
  RETURN 0 (OTHER)
  ```
- CONTROL_FLOW path clearly shows: read state → external call → return (no state written AFTER call)

**Feature that flags it:** `call_target_typed=0.0` on the function node — directly detected by Slither's IR analysis which recognises `.call.value()` as a `LowLevelCall`.

**What's NOT captured:** Whether the external call happens BEFORE or AFTER state updates (the actual CEA violation). The graph has no feature for "call position in CEI order". This is a fundamental limit — only the GNN's learned message-passing over the CFG topology can infer this pattern.

---

### Contract 2: DMarketNFTToken — Reentrancy (Solidity 0.8.1, ERC-721)

**Vulnerability:** `_checkOnERC721Received()` at line ~878:
```solidity
function _checkOnERC721Received(address from, address to, uint256 tokenID, bytes memory _data) 
    private returns (bool) {
    if (to.isContract()) {
        try IERC721Receiver(to).onERC721Received(...)   // ← external call to recipient
```

**Call chain:**
```
safeTransferFrom() → _safeTransfer() → _checkOnERC721Received() → IERC721Receiver.onERC721Received()
```

**Graph captures:**
- `_checkOnERC721Received` (node 114): `ext_calls=0.228` (1 external call)
- CFG node 118: `CFG_NODE_CALL` — `TRY retval = IERC721Receiver(to).onERC721Received(...)`
- CALLS edges: `safeTransferFrom → _safeTransfer → _checkOnERC721Received`

**Critical observation — `call_target_typed=1.0` here, NOT 0.0**

Despite being a reentrancy vector, `call_target_typed=1.0` because the call goes through a typed interface (`IERC721Receiver`), not a raw address. Slither classifies it as `HighLevelCall` with a typed receiver — so the "raw call" feature does NOT fire.

**What the GNN must learn instead:** The reentrancy signal comes from the CALLS edge topology (safeTransferFrom → _checkOnERC721Received) and the presence of CFG_NODE_CALL nodes inside a function reachable from `payable` entry points. This requires multi-hop message passing (3+ hops), which is exactly what JK connections (4 layers) are supposed to handle.

**Implication:** For modern Solidity (0.8.x) reentrancy via callback patterns, the only graph signal is the topological pattern (CALLS chain + CFG_NODE_CALL presence). For old Solidity (0.4.x) raw `.call.value()` patterns, there's a direct feature signal (`call_target_typed=0`).

---

### Contract 3: CryptoMinerTokenETHconnection — IntegerUO (Solidity 0.4.25)

**Vulnerability:** Integer overflow in token arithmetic — raw uint256 arithmetic without SafeMath in:
```solidity
uint256 fee = entryFee_ * incomingEthereum / 100;
uint256 tokenReward = tokensForEther(incomingEthereum - fee);
```
`entryFee_` is `uint8 = 10`, multiplied with `uint256` — overflow possible in Solidity 0.4.x where arithmetic is unchecked by default.

**Graph captures:**
- Multiple functions with high `ext_calls` (0.228–0.529): `sell()`, `transfer()`, `purchaseTokens()`
- `has_loop=1.0` on `sqrt()` function — loop for Newton's method
- `in_unchecked=0.0` everywhere — correct, because Solidity 0.4.x doesn't have `unchecked{}` blocks; ALL arithmetic is unchecked by default

**Critical observation — NO direct IntegerUO feature**

The feature vector has no signal for "uses unsafe arithmetic". The `in_unchecked` feature only captures the Solidity 0.8.x `unchecked{}` syntax. For 0.4.x contracts where ALL arithmetic is potentially overflowable:
- `in_unchecked=0.0` for everything → model gets no signal
- No "uses SafeMath" feature
- No "arithmetic operation count" feature

**What the GNN must learn instead:** The IntegerUO signal can only be learned from:
1. **Absence of SafeMath CALLS edges** — contracts using SafeMath have CALLS edges to SafeMath.mul, SafeMath.add, etc. Contracts without SafeMath don't. But this absence-of-edges signal is hard for GNNs to learn reliably.
2. **Token contract structural patterns** — IntegerUO correlates strongly with token contracts (high state var count, many public functions, ERC-20 pattern). But this is a confound, not the vulnerability itself.
3. **Complexity/LoC patterns** — higher complexity functions tend to have more arithmetic operations.

**Fundamental limitation:** IntegerUO in Solidity 0.4.x is essentially invisible in the current feature schema. The model must infer it from correlations (token contract pattern, no SafeMath), not from the actual vulnerability signal.

---

### Contract 4: Minaxis — Safe (Solidity 0.4.12)

**Source:** NonVulnerable directory. 567-line ERC-20 token with SafeMath library.

**Graph captures:**
- Multiple CALLS edges to SafeMath functions (`SafeMath.add`, `SafeMath.mul`, etc.) — the safe arithmetic pattern
- `call_target_typed=1.0` for all functions — no raw address calls
- `return_ignored=0.0` for all functions — all return values captured
- `payable=1.0` only on `fallback()` — no payable functions that could receive ETH and make external calls

**How the GNN should distinguish this from Reentrancy:**
- No `call_target_typed=0.0` functions → no raw address call signal
- No `payable` functions that also have high `ext_calls`
- SafeMath CALLS edges present → different graph topology

---

## Summary: What the Graph Encoding Can and Cannot Capture

### Features that directly encode vulnerability signals

| Feature | Vulnerability | Strength |
|---------|--------------|----------|
| `call_target_typed=0.0` | Reentrancy (old Solidity raw `.call.value()`) | **STRONG** — direct Slither IR detection |
| `return_ignored=1.0` | MishandledException | **STRONG** — direct IR scan |
| `payable=1.0` + `ext_calls>0` | Reentrancy (ETH-receiving + external calls) | **MODERATE** — combination needed |
| `has_loop=1.0` | GasException (unbounded loops) | **MODERATE** |
| `in_unchecked=1.0` | IntegerUO (Solidity 0.8.x only) | **WEAK** — only captures explicit `unchecked{}` |
| `CFG_NODE_CALL` count | Any external call pattern | **WEAK alone** — also present in safe contracts |

### Patterns the GNN must learn from topology (not features)

| Vulnerability | Topological signal | GNN hops needed |
|--------------|-------------------|----------------|
| Reentrancy (modern, callback) | CALLS chain → CFG_NODE_CALL inside callable function | 3–4 hops |
| IntegerUO (Solidity 0.4.x) | Absence of SafeMath CALLS edges | Indirect |
| DoS (unbounded loops) | CONTROL_FLOW loops + READS/WRITES on state vars in loops | 2 hops |
| UnusedReturn | `return_ignored=1.0` feature on function node | 1 hop (direct) |
| TransactionOrderDependence | READS of block.timestamp/block.number state vars | 2 hops |

### Structural limits

1. **Solidity 0.4.x IntegerUO is a dark spot.** The model cannot directly detect unchecked arithmetic for pre-0.8.x contracts. It learns correlations (token pattern, no SafeMath), not the actual vulnerability.

2. **CEA ordering is not encoded.** Whether an external call happens before or after state updates (Check-Effects-Interactions violation) is not directly captured. The CONTROL_FLOW edges encode ordering, but only within a single function — cross-function CEI requires multi-hop GNN.

3. **`loc` (lines of code) is not normalized.** Values range from 1 to 500+. This is 3-4 orders of magnitude larger than binary features and can dominate early layer dot products. The GNN weights compensate, but it's an initialization risk.

4. **96.6% token truncation.** The CodeBERT path sees at most 512 tokens per contract. For contracts with >2000 lines, the vulnerability-relevant functions (often in the middle or end) may be completely invisible to the text encoder. The GNN path is unaffected.

5. **`node_metadata[i]['type']` is wrong for all current .pt files** (all show STATE_VAR). This affects only debugging tools — training is unaffected since the model uses `x` tensor only.

---

## Why `reextract_graphs.py` Has the 280 Stale Graphs

The reextract script uses the LATEST PATCH of the declared minor version:
```python
_LATEST_PATCH = {"0.4": "0.4.26", "0.5": "0.5.17", ...}
# "0.4.15" → minor="0.4" → version="0.4.26"
```

But contracts with STRICT pragmas (no `^` or `>=`) like `pragma solidity 0.4.15;` fail with solc 0.4.26:
> "Source file requires different compiler version"

These contracts were compiled successfully in the ORIGINAL extraction (which likely set solc to exact match), but re-extraction with 0.4.26 fails → they become "fail" or "skip" status → old .pt files remain on disk → the 280 "stale v5.0" graphs.

**Impact:** 280/44,420 = 0.6% of the dataset. Labeled as "ghost" in the original docs but technically they're stale (were once valid v5.0 graphs that couldn't be updated to v2 schema). Their features are the 8-dim v5.0 schema, not the current 12-dim. If loaded by the trainer, the feature dimension check would catch this (assertion in extractor) — but the dataset loader doesn't re-validate schema versions on load. The `FEATURE_SCHEMA_VERSION` attribute in cached pickles should filter these out.

**Fix:** Add solc version fallback in `reextract_graphs.py` — if exact version fails, try dropping to exact pragma version. Or: re-check if all 280 are actually using the strict-only pragma pattern.
