# Schema Constants — Active Schema v9

Single source of truth for all graph schema constants used across the sentinel-data pipeline.
Verified against `ml/src/preprocessing/graph_schema.py:161,175,218` on 2026-06-08.

**If you change any constant here, you must:**
1. Bump `FEATURE_SCHEMA_VERSION` in `graph_schema.py`
2. Update `_schema_version_registry.json`
3. Re-extract all graphs (Stage 2 re-run)
4. Retrain the model (Run 11+ after Stage 8)

---

## Dimensions

| Constant | Value | Notes |
|---|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | Bump on any structural change |
| `NODE_FEATURE_DIM` | `12` | Was 11 in v8; feat[11] `in_unchecked_block` added |
| `NUM_NODE_TYPES` | `14` | Was 13 in v8; `CFG_NODE_ARITH=13` added |
| `NUM_EDGE_TYPES` | `12` | Was 11 in v8; `EXTERNAL_CALL=11` self-loop added |
| `_MAX_TYPE_ID` | `13.0` | Was 12.0 in v8 |
| `NUM_CLASSES` | `10` | **LOCKED** — class order matches all existing checkpoints |

---

## Feature Vector (NODE_FEATURE_DIM = 12)

| Index | Name | Description |
|---|---|---|
| 0 | `node_type_norm` | Normalised node type id (`type_id / _MAX_TYPE_ID`) |
| 1 | `visibility` | Function visibility: 0=public/external, 1=internal, 2=private |
| 2 | `uses_block_globals` | Count of `block.timestamp`, `block.number`, `now` reads (v9 fix: catches Solidity 0.4.x `now` keyword) |
| 3 | `external_call_count` | Number of external calls in this node's scope (v9 fix: includes Transfer + Send) |
| 4 | `state_var_writes` | Number of state variable write ops |
| 5 | `contract_size_norm` | Normalised contract line count |
| 6 | `loc` | Raw line count of this function / CFG node |
| 7 | `return_ignored` | 1.0 if a return value is silently dropped (fixed: checks `id()` in subsequent IR ops) |
| 8 | `call_target_typed` | 1.0 = typed HighLevelCall, 0.0 = raw low-level call |
| 9 | `has_loop` | 1.0 if this CFG node is inside a loop body |
| 10 | `payable` | 1.0 if the enclosing function is `payable` |
| 11 | `in_unchecked_block` | **NEW v9** — fraction of nodes in `unchecked{}` scope; pre-0.8 Solidity → 1.0 universally |

---

## Node Types (NUM_NODE_TYPES = 14)

| ID | Name | Notes |
|---|---|---|
| 0 | `STATE_VAR` | |
| 1 | `FUNCTION` | |
| 2 | `MODIFIER` | |
| 3 | `EVENT` | |
| 4 | `FALLBACK` | |
| 5 | `RECEIVE` | |
| 6 | `CONSTRUCTOR` | |
| 7 | `CONTRACT` | Node 0 in every graph is always CONTRACT |
| 8 | `CFG_NODE_CALL` | |
| 9 | `CFG_NODE_WRITE` | |
| 10 | `CFG_NODE_READ` | |
| 11 | `CFG_NODE_CHECK` | |
| 12 | `CFG_NODE_OTHER` | |
| 13 | `CFG_NODE_ARITH` | **NEW v9** — pure Binary arithmetic op nodes |

---

## Edge Types (NUM_EDGE_TYPES = 12)

| ID | Name | Notes |
|---|---|---|
| 0 | `CONTAINS` | |
| 1 | `CONTROL_FLOW` | |
| 2 | `DEF_USE` | |
| 3 | `CALL_ENTRY` | Internal calls only (cross-function external calls deferred to v2.1) |
| 4 | `RETURN_TO` | |
| 5 | `STATE_READ` | |
| 6 | `STATE_WRITE` | |
| 7 | `INHERITANCE` | |
| 8 | `MODIFIER_USE` | |
| 9 | `EMITS` | Open bug (Interp-6) — fix in Stage 7 seam swap |
| 10 | `REVERSE_CONTAINS` | Runtime-only; 0 on disk, built by GNNEncoder |
| 11 | `EXTERNAL_CALL` | **NEW v9** — self-loop on cross-contract call nodes |

---

## Class Order (LOCKED — NUM_CLASSES = 10)

| Index | Class | Notes |
|---|---|---|
| 0 | `Reentrancy` | v1.4: 4,622 retained (26.1%) — VERIFIED |
| 1 | `CallToUnknown` | v1.4: 245 retained (2.2%) — PROVISIONAL |
| 2 | `Timestamp` | v1.4: 1,197 retained (44.8%) — BEST-EFFORT |
| 3 | `ExternalBug` | v1.4: 614 retained (17.0%) — PROVISIONAL |
| 4 | `GasException` | v1.4: 2,814 retained (40.9%) — PROVISIONAL |
| 5 | `DenialOfService` | v1.4: 1,268 retained (10.2%) — BEST-EFFORT |
| 6 | `IntegerUO` | v1.4: 16,740 (100%) — VERIFIED |
| 7 | `UnusedReturn` | v1.4: 3,229 (100%) — VERIFIED |
| 8 | `MishandledException` | v1.4: 5,154 (100%) — VERIFIED |
| 9 | `NonVulnerable` | v1.4: 44,899 (+18,751 from verified FPs) |
