# Validated Code Audit — Self-Audition File

**Date:** 2026-06-01  
**Validated by:** Cross-referencing each claim against the actual source files  
**Status:** All 38 items checked; 36 CONFIRMED, 1 ALREADY FIXED, 1 PARTIALLY CONFIRMED (nuance only)

**Note on numbering:** A24 is absent from the audition file (skipped). A25 appears twice — once for `gnn_encoder.py` and once for `sentinel_model.py`. The second A25 is relabeled A25b here to avoid ambiguity.

---

## A1 — `graph_schema.py` — Type ID normalization missing range guard
**Status: CONFIRMED**  
`assert len(NODE_TYPES) == 13` exists at line 463. `assert max(NODE_TYPES.values()) == 12` does NOT exist. The max value in the live dict is 12 (CFG_NODE_OTHER), so the gap is real: count is guarded but max value is not. A new node type with id=13 would not be caught.

---

## A2 — `hash_utils.py` — `validate_hash` accepts uppercase hex
**Status: CONFIRMED**  
```python
try:
    int(hash_string, 16)
    return True
except ValueError:
    return False
```
Uses `int(..., 16)` — accepts uppercase A-F. `hashlib.md5().hexdigest()` always produces lowercase. Silent permissiveness confirmed.

---

## A3 — `graph_extractor.py` — `_MAX_TYPE_ID` is dynamic
**Status: CONFIRMED**  
Line 113:
```python
_MAX_TYPE_ID = float(max(NODE_TYPES.values()))
```
Dynamic — changes automatically if `NODE_TYPES` gains a new entry. Contradicts the `/12.0` normalization documented in graph_schema.py.

---

## A4 — `graph_extractor.py` — `assert` used for production invariant
**Status: CONFIRMED**  
Lines 1253-1257:
```python
assert len(node_metadata) == x.shape[0], (
    f"node_metadata length {len(node_metadata)} ≠ x.shape[0] {x.shape[0]} "
    f"for '{contract.name}'. This is a bug ..."
)
```
Uses `assert` — silent under `python -O`. Critical alignment invariant unguarded in optimised mode.

---

## A5 — `graph_extractor.py` — `except AttributeError` scope in `_compute_return_ignored`
**Status: PARTIALLY CONFIRMED (scope concern valid; "silent" characterisation incorrect)**  
The catch IS `except AttributeError` (not `except Exception`), which is more specific than claimed. However the scope concern is real: the `except` block covers the entire function body, meaning a refactoring-induced AttributeError from any inner expression would be caught and return the -1.0 sentinel. Correction: the catch is NOT silent — it calls `logger.warning(...)` before returning -1.0. The fix (tighten try scope) is still appropriate.

---

## A6 — `graph_extractor.py` — `except Exception: pass` in `_compute_call_target_typed`
**Status: CONFIRMED**  
Line 312-313:
```python
except Exception:
    pass  # type resolution failed; fall through to source scan
```
Bare `except Exception: pass`. Slither API changes would silently fall to the regex scan.

---

## A7 — `graph_extractor.py` — `_compute_in_unchecked` is dead code
**Status: CONFIRMED**  
Lines 331-360:
```python
# DEPRECATED (v7 BUG-L2) — safe to delete after v8 extraction is complete.
def _compute_in_unchecked(func: Any) -> float:
```
Function exists, is DEPRECATED, v8 is the current schema. Dead code.

---

## A8 — `graph_extractor.py` — `is True` identity check in `_compute_has_loop`
**Status: CONFIRMED**  
Line 376:
```python
if getattr(func, "is_loop_present", None) is True:
    return 1.0
```
Identity check `is True` — integer `1` or any truthy non-boolean missed.

---

## A9 — `graph_extractor.py` — string-based type check for `SolidityVariableComposed`
**Status: CONFIRMED**  
Line 424:
```python
if type(rv).__name__ == "SolidityVariableComposed":
```
String-based class check. Slither rename would silently zero out `uses_block_globals` for all Timestamp/TOD contracts.

---

## A10 — `graph_extractor.py` — `except Exception: pass` in `_cfg_node_type`
**Status: CONFIRMED**  
Lines 493-494:
```python
except Exception:
    pass
return NODE_TYPES["CFG_NODE_OTHER"]
```
All four priority classification branches (CALL/WRITE/READ/CHECK) are inside a single `except Exception: pass`. Any Slither API change silently makes every CFG node `CFG_NODE_OTHER`.

---

## A11 — `graph_extractor.py` — hardcoded parent feature indices in `_build_cfg_node_features`
**Status: CONFIRMED**  
Lines 542-547:
```python
p = parent_features or []
visibility  = p[1] if len(p) > 1 else 0.0
view        = p[3] if len(p) > 3 else 0.0
payable     = p[4] if len(p) > 4 else 0.0
complexity  = p[5] if len(p) > 5 else 0.0
has_loop    = p[9] if len(p) > 9 else 0.0  # dim[9] in v7 (was [10] in v6)
```
Raw integer indices. FEATURE_NAMES reorder → silent wrong-feature inheritance. Comment even acknowledges an earlier index shift.

---

## A12 — `graph_extractor.py` — `n.node_id` without `getattr` fallback in sort key
**Status: CONFIRMED**  
Lines 606-611 (sort key):
```python
key=lambda n: (
    n.source_mapping.lines[0]
    if n.source_mapping and n.source_mapping.lines else 0,
    n.node_id,
)
```
`n.node_id` accessed directly — no `getattr` default. Synthetic nodes without `node_id` raise `AttributeError` inside `sorted()`.

---

## A13 — `graph_extractor.py` — silently dropped CONTROL_FLOW edges not logged
**Status: CONFIRMED**  
Lines 639-641:
```python
for successor in (getattr(slither_node, "sons", None) or []):
    if successor in node_index_map:
        control_flow_edges.append((src_idx, node_index_map[successor]))
```
No `else` branch, no log when successor is absent from `node_index_map`. Silent drop.

---

## A14 — `graph_extractor.py` — RETURN_TO cartesian product includes impossible revert→normal paths
**Status: CONFIRMED**  
Lines 695-706:
```python
for son in call_site_sons:
    son_idx = local_map.get(son)
    if son_idx is None:
        continue
    for terminal_idx in callee_terminals:
        edges.append([terminal_idx, son_idx])
        edge_types.append(_RETURN_TO)
```
All `callee_terminals` (including revert/throw terminals) connected to all `call_site_sons`. No filtering on terminal type.

---

## A15 — `graph_extractor.py` — DEF_USE `def_map` keyed by variable name
**Status: CONFIRMED**  
Line 752:
```python
def_map.setdefault(lval.name, []).append(node_idx)
```
String key — variable name shadowing in nested scopes produces spurious DEF_USE edges.

---

## A16 — `graph_extractor.py` — `assert` for sentinel range in `_build_node_features`
**Status: CONFIRMED**  
Lines 856-857:
```python
assert return_ignored in (-1.0, 0.0, 1.0), f"return_ignored out of range: {return_ignored}"
assert call_target_typed in (-1.0, 0.0, 1.0), f"call_target_typed out of range: {call_target_typed}"
```
`assert` — removed under `python -O`.

---

## A17 — `graph_extractor.py` — Slither exception routing by string keyword matching
**Status: CONFIRMED**  
Lines 1059-1067:
```python
except Exception as exc:
    exc_lower = str(exc).lower()
    if any(kw in exc_lower for kw in ("compil", "syntax", "invalid solidity", "parsing", "solc")):
        raise SolcCompilationError(...) from exc
    raise SlitherParseError(...) from exc
```
String keyword matching exactly as described. Fragile in both directions.

---

## A18 — `graph_extractor.py` — `except Exception: pass` when building ICFG maps
**Status: CONFIRMED**  
Lines 1160-1173:
```python
try:
    from slither.core.cfg.node import NodeType as _SNT
    ...
    _func_entry_map[func_key] = cfg_node_map[_n]
    _func_terminal_map[func_key] = [...]
except Exception:
    pass
```
Bare `except Exception: pass` — no log. Missing ICFG maps produce zero CALL_ENTRY/RETURN_TO edges for all callers silently.

---

## A19 — `ast_extractor.py` — `get_solc_binary` uses `Path.cwd()`
**Status: CONFIRMED**  
Line 143:
```python
venv_path = Path.cwd() / ".venv" / ".solc-select" / "artifacts" / f"solc-{version}"
```
`get_project_root()` is defined at line 154-156 but NOT used here. CWD-dependent resolution.

---

## A20 — `ast_extractor.py` — `label=0` hardcoded for all contracts in batch extraction
**Status: CONFIRMED**  
Lines 307-311:
```python
worker = partial(
    self.contract_to_pyg,
    solc_binary=solc_bin,
    solc_version=version,
    label=0,
)
```
Every contract receives `label=0` regardless of actual vulnerability status. High severity — potential training data poisoning.

---

## A21 — `ast_extractor.py` — `print()` from worker processes
**Status: CONFIRMED**  
Lines 223 and 228:
```python
if self.verbose:
    print(f"  Skipped {Path(contract_path).name}: {exc}")
...
if self.verbose:
    print(f"  Unexpected error for {Path(contract_path).name}: {exc}")
```
`print()` from `mp.Pool` workers — interleaved output under concurrency.

---

## A22 — `ast_extractor.py` — `torch.save` no error handling
**Status: CONFIRMED**  
Line 328:
```python
torch.save(result, graph_file)
processed_hashes.add(result.contract_hash)
```
No `try/except`. Disk-full or I/O error aborts the entire batch loop; up to 499 processed contracts since last checkpoint lost.

---

## A23 — `gnn_encoder.py` — `last_weight_stds` comment says "0 if N=1" but PyTorch returns NaN
**Status: CONFIRMED**  
Line 123:
```python
self.last_weight_stds.copy_(w_nk.std(0).detach())   # [K] — 0 if N=1
```
`.std(0)` without `unbiased=False` — comment claims "0 if N=1" but PyTorch biased std returns NaN for N=1. Fix: `.std(0, unbiased=False).nan_to_num(0.0)`.

---

## A25 — `gnn_encoder.py` — `edge_index.max()` O(E) scan on every forward pass
**Status: CONFIRMED**  
Lines 389-393:
```python
if edge_index.numel() > 0 and edge_index.max() >= x.shape[0]:
    raise ValueError(...)
```
Full tensor scan on every forward call. Belongs at data-loading time, not inference hot path.

---

## A26 — `gnn_encoder.py` — `next(self.parameters())` called twice per forward pass
**Status: CONFIRMED**  
Line 398: `_param_dtype = next(self.parameters()).dtype`  
Line 521: `_proj_dtype = next(self.input_proj.parameters()).dtype`  
Two separate generator constructions per forward pass. `self._param_dtype` cached in `__init__` would eliminate both.

---

## A27 — `gnn_encoder.py` — `num_layers` stored but hardcoded to 8
**Status: CONFIRMED**  
Line 196: `self.num_layers = num_layers`  
Architecture is hardcoded to 8 layers; `num_layers` parameter is stored as metadata but has no effect on construction. Misleads checkpoint compatibility checks.

---

## A28 — `transformer_encoder.py` — `except (ImportError, ValueError)` catches real BERT load errors
**Status: CONFIRMED**  
Lines 142-147:
```python
except (ImportError, ValueError):
    self.bert = AutoModel.from_pretrained(
        "microsoft/graphcodebert-base",
        attn_implementation="sdpa",
    )
```
`ValueError` from corrupted config.json, missing files, or incompatible HuggingFace version silently falls through to SDPA retry.

---

## A29 — `transformer_encoder.py` — Python loop for prefix mask construction
**Status: CONFIRMED**  
Lines 241-242 and 284-285 (two code paths):
```python
for b in range(B):
    prefix_mask[b, :gnn_prefix_counts[b]] = 1
```
Appears twice. Vectorizable with `(torch.arange(K).unsqueeze(0) < gnn_prefix_counts.unsqueeze(1))`.

---

## A30 — `transformer_encoder.py` — `_word_embeddings` uses fragile hardcoded path
**Status: CONFIRMED**  
Lines 168-170:
```python
@property
def _word_embeddings(self) -> nn.Embedding:
    return self.bert.base_model.model.embeddings.word_embeddings
```
Five-level dotted path into peft internals. No validation at `__init__` time; failure surfaces at first forward pass.

---

## A31 — `fusion_layer.py` — `_scatter_to_dense` truncation overwrites real node at max_nodes-1
**Status: ALREADY FIXED (C2 fix)**  
Lines 107-116:
```python
# C2 fix: compute valid mask BEFORE clamping so excess nodes (local_idx >= max_nodes)
# are truly dropped. Without this, all excess nodes clamp to position max_nodes-1
# and overwrite each other (last-write-wins = random embedding at that slot).
valid     = local_idx < max_nodes
local_idx = local_idx.clamp(max=max_nodes - 1)
...
out[batch[valid], local_idx[valid]]  = x[valid]
mask[batch[valid], local_idx[valid]] = True
```
**Bug is already fixed.** The comment explicitly documents the old behaviour and the fix. Claim describes the pre-fix state.

---

## A32 — `sentinel_model.py` — `_MAX_TYPE_ID` dynamic recomputation decoupled from encoded .pt files
**Status: CONFIRMED**  
Line 75:
```python
_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 12.0 for v2 schema (ids 0–12)
```
Same class of bug as A3 — dynamic, not hardcoded. Comment acknowledges current value is 12.0 but no assertion enforces it. Same fix required: `assert _MAX_TYPE_ID == 12.0`.

---

## A33 — `sentinel_model.py` — `select_prefix_nodes` Python loop over batch dimension
**Status: CONFIRMED**  
Line 305:
```python
for g in range(num_graphs):
```
Python loop with `.tolist()`, list comprehension, and Python `sort()` inside each iteration. Vectorizable with tensor-based topk-per-graph.

---

## A34 — `sentinel_model.py` — secondary sort uses post-GAT embedding dim, not raw input feature
**Status: CONFIRMED**  
Line 326:
```python
sec = -g_embs[local_idx, _EXT_CALL_DIM].item() if t == _FUNCTION_ID else 0.0
```
`g_embs` is the 256-dim post-GAT output, not raw features. After 8 GAT layers, dimension 10 encodes a learned mixture of all features across the entire neighborhood — not `external_call_count`. IMP-M1's secondary sort is semantically wrong. Fix: use `graphs.x[global_node_idx, _EXT_CALL_DIM]`.

---

## A25b — `sentinel_model.py` — `compute_prefix_attention_mean` discards `gnn_prefix_counts`
**Status: CONFIRMED**  
Lines 544-546:
```python
gnn_prefix = self.select_prefix_nodes(node_embs, batch, node_type_ids, num_graphs)
if isinstance(gnn_prefix, tuple):
    gnn_prefix, _ = gnn_prefix
```
`node_counts` discarded with `_`. Diagnostic forward pass then averages attention over all K=48 positions instead of only the real-node positions, understating prefix attention. Diagnostic only — no training impact.

---

## A35 — `trainer.py` — `_FocalFromLogits` is an unpicklable local class
**Status: CONFIRMED**  
Lines 1066-1069 (inside `train()` function body):
```python
class _FocalFromLogits(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _focal(torch.sigmoid(logits.float()), targets)
loss_fn: nn.Module = _FocalFromLogits()
```
Locally-defined class — unpicklable. Currently safe (never pickled), but fragile for DDP or distributed checkpointing.

---

## A36 — `trainer.py` — `compute_pos_weight` re-reads label CSV
**Status: CONFIRMED**  
Lines 378-388:
```python
def compute_pos_weight(label_csv: str, ...) -> torch.Tensor:
    import pandas as pd
    df = pd.read_csv(label_csv)
```
Reads ~44K rows from CSV on every call. `DualPathDataset` already holds the same data in memory.

---

## A37 — `trainer.py` — threshold sweep O(N×C×19) every validation epoch
**Status: CONFIRMED**  
Lines 477-490 (sweep in `evaluate()`):
```python
if tune_thresholds:
    _candidates = np.linspace(0.1, 0.9, 19)
    for c in range(num_classes):
        for t in _candidates:
            ...
```
Line 1493 (training loop call site):
```python
tune_thresholds=True,  # BUG-M8: per-epoch threshold sweep
```
`tune_thresholds=True` is explicitly passed in the training loop — sweep runs every validation epoch. Comment labels it `BUG-M8`, confirming the author is already aware.

---

## A38 — `trainer.py` — NaN loss `backward()` runs before NaN check; NaN gradients bypass clip
**Status: CONFIRMED**  
Line 650: `loss.backward()`  
Line 713: `if not torch.isfinite(loss).item(): nan_loss_count += 1`  
`backward()` at line 650, finite check at line 713 — NaN gradients pass to `optimizer.step()` and can corrupt Adam momentum buffers (m1, m2) permanently for affected parameters.

---

## Summary

| ID | File | Status | Severity |
|----|------|--------|----------|
| A1 | graph_schema.py | CONFIRMED | Medium |
| A2 | hash_utils.py | CONFIRMED | Low |
| A3 | graph_extractor.py | CONFIRMED | Medium |
| A4 | graph_extractor.py | CONFIRMED | Medium |
| A5 | graph_extractor.py | CONFIRMED (not silent — logs warning) | Low |
| A6 | graph_extractor.py | CONFIRMED | Medium |
| A7 | graph_extractor.py | CONFIRMED | Low |
| A8 | graph_extractor.py | CONFIRMED | Low |
| A9 | graph_extractor.py | CONFIRMED | Medium |
| A10 | graph_extractor.py | CONFIRMED | Medium |
| A11 | graph_extractor.py | CONFIRMED | Medium |
| A12 | graph_extractor.py | CONFIRMED | Low |
| A13 | graph_extractor.py | CONFIRMED | Low |
| A14 | graph_extractor.py | CONFIRMED | Medium |
| A15 | graph_extractor.py | CONFIRMED | Medium |
| A16 | graph_extractor.py | CONFIRMED | Low |
| A17 | graph_extractor.py | CONFIRMED | Medium |
| A18 | graph_extractor.py | CONFIRMED | Medium |
| A19 | ast_extractor.py | CONFIRMED | Medium |
| A20 | ast_extractor.py | CONFIRMED | **High** |
| A21 | ast_extractor.py | CONFIRMED | Low |
| A22 | ast_extractor.py | CONFIRMED | Medium |
| A23 | gnn_encoder.py | CONFIRMED | Low |
| A25 | gnn_encoder.py | CONFIRMED | Low |
| A26 | gnn_encoder.py | CONFIRMED | Low |
| A27 | gnn_encoder.py | CONFIRMED | Low |
| A28 | transformer_encoder.py | CONFIRMED | Medium |
| A29 | transformer_encoder.py | CONFIRMED | Low |
| A30 | transformer_encoder.py | CONFIRMED | Low |
| A31 | fusion_layer.py | **ALREADY FIXED** (C2 fix) | — |
| A32 | sentinel_model.py | CONFIRMED | Medium |
| A33 | sentinel_model.py | CONFIRMED | Low |
| A34 | sentinel_model.py | CONFIRMED | Medium |
| A25b | sentinel_model.py | CONFIRMED | Low |
| A35 | trainer.py | CONFIRMED | Low |
| A36 | trainer.py | CONFIRMED | Low |
| A37 | trainer.py | CONFIRMED | Low |
| A38 | trainer.py | CONFIRMED | **Medium** |

**36 CONFIRMED, 1 ALREADY FIXED (A31), 1 with nuance (A5 — not silent).**  
**Highest priority for pre-Run-5:** A20 (data poisoning), A38 (NaN gradients corrupt Adam state), A10/A18 (silent total loss of CFG/ICFG signal).
