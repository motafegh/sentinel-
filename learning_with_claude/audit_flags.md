# Audit Flags — Learning With Claude

All issues found during teaching. Never delete entries — only append.
Format: one flag per entry, with file, location, description, severity, and status.

Severity scale: **Low** (code smell, clarity issue) | **Medium** (silent bug risk, design gap) | **High** (correctness bug, data corruption risk)
Status: **Open** (not yet fixed) | **Noted** (acknowledged, fix deferred) | **Fixed** (resolved in codebase)

---

## A1 — `graph_schema.py` — Type ID normalization missing range guard
**File:** `ml/src/preprocessing/graph_schema.py`
**Location:** `NODE_FEATURE_DIM` docstring + `FEATURE_NAMES` description of `type_id`
**Issue:** `type_id` is documented as `float(NODE_TYPES[kind]) / 12.0`, implying [0,1] range.
But there is no assertion that `max(NODE_TYPES.values()) == 12`. If a 14th node type is added
(id=13), the normalization silently produces values > 1.0, violating the feature range contract
and destabilising GNN dot products. The existing `assert len(NODE_TYPES) == 13` catches count
but not the max value.
**Fix:** Add `assert max(NODE_TYPES.values()) == 12` in `graph_schema.py` import-time checks.
**Severity:** Medium
**Status:** Open
**Raised:** Session 1

---

## A2 — `hash_utils.py` — `validate_hash` accepts uppercase hex
**File:** `ml/src/utils/hash_utils.py`
**Location:** `validate_hash()` function
**Issue:** Uses `int(hash_string, 16)` to validate hex format, which accepts uppercase
characters (A-F). But `hashlib.md5().hexdigest()` always returns lowercase. The validator
is silently permissive — it validates hashes the system never produces. If a component
accidentally produces uppercase hashes, `validate_hash` says "valid" but the hash won't
match any `.pt` filename (which are always lowercase), causing silent lookup failure.
**Fix:** Replace with `re.match(r'^[0-9a-f]{32}$', hash_string)` to precisely match the
actual contract of `hexdigest()`.
**Severity:** Low
**Status:** Open
**Raised:** Session 1

---

## A3 — `graph_extractor.py` — `_MAX_TYPE_ID` is dynamic, contradicts schema
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** Line 113: `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))`
**Issue:** This is a dynamic computation — it changes automatically whenever `NODE_TYPES`
gains a new entry. This contradicts `graph_schema.py`'s documented `/12.0` normalization.
If a new node type is added without retraining the model:
  1. ALL existing type_id normalizations shift (e.g. CFG_NODE_OTHER: 1.0 → 0.923).
  2. The new type collides with the old max type's normalized value (both → 1.0).
The model was trained with the old normalizations and silently receives wrong features.
No crash, no warning.
**Fix:** Hardcode `_MAX_TYPE_ID: float = 12.0` with a comment pointing to `graph_schema.py`
and the schema change policy. Adding a new node type should require a conscious change here too.
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 1

---

## A4 — `graph_extractor.py` — `assert` used for production invariant check
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** Line 1250: `assert len(node_metadata) == x.shape[0], (...)`
**Issue:** Python `assert` statements are silently removed when running with `python -O`
(optimization flag) or `PYTHONOPTIMIZE=1`, which is common in production batch deployments.
This check guards the critical invariant that `node_metadata` is index-aligned with `graph.x`.
If violated silently, node metadata lookups return data for the wrong node — e.g. the
cross-attention fusion routes the wrong transformer context to the wrong graph node.
**Fix:** Replace with an explicit check:
```python
if len(node_metadata) != x.shape[0]:
    raise SlitherParseError(
        f"node_metadata length {len(node_metadata)} ≠ x.shape[0] {x.shape[0]} "
        f"for '{contract.name}'. Bug in node construction."
    )
```
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 1

---

## A5 — `graph_extractor.py` — `except AttributeError` scope too broad in `_compute_return_ignored`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_return_ignored()`, outer `except AttributeError` block
**Issue:** The broad `except AttributeError` catches any AttributeError from inside the entire
function body — including programming errors like refactored field names. A bug introduced during
refactoring would silently return `-1.0` (sentinel) instead of crashing, hiding the error.
**Fix:** Tighten the try scope to only wrap `func.nodes` access. Inner loop errors should propagate.
**Severity:** Low
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A6 — `graph_extractor.py` — `except Exception: pass` in `_compute_call_target_typed`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_call_target_typed()`, line ~312
**Issue:** `except Exception: pass` is maximally broad — catches all exceptions including
`SystemExit` subclasses and hides Slither API changes. If Slither renames a field and
`recv_type.name` raises `AttributeError`, the code silently falls to the regex scan instead
of surfacing the API breakage. Should be `except (ImportError, AttributeError, TypeError)`.
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A7 — `graph_extractor.py` — `_compute_in_unchecked` is dead code
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** Lines 331–360, `_compute_in_unchecked()` function
**Issue:** Function is marked DEPRECATED with comment "safe to delete after v8 extraction is
complete." v8 is the current schema. The function is never called anywhere in the file.
Dead code adds maintenance burden and confusion.
**Fix:** Delete the function entirely. Verify no tests, docstrings, or scripts reference it first.
**Severity:** Low
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A8 — `graph_extractor.py` — `is True` identity check misses truthy non-booleans
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_has_loop()`, fallback: `getattr(func, "is_loop_present", None) is True`
**Issue:** `is True` uses identity comparison, not truthiness. If `is_loop_present` returns
integer `1` or any truthy non-boolean, the check silently returns `False` — loop presence missed.
**Fix:** Replace with `bool(getattr(func, "is_loop_present", False))`.
**Severity:** Low
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A9 — `graph_extractor.py` — string-based type check fragile on Slither class rename
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_uses_block_globals()`: `type(rv).__name__ == "SolidityVariableComposed"`
**Issue:** If Slither renames the class (not just moves it), this check silently returns 0.0
for all timestamp-related variables. The `Timestamp` and `TOD` vulnerability classes lose
their primary direct signal with no warning. No crash, no log — pure silent miss.
**Fix:** Try the import first, fall back to string check only on ImportError:
```python
try:
    from slither.core.variables.variable import SolidityVariableComposed as _SVC
    _is_svc = lambda rv: isinstance(rv, _SVC)
except ImportError:
    _is_svc = lambda rv: type(rv).__name__ == "SolidityVariableComposed"
```
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A10 — `graph_extractor.py` — `except Exception: pass` in `_cfg_node_type`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_cfg_node_type()`, lines 493–494: `except Exception: pass`
**Issue:** The entire priority-classification logic — all 4 priority branches — is wrapped in a
single `except Exception: pass`. If Slither renames `NodeType.IF`, moves `LowLevelCall`, or
changes `state_variables_written`, the exception is swallowed and every node silently becomes
`CFG_NODE_OTHER` (12). CALL nodes become OTHER: the GNN sees no external calls. WRITE nodes
become OTHER: reentrancy patterns vanish. Zero crash, zero log — silent total loss of CFG signal.
**Fix:** Move each import to module level with a clear ImportError guard. Inside the function,
catch only the specific exceptions that Slither attribute access can raise
(`AttributeError`, `TypeError`). Let unexpected exceptions propagate.
**Severity:** Medium
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A11 — `graph_extractor.py` — hardcoded parent feature indices in `_build_cfg_node_features`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_cfg_node_features()`, lines 543–547: `p[1]`, `p[3]`, `p[4]`, `p[5]`, `p[9]`
**Issue:** Parent feature dims are accessed by raw integer index. The comment names them
(visibility=1, view=3, payable=4, complexity=5, has_loop=9) but nothing enforces that contract.
If `graph_schema.py` reorders `FEATURE_NAMES` — e.g. inserting a new dim before index 3 — these
indices silently read the wrong feature (e.g. `p[3]` might now be `view` → `payable`, swapping
two values). The `len(p) > 9` guard for has_loop means if parent_features is shorter than 10
elements, has_loop falls back to 0.0 for every CFG node — all DoS loop detection degrades silently.
**Fix:** Use `FEATURE_NAMES.index("has_loop")` etc. to derive indices at parse time, or define
named constants `_F_VISIBILITY = 1` etc. in graph_schema.py alongside FEATURE_NAMES.
**Severity:** Medium
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A12 — `graph_extractor.py` — `n.node_id` accessed without `getattr` fallback in sort key
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_control_flow_edges()`, line 611: `n.node_id`
**Issue:** The CFG sort key uses `n.node_id` directly (no `getattr` with default). If a Slither
version does not attach `node_id` to synthetic nodes (ENTRY_POINT, BEGIN_LOOP, etc.), this raises
`AttributeError` inside the `sorted()` call, aborting CFG construction for the entire function
with no edges produced — silently dropped with zero graph signal for that function.
**Fix:** `getattr(n, "node_id", 0)` in the sort key lambda.
**Severity:** Low
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A13 — `graph_extractor.py` — silently dropped CONTROL_FLOW edges not logged
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_control_flow_edges()`, lines 639–641: `if successor in node_index_map`
**Issue:** When a CFG successor node is not in `node_index_map` (e.g. a cross-function edge or
Slither synthetic node not added in Pass 1), the edge is silently dropped. No log, no counter.
In a contract with complex control flow, entire branches may produce zero CONTROL_FLOW edges.
The GNN's Phase 2 (which uses CONTROL_FLOW(6) edges) operates on a structurally incomplete graph.
**Fix:** Log a debug/warning when a successor is not found:
`logger.debug("CFG edge dropped: successor %s not in node_index_map for func %s", successor, func.name)`
**Severity:** Low
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A14 — `graph_extractor.py` — RETURN_TO creates cartesian product including impossible revert→normal paths
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_add_icfg_edges()`, lines 699–706: cartesian product of callee_terminals × call_site_sons
**Issue:** Every terminal of the callee (including revert/throw terminals) gets a RETURN_TO edge
to every successor of the call site (post-call code). A revert terminal unwinds the call stack and
never transfers control to post-call code — but the graph models this as a possible path. The GNN
can form spurious reentrancy patterns: a node that always reverts appears to "reach" the code after
the call. A full ICFG would distinguish normal-return terminals from exceptional-exit terminals.
**Fix:** Classify callee terminal nodes as normal-exit or exceptional-exit (revert/throw). Only
wire normal-exit terminals to call-site successors. This requires Slither to expose CFG node type
for THROW/REVERT nodes, which it does via `NodeType.THROW`.
**Severity:** Medium
**Status:** Open
**Raised:** Session 4, Chunk 4

---

## A15 — `graph_extractor.py` — DEF_USE def_map keyed by variable name, not object identity
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_add_def_use_edges()`, line 752: `def_map.setdefault(lval.name, [])`
**Issue:** Two different `LocalVariable` objects with the same name (e.g. loop variable `i` in
nested scopes, or Solidity variable shadowing) are conflated in `def_map`. Reads of the inner
variable appear as uses of the outer variable's definitions, producing spurious DEF_USE edges
that model data flow that cannot occur. Slither's SSA partially mitigates this but `.name` does
not always include SSA suffixes depending on the Slither version.
**Fix:** Key `def_map` by the variable object itself (Python objects are hashable by identity)
rather than by `.name`: `def_map.setdefault(id(lval), []).append(node_idx)` and match on
`id(var)` in Pass 2 (requires building a secondary `var_id_to_name` lookup or using object keys).
**Severity:** Medium
**Status:** Open
**Raised:** Session 4, Chunk 4

---

## A16 — `graph_extractor.py` — `assert` for sentinel range in `_build_node_features`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_node_features()`, lines 856–857: `assert return_ignored in (-1.0, 0.0, 1.0)`
**Issue:** Same pattern as A4. Under `python -O` / `PYTHONOPTIMIZE=1`, these asserts are silently
removed. If a future change to `_compute_return_ignored` or `_compute_call_target_typed` returns
an out-of-range value, it passes through to the model's feature vector with no crash and no log.
**Fix:** Replace with explicit checks:
```python
if return_ignored not in (-1.0, 0.0, 1.0):
    raise SlitherParseError(f"return_ignored out of range: {return_ignored}")
if call_target_typed not in (-1.0, 0.0, 1.0):
    raise SlitherParseError(f"call_target_typed out of range: {call_target_typed}")
```
**Severity:** Low
**Status:** Open
**Raised:** Session 4, Chunk 4

---

## A17 — `graph_extractor.py` — Slither exception routing by string keyword matching
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `extract_contract_graph()`, lines 1059–1067
**Issue:** `Slither(...)` failures are routed to `SolcCompilationError` vs `SlitherParseError`
by checking if `str(exc).lower()` contains keywords like "compil", "syntax", "solc". This is
fragile in two directions: (1) A genuine Slither internal error whose message mentions "compiler"
gets misclassified as user fault → API returns HTTP 400 instead of HTTP 500. (2) A Solidity
syntax error whose message doesn't match keywords (non-English locale, future Slither version)
gets misclassified as system fault → API returns HTTP 500 for bad user input. Both mask the
real failure category.
**Fix:** Catch specific exception types from Slither's and crytic-compile's public exception
hierarchy: `slither.exceptions.SlitherError`, `crytic_compile.CryticCompileException` etc.
instead of pattern-matching the string representation.
**Severity:** Medium
**Status:** Open
**Raised:** Session 5, Chunk 5

---

## A18 — `graph_extractor.py` — `except Exception: pass` when building ICFG maps
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `extract_contract_graph()`, lines 1154–1167: inner `try/except Exception: pass`
**Issue:** The block that populates `_func_entry_map` and `_func_terminal_map` for a given
function is wrapped in a bare `except Exception: pass`. If the `NodeType` import fails or any
Slither attribute raises, the maps for this function are silently left empty. Then
`_add_icfg_edges` finds no entry point and no terminals for calls into this function,
producing zero CALL_ENTRY and RETURN_TO edges for all callers. The GNN's inter-procedural
reentrancy detection loses these edges with zero signal: no log, no counter, no warning.
**Fix:** At minimum, `logger.warning(...)` with the function name and exception. Ideally catch
only `ImportError, AttributeError` and let unexpected exceptions propagate.
**Severity:** Medium
**Status:** Open
**Raised:** Session 5, Chunk 5

---

## A19 — `ast_extractor.py` — `get_solc_binary` uses `Path.cwd()` instead of `get_project_root()`
**File:** `ml/src/data_extraction/ast_extractor.py`
**Location:** `get_solc_binary()`, line 143: `venv_path = Path.cwd() / ".venv" / ...`
**Issue:** Binary path resolution depends on the current working directory at call time. If the script is
invoked from any directory other than the project root (e.g. from `ml/src/`, a CI job, or a cron task
with a different cwd), `.venv/.solc-select/artifacts/` is not found. Every contract gets `solc_binary=None`,
silently falling back to system solc and bypassing version pinning entirely. `get_project_root()` is defined
4 lines below and resolves the project root from `__file__`, but is not used here.
**Fix:** Replace `Path.cwd()` with `get_project_root()`.
**Severity:** Medium
**Status:** Open
**Raised:** Session 7

---

## A20 — `ast_extractor.py` — `label=0` hardcoded for all contracts in batch extraction
**File:** `ml/src/data_extraction/ast_extractor.py`
**Location:** `extract_batch_with_checkpoint()`, lines 307–311: `partial(..., label=0)`
**Issue:** Every contract extracted by the batch pipeline receives `graph.y = tensor([0])` (safe), regardless
of the contract's actual vulnerability status. The dataframe `df` likely contains vulnerability labels but
they are never read. If vulnerable contracts are present in the parquet, they are mislabeled as safe —
producing poisoned training data. The model is trained to predict "safe" for patterns that are actually
vulnerable.
**Fix:** Pass the label per-row from `df`. Replace `pool.imap(worker, paths)` with `pool.imap(worker,
zip(paths, labels))` and unpack in `contract_to_pyg`.
**Severity:** High
**Status:** Open
**Raised:** Session 7

---

## A21 — `ast_extractor.py` — `print()` called from worker processes (verbose mode)
**File:** `ml/src/data_extraction/ast_extractor.py`
**Location:** `contract_to_pyg()`, lines 223 and 228: `print(f"  Skipped ...")` / `print(f"  Unexpected...")`
**Issue:** `contract_to_pyg` is called inside `mp.Pool` workers via `imap`. `print()` from 11 concurrent
worker processes writes to stdout simultaneously without serialization, producing interleaved/garbled log
lines. Under high load, consecutive log messages from different workers overwrite each other in the terminal.
**Fix:** Use `logging` with a `QueueHandler`/`QueueListener` pattern (designed for multiprocessing logging),
or pass a `multiprocessing.Queue` to workers for funneling messages to the main process.
**Severity:** Low
**Status:** Open
**Raised:** Session 7

---

## A22 — `ast_extractor.py` — `torch.save` has no error handling; disk-full crashes entire batch
**File:** `ml/src/data_extraction/ast_extractor.py`
**Location:** `extract_batch_with_checkpoint()`, line 329: `torch.save(result, graph_file)`
**Issue:** `torch.save` is called in the main process loop with no `try/except`. A disk-full error or
I/O error raises an uncaught exception that exits the `for result in tqdm(...)` loop and the `with mp.Pool()`
block. Up to 499 successfully processed contracts since the last checkpoint are lost (hash not in checkpoint,
`.pt` files may be partially written). The pool workers are left in undefined state.
**Fix:** Wrap `torch.save` in `try/except OSError` (covers disk-full, permission errors). On failure, log
to `failed_saves.json` and continue — don't abort the batch for a single save failure.
**Severity:** Medium
**Status:** Open
**Raised:** Session 7

---

## A23 — `gnn_encoder.py` — `last_weight_stds` comment says "0 if N=1" but PyTorch returns NaN
**File:** `ml/src/models/gnn_encoder.py`
**Location:** `_JKAttention.forward()`, line ~123: `w_nk.std(0).detach()`
**Issue:** `torch.Tensor.std()` on a single-element tensor (N=1) returns `NaN`, not `0.0`. If a graph
has exactly 1 node (e.g. a contract with only a constructor that has no CFG), `w_nk` is `[1, 3]`
and `w_nk.std(0)` produces `[nan, nan, nan]`, written into the `last_weight_stds` diagnostic buffer.
A monitoring dashboard or diagnostic script reading this buffer would see NaN and may fire incorrect
alerts or produce misleading plots. No training impact — buffer is diagnostic-only.
**Fix:** `w_nk.std(0, unbiased=False).nan_to_num(0.0).detach()` — biased std returns 0.0 for N=1,
which matches the intended semantics ("no spread when only one sample").
**Severity:** Low
**Status:** Open
**Raised:** Session 8

---

## A25 — `gnn_encoder.py` — `edge_index.max()` O(E) scan on every forward pass
**File:** `ml/src/models/gnn_encoder.py`
**Location:** `forward()`, line ~382: `if edge_index.numel() > 0 and edge_index.max() >= x.shape[0]`
**Issue:** `edge_index.max()` scans the full `[2, E]` edge tensor every forward pass. For a batch
with 10,000+ edges, this is a full tensor scan on the hot path. In a training loop of 1,000 steps/epoch
on 8 GPUs, this adds up. The check is correct but belongs at data-loading or extraction time where it
runs once per contract, not per training step.
**Fix:** Move to `graph_extractor.py` output validation (already has OOR checks) or to `DualPathDataset.__getitem__`.
In `forward()`, keep only behind `if __debug__:` or remove after pipeline is proven correct.
**Severity:** Low
**Status:** Open
**Raised:** Session 9

---

## A26 — `gnn_encoder.py` — `next(self.parameters())` called twice per forward pass
**File:** `ml/src/models/gnn_encoder.py`
**Location:** `forward()`, lines ~391 and ~513
**Issue:** `next(self.parameters()).dtype` and `next(self.input_proj.parameters()).dtype` each create
a new generator over module parameters and immediately advance it. Called on every forward pass.
Negligible individually but unnecessary — the dtype is fixed at construction and never changes.
**Fix:** Cache as `self._param_dtype = next(self.parameters()).dtype` in `__init__`; replace both
forward-pass lookups with `self._param_dtype`.
**Severity:** Low
**Status:** Open
**Raised:** Session 9

---

## A28 — `transformer_encoder.py` — `except (ImportError, ValueError)` catches real BERT load errors
**File:** `ml/src/models/transformer_encoder.py`
**Location:** Lines ~142–147: `except (ImportError, ValueError)` in Flash Attention 2 fallback
**Issue:** `from_pretrained` raises `ValueError` for many reasons beyond "flash attention not
supported" — e.g. corrupted config.json, incompatible HuggingFace version, missing model files.
These genuine errors are silently caught, the SDPA fallback is attempted, and the same error
fires again on the SDPA path — but now the user sees "SDPA active" in the log and a confusing
second traceback rather than the original root cause.
**Fix:** Check `"flash_attention_2" in str(exc)` before accepting the fallback; re-raise other
ValueErrors immediately. Alternatively, only catch `ImportError` (flash_attn not installed) and
let all ValueErrors propagate.
**Severity:** Medium
**Status:** Open
**Raised:** Session 10

---

## A29 — `transformer_encoder.py` — Python loop for prefix mask construction
**File:** `ml/src/models/transformer_encoder.py`
**Location:** `forward()`, lines ~239–243 (single-window) and ~282–287 (multi-window)
**Issue:** `for b in range(B): prefix_mask[b, :gnn_prefix_counts[b]] = 1` — Python loop over
batch dimension appears in both code paths. Vectorized replacement:
`(torch.arange(K).unsqueeze(0) < gnn_prefix_counts.unsqueeze(1)).to(mask_dtype)`
For small batches the overhead is negligible but the pattern is inconsistent with vectorized style
used elsewhere and doubles the forward-pass Python interpreter overhead for the prefix path.
**Fix:** Replace both occurrences with the vectorized arange comparison.
**Severity:** Low
**Status:** Open
**Raised:** Session 10

---

## A30 — `transformer_encoder.py` — `_word_embeddings` property uses fragile hardcoded path
**File:** `ml/src/models/transformer_encoder.py`
**Location:** `_word_embeddings` property, line ~170: `self.bert.base_model.model.embeddings.word_embeddings`
**Issue:** This attribute path is specific to peft's `PeftModel` wrapper around a `RobertaModel`.
If peft changes its internal wrapper structure (`base_model` or `model` attribute names) in a
future version, this raises `AttributeError` inside `forward()` with no informative message.
No validation occurs at `__init__` time — the failure only surfaces during the first prefix
injection forward pass.
**Fix:** Cache and validate at `__init__`: `self._word_emb_layer = self.bert.base_model.model.embeddings.word_embeddings`
wrapped in a try/except AttributeError with a clear message directing the user to check the peft version.
**Severity:** Low
**Status:** Open
**Raised:** Session 10

---

## A31 — `fusion_layer.py` — `_scatter_to_dense` truncation overwrites real node at max_nodes-1
**File:** `ml/src/models/fusion_layer.py`
**Location:** `_scatter_to_dense()`, line ~107: `local_idx = local_idx.clamp(max=max_nodes - 1)`
**Issue:** For a graph with more than `max_nodes=1024` nodes, all excess nodes are clamped to
`local_idx = 1023`. Multiple writes `out[batch, 1023] = x[i]` overwrite each other — only the
last truncated node's embedding survives. Additionally, the real node at position 1023 is also
overwritten. `mask[batch, 1023]` stays True (set by the real node), so pooling treats this
corrupted position as a valid real node. Affects <1% of corpus but produces silent data corruption.
**Fix:** Skip the write for out-of-bounds nodes:
`valid = local_idx < max_nodes`
`out[batch[valid], local_idx[valid]] = x[valid]`
`mask[batch[valid], local_idx[valid]] = True`
**Severity:** Medium
**Status:** Open
**Raised:** Session 11

---

## A27 — `gnn_encoder.py` — `num_layers` stored but hardcoded to 8
**File:** `ml/src/models/gnn_encoder.py`
**Location:** `GNNEncoder.__init__()`, line ~196: `self.num_layers = num_layers`
**Issue:** `num_layers` is stored as metadata but the 8 conv layers are hardcoded unconditionally.
Passing `num_layers=7` (pre-IMP-G3 value) constructs the same 8-layer network as `num_layers=8`.
If a checkpoint was saved with `num_layers=7` and loaded into a freshly constructed model with
`num_layers=8`, `load_state_dict` may succeed or fail depending on whether IMP-G3 (conv4c) added
new keys — but the caller has no reliable way to know which version the checkpoint requires.
**Fix:** Add `assert num_layers == 8, f"GNNEncoder v8 hardcodes 8 layers; got {num_layers}"` at
construction time, or make the layer count truly dynamic.
**Severity:** Low
**Status:** Open
**Raised:** Session 8
