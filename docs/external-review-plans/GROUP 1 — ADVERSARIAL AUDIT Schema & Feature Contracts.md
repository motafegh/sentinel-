Got it — the `__init__.py` files are empty, so I need to grep the actual source code to verify cross-file usage. Let me do exactly that for Group 1. I'll read both files fresh and then search for every exported constant/function across the entire `ml/` codebase.
Good, I have both files. Now let me search for every constant and function exported from these two files across the entire ML codebase:
Now let me also check the `__init__.py` for the preprocessing module and also read the `inference/preprocess.py` and the `gnn_encoder.py` to verify how they consume schema constants — particularly the `NUM_NODE_TYPES` dead code and the `type_id` normalisation issue:
Now let me check one more critical thing — the `graph_extractor.py` to verify the `type_id` normalisation (the v5.0 bug where raw 0-12 was used instead of /12.0):
Now let me also check the `_build_cfg_node_features` function in graph_extractor.py to verify type_id normalisation consistency there too:
Excellent. Now I have everything I need for a thorough Group 1 adversarial audit. Let me also quickly check the `predictor.py` to verify the hardcoded `NODE_FEATURE_DIM` values there:
Now I have a complete picture. Let me write the full adversarial audit for Group 1.
Now here's the full adversarial audit for **Group 1: Schema & Feature Contracts**.

---

# 🔴 GROUP 1 — ADVERSARIAL AUDIT: Schema & Feature Contracts

**Files:** `graph_schema.py`, `hash_utils.py`  
**Audit lens:** Hostile. Every assumption is wrong until proven otherwise. Every constant is a ticking bomb until verified.

---

## FINDING 1.1 — `NUM_NODE_TYPES`: Dead Code, Live Danger

**Severity: HIGH**

`NUM_NODE_TYPES = 13` is defined at line 125 of `graph_schema.py`, complete with a docstring saying "Used by GNNEncoder for the node type embedding." **It is never imported or referenced anywhere in the codebase.** The `__init__.py` re-exports `NUM_EDGE_TYPES` but not `NUM_NODE_TYPES`. The GNN encoder does not use a node-type embedding layer at all — it normalises `type_id / 12.0` as feature[0] and feeds it as a raw float.

**The hostile read:** Two possibilities, both bad:
1. **The v5.1 plan adds a node-type embedding** (which would need `NUM_NODE_TYPES`). If someone implements this, they'll reach for the constant — but it's 13 while the actual max `type_id` is 12. If they use `nn.Embedding(NUM_NODE_TYPES, d)`, they'd get indices 0–12, which is correct. But if they accidentally use `NUM_NODE_TYPES - 1 = 12` thinking IDs are 0-indexed with max=12 (like `NUM_EDGE_TYPES - 1`), they'd miss the CONTRACT(7) type. **This is a latent off-by-one trap.**
2. **The docstring is lying** — it says "Used by GNNEncoder" but GNNEncoder doesn't use it. A future developer reading the docstring will assume the constant is actively consumed and won't add a new assertion checking it.

**Recommendation:** Either (a) add `NUM_NODE_TYPES` to the `__init__.py` re-exports and add an `assert` in GNNEncoder linking it to the feature layout, or (b) delete it and the misleading docstring. Dead code with false documentation is worse than no code.

---

## FINDING 1.2 — `type_id` Normalisation: Hardcoded `/ 12.0` Magic Number

**Severity: MEDIUM**

Both `_build_node_features()` (line 563) and `_build_cfg_node_features()` (line 401) compute:
```python
float(type_id) / 12.0
```

The divisor `12.0` is the maximum `NODE_TYPES` ID value, but:
- It is **not** derived from `NUM_NODE_TYPES - 1` or any schema constant. It's a raw magic number.
- If a 13th node type is ever added (ID=13, requiring `NUM_NODE_TYPES=14`), the normalisation silently breaks: `13 / 12.0 = 1.083`, leaking outside the [0, 1] range the GNN comment claims.
- The `gnn_encoder.py` docstring (line 60-63) says: "x[:, 0] = type_id is normalised to [0, 1] in graph_extractor.py (/ 12.0)." This documentation would also silently become wrong.

**The hostile read:** The schema contract says features are in [0,1] or small ranges. The normalisation is **not enforced** by any runtime assertion. A schema extension would produce out-of-range features with no error, and the GNN would silently process them with potentially degraded attention scores.

**Recommendation:** Replace `12.0` with a computed constant:
```python
_MAX_TYPE_ID: int = max(NODE_TYPES.values())  # = 12
# Then: float(type_id) / float(_MAX_TYPE_ID)
```
And add an `assert 0.0 <= float(type_id) / float(_MAX_TYPE_ID) <= 1.0` in both functions.

---

## FINDING 1.3 — Three Scripts Duplicate `hash_utils` with Subtle Differences

**Severity: HIGH (data integrity)**

Three scripts bypass `hash_utils` and compute MD5 hashes inline:

| Script | What it does | How it differs from `hash_utils` |
|--------|-------------|----------------------------------|
| `reextract_graphs.py` (line 90-92) | `hashlib.md5(str(rel).encode("utf-8")).hexdigest()` | Uses `rel` (relative path) not the full path that `get_contract_hash()` uses (`str(contract_path)`). **Different input = different hash = file pairing failure.** |
| `verify_splits.py` (line 190) | `hashlib.md5(str(rel).encode()).hexdigest()` | Same as above — `rel` vs full path. Also missing `utf-8` encoding specification (platform-dependent default). |
| `dedup_multilabel_index.py` (lines 87, 92) | `hashlib.md5(str(rel).encode("utf-8")).hexdigest()` and `hashlib.md5(sol.read_bytes()).hexdigest()` | Path hash uses `rel`; content hash uses `sol.read_bytes()` (raw bytes) vs `hash_utils.get_contract_hash_from_content()` which takes a **string** and calls `.encode('utf-8')`. If any `.sol` file has a BOM or non-UTF-8 byte sequence, these produce **different hashes**. |

**The hostile read:** The `dedup_multilabel_index.py` script is the one that discovered the 34.9% cross-split leakage. It's computing `path_md5` and `content_md5` using slightly different semantics than the rest of the pipeline. The dedup script's `path_md5` uses relative paths while `get_contract_hash()` uses whatever path object is passed (which in `ast_extractor.py` is the full path). **If the dedup script's path-md5 values don't match the graph-filename hash values, the dedup is comparing the wrong keys and the leakage calculation itself could be wrong.**

Additionally, `verify_splits.py` omits the explicit `utf-8` encoding argument, relying on `locale.getpreferredencoding()`. On a Turkish or Greek locale, this could produce a different hash for the same path string.

**Recommendation:** All three scripts must import from `hash_utils`. The `get_contract_hash()` function must be used everywhere path-based hashing occurs, and `get_contract_hash_from_content()` everywhere content-based hashing occurs. No exceptions.

---

## FINDING 1.4 — `FEATURE_SCHEMA_VERSION` Not Bumped After v5.1 Phase 0 Fixes

**Severity: HIGH (inference correctness)**

`FEATURE_SCHEMA_VERSION = "v2"` was set when the v2 schema was created. The v5.1 Phase 0 fixes changed:
1. **Interface selection logic** (`_select_contract` now skips interfaces) — changes which contract is extracted, altering graph structure and features.
2. **Function-level pooling** — architectural change to how node embeddings are aggregated.
3. **`CFG_NODE_WRITE` mapping** — changed from `ReferenceVariable` to `StateVariable`, which changes which CFG nodes get type_id=9 vs type_id=12.

These are all feature-engineering changes that **invalidate cached inference results**. But `FEATURE_SCHEMA_VERSION` remains `"v2"`, meaning:
- `preprocess.py` line 253: `contract_hash = f"{content_hash}_{FEATURE_SCHEMA_VERSION}"` produces the **same cache key** for pre-fix and post-fix graphs.
- If the inference cache was populated before Phase 0 fixes, stale graphs will be served after the fix with **no invalidation**.

**The hostile read:** A production inference server with a warm cache would serve pre-fix graphs indefinitely after a code deployment that includes Phase 0 fixes. The model was retrained on fixed graphs but the cache returns unfixed ones. Silent accuracy regression.

**Recommendation:** Bump `FEATURE_SCHEMA_VERSION = "v2.1"` (or `"v3"`) immediately. Any deployment of v5.1 code must include this bump. Additionally, add a CI check: if `graph_extractor.py` or `graph_schema.py` changes, require `FEATURE_SCHEMA_VERSION` to be updated.

---

## FINDING 1.5 — Stale Doc Comments in `preprocess.py`

**Severity: LOW (misleading but not runtime-incorrect)**

`preprocess.py` line 39 and line 55 say:
```
NODE_FEATURE_DIM raw floats per node (currently 13 in v5, was 8 in v1/v4)
```
and:
```
graph.x  [N, NODE_FEATURE_DIM]  float32  (13 in v5; was 8 in v4)
```

But `NODE_FEATURE_DIM = 12`, not 13. This is a leftover from an intermediate draft that had `gas_intensity` as feature[12]. The comment is **wrong**. While it doesn't affect runtime, a developer debugging a shape mismatch would read "13 in v5" and think the code should produce 13 features, causing confusion.

**Recommendation:** Fix the comments to say "12 in v5 (v2 schema)".

---

## FINDING 1.6 — `predictor.py` Hardcodes `NODE_FEATURE_DIM` Values

**Severity: MEDIUM**

`predictor.py` lines 87-92 define:
```python
_ARCH_TO_NODE_DIM: dict[str, int] = {
    "three_eye_v5":         12,    # v5: NODE_FEATURE_DIM=12
    "cross_attention_lora": 8,     # v4: NODE_FEATURE_DIM=8
    "legacy":               8,
    "legacy_binary":        8,
}
```

These values are **not imported from `graph_schema.py`** — they're hardcoded. If `NODE_FEATURE_DIM` ever changes, this dict silently goes stale. The comment says "Do NOT hardcode 8 or 12 here; always derive from this map" — but the map itself hardcodes 12 and 8!

**The hostile read:** A v5.2 schema change that adds a feature (making `NODE_FEATURE_DIM=13`) would update `graph_schema.py` and `gnn_encoder.py` (which imports the constant), but `predictor.py` would still create dummy graphs with 12 features. The warmup forward pass would succeed (12 ≠ 13 would crash in GNN conv1), but only if the model was loaded with the new architecture. If a new architecture key is added without updating this dict, the fallback is 12 — potentially wrong.

**Recommendation:** Import `NODE_FEATURE_DIM` from `graph_schema.py` for the current architecture and only hardcode legacy values for backward compatibility:
```python
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM as _CURRENT_NODE_DIM
_ARCH_TO_NODE_DIM = {
    "three_eye_v5": _CURRENT_NODE_DIM,  # always in sync with schema
    ...
}
```

---

## FINDING 1.7 — Dual-Hash Semantic Mismatch (Path vs Content)

**Severity: HIGH (the original 34.9% leakage root cause, still partially unresolved)**

The pipeline uses two different hash schemes:

| Entry point | Hash function | Input | Format |
|---|---|---|---|
| `process(sol_path)` | `get_contract_hash(sol_path)` | Full path string | `{path_md5}` |
| `process_source(source_code)` | `get_contract_hash_from_content(source_code)` | Source code text | `{content_md5}_v2` |

The offline batch pipeline (`ast_extractor.py`) uses `get_contract_hash(contract_path)` — path-based. The inference cache uses content-based hashing. **These two hashes will never match.** This means:

1. A contract processed via `process()` gets cache key `{path_md5}`.
2. The same contract processed via `process_source()` gets cache key `{content_md5}_v2`.
3. These are different keys → cache miss → Slither runs again even though the graph was already computed.

This is by design (content-hash enables dedup, path-hash enables file pairing), but it creates a **semantic split**: if someone accidentally passes a path-based hash where a content-based hash is expected (or vice versa), the graph/tokens pairing breaks silently.

More critically, the `process()` path does NOT append `FEATURE_SCHEMA_VERSION`:
```python
# process() — line 191:
contract_hash = get_contract_hash(sol_path)  # NO schema version suffix

# process_source() — line 253:
contract_hash = f"{content_hash}_{FEATURE_SCHEMA_VERSION}"  # HAS schema version suffix
```

This means `process()` cache entries are **never invalidated by schema changes**, while `process_source()` entries are. A schema bump invalidates the content-hash cache but not the path-hash cache.

**The hostile read:** If both `process()` and `process_source()` ever share the same cache backend, `process()` writes a key without the schema version suffix. After a schema change, `process_source()` writes a new key WITH the suffix. The old key (no suffix) remains in the cache forever, and any `process()` call retrieves stale data.

**Recommendation:** `process()` must also include `FEATURE_SCHEMA_VERSION` in its cache key. The hash format should be unified: `"{hash}_{FEATURE_SCHEMA_VERSION}"` for both paths.

---

## FINDING 1.8 — `hash_utils.py` Dead Code: 3 Functions Never Used Externally

**Severity: LOW (maintenance burden, not a correctness issue)**

Three functions in `hash_utils.py` are never called outside the module:

| Function | Status |
|---|---|
| `validate_hash()` | Only used internally by `extract_hash_from_filename()` and self-test |
| `get_filename_from_path()` | Only used in self-test |
| `extract_hash_from_filename()` | Only used in self-test |

These aren't harmful, but they're maintenance surface. If the hash format ever changes (e.g., switching from MD5 to SHA256 for collision resistance), all three must be updated alongside the two active functions — increasing the chance of a missed update.

**Recommendation:** Either remove the dead functions or mark them as `# KEEP: planned use in <specific feature>` with a ticket reference.

---

## FINDING 1.9 — Sentinel Value `-1.0` Not Documented in Schema Asserts

**Severity: MEDIUM**

Features `[7]` (return_ignored) and `[8]` (call_target_typed) use `-1.0` as a sentinel for "IR unavailable" / "source unavailable." The schema docstring documents this, but there is **no runtime assertion** that these values stay in the expected range `{-1.0, 0.0, 1.0}`.

In `_build_node_features()`, `_compute_return_ignored()` and `_compute_call_target_typed()` are called but their return values are not range-checked. If a Slither version change causes these functions to return, say, `2.0`, the feature vector would contain an out-of-range value with no error.

Meanwhile, `in_unchecked` is ALWAYS `0.0` for CFG_NODE nodes (line 410). The schema docstring says "CFG_NODE in_unchecked [9] is always 0.0 — never inherited from the parent function." This is an **implicit contract** not enforced by any assertion.

**The hostile read:** A future contributor modifying `_compute_in_unchecked()` to propagate the function-level flag to CFG nodes (seems "obvious" — why wouldn't you?) would violate this contract with no error, and the GNN would receive in_unchecked=1.0 for CFG nodes outside unchecked blocks — exactly the false-positive pattern the comment warns against.

**Recommendation:** Add a feature-range assertion at the end of `_build_node_features()`:
```python
result = [float(type_id) / 12.0, visibility, ...]
assert result[7] in (-1.0, 0.0, 1.0), f"return_ignored out of range: {result[7]}"
assert result[8] in (-1.0, 0.0, 1.0), f"call_target_typed out of range: {result[8]}"
return result
```
And add a similar assertion in `_build_cfg_node_features()` that `result[9] == 0.0`.

---

## FINDING 1.10 — MD5 Collision Surface for 68K+ Contracts

**Severity: LOW (statistical, not practical)**

`hash_utils.py` uses MD5 (128-bit) for contract identification. The docstring claims "Collision probability: ~0% for millions of files." This is technically true for random inputs (birthday bound: ~2^64 files for a 50% collision chance), but:

1. The path-based hash (`get_contract_hash`) operates on path strings that share long common prefixes (e.g., `BCCC-SCsVul-2024/SourceCodes/Reentrancy/contract_001.sol` through `contract_999.sol`). Structured inputs with common prefixes don't meaningfully increase MD5 collision risk (MD5 handles prefix similarity well), but this isn't stated.
2. The content-based hash (`get_contract_hash_from_content`) operates on Solidity source code, which often has large common boilerplate. Two contracts differing only in a single variable name would have different hashes — MD5 is sensitive to single-byte changes.
3. The `validate_hash()` function checks for valid hex but does NOT check for known MD5 collision pairs (which exist since 2004). If someone deliberately crafts two Solidity files with the same MD5 hash (using chosen-prefix collision), the dedup pipeline would treat them as the same contract.

**The hostile read:** MD5 chosen-prefix collisions cost ~$0.10 on modern GPUs. An adversary could craft a malicious contract with the same content-md5 as a known-safe contract, bypassing the dedup system. This is a theoretical attack but relevant for a **vulnerability detector** — the system being attacked is itself a security tool.

**Recommendation:** For the current dataset size (44K contracts), MD5 is fine. For production, consider SHA-256 for content hashing (path hashing can stay MD5 since it's just for file naming, not security). At minimum, add a comment acknowledging the collision surface.

---

## FINDING 1.11 — `graph_schema.py` Slither Version Check Has a Gap

**Severity: MEDIUM**

Lines 62-73 check `slither-analyzer >= 0.9.3` but:
1. The upper bound is `< 0.11` (line 70), but `NODE_TYPES` values 8-12 (CFG subtypes) may change in future Slither versions. There's no lower-bound test for the specific `NodeType.STARTUNCHECKED` enum — just a version string comparison.
2. If Slither 0.11.0 renames `NodeType.STARTUNCHECKED` to `NodeType.UNCHECKED_BLOCK`, the version check passes (0.11.0 is NOT < 0.11... wait, the check is `< 0.11`, so 0.11.0 would NOT be caught). Actually the check is `_version < (0, 9, 3)` raises error — there's no upper bound check that raises an error. The upper bound `<0.11` in the error message is just a recommendation, not enforced.
3. The `except PackageNotFoundError: pass` at line 72-73 means if Slither isn't installed at all, the version check is silently skipped. This is fine for inference-only deploys, but if someone runs graph extraction without Slither installed, they'll get an ImportError from `ast_extractor.py` line 82-85, not a clear "you need Slither >=0.9.3" message.

**Recommendation:** Add an explicit upper-bound warning (not error) when Slither version exceeds tested range. And add a runtime check in `graph_extractor.py` that actually imports `NodeType.STARTUNCHECKED` and raises if it doesn't exist, rather than relying on version string parsing.

---

## FINDING 1.12 — `VISIBILITY_MAP` Collapses `public` and `external`

**Severity: LOW (by design, but worth noting)**

```python
VISIBILITY_MAP: dict[str, int] = {
    "public":   0,
    "external": 0,  # same ordinal as public
    "internal": 1,
    "private":  2,
}
```

`public` and `external` both map to 0. The docstring justifies this as "open-to-closed axis." But from a vulnerability perspective, `external` and `public` have different call semantics:
- `external`: can only be called from outside the contract (no internal calls).
- `public`: can be called both internally and externally.

For reentrancy detection, this distinction matters: an `external` function that makes an external call creates a direct reentrancy path, while a `public` function called internally might not. The model receives the same feature[1] value for both, losing this signal.

**The hostile read:** The v5.0 behavioral test showed 0% specificity. Part of the reason might be that the model can't distinguish `external` from `public` entry points, making it unable to learn "external functions with state writes after calls are dangerous" vs "public functions with the same pattern that are safe (internal-only calls)."

**Recommendation:** Consider splitting: `public=0`, `external=1`, `internal=2`, `private=3`. This adds one ordinal step but preserves the ordering while giving the model the distinction it needs. This would be a schema change requiring FEATURE_SCHEMA_VERSION bump and full retrain.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1.1 | `NUM_NODE_TYPES` dead code with misleading docstring | HIGH | Dead code / latent bug |
| 1.2 | `/ 12.0` magic number, not derived from schema | MEDIUM | Schema contract gap |
| 1.3 | 3 scripts duplicate hash_utils with semantic differences | HIGH | Data integrity |
| 1.4 | `FEATURE_SCHEMA_VERSION` not bumped after Phase 0 fixes | HIGH | Cache invalidation |
| 1.5 | Stale "13 in v5" doc comments in preprocess.py | LOW | Documentation |
| 1.6 | predictor.py hardcodes NODE_FEATURE_DIM values | MEDIUM | DRY violation |
| 1.7 | Dual-hash mismatch + process() missing schema suffix | HIGH | Cache correctness |
| 1.8 | 3 dead functions in hash_utils.py | LOW | Maintenance |
| 1.9 | Sentinel value -1.0 not runtime-asserted | MEDIUM | Implicit contract |
| 1.10 | MD5 collision surface for security tool | LOW | Theoretical risk |
| 1.11 | Slither version check has upper-bound gap | MEDIUM | Dependency contract |
| 1.12 | VISIBILITY_MAP collapses public/external | LOW | Feature engineering |

**4 HIGH, 4 MEDIUM, 4 LOW** — The four HIGH findings (1.3, 1.4, 1.7, and 1.1's latent risk) should be addressed before v5.1 training begins, as they affect data integrity and cache correctness.

---

Ready for **Group 2: Graph Extraction Engine** (`graph_extractor.py`, `ast_extractor.py`) whenever you want to continue.