# ADR-0001: Schema versioning with `FEATURE_SCHEMA_VERSION`

**Status:** Accepted
**Date:** 2026-06-06
**Deciders:** Ali Motafegh, Claude (assistant)
**Supersedes:** None
**Superseded by:** None (current)

## Context

SENTINEL's ML pipeline produces several long-lived artifacts that must stay in sync with
the graph schema:

- **Graph `.pt` files** (one per contract, 41,576 in current corpus) — encode `x` shape
  `[N, NODE_FEATURE_DIM]`, `edge_attr` values in `[0, NUM_EDGE_TYPES)`, and `x[:,0]`
  as `float(type_id) / _MAX_TYPE_ID`.
- **Token `.pt` files** (one per contract) — encoder-windowed, schema-independent.
- **Cache `cached_dataset_vN.pkl`** (2.5 GB) — joins graphs + tokens + labels into a
  single `list[Data]` for fast DataLoader startup.
- **Checkpoints** (`ml/checkpoints/GCB-P1-RunN-..._best.pt`) — store GNNEncoder weights
  whose input projection expects `in_channels=NODE_FEATURE_DIM` and an `nn.Embedding`
  sized to `NUM_EDGE_TYPES`.
- **Inference cache keys** (`{content_md5}_{FEATURE_SCHEMA_VERSION}`) — what
  `ContractPreprocessor` returns when called twice with the same source.

These artifacts have all changed shape **six times** during the project's life
(v4→v5→v6→v7→v8→v9). Each bump requires re-extraction (~45 min on 8 workers), cache
rebuild (~3 min), and — for schema-breaking changes — a fresh training run from scratch
(v8 checkpoints cannot be loaded into a v9 model).

The cost of **silent staleness** is high. In the pre-Run-9 audit, the `--relabel-timestamp`
fix was correctly applied to the labels but **never re-run on the cache**, so training
silently consumed v8-schema graphs for 22 epochs. Run 8 plateaued at test F1=0.2307
because the model never saw the fix's effect.

## Decision

We use a **single string constant** `FEATURE_SCHEMA_VERSION` (currently `"v9"`) defined
in `ml/src/preprocessing/graph_schema.py:160`. This string:

1. **Suffixes every inference cache key** as `{content_md5}_{FEATURE_SCHEMA_VERSION}`.
   Bumping the string invalidates all cached (graph, tokens) pairs at the key level —
   no Python code needs to know which keys are stale.
2. **Is referenced from every `ml/scripts/` cache builder** via
   `create_cache.py` and the inference cache loader. Both raise `RuntimeError` on
   version mismatch rather than silently returning stale data.
3. **Is documented at the top of `graph_schema.py`** with a one-line rule: "Bump this
   string whenever `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES`, or `FEATURE_NAMES`
   change, or whenever `_build_node_features()` logic changes in `graph_extractor.py`."

We also use **module-level asserts** to catch schema drift at import time:

- `graph_schema.py` asserts `len(FEATURE_NAMES) == NODE_FEATURE_DIM` and
  `len(EDGE_TYPES) == NUM_EDGE_TYPES`.
- `graph_extractor.py` asserts `_MAX_TYPE_ID == max(NODE_TYPES.values())`.
- `sentinel_model.py` asserts `_MAX_TYPE_ID == 13.0` with a versioned message.

These asserts turn silent shape mismatches (which would manifest deep inside a training
loop as a 5,000-iteration crash) into immediate `ImportError` at the first `import`.

## Consequences

### Positive

- **Cache invalidation is a one-line change.** A schema bump forces every cache
  consumer to rebuild without manual coordination.
- **Silent staleness is impossible** by construction. The `dual_path_dataset.py` cache
  loader raises `RuntimeError` on version mismatch, not a silent return.
- **The audit trail is implicit.** Every checkpoint, cache file, and pkl includes the
  schema version in its name. `ls ml/data/cached_dataset_*.pkl` shows the version at a
  glance.
- **Cross-team portability.** If a colleague clones the repo, the version mismatch
  between their environment and any artifact is immediately diagnosable.

### Negative

- **Bumping is expensive.** v8→v9 cost: ~45 min re-extraction + ~3 min cache rebuild +
  ~22 min/epoch for a new training run. We accept this because the alternative
  (silent staleness) is worse.
- **The version number is opaque.** `"v9"` does not encode what changed. We rely on
  `docs/changes/INDEX.md` and (forthcoming) ADRs to record what each bump contains.
- **Multiple definitions exist.** `FEATURE_SCHEMA_VERSION` in `graph_schema.py` is the
  source of truth, but `ARCHITECTURE/MODEL_VERSION` strings (e.g. `"v8.1"`, `"v9"`)
  in `train.py` and `sentinel_model.py` are partially redundant and can drift.

### Neutral

- A new constant `ARCH_VERSION` could be added later to separate data schema from
  model architecture. Not done yet — they're versioned in lockstep.

## Alternatives Considered

**1. No version string. Trust callers to invalidate caches manually.**

Rejected. This is what we did before v7. The `--relabel-timestamp` audit (Run 8) showed
the failure mode: a fix is applied, a label CSV is regenerated, but the cache is not
rebuilt because no one remembers to. The cost of the bug is measured in days of wasted
training. The cost of the version string is one constant.

**2. Hash the schema definition (e.g. SHA256 of the graph_schema.py file content).**

Rejected. A hash changes every time someone adds a comment. A semantic version like
`"v9"` changes only when behavior changes, which matches what we actually want to
invalidate. The hash approach also makes cache keys unreadable.

**3. Store the schema version inside the cache file itself (e.g. as a pickle metadata
attribute).**

Partially adopted. Inference cache keys include the version. But the *graph* and
*token* `.pt` files on disk do not — they would require a new field and re-extraction.
The string-in-key approach is simpler and equally safe as long as every artifact
naming path includes the version.

**4. Use Python's PEP 440 semantic versioning (e.g. `"8.1.0"`).**

Rejected. We never bump in patch versions. The integer-bump style matches our cadence
(one major schema per ~2 runs) and the changelog convention.

## References

- Source: `ml/src/preprocessing/graph_schema.py:160-168` (constant + docstring)
- Source: `ml/src/datasets/dual_path_dataset.py:221-229` (cache version check)
- Source: `ml/src/inference/preprocess.py` (cache key construction)
- Audit: `docs/pre-run9-fixes/00-overview.md` (Finding A: `--relabel-timestamp` never
  applied to v10)
- Audit: `docs/pre-run9-fixes/PIPELINE.md` (fresh-build execution path)
- Changelog: `docs/changes/2026-05-19-v8-graph-extension-and-full-reextraction.md`
  (v8 schema introduction)
- Related: ADR-0002 (multi-label formulation, interacts with class dimension)
- Related: ADR-0003 (Four-Eye architecture, must match schema dimensions)
