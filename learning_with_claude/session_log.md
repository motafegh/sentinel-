# Session Log — Learning With Claude

Chronological record of all teaching sessions. Never delete — only append.
Each entry: what was taught, what was answered, gaps closed, audit flags raised.

---

## Session 1 — Phase 1: `graph_schema.py` + `hash_utils.py`

**Files covered:**
- `ml/src/preprocessing/graph_schema.py`
- `ml/src/utils/hash_utils.py`

**Concepts taught:**
- Single source of truth pattern — why feature drift between training and inference is a killer bug
- `FEATURE_SCHEMA_VERSION` — how it invalidates inference caches on schema change
- Node feature vector (11 dims, v7 schema) — each feature's purpose and normalization rationale
- Log normalization (`log1p`) — why raw values dominate dot products and how log fixes it
- Sentinel values (`-1.0`) — encoding "unknown" vs "safe" vs "unsafe" in a float
- Node types (13) — declaration-level vs CFG-level, priority ordering
- `NodeType` IntEnum — type safety over raw ints
- `STRUCTURAL_PREFIX_TYPES` — why declaration-level only for GNN prefix injection
- Edge types (11) — 3-phase routing, REVERSE_CONTAINS runtime-only invariant
- Compile-time assertions — fail-fast at import vs silent shape mismatch in training loop
- MD5 path hashing — deterministic file pairing, why path not content for training data
- Content hashing — inference cache key design (`content_hash + schema_version`)
- `validate_hash` — int vs regex, permissiveness tradeoff

**Challenge questions answered:** Q1–Q6

**Gaps found and closed:**
- Why hardcoding schema constants is safer than dynamic computation (train/inference mismatch)
- `nn.Embedding` shape contract: `[E,1]` produces `[E,1,d]` (3D) → shape mismatch in GATConv
- REVERSE_CONTAINS serialization: duplication (not conflict) → 2× edges → silent graph corruption
- Path hashing staleness + content hashing write-collision problem
- Graph-level vs node-level features: why GNN needs local signals for structural pattern detection
- `int(hex, 16)` accepts uppercase; `hexdigest()` always lowercase → validator is over-permissive

**Audit flags raised:** A1, A2

---

## Session 2 — Phase 2: `graph_extractor.py` (Chunk 1)

**File:** `ml/src/preprocessing/graph_extractor.py` (lines 1–213)

**Concepts taught:**
- Module purpose: canonical Solidity → PyG graph converter, shared by batch + inference
- Exception hierarchy: `GraphExtractionError` → `SolcCompilationError` / `SlitherParseError` /
  `EmptyGraphError` — typed exceptions map to HTTP 400 vs 500
- Why typed exceptions > `None` returns for dual-use (batch + API) code
- `GraphExtractionConfig` dataclass — all fields and their roles
- `multi_contract_policy` — 3 heuristics, accuracy comparison, "most_derived" rationale
- `_MAX_TYPE_ID` dynamic computation — contradiction with `graph_schema.py` hardcoded `/12.0`
- PyG COO format — why sparse over dense adjacency matrix
- `node_metadata` index-alignment invariant — what breaks if violated
- `include_edge_attr=False` — PyG `Data.__getattr__` hard crash vs silent degradation tradeoff

**Preferences added mid-session:** P4, P5, P6, P7, P8

**Challenge questions answered:** Q1–Q5

**Gaps found and closed:**
- Concrete example of `EmptyGraphError` (imports-only Solidity file)
- "Compatibility" = existing configs reference the policy, not just documentation
- Exact numerical shift when `_MAX_TYPE_ID` changes: type collision at 1.0 + all types shift
- `python -O` disables `assert` → production invariant checks must use `if/raise`
- PyG `Data` raises `AttributeError` for missing attrs (not `None`) → hard crash is good

**Audit flags raised:** A3, A4

---

## Session 3 — Phase 2: `graph_extractor.py` (Chunk 2) + preferences P11–P14

**File:** `ml/src/preprocessing/graph_extractor.py` (lines 218–432)

**New preferences added:** P11 (Solidity domain knowledge inline), P12 (expand abbreviations),
P13 (specify learning mode per code block), P14 (explain mechanism of complex code)

**Concepts taught:**
- 6 feature computation helpers and which feature index each produces
- `_compute_return_ignored`: IMP-D1 fix — why global-set approach had false negatives,
  why sequential CFG-order scan fixes it, sentinel -1.0 semantics
- `_compute_call_target_typed`: two-pass IR + regex fallback, closed-world assumption guard,
  negative lookahead regex for address(this) exclusion
- `_compute_in_unchecked`: dead code, deprecated, should be deleted
- `_compute_has_loop`: Slither NodeType loop markers, `is True` vs bool() issue
- `_compute_external_call_count`: why Transfer/Send counted separately, log1p normalization
- `_compute_uses_block_globals`: why feature exists (no READS edge for block globals),
  string-based vs isinstance type check tradeoff
- Broad `except Exception` anti-pattern across all helpers
- Alternative approaches: symbolic execution vs IR traversal for return-value tracking

**Warm-up recall:** Q1 slightly off (compatibility = configs not file versioning) — gap closed
**Learning questions answered:** high/low level calls explained, sequential scan disambiguation
**Challenge questions answered:** Q1–Q5 with gaps closed on all

**Audit flags raised:** A5, A6, A7, A8, A9

---

## Session 4 — Phase 2: `graph_extractor.py` (Chunk 3)

**File:** `ml/src/preprocessing/graph_extractor.py` (lines 435–643)

**Concepts taught:**
- `_cfg_node_type()` — 5-priority CFG node classifier: CALL > WRITE > READ > CHECK > OTHER
- Priority ordering rationale — most vulnerability-relevant op wins when Slither merges IR nodes
- `_build_cfg_node_features()` — 11-dim feature vector for CFG (statement-level) nodes
- BUG-C3 fix — CFG nodes inherit visibility, view, payable, complexity, has_loop from parent FUNCTION
- Why in_unchecked is never inherited (function-level flag would cause false positives on all child nodes)
- `_build_control_flow_edges()` — 2-pass edge construction for CFG
- Pass 1: index assignment + feature vector + metadata (safe to build edges in pass 2)
- Pass 2: CONTROL_FLOW edge wiring using indices from pass 1
- Why 2-pass is necessary — backward CFG edges (loop end → loop condition) need the target's index
- `len(x_list)` as global index — why `len(node_index_map)` would give wrong indices
- Deterministic node ordering — sort by (source_line, node_id) for cross-run stability
- Slither `node_id` stability risk — different Slither versions may assign different node_id values

**Warm-up recall (from Chunk 2):**
- Q1: _compute_return_ignored — sequential CFG scan (correct)
- Q2: sentinel -1.0 meaning — "feature could not be determined" (correct)
- Q3: _compute_call_target_typed — two-pass + regex fallback (correct)
- Q4: _compute_has_loop Slither markers (partial: NodeType.STARTLOOP/ENDLOOP loop markers named but is_loop_present fallback missed)
- Q5: log1p normalization for external_call_count (correct)

**Challenge questions answered:**
- Q1: DEF_USE edge not catching writes through reference variables ✓ (user got this exactly right)
- Q2: return_ignored as function-level not per-statement — user said "misleading" without the mechanism
  - Gap closed: function-level sufficient because FUNCTION node carries signal + CONTAINS edges propagate it to CFG children; per-statement would need IMP-D1 scan per call node, requiring forward-reachability per-call
- Q3: why 2-pass needed — user had no idea
  - Gap closed: backward CFG edges from loop-end→loop-condition require loop-condition's index to already exist; 1-pass would hit unknown target index; 2-pass indexes all nodes first, builds edges second
- Q4: parent_features `len(p) > 9` — user had no idea
  - Gap closed: if parent_features has < 10 elements, has_loop silently falls back to 0.0 for all CFG children; GNN Phase 2 loses loop detection signal; DoS loop patterns degrade with no error
- Q5: deterministic ordering guarantee across Slither versions — user had correct intuition (sorting) but did not identify the node_id stability risk
  - Gap closed: node_id values can differ across Slither versions; training on v0.9.x and inference on v0.10.x can produce different node orderings → wrong edge wiring

**Audit flags raised:** A10, A11, A12, A13

---

## Session 5 — Phase 2: `graph_extractor.py` (Chunk 4)

**File:** `ml/src/preprocessing/graph_extractor.py` (lines 646–979)

**Concepts taught:**
- `_add_icfg_edges()`: ICFG-Lite — stitches per-function CFGs via CALL_ENTRY(8) + RETURN_TO(9)
- ICFG vs CFG — inter-procedural vs intra-procedural; why cross-function connectivity matters for reentrancy detection
- "Lite" scope: only internal calls; full ICFG alternative with call-return matching
- RETURN_TO cartesian product: all callee terminals × all call-site successors (including impossible revert→normal paths, A14)
- `_add_def_use_edges()`: DEF_USE(10) edges for LocalVariable only (not TemporaryVariable/StateVariable)
- SSA and TemporaryVariable — why SSA temporaries need no edges
- 2-pass DEF_USE: Pass 1 builds def_map, Pass 2 emits edges with seen_pairs deduplication
- def_map keyed by name not object identity — variable name collision risk (A15)
- Reaching definitions analysis alternative — over-approximation vs precision tradeoff
- `_build_node_features()`: 11-dim declaration-level feature vector assembly
- Duck-typing _is_function with hasattr instead of isinstance
- type_id override for constructor/fallback/receive — Solidity special function types
- assert for sentinel range (A16, same pattern as A4)
- `_select_contract()`: contract selection from multi-contract file
- is_from_dependency() filter, is_interface filter
- most_derived heuristic (≥92%) vs last (87.4%) vs most_funcs (52.6% — worse than random)
- Derivation score: (n_in_file_ancestors, source_order_index) — tiebreak rationale
- Fallback chain: by_name → most_derived → last

**Warm-up recall (from Chunk 3):** Questions posted; answers pending

**Challenge questions:** Posted below in teaching response

**Audit flags raised:** A14, A15, A16

---

## Session 6 — Phase 2: `graph_extractor.py` (Chunk 5) — COMPLETE

**File:** `ml/src/preprocessing/graph_extractor.py` (lines 981–1329)

**Concepts taught:**
- `_build_solc_args()`: --allow-paths flag, pre-0.5 solc version guard
- `extract_contract_graph()`: main public API — single canonical .sol → PyG converter
- Slither instantiation: detectors_to_run=[], solc_binary override, exception routing
- Exception routing via keyword string matching — fragility and fix (A17)
- Shared state design: x_list / node_metadata / node_map / edges / edge_types (all parallel)
- `_add_node` inner function: duplicate guard, type_id reverse-decode (round(x*12))
- Node insertion order: CONTRACT → parents → STATE_VARs → FUNCTIONs+CFG → MODIFIERs → EVENTs
- BUG-H8: parent CONTRACT nodes pre-added so INHERITS edges can resolve
- Per-function loop: _add_node → _build_control_flow_edges → accumulate ICFG maps
- Duplicate function handling (inherited functions in contract.functions)
- ICFG map accumulation: entry = NodeType.ENTRYPOINT, terminals = nodes with no sons
- except Exception: pass in ICFG map accumulation (A18)
- CFG failure rate monitoring: 5% threshold, dynamic log level selection
- MODIFIERs and EVENTs added last — CFG-free, spatial locality in x_list
- EmptyGraphError guard (zero x_list after all filtering)
- Feature tensor: torch.tensor(x_list) → [N, 11], dimension guard, OOR validation (BUG-L4)
- OOR = out-of-range: warn not raise (single bad contract must not abort batch run)
- node_metadata alignment assert (A4 pattern)
- Declaration-level edges: CALLS, READS, WRITES using Slither pre-computed summaries
- _add_edge silent skip for cross-contract/missing endpoints
- EMITS dual-path: events_emitted API (>=0.4.21) + EventCall IR scan fallback (BUG-H7)
- INHERITS using pre-added parent nodes
- PyG Data assembly: torch.tensor(edges).t().contiguous() — [E,2] → [2,E] COO
- .contiguous() necessity after .t() (non-contiguous view → crash in PyG C++ kernels)
- include_edge_attr flag: edge_attr attached only when True; missing attr = AttributeError (fail-fast)

**Phase 2 status:** graph_extractor.py COMPLETE (all 5 chunks)

**Warm-up recall (from Chunk 4):** Questions posted; answers pending

**Challenge questions:** Posted below

**Audit flags raised:** A17, A18

---

## Session 8 — Phase 5: `gnn_encoder.py` (Chunk 1)

**File:** `ml/src/models/gnn_encoder.py` (lines 1–337)

**Concepts taught:**
- GAT (Graph Attention Network): learned per-edge attention weights vs uniform aggregation
- Multi-head attention in GAT: `concat=True` (concatenate heads) vs `concat=False` (average/single head)
- `out_channels` in GATConv is per-head, not total — total = `out_channels × heads` when concat=True
- Over-smoothing in deep GNNs; JK connections as mitigation
- `_JKAttention`: `nn.Linear(channels, 1, bias=False)` as attention scorer; stack→score→softmax→weighted sum
- `register_buffer` vs plain attribute: device movement, state_dict serialization, no gradients
- JK entropy term: `-(w·log(w)).sum(dim=1).mean()` — measures attention collapse, gradient-attached
- `last_node_weights` as plain attribute (not buffer): shape varies per batch, eval-only diagnostic
- `GNNEncoder.__init__`: 8 conv layers across 3 phases (2+3+3), named conv1/2/3/3b/3c/4/4b/4c
- `_head_dim = hidden_dim // heads` — ensures total Phase 1 output = hidden_dim after concat
- `nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)` — edge type lookup table, incorporated into GATConv via edge_dim
- IMP-G2: `input_proj = nn.Linear(11, 256, bias=False)` skip connection — prevents raw feature loss at init
- Phase 1: `add_self_loops=True, heads=8, concat=True` — 8 parallel structural views
- Phase 2: `add_self_loops=False, heads=1, concat=False` — directional CFG/ICFG, no self-loops
- Phase 3: same as Phase 2 — bidirectional CONTAINS (up × 2, down × 1, IMP-G3)
- IMP-G1: three separate Phase 2 layers (CF-only, ICFG-only, joint) for distinct representations
- Per-phase LayerNorm: equalizes norms before JK scoring to prevent scale dominance
- `nn.ModuleList` vs plain Python list for sub-modules

**Warm-up recall (from Session 7):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A23, A24 (renumbered A27)

---

## Session 9 — Phase 5: `gnn_encoder.py` (Chunk 2)

**File:** `ml/src/models/gnn_encoder.py` (lines 338–582)

**Model answers delivered for:** Session 7 warm-up Q1–Q3, Session 8 challenge Q1–Q5

**Concepts taught:**
- Three input guards: feature dim (stale .pt), use_edge_attr+None (silent Phase 2 off), OOB edge_index (CUDA corruption)
- Dtype normalization: BF16 global dtype pollution from BERT loading, `next(self.parameters()).dtype` pattern
- Edge embedding OOB clamping: clamp-and-warn vs raise — batch robustness policy
- Edge mask computation: struct_mask (≤5), cfg_mask (6,8,9,10), contains_mask (==5)
- Why DEF_USE(10) goes into Phase 2 joint layer (conv3c) only, not cf_only or icfg_only
- `fwd_contains_ei.flip(0)` — reverses COO edge direction for Phase 3 upward pass
- Type-7 embeddings synthesized at runtime for reverse direction (not stored on disk)
- `_live` vs `_intermediates` duality: gradient-attached vs detach+clone diagnostic snapshots
- Phase 1 forward: IMP-G2 skip inside ReLU (unusual), Layer 2 standard post-activation residual
- Phase 2 forward: IMP-G1 three layers with CF-only / ICFG-only / joint subsets
- Phase 3 forward: two up-hops (CFG→FUNCTION→CONTRACT), one down-hop IMP-G3 (FUNCTION→CFG)
- Zero-message behavior: GATConv on empty [2,0] returns zero → residual preserves identity
- JK aggregation: `_live` list passed to `_JKAttention`, `_jk_entropy` scalar returned
- `return_intermediates` 4-tuple vs default 3-tuple return
- `x.new_zeros(1).squeeze()` scalar zero for use_jk=False path

**Warm-up recall (from Session 8):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A25, A26, A27 (renaming A24→A27 to maintain append order)

---

## Session 10 — Phase 5: `transformer_encoder.py` (full file)

**File:** `ml/src/models/transformer_encoder.py` (lines 1–350)

**Concepts taught:**
- LoRA (Low-Rank Adaptation): A[768,r] + B[r,768] matrices injected into frozen Q+V projections;
  rank-16 update ≈ 590K trainable params vs 125M frozen; why full fine-tune and frozen both fail
- Module-level hard peft requirement check: raise at import vs warn-then-fallback rationale
- Flash Attention 2 vs SDPA: full attention matrix not materialized, memory savings for 512-token batches
- Global dtype pollution: `from_pretrained(torch_dtype=bfloat16)` calls `set_default_dtype` as side effect;
  save/restore pattern in try/finally
- `get_peft_model()` three operations: freeze backbone, inject A/B matrices, wrap model
- Why no `torch.no_grad()` around `self.bert()`: would kill LoRA gradients
- MLflow string deserialization guard for `lora_target_modules`
- Standard single-window path: `outputs.last_hidden_state` [B, L, 768]
- Multi-window path: `[B, W, L]` → flatten → BERT → `[B, W*L, 768]`
- `inputs_embeds` vs `input_ids`: bypasses embedding lookup, enables continuous GNN vector injection
- Prefix injection: `code_budget = L - K`, truncates last K tokens silently to make room
- IMP-M3 prefix count mask: zero-padded prefix positions masked as PAD in attention
- Position IDs: prefix at pos=1 (RoBERTa padding slot — no positional bias), code at pos 3+
- `output_attentions=True` diagnostic: slice `attn[:, :, :, K:, :K].mean()` for prefix attention monitoring
- Multi-window + prefix: same prefix shared across all W windows; prefix expanded via unsqueeze+expand
- `WindowAttentionPooler`: CLS at `i*window_size + prefix_k`; learned attention over W window-CLS tokens
- `_word_embeddings` property: accesses BERT word embedding table for code token conversion to embeddings

**Warm-up recall (from Session 9):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A28, A29, A30

---

## Session 7 — Phase 3: `ast_extractor.py` (single chunk)

**File:** `ml/src/data_extraction/ast_extractor.py` (lines 1–437)

**Concepts taught:**
- Pre-refactor train/inference feature drift problem — duplicate extraction logic in ast_extractor.py + preprocess.py
- Post-refactor thin wrapper design — graph logic delegated entirely to graph_extractor.py
- `parse_solc_version` — (major, minor, patch) tuple, never-raise contract, returns (0,0,0) on failure
- `solc_supports_allow_paths` — 0.5.0 version gate for --allow-paths flag
- `get_solc_binary` — solc-select binary resolution under .venv/.solc-select/artifacts/
- `contract_to_pyg` — three-tier error policy: RuntimeError re-raise, GraphExtractionError skip, Exception skip
- Offline metadata attachment: contract_hash, contract_path, graph.y label tensor
- `extract_batch_with_checkpoint` — checkpoint/resume idiom for interruptible batch pipelines
- `pool.imap` vs `pool.map` — streaming vs collect-all, memory implications at scale
- `functools.partial` as worker factory for per-version-group configuration
- Why group by detected_version — shared solc binary per group avoids per-contract binary resolution
- Checkpoint race condition: hash added to processed_hashes after torch.save; kill between them is safe (idempotent re-process)
- Schema compatibility note: edge_attr [E,1] vs [E] shapes in legacy vs new .pt files

**Warm-up recall (from Session 6/Chunk 5):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A19, A20, A21, A22

---

## Session 11 — Phase 5: `fusion_layer.py` (single chunk)

**File:** `ml/src/models/fusion_layer.py` (lines 1–277)

**Concepts taught:**
- Why concat+MLP was replaced: pooling before fusion collapses per-node/per-token detail; cross-attention needs full sequences
- `_scatter_to_dense`: replaces PyG `to_dense_batch` to eliminate `GuardOnDataDependentSymNode` torch.compile graph breaks; static `max_nodes=1024` constant; output `(out [B, max_nodes, d], mask [B, max_nodes])`
- `torch.compile` graph break mechanism: data-dependent shapes force eager mode fallback; static shapes avoid it entirely
- `CrossAttentionFusion.__init__`: `node_proj`, `token_proj`, `token_norm`, `node_to_token_attn`, `token_to_node_attn` MHA modules, concat+project output
- BUG-C2 fix: `token_norm = nn.LayerNorm(token_dim)` applied BEFORE `token_proj` — prevents CodeBERT L2 norm (~10-15) from dominating cross-attention dot products
- Fix #26: `need_weights=False` on both MHA calls — saves ~33 MB VRAM per forward pass by skipping attention weight matrix materialization
- MHA `key_padding_mask` convention: `True = IGNORE` (inverted from attention_mask); must invert with `~node_real_mask` and `(attention_mask == 0)`
- Node→Token cross-attention: Q=nodes, K=V=tokens; each node attends across full 512-token sequence
- Fix #8: explicit `enriched_nodes = enriched_nodes * node_real_mask.float().unsqueeze(-1)` — zeroes padded node positions after cross-attention, preventing PAD embeddings from leaking into masked mean pool
- Token→Node cross-attention: Q=tokens, K=V=nodes; each token attends across all real graph nodes
- Masked mean pooling: divide by `mask.sum(dim=1, keepdim=True).clamp(min=1.0)` — `clamp` prevents division by zero for ghost graphs (zero function nodes)
- Full 6-step forward pipeline: project → scatter-to-dense → Node→Token attn → Token→Node attn → masked pool both sides → cat+project → `[B, 128]`
- Cross-attention vs self-attention: Q and K/V come from different sequences; used here for bidirectional info flow

**Warm-up recall (from Session 10):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A31

---

## Session 12 — Phase 5: `sentinel_model.py` (Chunk 1)

**File:** `ml/src/models/sentinel_model.py` (lines 1–332)

**Concepts taught:**
- Three-eye architecture overview: GNN eye + Transformer eye + Fused eye → concat [B, 384] → classifier
- `_MAX_TYPE_ID`: derived from `max(NODE_TYPES.values())` at import — decodes feature[0] back to integer type_id; same schema-drift risk as A3
- `_FUNC_TYPE_IDS` frozenset: declaration-level nodes (FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR) pooling target; why not all nodes (CFG_RETURN dominance)
- `_FUNC_IDS_CPU`: pre-built CPU tensor at module level (BUG-L1 perf fix); compile-time assert for integrity
- `_PREFIX_NODE_PRIORITY` vs `_PREFIX_TYPE_IDX`: different dicts — priority controls selection order, type_idx controls embedding slot; must not confuse them
- `SentinelModel.__init__`: sub-module wiring (GNNEncoder, TransformerEncoder, CrossAttentionFusion)
- Conditional prefix modules: `gnn_to_bert_proj` and `prefix_type_embedding` only created when `gnn_prefix_k > 0`
- `_current_epoch` int attribute: set by trainer, controls prefix gating, hardcoded to 9999 at inference
- GNN eye projection: cat([global_max_pool, global_mean_pool]) [B, 512] → Linear(512,128) → [B, 128]
- Auxiliary heads: three Linear(128, num_classes), one per eye; training only, discarded at inference
- Aux loss rationale: λ=0.3 * (loss_gnn + loss_tf + loss_fused) keeps all three eye gradients alive
- `select_prefix_nodes`: step-by-step — mask per graph, eligible_local list, two-key sort (primary=type_priority, secondary=-ext_call_count for FUNCTION only), truncate to K
- `gnn_to_bert_proj(selected_embs)` + `prefix_type_embedding(type_indices)` bias addition → [B, K, 768]
- IMP-M3: `node_counts[g] = n_sel` tracks real (non-padded) nodes per graph; returned alongside prefix
- Ghost graph handling: `continue` on no eligible nodes → prefix stays all-zero; node_counts stays 0

**Warm-up recall (from Session 11):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A32, A33

---

## Session 13 — Phase 5: `sentinel_model.py` (Chunk 2)

**File:** `ml/src/models/sentinel_model.py` (lines 334–561)

**Model answers delivered for:** Session 12 challenge Q1–Q5

**Concepts taught:**
- `forward()` step-by-step: flat_mask construction, GNN path, BF16-safe type decode (.float()+.round()+.long()), func_mask via torch.isin
- Empty batch guard: zero logits returned when batch.numel()==0, correct return type for (logits, aux) unpacking
- Ghost graph fix (BUG-H2): `pool_mask = func_mask`; `global_max/mean_pool` returns zero row for graphs with no function nodes; old fallback (STATE_VAR pooling) was actively harmful for 9% training samples
- `size=num_graphs` argument to global_max/mean_pool: forces correct [B, d] output shape when last graph is a ghost
- GNN prefix gating: both `gnn_prefix_k > 0` AND `_current_epoch >= warmup` required; during warmup prefix is None, gnn_to_bert_proj gets zero gradient
- Transformer path: token_embs [B, W*L, 768]; prefix=None during warmup triggers standard (non-prefix) TransformerEncoder path
- Fused eye: `self.fusion(node_embs, batch, token_embs, flat_mask)` — flat_mask is the W*L-length mask from Step 1
- `size=num_graphs` critical: without it, global_max_pool omits ghost graphs at the end of a batch
- Three-eye concat → classifier → `num_classes==1` squeeze for binary
- `return_aux` pattern: aux dict contains gnn, transformer, fused logits + `jk_entropy` scalar piggybacked in
- `compute_prefix_attention_mean`: diagnostic @torch.no_grad() forward; `isinstance(gnn_prefix, tuple)` check as backward-compat guard; IMP-M3 bypass (A25 confirmed)
- `parameter_summary()`: trainable vs frozen per sub-module, conditional on gnn_prefix_k > 0
- Phase 5 full architecture data flow recap (graphs → GNN → pool/prefix → Transformer → Fusion → three eyes → classifier)

**Warm-up recall (from Session 12):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A25 (confirmed from roadmap), A34 (select_prefix_nodes sorts on post-GAT embedding dim, not raw ext_call_count)

---

## Session 14 — Phase 6: `focalloss.py` + `losses.py`

**Files:**
- `ml/src/training/focalloss.py` (lines 1–143)
- `ml/src/training/losses.py` (lines 1–126)

**Concepts taught:**
- Multi-label imbalance problem: 44K×10 = 440K cells, 85%+ negative; why standard BCE fails
- Positive class frequency in Sentinel: Reentrancy ~8%, DoS ~0.6%, Timestamp ~3.4%
- `FocalLoss`: pt = model confidence on correct class (p for y=1, 1-p for y=0); focal weight (1-pt)^gamma; alpha for pos/neg balance; BF16 guard (.float() cast); post-sigmoid contract
- `gamma` effect: gamma=2 crushes pt=0.95 to 0.0025×, keeps pt=0.05 at 0.90×
- `_FocalFromLogits` wrapper: inline class inside train() for sigmoid→focal chaining; unpicklable (A35)
- `MultiLabelFocalLoss`: per-class List[float] alpha; raw logits input; sigmoid applied internally; register_buffer for alpha
- alpha_t bug fix: old code applied same alpha_c to both pos AND neg; for rare class alpha_c=0.9, negatives were over-weighted (0.9 instead of 0.1); fixed by `torch.where(targets==1, alpha, 1-alpha)`
- `AsymmetricLoss` (Ridnik et al. ICCV 2021): gamma_neg > gamma_pos asymmetry; probability clip mechanism: `prob_neg = (prob - clip).clamp(0)` → zero gradient for confident negatives
- ASL step-by-step: prob → prob_neg (shifted) → log_pos (full prob) / log_neg (shifted prob) → focal_pos + focal_neg → per-cell loss
- Production settings: gamma_neg=2.0 (not 4.0 — BUG-C4 all-zeros collapse), clip=0.01 (not 0.05 — BUG-M2 oscillation)
- Why pos_weight NOT passed to ASL: double-amplification (DoS pos_weight × ASL gamma_neg) collapsed GNN gradient share to 24% in Run 3
- `register_buffer` vs plain attribute: automatic device placement + state_dict inclusion
- BCE+pos_weight vs ASL comparison: amplify positives vs silence negatives

**Warm-up recall (from Session 13):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A35

---

## Session 15 — Phase 6: `trainer.py` Chunk 1

**File:** `ml/src/training/trainer.py` (lines 1–492)

**Model answers delivered for:** Session 14 challenge Q1–Q5

**Concepts taught:**
- `CLASS_NAMES` order sensitivity: index 1 = DoS, used by BUG-H6 gradient scaling; CLASS_NAMES reorder = silent label mis-assignment
- 10 vulnerability class descriptions: Reentrancy, IntegerUO, DoS, Timestamp, TOD, GasException, MishandledException, CallToUnknown, UnusedReturn, ExternalBug
- VRAM helpers: `memory_reserved()` vs `memory_allocated()` — reserved includes caching allocator free pool; `empty_cache()` operates on the gap
- `_parse_version()`: lstrip + split + int tuple; tuple comparison for version ordering
- `TrainConfig` dataclass: `@dataclass`, field defaults, `field(default_factory=lambda: ...)` for mutable defaults
- `eval_threshold=0.35` vs `threshold=0.5`: minority class probability clustering near boundary; patience noise at 0.5 caused premature stopping at ep30
- Per-class label smoothing: `label*(1-eps) + 0.5*eps`; symmetric around 0.5; DenialOfService eps=0.18 (Slither detection rule drift); Reentrancy eps=0.14 (confirmed 14% noise)
- `gradient_accumulation_steps=8`: batch=8 × 8 = effective 64; why 8 GB GPU limit forces small micro-batch
- `asl_gamma_neg=2.0` (not 4.0): BUG-C4 all-zeros collapse; `asl_clip=0.01` (not 0.05): BUG-M2 boundary oscillation
- `dos_loss_weight=0.5`: DoS had 3 samples historically; fractional gradient blend prevents zero-signal
- `__post_init__`: gnn_layers hard error vs warning tiers; class_label_smoothing name validation (set difference)
- `compute_pos_weight`: sqrt-scaled ratio `sqrt((N-pos)/pos)`; pos_weight_min_samples cap (Reentrancy 3.4× → 1.0 prevents behavioral collapse v5.2); pos_weight_cap=10.0 ceiling
- `evaluate()`: `model.eval()` vs `torch.no_grad()` — different purposes, both required; `sigmoid(logits.float())` BF16 guard post-AMP
- Hamming loss definition: fraction of wrong label cells across N×C
- Macro F1 vs Micro F1: macro = per-class independent F1 averaged (rare classes count equally); micro = global TP/FP/FN (dominated by frequent classes)
- Threshold tuning (BUG-M8): 19 candidates per class, per-class optimal threshold; every epoch in main loop

**Warm-up recall (from Session 14):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A36, A37

---

## Session 16 — Phase 6: `trainer.py` Chunk 2

**File:** `ml/src/training/trainer.py` (lines 498–698: `train_one_epoch` + `_grad_norm`)

**Model answers delivered for:** Session 15 challenge Q1–Q5

**Concepts taught:**
- `train_one_epoch` signature: 10+ params including class_eps, dos_loss_weight, jk_entropy_reg_lambda
- `label_smoothing` backward-compat: kept alongside class_eps but overridden when class_eps is provided
- Per-class label smoothing application: `labels = labels*(1-class_eps) + 0.5*class_eps` — non-destructive broadcast
- AMP autocast block: `torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)`
- DoS gradient scaling (BUG-H6): `w*logit + (1-w)*logit.detach()` — partial detach scales gradient by w; predictions unaffected; same pattern applied to aux heads
- `.detach()` mechanism: shares storage, no grad_fn; multiply by scalar → gradient path sees w× of normal gradient
- `_actual_window` tail-window fix: `min(accum_steps, len(loader) - window_start)` — prevents underscaling last optimizer step
- Loss composition: `(main_loss + aux_loss_weight*aux_loss) / _actual_window + jk_reg/_actual_window`
- JK entropy regularizer (C-3): `lambda*(log(3) - H.clamp(max=log(3)))`; penalty=0 at max entropy, ≈0.0055 at collapse; lambda=0.005 (0.01 forced rigid 33/33/33)
- Fix #28 gradient norm ordering: clip → read norms → step → zero_grad; old code read after zero_grad (set_to_none=True) → all norms 0.000, GNN collapse detection broken
- `set_to_none=True` vs False: None frees memory entirely; zero fills in-place; None is faster (avoids read+zero pass, next backward allocates fresh)
- `_grad_norm()`: L2 norm of `.grad.detach().float()` — float() needed for BF16 training
- GNN collapse detection: streak counter, 3 consecutive log_interval periods < 10% share → warning; resets to 0 on any recovery
- `is_accum_step` condition: `(batch_idx+1) % accum_steps == 0 or is_last_batch`
- NaN loss handling: count nan batches, exclude from avg; backward still runs before check (A38)
- `_grad_norm()` returns 0.0 for modules with `p.grad is None` — important for set_to_none=True timing

**Warm-up recall (from Session 15):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** A38 (NaN backward runs before NaN check; clip_grad_norm_ doesn't catch NaN; Adam momentum corruption risk)

---

## Session 17 — Phase 6: `trainer.py` Chunk 3

**File:** `ml/src/training/trainer.py` (lines 744–1200: `train()` setup)

**Model answers delivered for:** Session 16 challenge Q1–Q5

**Concepts taught:**
- Environment setup inside train(): TRANSFORMERS_OFFLINE, TF32, cudnn.benchmark — why inside train() not module level
- TF32: Ampere GPU format, 10 mantissa bits, ~8× matmul throughput, negligible accuracy loss
- Shared cache pattern: single `_shared_cache` dict assigned by reference to both datasets → halves RAM (2.28 GB once vs twice); fork CoW: workers inherit but never write → no physical copy
- `multiprocessing_context="fork"` CUDA safety: safe only if workers never call CUDA; forked child inherits partial CUDA context → crash or corruption if CUDA touched
- DataLoader kwargs: `pin_memory` (locked RAM for DMA), `persistent_workers` (avoid respawn overhead), `prefetch_factor=4` (keep GPU fed)
- `sampler` vs `shuffle` mutual exclusion in PyTorch DataLoader
- WeightedRandomSampler: 3× weight for any-vuln rows (mode="positive"); shifts effective pos/neg from 40/60 to 60/40
- SentinelModel construction + C-1 dtype check: `next(model.gnn.conv1.parameters()).dtype` catches BF16 pollution regression post-construction
- Loss function selection: aux_loss always plain BCE (no pos_weight); pos_weight NOT passed to ASL (double-amplification caused GNN share collapse to 24% in Run 3)
- Per-group optimizer: 5 groups (GNN×2.5, LoRA×0.3, Fusion×0.5, PrefixProj×5.0, Other×1.0); `_seen_param_ids` prevents double-counting; non-empty group guards prevent `_max_lrs` misalignment
- LoRA `weight_decay=0.0`: L2 decay opposes adaptation signal on small-magnitude LoRA matrices
- `fused=True` AdamW: single CUDA kernel for all parameter updates; avoids per-parameter kernel launch overhead
- torch.compile submodule strategy: compile gnn/fusion/classifier/aux, skip transformer (HuggingFace control flow causes graph breaks); `dynamic=True` for variable N/E; cache_size_limit=256; suppress_errors=True
- OneCycleLR: ramp to max_lr over pct_start*total_steps, then cosine decay; `max_lr` list must match param_group order exactly
- Fix #32: `epochs=config.epochs` always (not remaining_epochs); checkpoint scheduler.total_steps must match for state restoration
- Checkpoint resume: `strict=False` + LoRA key split (warn) vs non-LoRA missing (raise); optimizer group count guard; scheduler total_steps guard

**Warm-up recall (from Session 16):** Questions posted; answers pending

**Challenge questions:** Q1–Q5 posted; answers pending

**Audit flags raised:** none (no new flags this session)
