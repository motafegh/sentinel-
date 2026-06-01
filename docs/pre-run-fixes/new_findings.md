CRITICAL / HIGH Severity
1. EMITS Edge Key Mismatch (graph_extractor.py)
The EventCall IR fallback path (BUG-H7 fix for old Solidity) uses a short name like "Transfer" but node_map stores canonical names like "ERC20.Transfer". All EMITS edges are silently dropped whenever the fallback path fires. This completely defeats the BUG-H7 fix for Solidity <0.4.21 contracts.

2. _add_node Reverse-Normalization Hardcodes 12 (graph_extractor.py line 1094)
actual_type_id = int(round(x_list[-1][0] * 12)) uses hardcoded 12 instead of _MAX_TYPE_ID. If a new node type is added, metadata is silently corrupted — same class of bug as A3/A32 but on the decode side.

3. build_multilabel_index.py Falls Back to graph.y for Non-BCCC Contracts
Since A20 makes graph.y always 0, any non-BCCC contract (SolidiFI, SmartBugs, augmented) is automatically labeled safe. This is a downstream propagation of A20 that affects training data even when DualPathDataset uses the CSV workaround.

4. gnn_layers Default Mismatch (train.py vs TrainConfig)
train.py defaults --gnn-layers to 7, but TrainConfig defaults to 8. Running via CLI gives a different architecture than using TrainConfig directly. Run 4 used 8 layers — anyone re-running with train.py gets 7 by default.

5. aux_phase2_loss_weight Not Exposed as CLI Arg in train.py
Users cannot override the 0.10 default from the command line. This is the most important new hyperparameter for Run 5 and must be tunable.

MEDIUM Severity
6. Phase 2 Layer-Specific Edges Ignore phase2_edge_types Ablation (gnn_encoder.py)
Layers 3 and 4 build their edge subsets unconditionally from edge_attr, ignoring the phase2_edge_types config. This means ablation experiments (like excluding DEF_USE from Phase 2) silently include them in Layers 3/4 anyway.

7. _compute_external_call_count and _compute_uses_block_globals Return 0.0 on Failure (not sentinel -1.0)
When these fail, they return 0.0 ("no external calls" / "no block globals") instead of -1.0 ("unknown"). This creates false negatives for DoS, MishandledException, and Timestamp detection — the model assumes no vulnerability signal when the truth is unknown.

8. Empty Batch Guard Returns Incomplete Aux Dict (sentinel_model.py)
When an empty batch occurs, the aux dict is missing "phase2" and "jk_entropy" keys that the normal return path includes. This creates an inconsistency in the return contract.

9. global_max_pool Returns -inf for Ghost Graphs (sentinel_model.py)
The comment says it returns zero, but PyG's scatter max initializes with -inf. This only works because ReLU converts -inf→0. If the activation were ever changed to GELU/SiLU, ghost graphs would produce NaN.

10. BF16 Precision Loss in Prefix Tensor Storage (sentinel_model.py)
The prefix tensor is allocated in BF16, but the projection+type_embedding result is float32. Storing into BF16 truncates precision, which can zero out small but meaningful components early after warmup — exactly when the projection is most fragile.

11. AdamW(fused=True) Crashes on CPU (trainer.py)
If running on a CPU-only machine, fused=True raises RuntimeError. Should be fused=(device == "cuda").

12. Predictive Inference: Window Truncation and Random Edge Embeddings (predictor.py)

Preprocessor can produce up to 8 windows but predictor silently truncates to 4 with no warning
Old checkpoints with fewer edge types get randomly initialized embeddings for new edge types with no warning
13. Duplicate Function Handling Attaches CFG to Wrong Parent (graph_extractor.py)
When _add_node returns None (duplicate key), CFG nodes of the second function get attached as children of the first function with the same name, conflating their CFG structures.

14. _add_edge Silently Drops ALL Edge Types (graph_extractor.py)
Like A13 but for all edge types (CALLS, READS, WRITES, EMITS, INHERITS), not just CONTROL_FLOW. No counter, no logging. Unknown aggregate data loss across the training set.

15. Checkpoint JSON Not Crash-Safe (ast_extractor.py)
Written directly to target file — if killed during write, partial JSON corrupts the checkpoint and forces a --force restart.

LOW Severity (But Worth Tracking)
Dead _raw variable computed but never used in trainer Phase 2 loss
select_prefix_nodes API lacks raw features parameter (needed for A34 fix)
Dead graph_has_func variable in sentinel_model.py
parameter_summary() missing aux_phase2 module
Label CSV read twice in trainer
Stale docstrings (graph_extractor: "12 dimensions", "8 edge types")
_BLOCK_GLOBALS set recreated on every call instead of module constant
weights_only inconsistency across files (dataset=True, label_cleaner=False, predictor=False)
Hardcoded feature dimension indices in label_cleaner.py (same fragility as A11)
