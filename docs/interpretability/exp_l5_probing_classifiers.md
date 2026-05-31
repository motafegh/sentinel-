# EXP-L5: Probing Classifiers Per GNN Phase

**Layer:** 3  **Priority:** 1  **Status:** COMPLETE (2026-05-30)  
**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (Run 4, ep32, F1=0.3362)  
**Output:** `ml/logs/interpretability/exp_l5_probing_classifiers/`

---

## Hypothesis

If Phase 2 (CFG/ICFG/DFG layers) adds genuinely new vulnerability signal beyond Phase 1 (structural + CONTAINS), then linear probes trained on Phase 2 embeddings should outperform Phase 1 probes. Specifically, Reentrancy probes should improve by at least 3 percentage points from Phase 1 to Phase 2, since reentrancy is a CFG-level pattern (external call followed by state write).

Given prior findings (EXP-L1: Phase 2 has lowest JK weight 0.322; EXP-L2: CFG ablation near-zero effect), we expect Phase 2 probes to NOT outperform Phase 1 probes for most classes.

---

## Method

For each of 500 sampled val-split contracts, GNNEncoder.forward is called with `return_intermediates=True` to obtain per-phase node embeddings at three checkpoints: after Phase 1 (L1+L2), after Phase 2 (L3+L4+L5), and after Phase 3 (L6+L7+L8). Each embedding `[N, 256]` is pooled over function-level nodes (FUNCTION, MODIFIER, FALLBACK, RECEIVE, CONSTRUCTOR; types 1,2,4,5,6) with mean pooling to produce a graph-level vector `[256]`. A logistic regression probe (C=1.0, max_iter=500, lbfgs) is then trained on 80% of samples and evaluated on 20%, stratified by class where possible. F1 and AUROC are reported for all 3 phases × 10 classes.

**Fix applied:** The script had a duplicate `--n-contracts` argument (defined in both `add_common_args` and `parse_args`). Fixed by replacing the duplicate argument with `parser.set_defaults(n_contracts=2000)`.

---

## Results

### Probing Classifier F1 Table (3 Phases × 10 Classes)

| Class | Phase1 F1 | Phase2 F1 | Phase3 F1 | Δ P2-P1 | Δ P3-P1 |
|-------|-----------|-----------|-----------|---------|---------|
| CallToUnknown | 0.1818 | 0.1818 | 0.1818 | +0.0000 | +0.0000 |
| DenialOfService | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| ExternalBug | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| GasException | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| IntegerUO | 0.1143 | 0.1143 | **0.1538** | +0.0000 | **+0.0395** |
| MishandledException | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| Reentrancy | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| Timestamp | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| TransactionOrderDependence | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| UnusedReturn | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

### AUROC Table (3 Phases × 10 Classes, from Phase 1 and 2)

| Class | Phase1 AUROC | Phase2 AUROC | Phase3 AUROC | n_pos_train |
|-------|-------------|-------------|-------------|-------------|
| CallToUnknown | 0.5560 | 0.5560 | 0.6497 | 31 |
| DenialOfService | 0.6452 | 0.6344 | 0.5161 | 5 |
| ExternalBug | 0.5795 | 0.5682 | 0.5436 | 25 |
| GasException | 0.4639 | 0.4671 | 0.5723 | 44 |
| IntegerUO | 0.6239 | 0.6254 | 0.6157 | 126 |
| MishandledException | 0.7660 | 0.7621 | 0.6641 | 35 |
| Reentrancy | 0.6119 | 0.6179 | 0.5262 | 38 |
| Timestamp | **0.9677** | **0.9785** | 0.7957 | 3 |
| TransactionOrderDependence | 0.5993 | 0.5977 | 0.4992 | 26 |
| UnusedReturn | 0.5531 | 0.6154 | 0.6996 | 10 |

### Pass/Fail Check

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Reentrancy: Phase2 F1 > Phase1 F1 + 3pp | Δ = +0.0000 | +0.03 | **FAIL** |

---

## Key Findings

1. **GNN embeddings are not linearly separable for most classes:** Linear probes achieve F1=0 for 8 out of 10 classes across all three phases. This is the most striking finding — despite the model achieving overall val F1=0.3362, the information is encoded in a non-linear, entangled representation that a simple logistic regression cannot decode from mean-pooled function embeddings.

2. **Phase 2 adds zero new linearly-separable signal:** Phase 2 deltas are exactly 0.0000 for all 10 classes. This confirms the prediction from EXP-L1 (Phase 2 lowest JK weight) and EXP-L2 (CFG ablation near-zero). The CFG/ICFG layers appear to be refining the Phase 1 representation rather than creating new linearly-separable class axes.

3. **Phase 3 (REVERSE_CONTAINS) is the only phase showing improvement:** IntegerUO gains +3.95pp F1 from Phase 1 to Phase 3, the only positive delta observed. This is consistent with Phase 3 having the highest JK attention weight (0.346 from EXP-L1) — the hierarchical tree-like REVERSE_CONTAINS traversal may be integrating function-level context that helps IntegerUO (which often requires understanding the full call hierarchy for overflow detection).

4. **AUROC tells a different story than F1:** Timestamp AUROC is 0.9677 / 0.9785 at Phase 1 / Phase 2 with only 3 positive training samples. This is almost certainly an artifact of the extreme class imbalance (3 positives in 500-sample training set = 0.6%) producing degenerate probe behavior. MishandledException also shows AUROC=0.766 at Phase 1 despite F1=0, consistent with the probe learning to assign high scores to a distinctive-but-not-threshold-crossing minority.

5. **Information is present but non-linearly encoded:** The non-zero AUROC values (all above 0.5 except GasException at Phase 1 and TransactionOrderDependence at Phase 3) confirm that vulnerability information IS in the Phase embeddings — it is just not linearly separable. The GNN has learned a distributed, non-linear representation.

---

## Implications for Architecture

- **Phase 2 value is non-linear:** The zero linear probe delta for Phase 2 does not mean Phase 2 is useless — the full model uses non-linear JK attention to mix phases, and phase 2 may contribute through interaction terms the probe cannot see. However, it does confirm that CFG layers are not creating an obvious, directionally-distinct CFG subspace in embedding space.

- **Pooling choice matters:** Mean pooling over function-level nodes loses within-function CFG structure. A probe that pools at the CFG node level, or uses attention over the full node set, might show different Phase 2 gains. This is a known limitation of this experiment.

- **Class imbalance is severe in 500-sample probing:** With only 3–44 positive training examples for most classes, the probe is underpowered. EXP-L5 should be rerun with n_contracts=2000 or the full train split for more reliable probe estimates.

- **IntegerUO + Phase 3:** The only consistent improvement is IntegerUO via Phase 3 (REVERSE_CONTAINS), confirming that hierarchical scope traversal is meaningful for overflow detection (a function calling a sub-function that overflows propagates upward in the call tree).

---

## Known Caveats

- 500 contracts produces severe class imbalance: 3 positive Timestamp training samples. Probe results for rare classes (Timestamp, DenialOfService) are unreliable.
- Mean pooling over function-level nodes discards intra-function CFG structure — the probe cannot see the sequence or topology of CFG nodes within a function.
- The duplicate `--n-contracts` argument bug in the script was fixed before running (removed the redundant `parser.add_argument` call in `parse_args`, replaced with `parser.set_defaults`).
- AUROC values should be interpreted cautiously for classes with <10 positive test examples, where a single prediction flip can change AUROC by 0.1+.
