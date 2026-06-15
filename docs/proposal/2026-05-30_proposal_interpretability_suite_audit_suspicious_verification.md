
## Interpretability Suite Audit — Suspicious Items & Verification Flags

**Date:** 2026-06-01  


---

### TIER 1 — Confirmed Bugs (Results Are Wrong)

**ITEM 1: EMITS extraction bug — 15.46× enrichment is real but completely unexploitable**

The root cause is confirmed in `graph_extractor.py`: event nodes are registered into `node_map` ~100 lines **after** EMITS edge creation runs. `_add_edge()` silently drops any edge whose destination isn't in `node_map` yet. All EMITS edges lost. The fix is a 2-line reorder: move event node registration before edge creation begins.

**Impact:** The 15.46× enrichment ratio for UnusedReturn, cited as a potential signal, is real but completely based on 1 surviving edge across 41K contracts. If the extractor were fixed, EMITS might be the cleanest structural signal for UnusedReturn — but every conclusion drawn from "EMITS enrichment" in S2 is currently meaningless as a training signal.

---

**ITEM 2: EXP-L3 "PASS" is a false positive — 100% CF is mathematically guaranteed**

The doc says "PASS — 11/11 contracts at 100% CF fraction in top-20 attention edges." The doc itself acknowledges:
> *"The 100% CF fraction result is structurally guaranteed by the architecture: conv3 receives only CF edges in its edge_index. This is not a learned behavior but a hardcoded architectural constraint."*

`conv3` physically cannot attend to non-CF edges because its `edge_index` only contains CONTROL_FLOW edges. You're asking "do 100% of the attention edges in a graph that only has CF edges come from CF edges?" and reporting the answer as a PASS. The real finding buried in the doc is more damning: **all GAT attention weights = 1.0 (uniform)** — the model hasn't learned any selective attention within the CFG at all. That should have been the headline, not the guaranteed 100%.

---

**ITEM 3: EXP-L4 results have wrong feature labels for dims 3–9**

L4 was run on 2026-05-30. The FEATURE_NAMES bug was identified and fixed on 2026-05-31. The index explicitly says: *"Rerun needed for corrected per-dim attribution."* But the per-class saliency table in the master report (external_call_count dominates all classes) was produced with the stale labels. The finding that `external_call_count` ranks #1 for all 10 classes may be correct in substance, but you cannot trust the feature names for ranks 2–11 or any feature beyond `type_id_norm` (dim 0) and `external_call_count` (whichever dim that is). **L4 was never actually rerun.**

---

**ITEM 4: EXP-L2 "structural ablation" numbers are not from actual measurements**

The script only implements embedding zeroing. Searching the script for `edge_index` removal code returns nothing. The `0.0048` and `0.014` numbers in the doc and master report are described as *"approximately"* and appear to be theoretical estimates, not measured results. The doc says:

> *"A proper structural ablation... produces approximately 450× larger effects: CF edge removal yields a drop of approximately 0.0048..."*

The word "approximately" twice and no corresponding code path = these numbers were estimated, not run. The master report presents them as experimental results. That's a misleading representation.

---

### TIER 2 — Methodological Issues (Results May Be Biased)

**ITEM 5: EXP-B1 gradient norms computed through raw logits, not loss**

The script runs `logits[0, class_idx].backward()` — backpropagating through a single raw logit, not through a loss function. During training, gradients flow through `BCEWithLogitsLoss` (which applies sigmoid). The raw logit gradient has a different magnitude and shape than the training-time gradient. The finding that Phase1 > Phase2 > Phase3 gradient norms may be correct in relative ordering, but the absolute numbers and ratios (e.g., "Phase 2 receives 71–87% of Phase 1 gradient") are not representative of what happens during actual training. This is a non-trivial distinction because the claim "no gradient starvation" is used to argue that Phase 2's failure to learn isn't a gradient flow issue — but that argument rests on gradients computed differently from training.

---

**ITEM 6: EXP-L4 saliency is identical across all 10 classes — probable methodology artifact**

Every single class shows `external_call_count` as rank 1 at ~21–24%, with nearly identical percentages. Reentrancy = 23.5%, GasException = 23.9%, IntegerUO = 21.5%, Timestamp = 21.4%. When 10 classes with completely different detection requirements all produce the same top feature at the same magnitude, the saliency isn't measuring class-discriminative signal — it's measuring which features generally push all logits up. This is a known failure mode of input gradient saliency (as opposed to integrated gradients or SHAP): it captures global feature sensitivity, not class-specific reasoning.

---

**ITEM 7: EXP-E4 direction sensitivity test may not isolate individual edge types**

All 4 edge types give exactly 89.1% distinguishability with exactly 0.0% difference. The same 92 pairs are used for all 4 tests. The 89.1% is the baseline WL distinguishability of those Reentrancy pairs using the full graph. If you make CONTROL_FLOW undirected but keep DEF_USE, CALL_ENTRY, RETURN_TO directed, the pairs that are already distinguishable through **other** features (node types, other edge types) will still be distinguishable. You'd only see a change if those pairs are exclusively distinguishable through the specific edge type's direction. The identical result across all 4 types suggests the test is measuring "overall graph distinguishability" not "contribution of this specific edge type's direction." The conclusion "direction adds zero discriminative power" is likely correct but for the wrong reason.

---

### TIER 3 — Sample Size Issues (Numbers Unreliable)

**ITEM 8: EXP-A4 F1 numbers for Timestamp, DoS, UnusedReturn are statistically invalid**

| Class | Positives in 470-sample eval | F1 interpretation |
|-------|---------------------------|-------------------|
| Timestamp | **4** | ±0.25 per misprediction |
| DenialOfService | **6** | ±0.17 per misprediction |
| UnusedReturn | **13** | ±0.08 per misprediction |

With 4 Timestamp positives, "Main F1 = 0.571" means the model got some fraction of 4 contracts right. One different contract changes F1 by 0.25. These numbers have no statistical validity and should not be used to draw conclusions about class-level GNN usefulness. The full 6,236-sample val split should be used for these evaluations.

---

**ITEM 9: EXP-L6 counterfactual verdict rests on 4 hand-crafted contracts**

The strongest conclusion in the report — "the model cannot detect CEI violation, unchecked overflow, or timestamp branching" — comes from 4 minimalist synthetic contracts. These contracts are small (3–16 nodes per the L3 doc), while production contracts in the training set have mean ~124–344 nodes depending on class. The model may be responding to the extreme size mismatch between these synthetic contracts and what it was trained on, rather than failing to detect the semantic vulnerability. Three of 4 pairs FAILing could be a distribution shift artifact.

---

**ITEM 10: EXP-L9 only 3 test contracts for attention rollout**

The attention rollout verdict ("safe CW > vulnerable CW, FAIL") is based on 3 test contracts. With n=3 and mean attribution scores differing by delta=−0.00654, this is close to noise. The retracted original PASS was also based on a criterion satisfied by both contracts — the replacement criterion is better but the sample size makes any quantitative claim fragile.

---

### TIER 4 — Inconsistencies & Status Discrepancies

**ITEM 11: EXP-L3 and EXP-L4 status conflicts between docs, index, and master report**

| Experiment | Individual Doc | EXPERIMENT_INDEX | MASTER_REPORT |
|-----------|---------------|-----------------|---------------|
| L3 | PASS (run 2026-05-30) | PENDING — requires checkpoint | PENDING |
| L4 | COMPLETE (run 2026-05-30) | COMPLETE but "rerun needed" | PENDING |

L3 was actually run and the results are in the doc. L4 was run but needs a rerun after the FEATURE_NAMES fix. Neither of these matches "PENDING" in the master report. You can't trust the master report's status table as a source of truth for which experiments have valid results.

---

**ITEM 12: EXP-S3 Cohen's d for Timestamp — three different numbers in three places**

| Source | Number | For what |
|--------|--------|---------|
| EXPERIMENT_INDEX | d = 1.592 | cfg_call_count, presumably train |
| MASTER_REPORT | d = 0.643 | val split |
| VALIDATION_SUMMARY | "original 1.657 was wrong, val is 0.643" | val split |

1.657, 1.592, and 0.643 are all cited in different places. The train/val split distinction is real (train has 2.34× size ratio), but the inconsistency across documents means you can't read any single doc and trust the cited number without checking which split it refers to.

---

**ITEM 13: EXP-E1 KeyError crash at line 521**

The script has a key mismatch: `results["analysis2_function_aggregation"]` but the key stored is `"analysis2_function_cfg_coverage"`. This crashes the JSON output section. The reported 85.5% CONTAINS coverage result from the redesigned A2 analysis may have been obtained from an intermediate print or an earlier version of the script — the current script cannot produce its output JSON without crashing.

---

### Summary Table

| # | Item | Severity | What to do |
|---|------|---------|-----------|
| 1 | EMITS extractor bug — event nodes registered after edge creation | **Critical bug** | Fix node registration order in graph_extractor.py |
| 2 | EXP-L3 PASS is architecturally guaranteed, not learned | **False positive** | Reclassify as N/A; true finding is uniform attention |
| 3 | EXP-L4 per-feature labels wrong (pre-fix labels), never rerun | **Wrong results** | Rerun after FEATURE_NAMES fix before trusting feature rankings |
| 4 | EXP-L2 "structural ablation" numbers not from actual code | **Unverified claim** | Either implement structural ablation or remove the 0.014 number |
| 5 | EXP-B1 gradient through raw logit ≠ training-time gradient | **Method bias** | Rerun with BCEWithLogitsLoss for valid training-comparable norms |
| 6 | EXP-L4 all classes show identical top feature — not class-discriminative | **Method artifact** | Flag as global sensitivity, not class-specific; use integrated gradients |
| 7 | EXP-E4 doesn't isolate individual edge type direction contribution | **Method issue** | Conclusions valid directionally but reasoning may be wrong |
| 8 | EXP-A4 F1 meaningless for Timestamp (n=4), DoS (n=6) | **Small sample** | Rerun on full val split |
| 9 | EXP-L6 counterfactual on 4 synthetic tiny contracts | **Small + distribution shift** | Flag as directional evidence only |
| 10 | EXP-L9 rollout on 3 contracts | **Small sample** | Flag as directional evidence only |
| 11 | L3/L4 status conflicts across documents | **Documentation** | Fix master report status table |
| 12 | Timestamp Cohen's d — three numbers cited | **Documentation** | Standardise to one number with split clearly labelled |
| 13 | EXP-E1 KeyError crash at line 521 | **Code bug** | Fix key name before next run |

---

The conclusions that survive this audit intact: Phase 2 is clearly underutilised (confirmed by L1, L2, A3, A4 convergently), the Timestamp size shortcut is real (confirmed by L7 even if the magnitude is overstated elsewhere), and temperature calibration was correctly diagnosed (B2 clean). The critical items before Run 5 conclusions are solidified: fixing EMITS extraction, rerunning L4 with correct feature names, and actually implementing the structural ablation in L2.

