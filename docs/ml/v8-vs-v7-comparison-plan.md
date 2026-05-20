# v8 vs v7 Model Comparison Plan
**Goal:** Determine *why* v8.0-AB (F1=0.2593) fails to outperform v7.0 (F1=0.2651) despite a richer graph schema, and decide what to fix before running further ablations.

**Trigger:** Execute after killing v8.0-AB (when patience=10/30 ~ep32).

---

## Hypotheses (ranked by prior probability)

| ID | Hypothesis | Evidence for | Evidence against |
|----|-----------|-------------|-----------------|
| **H1** | Phase 2 conv hops designed for CF(6) only; adding CE(8)/RT(9)/DU(10) to the same hops dilutes the learned CEI pattern | JK Phase2 up (+45%) but F1 down; fused eye not improving past v7 | Phase 2 weight rising suggests signal, not noise |
| **H2** | DEF_USE is intra-function only — cross-function reentrancy/external-bug patterns still invisible | Reentrancy F1 not improving despite new call-graph edges | CALL_ENTRY/RETURN_TO are cross-function, just not DEF_USE |
| **H3** | Label ceiling: val set noise/ambiguity limits both models equally around F1=0.26 | Both models plateau at same point; oscillation not collapse | v8 schema should push past v7 ceiling if labels are the only limit |
| **H4** | Probability calibration differs between v7 and v8; default 0.5 threshold suboptimal for v8 | Easy to test; common when architecture changes | v7 and v8 share same classifier head structure |
| **H5** | v8 helps some classes but hurts others; net effect is slightly negative | Different JK weights across training mean different feature emphasis | Would show in per-class F1 comparison |
| **H6** | Phase 2 edge type count (4 types vs 1) makes the 3-hop convolutions less focused; attention heads spread over more edge types | Phase 2 uses edge_emb(11,64) vs v7's edge_emb(8,64) — marginal change | Same GATConv structure, only embedding table size changed |

---

## Test matrix

| Test | Reveals | Script |
|------|---------|--------|
| Per-class F1/precision/recall at default threshold | H1, H5 — which classes v8 wins/loses | `compare_checkpoints.py --mode metrics` |
| Per-class F1 at optimized threshold | H4 — does calibration explain the gap? | `tune_threshold.py` × 2, then compare |
| Prediction overlap matrix | H3, H5 — where do they agree vs disagree? | `compare_checkpoints.py --mode overlap` |
| Probability distribution per class | H4 — are v8 probs systematically shifted? | `compare_checkpoints.py --mode probs` |
| JK weights from saved checkpoints | H1, H6 — confirm Phase 2 weight, not just inference | `compare_checkpoints.py --mode jk` |
| Hand-crafted contract behavioral test | H2 — can v8 detect cross-function reentrancy that v7 misses? | `manual_test.py` × 2 |
| Per-sample disagreement analysis | H1, H3 — what kind of contracts does each model uniquely get right? | `compare_checkpoints.py --mode errors` |
| AUC-ROC per class | H4 — threshold-independent ranking quality | `compare_checkpoints.py --mode metrics` |

---

## Execution plan

### Step 0 — Kill v8-AB and verify checkpoints

```bash
# Kill training (when patience reaches 10/30, around epoch 32)
kill $(pgrep -f "train.py.*sentinel-v8")

# Verify both checkpoints exist
ls -lh ml/checkpoints/v7.0_best.pt ml/checkpoints/v8.0-AB_best.pt
```

Expected: v7 ~258 MB, v8-AB similar size. Both files must exist before proceeding.

---

### Step 1 — Full metric comparison on val + test splits

**Write and run** `ml/scripts/compare_checkpoints.py` (see implementation spec below).

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compare_checkpoints.py \
    --ckpt-a ml/checkpoints/v7.0_best.pt \
    --ckpt-b ml/checkpoints/v8.0-AB_best.pt \
    --label-a "v7.0 (ep23)" \
    --label-b "v8.0-AB (ep22)" \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --split test \
    --out ml/logs/comparison_v7_v8AB.json
```

**Primary output:** per-class F1, precision, recall, AUC-ROC for both models on the test split.

**What to look for:**
- Which classes does v8 win? Which does it lose?
- Is the gap concentrated in specific classes (H5) or spread evenly?
- Does v8's AUC beat v7 for any class even when F1 doesn't? (H4 — threshold problem)

---

### Step 2 — Threshold optimisation for both models

```bash
# Tune v7 thresholds on val split
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/v7.0_best.pt \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped

# Tune v8-AB thresholds on val split
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/v8.0-AB_best.pt \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped
```

**Note:** Both use the v8 cache — v7 graphs are a strict subset of v8 (same contracts, just fewer edge types). The v7 checkpoint contains its own config; `load_model_from_checkpoint` will reconstruct the v7 architecture (edge_emb(8,64)) correctly.

**What to look for (H4):**
- If v8's tuned F1 matches or beats v7's tuned F1 → the default threshold was the problem, not the schema
- If v8's tuned F1 is still below v7 → calibration is not the issue

**Then re-run compare_checkpoints using tuned thresholds:**
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compare_checkpoints.py \
    --ckpt-a ml/checkpoints/v7.0_best.pt \
    --ckpt-b ml/checkpoints/v8.0-AB_best.pt \
    --thresholds-a ml/checkpoints/v7.0_best_thresholds.json \
    --thresholds-b ml/checkpoints/v8.0-AB_best_thresholds.json \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --split test \
    --out ml/logs/comparison_v7_v8AB_tuned.json
```

---

### Step 3 — Prediction overlap and disagreement analysis

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compare_checkpoints.py \
    --ckpt-a ml/checkpoints/v7.0_best.pt \
    --ckpt-b ml/checkpoints/v8.0-AB_best.pt \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --split test \
    --mode errors \
    --out ml/logs/comparison_v7_v8AB_errors.json
```

**What to look for:**
- **v7-only correct**: Contracts where v7 gets it right but v8 doesn't. What do they have in common? (H1 — did v8 break the CEI pattern?)
- **v8-only correct**: Contracts where v8 uniquely succeeds. Which classes? (H2 — are these cross-function call patterns?)
- **Both wrong**: The irreducible error — these are the label-noise ceiling (H3)
- **Both right**: Agreement zone — not informative for the gap

---

### Step 4 — JK weight and eye score extraction

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compare_checkpoints.py \
    --ckpt-a ml/checkpoints/v7.0_best.pt \
    --ckpt-b ml/checkpoints/v8.0-AB_best.pt \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --split test \
    --mode internals \
    --out ml/logs/comparison_v7_v8AB_internals.json
```

**Internals extracted:**
- JK `last_weights` buffer from each checkpoint (not inference-time — from the saved state dict)
- Per-batch eye loss breakdown (GNN/TF/fused) on the same 500 test samples
- Edge type activation rates in v8 (how often do CALL_ENTRY/RETURN_TO/DEF_USE fire per graph?)

**What to look for (H1, H2, H6):**
- Edge type activation: if CALL_ENTRY(8)/RETURN_TO(9) fire in <20% of graphs, the ICFG edges are sparse — the model can't learn from rare signals
- DEF_USE activation rate: sparse → H2 confirmed (intra-function limitation)
- JK weights in v7 saved checkpoint: should match the logged ep23 values (P1≈0.05, P2≈0.18, P3≈0.77)

---

### Step 5 — Behavioral test on hand-crafted contracts

```bash
# v7 behavioral test
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v7.0_best.pt \
    --contracts ml/scripts/test_contracts/

# v8-AB behavioral test
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v8.0-AB_best.pt \
    --contracts ml/scripts/test_contracts/
```

**What to look for (H2):**
- Reentrancy test contracts: does v8 detect reentrancy patterns that v7 misses?
- If v8 misses reentrancy that v7 catches → H1 confirmed (CEI pattern degraded)
- If v8 catches reentrancy that v7 misses → H2 partially refuted (ICFG edges help after all)

---

### Step 6 — Probability distribution comparison

Look at the raw probability distributions per class across the full test set.

**What to look for (H4):**
- If v8 has systematically lower probabilities for true positives → tuning the threshold lower would fix it
- If v8's positive-class probability distributions overlap heavily with negative-class → v8 has worse discrimination (AUC problem, not threshold problem)

This is surfaced in the `--mode metrics` output (AUC-ROC per class).

---

## compare_checkpoints.py implementation spec

The script should be self-contained and reuse existing infrastructure from `tune_threshold.py`:

```
ml/scripts/compare_checkpoints.py
    args:
        --ckpt-a, --ckpt-b          — two checkpoint paths
        --label-a, --label-b        — display names (default: checkpoint stem)
        --thresholds-a, --thresholds-b  — optional per-class threshold JSON files
        --cache                     — path to cached_dataset_v8.pkl
        --splits-dir                — path to splits/deduped/
        --split                     — "val" | "test" (default: test)
        --mode                      — "metrics" | "overlap" | "errors" | "internals" | "all"
        --n-error-samples           — how many disagreement cases to log (default: 20)
        --out                       — JSON output path

    reuses:
        load_model_from_checkpoint()    from tune_threshold.py
        build_val_loader()              from tune_threshold.py (use test split too)
        CLASS_NAMES, NUM_CLASSES        from trainer.py
        DualPathCachedDataset           from trainer.py / dual_path_dataset.py

    outputs (per mode):
        "metrics":   per-class F1/precision/recall/AUC + macro averages, both models side-by-side
        "overlap":   per-class confusion (both_right / a_only / b_only / both_wrong)
        "errors":    sample indices + class labels for each disagreement bucket
        "internals": JK last_weights from state_dict; edge-type activation rates (v8 only)
        "all":       all of the above
```

**Key implementation note on v7 vs v8 cache compatibility:** The v8 cache contains v8 graphs (edge types 0–10). The v7 model's GATConv layers are parameterized with `edge_emb = Embedding(8, 64)` (types 0–7 only). If v8 cache graphs contain edge types 8/9/10, the v7 model's embedding lookup will fail with an out-of-range index.

**Fix:** In compare_checkpoints.py, detect from the checkpoint config which NUM_EDGE_TYPES the model was trained with, and clamp `graph.edge_attr` to `[0, num_edge_types-1]` before batching. For v7 (num_edge_types=8), edge types 8/9/10 are clamped to 7 (treated as "unknown/other"). This matches how v7 was trained — it never saw those types.

Alternatively, use the v7 cache (`ml/data/cached_dataset_deduped.pkl`) for v7 inference and v8 cache for v8 inference. The test split indices (`test_indices.npy`) were generated from the deduplicated CSV — the stem-to-index mapping is consistent across caches, so the same index array can be used with either cache.

---

## Interpretation decision matrix

After running all steps, use this matrix to decide the next action:

| Observation | Diagnosis | Next action |
|-------------|-----------|-------------|
| v8 tuned F1 ≥ v7 tuned F1 | H4 confirmed — threshold calibration was the problem | Keep v8-AB as primary; proceed to PLAN-3A with tuned thresholds |
| v8 loses on Reentrancy AND gains on IntegerUO | H5 confirmed — class-specific tradeoff | Weighted loss adjustment, or separate thresholds per edge config |
| v8 better AUC but worse F1 at default 0.5 | H4 partially confirmed | Threshold tuning closes most of the gap |
| CALL_ENTRY/RETURN_TO fire in <30% of test graphs | H2 confirmed — ICFG edges too sparse | Need PLAN-1D: true cross-function CFG with denser edge coverage |
| Both models wrong on same contracts | H3 confirmed — data/label ceiling | Label re-audit before architecture changes |
| v8 worse on Reentrancy despite new call edges | H1 confirmed — CEI pattern degraded | PLAN-3A (ICFG-only without DEF_USE) likely to recover |
| v7-only correct contracts are all short/simple | H6 — v8 Phase2 noise hurts on simple contracts | Use --phase2-edge-types 6 only for v8 training (i.e., v7 Phase2 config) |

---

## Files to create

| File | Purpose |
|------|---------|
| `ml/scripts/compare_checkpoints.py` | Main comparison script (NEW) |
| `ml/logs/comparison_v7_v8AB.json` | Raw metrics output |
| `ml/logs/comparison_v7_v8AB_tuned.json` | Tuned threshold metrics |
| `ml/logs/comparison_v7_v8AB_errors.json` | Disagreement sample analysis |
| `ml/logs/comparison_v7_v8AB_internals.json` | JK weights + edge activation |
| `docs/ml/v8-vs-v7-comparison-results.md` | Written findings (after running) |

---

## Time estimates

| Step | Time |
|------|------|
| Step 0: kill + verify | 5 min |
| Step 1: full metric comparison | ~15 min (test set inference × 2 models) |
| Step 2: threshold tuning both | ~20 min each = 40 min |
| Step 3: overlap/errors | included in Step 1 run with --mode all |
| Step 4: JK + internals | ~15 min |
| Step 5: behavioral test | ~10 min |
| Step 6: interpret + write findings | 30–60 min |
| **Total** | **~2–3 hours** |

---

## Prerequisite: what v8-AB final epoch we need

The v8-AB checkpoint saved is epoch 22 (F1=0.2593). If training produces a new best before being killed, the checkpoint will have been overwritten to the new best automatically. The comparison uses whatever is in `v8.0-AB_best.pt` at kill time — do not copy/rename before analysis is complete.
