# GNN Interpretability Fix Proposals

**Based on:** Interpretability Suite 2026-05-30 (`docs/interpretability/`)
**Model baseline:** GCB-P1-Run4-no-asl-pw_best.pt — ep32 — macro-F1 = 0.3362
**Goal:** Break F1 = 0.40 barrier in Run 5

---

## Executive Summary

The interpretability study confirmed that the model's F1 = 0.3362 ceiling is caused by three mutually reinforcing problems. First, the GNN's Phase 2 (control-flow / ICFG / data-flow) contributes near-zero discriminative signal despite the graph data being structurally correct: CFG edge ablation at inference changes model outputs by only 1.08 × 10⁻⁶ (threshold: 0.03), and the GNN eye alone achieves F1 = 0 for 7 of 10 vulnerability classes. Second, the model is severely miscalibrated on all 10 classes (ECE range 0.205–0.310, mean 0.252), which means threshold tuning is noisy and a calibration layer is a zero-training-cost quick win. Third, the Timestamp class has a confirmed size shortcut (Cohen's d = 1.657; F1 = 1.0 on small/medium contracts, 0.364 on large) that inflates macro-F1 and will break on out-of-distribution contracts.

The fixes below address these three root causes in order of implementation cost versus expected impact.

---

## Finding-to-Fix Map

| Finding | Source experiment | Proposed fix | Expected F1 impact |
|---------|------------------|-------------|-------------------|
| Phase 2 carries near-zero signal; Reentrancy GNN eye F1=0.182 | EXP-L2, EXP-A4, EXP-L1 | Fix 3: CEI auxiliary loss on Phase 2 pooled embeddings | +0.03–0.05 on Reentrancy/TOD/ExternalBug |
| ECE 0.205–0.310 for all 10 classes | EXP-L7 | Fix 1: Temperature scaling calibration layer | +0.02–0.04 macro-F1 via better thresholds |
| Timestamp F1 collapses from 1.0 to 0.364 on large contracts | EXP-L7, EXP-S3 | Fix 2: Timestamp size normalisation (data fix) | +0.02–0.04 on Timestamp alone |
| solc-select outdated; EXP-L6 fully blocked | EXP-L6 | Fix 4: 1-command solc-select upgrade | Unblocks counterfactual validation |
| max_nodes=1024 truncates up to 1414-node contracts | EXP-L7 (C-4 warning) | Fix 5: Raise max_nodes to 2048 + IMP-D1 re-extraction | Better large-contract coverage |
| EMITS edges: 12 total across 41K contracts | EXP-S2 | Fix 6: Fix graph_extractor.py EMITS extraction | Possible UnusedReturn signal |

---

## Fix 1: Calibration Layer (Quick Win — 0 Training Cost)

### What

Add post-hoc temperature scaling calibration before threshold decision. A single scalar T (or per-class vector T_c) is learned on the val set by minimising NLL of calibrated probabilities. At inference: `p_calibrated = sigmoid(logit / T)`.

### Why

EXP-L7 measured ECE for all 10 classes:

| Class | ECE |
|-------|-----|
| DenialOfService | 0.310 (worst) |
| CallToUnknown | 0.280 |
| Reentrancy | 0.270 |
| UnusedReturn | 0.257 |
| TransactionOrderDependence | 0.249 |
| MishandledException | 0.247 |
| GasException | 0.247 |
| ExternalBug | 0.250 |
| Timestamp | 0.207 |
| IntegerUO | 0.205 (best) |

Mean ECE = 0.252. Well-calibrated models have ECE < 0.05. The current miscalibration means the thresholds learned by `tune_threshold.py` are fitting to a distorted probability landscape — calibration first, then threshold tuning, would yield meaningfully better thresholds.

### How

```python
# ml/scripts/calibrate_temperature.py
# Run ONCE after Run 5 checkpoint is available, before tune_threshold.py

import torch
import torch.nn as nn
from torch.optim import LBFGS

class TemperatureScaler(nn.Module):
    """
    A single-scalar temperature scaling wrapper.
    Usage:
        scaler = TemperatureScaler(model)
        scaler.set_temperature(val_loader)  # fits T on val set
        calibrated_logits = scaler(input_ids, attention_mask, graph)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # One temperature per class for independent calibration.
        # Start with scalar T=1.5 (typical for overconfident models).
        self.temperature = nn.Parameter(
            torch.ones(10) * 1.5  # shape [10], one per class
        )

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)  # [B, 10]
        return logits / self.temperature.unsqueeze(0)  # broadcast [B, 10]

    @torch.no_grad()
    def set_temperature(self, val_loader, device="cuda"):
        """Fit temperature on validation set via NLL minimisation."""
        self.model.eval()
        all_logits, all_labels = [], []

        for batch in val_loader:
            logits = self.model(**batch)  # raw [B, 10] before sigmoid
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

        all_logits = torch.cat(all_logits)   # [N, 10]
        all_labels = torch.cat(all_labels)   # [N, 10] float

        # NLL for multi-label: BCE with logits
        criterion = nn.BCEWithLogitsLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature.unsqueeze(0)
            loss = criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        print(f"Fitted temperatures: {self.temperature.detach().tolist()}")
        return self
```

**Integration point:** Wrap the final model in `TemperatureScaler` and call `set_temperature(val_loader)` after training, before `tune_threshold.py`. The fitted `temperature` vector is saved alongside the checkpoint.

**Expected impact:** Calibrated probabilities reduce ECE from ~0.252 to < 0.05. More importantly, `tune_threshold.py` will find better thresholds on calibrated outputs, translating directly to +0.02–0.04 macro-F1 on the val set. For MishandledException and UnusedReturn — where AUC-ROC is 0.834 and 0.965 respectively but F1 is near 0 — calibration may be the single factor converting ranked predictions into useful binary classifications.

---

## Fix 2: Timestamp Size Normalisation (Data Fix)

### What

The Timestamp class learns "large contract" as a proxy for `block.timestamp` misuse. Fix by (a) explicitly adding contract-size-normalised features to break the size proxy, and (b) running Sol-3 data gating to ensure the Timestamp training set includes large contracts that do not use `block.timestamp`.

### Why

EXP-S3 found Cohen's d = 1.657 for Timestamp positive vs. negative contracts on total node count. EXP-L7 confirmed this shortcut is active in predictions:

| Stratum | Contracts | Timestamp F1 |
|---------|-----------|-------------|
| Small (<30 nodes) | 57 | 1.000 |
| Medium (30–150 nodes) | 631 | 1.000 |
| Large (>150 nodes) | 248 | **0.364** |

The F1 gap of 0.636 is the worst of any class. Training data statistics confirm the training-time bias: Timestamp-positive contracts have mean 274.6 CFG nodes vs. 88.5 for negatives (2.75× ratio), Cohen's d = 1.672.

### How

**Step A — Size-normalised node features (code change, affects re-extraction):**

In `ml/src/preprocessing/graph_extractor.py`, add a graph-level normalisation pass after extraction:

```python
# In GraphExtractor.extract() after all nodes are added:

graph_num_nodes = g.num_nodes()  # total node count for this contract

for node_id in g.nodes():
    feats = g.nodes[node_id]["x"]  # torch.Tensor [11]
    # dim 3 (cfg_count_norm): currently raw count, normalise by contract size
    # This converts "this contract has 300 CFG nodes" to
    # "this node is in a contract where CFG nodes are X% of all nodes"
    raw_cfg_count = feats[3].item()
    feats[3] = raw_cfg_count / max(graph_num_nodes, 1)
    g.nodes[node_id]["x"] = feats
```

**Step B — Sol-3 data gating (labelling fix):**

Following `docs/sentinel-c2-concrete-data-fixing-solutions.md` Sol-3: audit Timestamp-positive labels for contracts >200 nodes. Contracts where `block.timestamp` only appears in `emit` statements (not in branch conditions) should be relabelled or removed. This requires running `solc-select` (Fix 4 below).

**Expected impact:** Breaking the size shortcut removes the inflated Timestamp F1 for medium contracts and forces the model to learn genuine `block.timestamp`-in-branch patterns. Short-term this may *reduce* Timestamp F1 on the existing val set (which has the same bias), but produces a model that generalises. Net macro-F1 change is uncertain; Timestamp F1 on unbiased test contracts is expected to improve by +0.02–0.04.

---

## Fix 3: CEI Auxiliary Loss (Training Fix — Highest Expected Impact)

### What

Add a per-phase auxiliary loss head that trains Phase 2 GNN representations to be discriminative for CFG-dependent vulnerability classes. The primary target is Reentrancy (CEI pattern: CALL_ENTRY chain with subsequent state write), with secondary benefit to ExternalBug, TOD, and UnusedReturn.

### Why

The evidence chain for Phase 2 underuse is mutually reinforcing and validated across four experiments:

1. **EXP-L1:** Phase 2 JK weight = 0.322 (lowest phase, consistent across all 10 classes at ep32).
2. **EXP-L2:** CFG edge ablation effect = 1.08 × 10⁻⁶ (five orders of magnitude below the 0.03 threshold). CALL_ENTRY specifically: −5.3 × 10⁻⁷ on Reentrancy logit.
3. **EXP-A4:** GNN eye F1 = 0.182 for Reentrancy (barely above baseline 0.170). GNN F1 = 0 for ExternalBug and TOD.
4. **EXP-S4:** 76% of Reentrancy-positive contracts have CALL_ENTRY edges and 69% have the full CALL_ENTRY + RETURN_TO chain — so the failure is not a data gap.

The root cause (Section 7, INTERPRETABILITY_MASTER_REPORT.md): without an explicit auxiliary loss on Phase 2 features, the JK entropy regulariser keeps Phase 2 "alive" (entropy = 99.98% of max) but does not force Phase 2 to carry discriminative signal. Phase 1/3 structural hierarchy provides a sufficient gradient signal early in training; Phase 2 never gets a direct learning signal specific to CFG-dependent patterns.

### How

**Step A — Extract per-phase GNN embeddings in `gnn_encoder.py`:**

The `GNNEncoder.forward()` currently returns `(x, batch, jk_entropy)` — the JK-aggregated output. To add Phase 2 supervision, the Phase 2 subgraph embedding must be exposed before JK aggregation. With `return_intermediates=True` the encoder already returns per-phase outputs as the 4th element. Use this path in the trainer:

```python
# In GNNEncoder.forward(), when return_intermediates=True,
# phase_outputs is a list: [phase1_x, phase2_x, phase3_x] each [N, 256]
# (already implemented — see gnn_encoder.py lines ~490–510)
gnn_out, batch_vec, jk_entropy, phase_outputs = model.gnn(
    x, edge_index, edge_type, batch,
    return_intermediates=True
)
```

**Step B — Add `aux_phase2` classification head to `SentinelModel`:**

```python
# In ml/src/models/sentinel_model.py __init__():
# Phase 2 aux head: pool Phase 2 node embeddings → classify
self.aux_phase2 = nn.Sequential(
    nn.Linear(self.gnn_hidden_dim, self.gnn_hidden_dim // 2),  # 256 → 128
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(self.gnn_hidden_dim // 2, num_classes),          # 128 → 10
)

# In forward():
# Pool Phase 2 embeddings at graph level (mean pool over FUNCTION nodes)
phase2_x = phase_outputs[1]  # [N, 256] node embeddings after Phase 2 layers
# Reuse the existing function-node mask logic from GNN eye pooling
phase2_pooled = global_mean_pool(phase2_x, batch_vec)  # [B, 256]
aux_phase2_logits = self.aux_phase2(phase2_pooled)     # [B, 10]
```

**Step C — Add aux Phase 2 loss in `trainer.py`:**

```python
# In Trainer._compute_loss() or training loop, after main loss:

# Weight: start at 0.05 (5% of main loss); increase to 0.15 after warmup
# CEI-specific weighting: Reentrancy, ExternalBug, TOD get 3× weight
# because these classes most depend on Phase 2 CFG structure

CEI_CLASS_INDICES = [
    CLASS_NAMES.index("Reentrancy"),
    CLASS_NAMES.index("ExternalBug"),
    CLASS_NAMES.index("TransactionOrderDependence"),
]
OTHER_CLASS_INDICES = [i for i in range(NUM_CLASSES) if i not in CEI_CLASS_INDICES]

# Compute per-class aux loss
aux_loss_raw = F.binary_cross_entropy_with_logits(
    aux_phase2_logits,  # [B, 10]
    labels.float(),
    reduction="none"
)  # [B, 10]

# Apply CEI class weighting
cei_weight = torch.ones(NUM_CLASSES, device=labels.device)
cei_weight[CEI_CLASS_INDICES] = 3.0
aux_loss = (aux_loss_raw * cei_weight.unsqueeze(0)).mean()

# Epoch-dependent weight (ramp up after prefix warmup)
phase2_aux_weight = 0.05 if epoch < gnn_prefix_warmup_epochs else 0.15
total_loss = main_loss + phase2_aux_weight * aux_loss + jk_entropy_loss

# Log separately for monitoring
log_metric("aux_phase2_loss", aux_loss.item())
log_metric("phase2_aux_weight", phase2_aux_weight)
```

**Step D — Monitor Phase 2 GNN eye F1 separately during training:**

Add `aux_phase2_logits` to the validation loop to track Phase 2 eye F1 per class. The target: Reentrancy GNN Phase 2 eye F1 should rise from 0.182 toward 0.30+ within the first 20 epochs of Run 5. If it does not, increase `cei_weight` or `phase2_aux_weight`.

### Expected Impact

The Transformer eye already achieves Reentrancy F1 = 0.389 without any graph structure. If Phase 2 learns to complement the Transformer eye with CEI-path evidence, the fused and main head should improve substantially. Conservative estimate based on the current gap:

| Class | Current GNN eye F1 | Target GNN Phase 2 eye F1 | Expected main head F1 uplift |
|-------|-------------------|--------------------------|------------------------------|
| Reentrancy | 0.182 | 0.30+ | +0.03–0.05 |
| ExternalBug | 0.000 | 0.15+ | +0.02–0.04 |
| TOD | 0.000 | 0.15+ | +0.02–0.04 |
| UnusedReturn | 0.000 | 0.10+ | +0.01–0.03 |

Macro-F1 expected improvement from Phase 2 alone: +0.03–0.05. Combined with Fix 1 (calibration): +0.05–0.09 total.

---

## Fix 4: Fix solc-select (1-Command Unblock)

### What

Update `solc-select` and install compiler version 0.8.25. This unblocks EXP-L6 (counterfactual contracts) and EXP-S1 (structural trace on test contracts).

### Why

EXP-L6 status: BLOCKED. All 4 test contract pairs failed with:

```
argparse.ArgumentTypeError: solc-select is out of date. Please run `solc-select upgrade`
```

EXP-L6 is the gold-standard test for whether the model correctly distinguishes semantically vulnerable contracts from patched equivalents. It answers the question: "does removing the reentrancy vulnerability from a contract cause SENTINEL to lower its Reentrancy score?" Without this test, we cannot verify that the model responds to genuine vulnerability-specific changes rather than irrelevant structural features.

### How

```bash
# Run in the project venv
source ml/.venv/bin/activate
pip install --upgrade solc-select
solc-select upgrade
solc-select install 0.8.25
solc-select use 0.8.25
# Verify:
solc --version
```

Then re-run EXP-L6:

```bash
PYTHONPATH=. python ml/scripts/interpretability/exp_l6_counterfactual_contracts.py \
    --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \
    --cache ml/data/cached_dataset_v8.pkl
```

**Expected cost:** < 5 minutes. **Unblocks:** EXP-L6 (4 contract pairs), EXP-S1 test-contract portion.

---

## Fix 5: max_nodes Increase + IMP-D1 Re-extraction

### What

Raise `max_nodes` from 1024 to 2048 in `graph_extractor.py` and re-run `reextract_graphs.py` to rebuild all 41K graphs.

### Why

EXP-L7 observed a C-4 warning during inference: a val-split contract with 1,207 nodes was truncated to 1024. The EXP-L7 node count distribution shows:

| Metric | Value |
|--------|-------|
| Min nodes | 6 |
| Median | 94 |
| Max observed in val | 1,414 |

Contracts with >1024 nodes are silently truncated. For large Timestamp-positive contracts (mean 344 nodes, std 294 — so the tail extends well above 1024), truncation may remove the specific subgraph region containing the `block.timestamp` misuse. This contributes to the large-contract F1 collapse (Timestamp F1 = 0.364 for large contracts).

MEMORY.md pending item C-4 already flags this: "quantify % corpus > 1024 nodes; consider raising to 2048 before Phase 2."

### How

```python
# In ml/src/preprocessing/graph_extractor.py:
# Change:
MAX_NODES = 1024
# To:
MAX_NODES = 2048

# Or pass via config:
# GraphExtractionConfig(max_nodes=2048)
```

Then rebuild the full cache:

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/reextract_graphs.py \
    --output-dir ml/data/graphs/ \
    --max-nodes 2048
# Then rebuild cache:
PYTHONPATH=. python ml/scripts/build_cache.py \
    --graphs-dir ml/data/graphs/ \
    --output ml/data/cached_dataset_v8_2048.pkl
```

**Memory impact:** Graphs with 2048 nodes require ~2× the GPU memory per batch node. With `batch_size=8` and `grad_accum=8`, the effective batch is 64 graphs. The RTX 3070 8GB should handle 2048-node graphs at batch_size=4 or 6 — test before committing to the full re-extraction.

**Expected impact:** Reduces large-contract truncation; combined with Fix 2 (size normalisation), expected to improve large-contract Timestamp F1 by +0.02–0.04.

---

## Fix 6: EMITS Edge Extraction (Graph Extractor Fix)

### What

Fix `graph_extractor.py` to correctly extract EMITS edges (Solidity event emissions).

### Why

EXP-S2 found that EMITS (edge type 3) has only 12 total edges across 41,577 contracts:

| Edge | Baseline | UnusedReturn enrichment |
|------|---------|------------------------|
| EMITS (type 3) | 12 total (0.051%) | 15.46× |

The 15.46× enrichment for UnusedReturn is the highest enrichment ratio in the entire dataset. EMITS edges occur when a contract emits an event — and contracts that call external functions and ignore return values are often contracts that also emit events (e.g., ERC20 `Transfer` events). If EMITS extraction were working correctly, we would expect thousands of EMITS edges in 41K contracts (most Solidity contracts use events). The current near-zero count is a graph extractor bug, not a data property.

### How

In `ml/src/preprocessing/graph_extractor.py`, locate the edge extraction loop for event emissions. The current code likely queries AST nodes for `emit` statements but uses an incorrect AST node type name.

```python
# Likely incorrect (current):
for node in slither_contract.nodes:
    if node.type == NodeType.EMIT:  # This may not match Slither's NodeType enum
        # ... extract EMITS edge

# Correct approach using Slither's API:
from slither.core.cfg.node import NodeType

for func in slither_contract.functions + slither_contract.modifiers:
    for node in func.nodes:
        if node.type == NodeType.EXPRESSION:
            # Check for emit statements in node's IR
            from slither.slithir.operations import EventCall
            for ir in node.irs:
                if isinstance(ir, EventCall):
                    # Add EMITS edge: caller_node → event_node
                    src_id = node_id_map[node]
                    # EventCall.name is the event name
                    event_node_id = get_or_create_event_node(ir.name)
                    g.add_edge(src_id, event_node_id, edge_type=EDGE_TYPES["EMITS"])
```

**Expected impact:** Once EMITS extraction works correctly, EXP-S2 enrichment analysis should show the 15.46× UnusedReturn ratio becomes a usable signal. Phase 2 auxiliary loss (Fix 3) combined with EMITS edges could provide a discriminative structural signature for UnusedReturn. GNN AUC-ROC for UnusedReturn is already 0.929 — the signal is there; the structural evidence just needs to reach the model.

---

## Implementation Priority

In recommended execution order:

| Priority | Fix | Effort | Expected F1 impact | Blocking dependency |
|----------|-----|--------|--------------------|-------------------|
| 1 | Fix 4: solc-select upgrade | < 5 min | Unblocks EXP-L6 | None |
| 2 | Fix 1: Temperature scaling calibration | 1–2 hours | +0.02–0.04 macro-F1 | Checkpoint loaded |
| 3 | Fix 3: CEI auxiliary loss | 1–2 days | +0.03–0.05 on hard classes | Architecture change requires Run 5 |
| 4 | Fix 2: Timestamp size normalisation | 2–3 days | +0.02–0.04 on Timestamp | Sol-3 data gating + re-extraction |
| 5 | Fix 5: max_nodes to 2048 + re-extraction | 1 day (compute) | Better large-contract coverage | Memory test first |
| 6 | Fix 6: EMITS edge extraction | 2–4 hours | Possible UnusedReturn signal | IMP-D1 re-extraction already pending |

**One-session quick win (today):** Fix 4 (solc-select, 5 min) + Fix 1 (calibration, 2 hours) can be done immediately against the ep32 checkpoint without any training. Together these could push the effective val-set macro-F1 above 0.36 before Run 5 is launched.

**Run 5 training changes:** Fix 3 (CEI aux loss) is the highest-leverage change for the next training run. It requires modifying `sentinel_model.py` and `trainer.py` before Run 5 is launched. This is a one-day implementation task.

---

## Validation Plan

How to confirm each fix worked:

### Fix 1 (Calibration)
- Run `calibrate_temperature.py` on val set → report fitted T values per class
- Recompute ECE post-calibration: expect ECE < 0.05 for all classes
- Re-run `tune_threshold.py` on calibrated outputs → compare thresholds before/after
- Run val-set evaluation with calibrated thresholds → expect macro-F1 > 0.36

### Fix 2 (Timestamp size normalisation)
- Recompute Cohen's d for Timestamp in EXP-S3: target d < 0.5 after normalisation
- Recompute size-stratified F1 (EXP-L7): target Timestamp F1 gap < 0.20 (down from 0.636)

### Fix 3 (CEI aux loss)
- Monitor `aux_phase2_loss` per epoch: should decrease monotonically after warmup
- Track GNN Phase 2 eye F1 per epoch: Reentrancy Phase 2 eye F1 target > 0.30 by ep20
- Track JK phase weights: Phase 2 should increase from 0.322 toward 0.34+ as aux loss forces it to be more discriminative
- Full val-set macro-F1: target > 0.38 at Run 5 best checkpoint

### Fix 5 (max_nodes increase)
- Before re-extraction: count `% contracts > 1024 nodes` in current corpus
- After re-extraction: verify 0 C-4 warnings at max 2048 nodes
- Rerun EXP-L7 size-stratified analysis: large-contract F1 should improve for Timestamp and Reentrancy

### Fix 6 (EMITS edges)
- After fix: rerun cache stats → expect EMITS edge count > 1000 (from 12)
- Rerun EXP-S2 UnusedReturn enrichment: expect EMITS ratio to remain high (15×) but now with real sample size
- Rerun EXP-A4 UnusedReturn GNN eye F1: expect improvement from 0.000

---

## Appendix: Architecture Locations

| Fix | Primary files |
|-----|--------------|
| Fix 1 | `ml/scripts/calibrate_temperature.py` (new) · `ml/scripts/tune_threshold.py` (add calibration step) |
| Fix 2 | `ml/src/preprocessing/graph_extractor.py` · `ml/data/multilabel_index_cleaned.csv` (Sol-3) |
| Fix 3 | `ml/src/models/sentinel_model.py` · `ml/src/training/trainer.py` |
| Fix 4 | Shell only — no code change |
| Fix 5 | `ml/src/preprocessing/graph_extractor.py` (MAX_NODES) · `ml/scripts/reextract_graphs.py` |
| Fix 6 | `ml/src/preprocessing/graph_extractor.py` (EMITS extraction loop) |

---

*Generated: 2026-05-30*
*Based on: GNN Interpretability Suite — 21 experiments, 22 scripts, 24 docs*
*Checkpoint: GCB-P1-Run4-no-asl-pw_best.pt (ep32, F1=0.3362)*
