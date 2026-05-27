# SENTINEL C-2: Concrete Data Quality Fixes
**Last updated:** 2026-05-25 (Sol-8 added — external dataset integration)
**Problem:** Three architecturally different runs hit the same F1 ceiling (0.2875–0.2877). 0/3 safe contracts score clean. The training signal is wrong.
**Constraint:** Solutions 1–7 work with the EXISTING `.pt` graph files, existing `label_cleaner.py` architecture, and existing Slither pipeline — no full re-extraction required unless explicitly noted. Solution 8 (external data) requires extraction of new contracts.

---

## Solution 1 — CEI Order Detection for Reentrancy (Graph Traversal)
**Target:** Removes remaining Reentrancy Tier-2 noise: contracts with ext_call + WRITES but in safe CEI order
**Expected removal:** ~200–400 additional Reentrancy=1 labels
**Requires:** No re-extraction — uses existing CFG edges in .pt files
**Effort:** ~1 day

### The Problem
Current `check_reentrancy()` asks: "does this contract have external calls AND state writes?" That is the necessary condition. The exploitable condition is: "does a state WRITE come AFTER an external CALL in the control flow graph?" A contract that writes state BEFORE calling out (correct CEI order: Checks → Effects → Interactions) cannot be reentrancy-exploited, even though it has both edges.

### The Implementation

Add to `label_cleaner.py`, after the existing checks:

```python
def check_reentrancy_cei_order(data) -> bool:
    """
    TIER-2 Reentrancy filter: checks whether any execution path has a state
    WRITE reachable AFTER an external CALL via CONTROL_FLOW edges.

    True  → CFG shows call-before-write (CEI violation) → label is plausible.
    False → all writes precede all calls (safe CEI) OR no path exists between
            them → label is structurally implausible → remove.

    Reads from the existing v7 .pt graph using:
      NODE_TYPES["CFG_NODE_CALL"]  = 8   (statement with external call)
      NODE_TYPES["CFG_NODE_WRITE"] = 9   (statement writing state var)
      EDGE_TYPES["CONTROL_FLOW"]   = 6   (directed CFG successor edges)
      EDGE_TYPES["CONTAINS"]       = 5   (FUNCTION → CFG children)
    
    The _MAX_TYPE_ID normalisation: type_id stored as float(type_id)/12.0
    So CFG_NODE_CALL stored as 8/12 = 0.6667, CFG_NODE_WRITE as 9/12 = 0.75
    """
    import torch
    from torch_geometric.utils import subgraph as pyg_subgraph

    _MAX_TYPE_ID = 12.0
    _CALL_NORM  = 8.0 / _MAX_TYPE_ID   # 0.6667
    _WRITE_NORM = 9.0 / _MAX_TYPE_ID   # 0.75
    _FUNC_NORM  = 1.0 / _MAX_TYPE_ID   # FUNCTION node type
    _MOD_NORM   = 2.0 / _MAX_TYPE_ID   # MODIFIER
    _FB_NORM    = 4.0 / _MAX_TYPE_ID   # FALLBACK
    _RCV_NORM   = 5.0 / _MAX_TYPE_ID   # RECEIVE
    _CON_NORM   = 6.0 / _MAX_TYPE_ID   # CONSTRUCTOR

    EDGE_CONTAINS = 5
    EDGE_CF       = 6

    if data.edge_index.size(1) == 0 or data.edge_attr is None:
        return False

    x  = data.x               # [N, 11]
    ei = data.edge_index       # [2, E]
    ea = data.edge_attr        # [E]
    if ea.dim() > 1:
        ea = ea.squeeze(-1)

    type_col = x[:, 0]  # normalised type_id

    # ── Step 1: identify FUNCTION-level node indices ──────────────────────
    func_norm_set = {_FUNC_NORM, _MOD_NORM, _FB_NORM, _RCV_NORM, _CON_NORM}
    func_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for fn in func_norm_set:
        func_mask |= (type_col - fn).abs() < 0.01

    func_nodes = func_mask.nonzero(as_tuple=True)[0].tolist()
    if not func_nodes:
        return False  # ghost graph — no functions

    # ── Step 2: per-function CEI check ────────────────────────────────────
    contains_mask = ea == EDGE_CONTAINS
    contains_src  = ei[0, contains_mask]  # FUNCTION nodes
    contains_dst  = ei[1, contains_mask]  # CFG children

    cf_mask = ea == EDGE_CF
    cf_ei   = ei[:, cf_mask]   # [2, E_cf] — CONTROL_FLOW edges only

    for func_node in func_nodes:
        # All CFG children of this function
        child_mask = contains_src == func_node
        children   = contains_dst[child_mask]  # [K]
        if children.numel() == 0:
            continue

        child_set = set(children.tolist())

        # Find CALL and WRITE nodes within this function's CFG
        call_nodes  = []
        write_nodes = []
        for c in child_set:
            t = type_col[c].item()
            if abs(t - _CALL_NORM) < 0.01:
                call_nodes.append(c)
            elif abs(t - _WRITE_NORM) < 0.01:
                write_nodes.append(c)

        if not call_nodes or not write_nodes:
            continue  # this function can't have CEI violation

        # Build adjacency dict for CONTROL_FLOW within this function
        # Only include edges where BOTH src and dst are children of this function
        adj: dict[int, list[int]] = {c: [] for c in child_set}
        for e_idx in range(cf_ei.shape[1]):
            src = cf_ei[0, e_idx].item()
            dst = cf_ei[1, e_idx].item()
            if src in child_set and dst in child_set:
                adj[src].append(dst)

        # BFS/DFS from each CALL node: can we reach any WRITE node?
        for call_node in call_nodes:
            visited = set()
            queue   = [call_node]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                if curr in write_nodes and curr != call_node:
                    return True  # CALL → WRITE path found: CEI violation
                queue.extend(adj.get(curr, []))

    return False  # no CEI violation found across all functions


# In PRECONDITIONS dict, replace check_reentrancy with check_reentrancy_cei_order:
# (or chain them: first check structural, then check CEI)
PRECONDITIONS["Reentrancy"] = check_reentrancy_cei_order
```

### Expected Impact
- Current `check_reentrancy` keeps contracts that have both ext_call and WRITES (passes structural gate).
- `check_reentrancy_cei_order` additionally rejects contracts where the CFG shows writes before calls in all execution paths.
- Estimated removals: based on the data that ~14% of remaining Reentrancy=1 after structural cleaning are CEI-safe, this should remove another 200–400 labels from ~3,000 remaining Reentrancy positives.
- **F1 impact:** Reentrancy improvement +0.01–0.03 (closes the gap between PLAN-3A 0.291 and v7 0.303).

---

## Solution 2 — Solidity ≥0.8.0 IntegerUO Cleaning
**Target:** Contracts labeled IntegerUO=1 that compile with Solidity ≥0.8.0 (built-in overflow protection)
**Expected removal:** ~1,000–3,000 IntegerUO=1 labels (BCCC has significant 0.8+ contracts)
**Requires:** Access to original .sol files (via `data.contract_path`) OR re-run with version stored
**Effort:** ~2–3 hours

### The Problem
Solidity ≥0.8.0 has built-in integer overflow/underflow protection. Any arithmetic that would overflow reverts. The only exception is `unchecked{}` blocks. Since `in_unchecked` was dropped from v7 features (BUG-L2, dead for 87.9% of dataset), we have no signal for this. But we CAN parse the pragma directly from the source file — the .pt files store `data.contract_path` for exactly this purpose.

### The Implementation

```python
import re
from pathlib import Path

def _parse_solc_version_from_source(sol_path: str) -> tuple[int, int, int]:
    """
    Parse the Solidity version pragma from the source file.
    Returns (major, minor, patch) or (0, 0, 0) on failure.
    
    Handles: ^0.8.0, >=0.8.0, =0.8.19, ~0.8, 0.8.0
    Conservative: if the pragma specifies a RANGE, use the MINIMUM version
    (a pragma like '>=0.7.0 <0.9.0' means the contract MAY run on 0.7,
    so we cannot assume 0.8 overflow protection).
    """
    try:
        content = Path(sol_path).read_text(errors='ignore')
        # Find all version constraints in the pragma
        pragma_match = re.search(
            r'pragma\s+solidity\s+([^;]+);', content
        )
        if not pragma_match:
            return (0, 0, 0)
        
        pragma_str = pragma_match.group(1).strip()
        
        # Extract all version numbers mentioned
        versions = re.findall(r'(\d+)\.(\d+)(?:\.(\d+))?', pragma_str)
        if not versions:
            return (0, 0, 0)
        
        # If the pragma has a lower bound only (^0.8, >=0.8, =0.8):
        # check if the MINIMUM stated version is >= 0.8.0
        # ^0.8.x means >=0.8.x <0.9.0 — minimum is 0.8.x ✓
        # >=0.8.0 — minimum is 0.8.0 ✓
        # >=0.7.0 <0.9.0 — minimum is 0.7.0 ✗ (cannot assume 0.8)
        
        # Get the minimum version number in the pragma
        min_ver = None
        for v in versions:
            major, minor = int(v[0]), int(v[1])
            patch = int(v[2]) if v[2] else 0
            ver = (major, minor, patch)
            if min_ver is None or ver < min_ver:
                min_ver = ver
        
        return min_ver or (0, 0, 0)
    except Exception:
        return (0, 0, 0)


def _source_has_unchecked_block(sol_path: str) -> bool:
    """
    Returns True if the source code contains an unchecked{} block.
    unchecked blocks in Solidity >=0.8.0 opt out of overflow protection.
    If present, IntegerUO is still possible even on 0.8.0+.
    """
    try:
        content = Path(sol_path).read_text(errors='ignore')
        return bool(re.search(r'\bunchecked\s*\{', content))
    except Exception:
        return True  # conservative: assume unchecked exists if unreadable


def check_integer_uo(data) -> bool:
    """
    IntegerUO is structurally impossible in Solidity >=0.8.0 contracts UNLESS
    the contract uses unchecked{} blocks (which opt out of overflow protection).
    
    Returns False (remove label) if:
      - The .sol source is accessible via data.contract_path
      - The minimum pragma version is >= 0.8.0
      - The source contains no unchecked{} blocks
    
    Returns True (keep label) if:
      - Version < 0.8.0 (overflow is possible)
      - Version cannot be determined (conservative: keep)
      - unchecked{} block found (overflow possible in those blocks)
    """
    sol_path = getattr(data, 'contract_path', None)
    if sol_path is None:
        return True  # no path info: can't determine → keep label conservatively

    min_ver = _parse_solc_version_from_source(sol_path)
    if min_ver < (0, 8, 0):
        return True  # pre-0.8: overflow protection not built-in → label plausible

    # >= 0.8.0: overflow impossible unless unchecked{} present
    if _source_has_unchecked_block(sol_path):
        return True  # unchecked block found → still possible

    return False  # >= 0.8.0 with no unchecked{} → IntegerUO impossible
```

Add `"IntegerUO": check_integer_uo` to `PRECONDITIONS`.

### Expected Impact
- The BCCC dataset spans Solidity 0.4.x through 0.8.x. Based on dataset composition, ~15–25% of IntegerUO=1 contracts are likely 0.8.0+ without unchecked blocks.
- IntegerUO currently has 13,797 training positives. Removing 2,000–3,000 mislabels gives the model cleaner signal.
- **F1 impact:** IntegerUO has been stable at 0.699–0.715 — this class has enough positives that noise hasn't fully corrupted it. Cleaning will mostly reduce false-positive rate on 0.8+ contracts at inference.

---

## Solution 3 — Timestamp CFG-Path Gating Filter
**Target:** Timestamp=1 contracts where `block.timestamp` is used but NOT in a condition that gates value transfers
**Expected removal:** ~100–200 additional Timestamp=1 labels beyond the current structural check
**Requires:** No re-extraction
**Effort:** ~4–6 hours

### The Problem
The current `check_timestamp()` requires `uses_block_globals AND (ext_call OR payable)`. This still keeps contracts where block.timestamp is used in non-security contexts (event emission timestamps, informational expiry logs) that happen to ALSO have payable functions. The actual vulnerability requires: timestamp used in a CONDITION (CFG_NODE_CHECK) that then gates an external call or ETH transfer.

### The Implementation

```python
def check_timestamp_gated_path(data) -> bool:
    """
    Stricter Timestamp check: requires that block.timestamp (uses_block_globals)
    appears in a CFG_NODE_CHECK node that has a CONTROL_FLOW path to a
    CFG_NODE_CALL node in the same function.
    
    Pattern: [CHECK with block.timestamp] →(CF)→ [CALL or payable transfer]
    This is the actual dangerous pattern: a timestamp condition gates a call.
    
    Contracts where timestamp is only used in CFG_NODE_OTHER or CFG_NODE_READ
    nodes (logging, assignment) WITHOUT a gating check → remove label.
    
    Feature layout reminder:
      x[:, 2] = uses_block_globals (1.0 = reads block.timestamp/number/etc)
      NODE_TYPES["CFG_NODE_CHECK"] = 11 → normalised 11/12 = 0.9167
      NODE_TYPES["CFG_NODE_CALL"]  = 8  → normalised 8/12  = 0.6667
    """
    _MAX_TYPE_ID  = 12.0
    _CHECK_NORM   = 11.0 / _MAX_TYPE_ID   # 0.9167
    _CALL_NORM    = 8.0  / _MAX_TYPE_ID   # 0.6667
    EDGE_CONTAINS = 5
    EDGE_CF       = 6

    if data.edge_index.size(1) == 0 or data.edge_attr is None:
        return False

    x  = data.x
    ei = data.edge_index
    ea = data.edge_attr
    if ea.dim() > 1:
        ea = ea.squeeze(-1)

    type_col    = x[:, 0]
    globals_col = x[:, 2]   # uses_block_globals

    # Find all CHECK nodes that also read block globals
    check_with_timestamp = [
        i for i in range(x.shape[0])
        if abs(type_col[i].item() - _CHECK_NORM) < 0.01
        and globals_col[i].item() > 0.5
    ]
    if not check_with_timestamp:
        return False  # no timestamp-gated checks at all

    # Build CONTROL_FLOW adjacency
    cf_mask = ea == EDGE_CF
    cf_src  = ei[0, cf_mask].tolist()
    cf_dst  = ei[1, cf_mask].tolist()
    adj: dict[int, list[int]] = {}
    for s, d in zip(cf_src, cf_dst):
        adj.setdefault(s, []).append(d)

    # Find all CALL nodes
    call_nodes = set(
        i for i in range(x.shape[0])
        if abs(type_col[i].item() - _CALL_NORM) < 0.01
    )
    if not call_nodes:
        return False

    # For each timestamp-gated CHECK, check if any CALL node is reachable
    for check_node in check_with_timestamp:
        visited = set()
        queue   = [check_node]
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            if curr in call_nodes and curr != check_node:
                return True   # timestamp CHECK → CALL path: label plausible
            queue.extend(adj.get(curr, []))

    return False  # no timestamp-gated call path found


# Replace existing Timestamp check:
PRECONDITIONS["Timestamp"] = check_timestamp_gated_path
```

---

## Solution 4 — Cross-Checkpoint Ensemble Label Audit
**Target:** All 10 classes — find labels where all 3 trained models consistently disagree
**Expected removals:** ~500–1,500 labels flagged for review, ~70% confirmed mislabels
**Requires:** The 3 trained checkpoints (v7, v8-AB, PLAN-3A) already on disk
**Effort:** ~1–2 days (script writing + runtime)

### The Concept
Three architecturally different models trained on the same data have different biases. Where they ALL agree that a label=1 sample is p < 0.10, the probability all three are wrong is very low. This is model-ensemble-based label noise detection — a standard technique in learning with noisy labels (Northcutt et al., 2021, "Confident Learning").

### The Script (`ml/scripts/ensemble_label_audit.py`)

```python
"""
ensemble_label_audit.py — Use 3 trained checkpoints to find systematic label disagreements.

For each (sample, class) in the training set where label=1 and ALL 3 models
predict p < LOW_CONF_THRESHOLD: flag as high-confidence mislabel candidate.
For each (sample, class) where label=0 and ALL 3 models predict p > HIGH_CONF_THRESHOLD:
flag as possible missed label.

Output: JSON with flagged samples, sorted by ensemble confidence.

Usage:
    python ml/scripts/ensemble_label_audit.py \
        --checkpoints ml/checkpoints/v7-best.pt ml/checkpoints/v8AB-best.pt ml/checkpoints/plan3a-best.pt \
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \
        --split-indices ml/data/splits/deduped/train_indices.npy \
        --output ml/data/processed/ensemble_label_audit.json
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.inference.predictor import Predictor
from ml.src.training.trainer import CLASS_NAMES

LOW_CONF_THRESHOLD  = 0.10  # model says <10% probability but label=1 → suspect
HIGH_CONF_THRESHOLD = 0.90  # model says >90% probability but label=0 → possible miss


def audit(checkpoints, label_csv, split_npy, output_path, graphs_dir, tokens_dir, device):
    train_indices = np.load(split_npy).tolist()
    
    dataset = DualPathDataset(
        graphs_dir=graphs_dir,
        tokens_dir=tokens_dir,
        indices=train_indices,
        label_csv=Path(label_csv),
        cache_path=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=dual_path_collate_fn,
        num_workers=2,
    )

    # Collect predictions from each checkpoint
    all_probs = []   # [num_checkpoints, N, C]
    all_labels = None

    for ckpt_path in checkpoints:
        predictor = Predictor(checkpoint=ckpt_path, device=device)
        model = predictor.model.eval()
        ckpt_probs = []
        labels_list = []

        with torch.no_grad():
            for batch in loader:
                graphs, tokens, labels = batch
                graphs    = graphs.to(device)
                input_ids = tokens["input_ids"].to(device)
                attn_mask = tokens["attention_mask"].to(device)
                logits    = model(graphs, input_ids, attn_mask)
                probs     = torch.sigmoid(logits.float()).cpu().numpy()
                ckpt_probs.append(probs)
                if all_labels is None:
                    labels_list.append(labels.numpy())

        all_probs.append(np.concatenate(ckpt_probs, axis=0))   # [N, C]
        if all_labels is None and labels_list:
            all_labels = np.concatenate(labels_list, axis=0)

    # all_probs: [3, N, C]  all_labels: [N, C]
    probs_array = np.stack(all_probs, axis=0)  # [K, N, C]
    K, N, C     = probs_array.shape
    
    # Ensemble: geometric mean of probabilities (more conservative than arithmetic)
    ensemble_probs = np.exp(np.log(probs_array + 1e-8).mean(axis=0))  # [N, C]

    flagged_fp = []   # flagged false positives (label=1 but ensemble disagrees)
    flagged_fn = []   # flagged false negatives (label=0 but ensemble agrees positive)

    md5_list = dataset.paired_hashes   # list of md5 stems, index-aligned with dataset

    for n in range(N):
        for c in range(C):
            label     = int(all_labels[n, c])
            ens_prob  = float(ensemble_probs[n, c])
            per_model = [float(all_probs[k][n, c]) for k in range(K)]
            
            if label == 1 and all(p < LOW_CONF_THRESHOLD for p in per_model):
                flagged_fp.append({
                    "md5":           md5_list[n],
                    "class":         CLASS_NAMES[c],
                    "label":         1,
                    "ensemble_prob": round(ens_prob, 4),
                    "per_model":     [round(p, 4) for p in per_model],
                    "confidence":    round(1.0 - ens_prob, 4),   # confidence this is a FP
                    "action":        "CANDIDATE_REMOVE",
                })
            
            elif label == 0 and all(p > HIGH_CONF_THRESHOLD for p in per_model):
                flagged_fn.append({
                    "md5":           md5_list[n],
                    "class":         CLASS_NAMES[c],
                    "label":         0,
                    "ensemble_prob": round(ens_prob, 4),
                    "per_model":     [round(p, 4) for p in per_model],
                    "confidence":    round(ens_prob, 4),
                    "action":        "CANDIDATE_ADD",
                })

    # Sort by confidence (most certain disagreements first)
    flagged_fp.sort(key=lambda x: x["confidence"], reverse=True)
    flagged_fn.sort(key=lambda x: x["confidence"], reverse=True)

    result = {
        "summary": {
            "total_train_samples":   N,
            "flagged_fp":            len(flagged_fp),
            "flagged_fn":            len(flagged_fn),
            "low_conf_threshold":    LOW_CONF_THRESHOLD,
            "high_conf_threshold":   HIGH_CONF_THRESHOLD,
            "checkpoints_used":      [str(c) for c in checkpoints],
        },
        "false_positive_candidates": flagged_fp,
        "false_negative_candidates": flagged_fn,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"Flagged FP: {len(flagged_fp)}, FN: {len(flagged_fn)}")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--label-csv",   required=True)
    parser.add_argument("--split-indices", default="ml/data/splits/deduped/train_indices.npy")
    parser.add_argument("--graphs-dir",  default="ml/data/graphs")
    parser.add_argument("--tokens-dir",  default="ml/data/tokens_windowed")
    parser.add_argument("--output",      default="ml/data/processed/ensemble_label_audit.json")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    audit(
        checkpoints   = args.checkpoints,
        label_csv     = args.label_csv,
        split_npy     = args.split_indices,
        output_path   = args.output,
        graphs_dir    = args.graphs_dir,
        tokens_dir    = args.tokens_dir,
        device        = args.device,
    )
```

### How to Apply the Audit Results

```python
# ml/scripts/apply_ensemble_audit.py
# After manual review of ensemble_label_audit.json, apply confirmed removals:

import json, csv
from pathlib import Path

REVIEW_THRESHOLD = 0.95   # only auto-apply when ensemble confidence > 95%

audit    = json.loads(Path("ml/data/processed/ensemble_label_audit.json").read_text())
auto_removes = {
    (item["md5"], item["class"])
    for item in audit["false_positive_candidates"]
    if item["confidence"] >= REVIEW_THRESHOLD
}
print(f"Auto-removing {len(auto_removes)} high-confidence mislabels (conf >= {REVIEW_THRESHOLD})")

# Then patch the cleaned CSV:
rows = list(csv.DictReader(open("ml/data/processed/multilabel_index_cleaned.csv")))
changes = 0
for row in rows:
    for cls in CLASS_NAMES:
        if (row["md5_stem"], cls) in auto_removes and row.get(cls) == "1":
            row[cls] = "0"
            changes += 1
print(f"Applied {changes} label changes")
```

### Expected Impact
Based on published Confident Learning results on noisy datasets, this method typically finds 60–80% of true mislabels with ~70% precision at the 95th confidence percentile. For SENTINEL:
- Expected ~500–800 high-confidence FP flags
- ~350–560 of those are genuine mislabels (70% precision)
- Distributed across Timestamp, Reentrancy, MishandledException primarily

---

## Solution 5 — Safe Contract Injection (Fixes 0/3 Behavioral Test Failure)
**Target:** The 0/3 safe contracts score clean failure — teaches the model what "no vulnerability" means
**Expected impact:** Behavioral test → 3/3 clean contracts score clean (direct fix for the production alarm)
**Requires:** ~100 contracts, 1 re-run of label_cleaner on augmented data
**Effort:** ~2–3 days

### The Root Cause
The model fires 7/10 classes on a clean ERC20 because it was NEVER trained on contracts with explicit all-zero labels. The BCCC dataset's benign contracts are labeled as vulnerable because of folder-level OR-labeling. The model never saw a ground-truth negative anchor.

### Exact Implementation Strategy

**Step 1: Source the contracts** (not "go find data" — specific sources with explicit retrieval commands)

```bash
# Source 1: OpenZeppelin v4 base contracts (audited, known-safe)
git clone --depth 1 --branch v4.9.0 \
    https://github.com/OpenZeppelin/openzeppelin-contracts.git \
    /tmp/oz_contracts

# The following files are definitively safe:
OZ_SAFE_FILES=(
    "contracts/token/ERC20/ERC20.sol"
    "contracts/token/ERC20/extensions/ERC20Burnable.sol"
    "contracts/token/ERC721/ERC721.sol"
    "contracts/token/ERC1155/ERC1155.sol"
    "contracts/access/Ownable.sol"
    "contracts/access/AccessControl.sol"
    "contracts/utils/math/SafeMath.sol"
    "contracts/finance/PaymentSplitter.sol"
    "contracts/governance/TimelockController.sol"
    "contracts/proxy/transparent/TransparentUpgradeableProxy.sol"
    # ... add ~40 more
)

# Source 2: Solmate (audited DeFi primitives, all safe)
git clone --depth 1 https://github.com/transmissions11/solmate.git /tmp/solmate
# Use: src/tokens/ERC20.sol, src/tokens/ERC721.sol, src/auth/Auth.sol, etc.

# Source 3: Your existing augmented DoS-safe contracts (already in pipeline)
# ml/scripts/test_contracts/12_safe_contract.sol ... 19_safe_with_transfer.sol
# These 8 existing files should also get all-zero labels added
```

**Step 2: Inject with all-zero labels via `inject_augmented.py`**

```python
# Extend ml/scripts/inject_augmented.py:
# Instead of only injecting DoS augmented contracts,
# add a --clean-negatives mode:

def inject_clean_negatives(
    sol_files: list[Path],
    graphs_dir: Path,
    tokens_dir: Path,
    label_csv:  Path,
    output_csv: Path,
) -> None:
    """
    Extract graphs + tokens for confirmed-clean contracts and add them
    to the dataset with all CLASS_NAMES labels = 0.
    
    All-zero rows are the negative anchors the model currently lacks.
    """
    import hashlib
    
    new_rows = []
    for sol_path in sol_files:
        # Compute md5 (same as production pipeline)
        content = sol_path.read_bytes()
        md5     = hashlib.md5(content).hexdigest()
        
        # Extract graph (same pipeline as training data)
        try:
            from ml.src.preprocessing.graph_extractor import (
                extract_contract_graph, GraphExtractionConfig
            )
            config = GraphExtractionConfig(include_edge_attr=True)
            graph  = extract_contract_graph(str(sol_path), config)
            torch.save(graph, graphs_dir / f"{md5}.pt")
        except Exception as e:
            print(f"SKIP {sol_path.name}: {e}")
            continue
        
        # Tokenize (same pipeline)
        # ... (calls existing tokenizer)
        
        # Build all-zero label row
        row = {"md5_stem": md5, "filename": sol_path.name}
        for cls in CLASS_NAMES:
            row[cls] = "0"   # EXPLICIT ZERO — not missing, not ambiguous
        row["is_clean_anchor"] = "1"   # audit trail
        new_rows.append(row)
    
    # Append to cleaned CSV
    existing = list(csv.DictReader(open(label_csv)))
    all_rows  = existing + new_rows
    # write to output_csv ...
```

**Step 3: Weighted sampler must include clean anchors**

The `_build_weighted_sampler()` currently gives 3× weight to any-vuln rows and 1× weight to zero-label rows. The 100 clean anchors will get 1× weight — but they need HIGHER weight to counteract the 44K training samples. Modify:

```python
# In _build_weighted_sampler():
elif mode == "positive":
    has_vuln    = any(float(row.get(cls, 0)) == 1.0 for cls in CLASS_NAMES)
    is_anchor   = float(row.get("is_clean_anchor", 0)) == 1.0
    if is_anchor:
        w = 15.0   # 15× weight: 100 anchors compete with 44K samples
    elif has_vuln:
        w = 3.0
    else:
        w = 1.0
```

**Why 15×?** 100 anchor contracts × 15 = 1,500 effective samples. Training has ~44K samples. Clean anchors will represent ~3.3% of the effective training distribution — enough signal without dominating.

---

## Solution 6 — Pragma-Aware Temporal Data Splitting
**Target:** Eliminate train/val contamination from version-correlated contract groups
**Expected impact:** Makes reported F1 numbers trustworthy; prevents val-set overfitting
**Requires:** Access to Solidity version in the splits script
**Effort:** ~1 day

### The Problem
The current split is by deduplication (hash-based). But contracts from the same BCCC folder (same vulnerability category) tend to have the same Solidity version and similar code structure. A model can overfit to "version 0.4.22 + IntegerUO folder → predict IntegerUO" rather than learning the actual vulnerability pattern. The val set may have the same version distribution as train, making val F1 an overly optimistic estimate.

### The Fix in `create_splits.py`

```python
# ml/scripts/create_splits.py — add version-stratified splitting

import re, pandas as pd, numpy as np
from pathlib import Path

def extract_solc_version_from_path(contract_path: str) -> str:
    """Extract version string from BCCC folder structure or pragma."""
    # BCCC paths often contain version info: .../solidity-0.4.22/...
    match = re.search(r'solidity-?(\d+\.\d+\.\d+)', str(contract_path))
    if match:
        return match.group(1)
    return "unknown"

# When building the splits:
# 1. Group contracts by (vulnerability_class, solc_version_group)
# 2. Split each group 70/15/15 and combine
# This ensures train and val see the same version distribution
# and prevents the model from learning "version = vulnerability" shortcuts

def stratified_split_by_version(df, test_frac=0.15, val_frac=0.15, seed=42):
    """
    Splits that ensure each Solidity version group appears in both 
    train and val — no version group is entirely in train or entirely in val.
    """
    df = df.copy()
    df["version_group"] = df["contract_path"].apply(
        lambda p: ".".join(extract_solc_version_from_path(p).split(".")[:2])
    )  # group by major.minor (0.4, 0.5, 0.6, 0.7, 0.8)
    
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)
    
    for version, group in df.groupby("version_group"):
        n     = len(group)
        n_val = max(1, int(n * val_frac))
        n_tst = max(1, int(n * test_frac))
        perm  = rng.permutation(n)
        idx   = group.index.to_numpy()
        test_idx.extend(idx[perm[:n_tst]])
        val_idx.extend(idx[perm[n_tst:n_tst + n_val]])
        train_idx.extend(idx[perm[n_tst + n_val:]])
    
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)
```

---

## Solution 7 — Threshold Validation on Held-Out Test Set
**Target:** Make the reported tuned F1-macro numbers actually trustworthy
**Impact:** Reduces reported F1 by ~0.01–0.03 (makes it honest), prevents over-optimistic deployment decisions
**Requires:** The existing 3-way split in `ml/data/splits/deduped/` (if test_indices.npy exists)
**Effort:** ~2–3 hours

### The Problem
`tune_threshold.py` currently sweeps thresholds on the VAL set — the same set used to select the best checkpoint. This double-dips: the threshold is optimized on data that was already used to pick the model. True performance estimate requires the test set to be untouched until final evaluation.

### The Fix in `tune_threshold.py`

```python
# ml/scripts/tune_threshold.py — change the threshold tuning data source

# CURRENT (wrong):
# thresholds tuned on val split → same split used for checkpoint selection

# CORRECT:
# Step 1: select checkpoint using VAL split (already done)
# Step 2: tune thresholds on a SEPARATE held-out portion of the VAL split
#         (use only first half of val for checkpoint selection,
#          second half for threshold tuning)
# Step 3: final reported performance on TEST split (never touched before)

def tune_on_holdout(
    model, val_loader, test_loader, device, use_amp=True
) -> dict:
    """
    Tunes per-class thresholds on the FIRST HALF of val,
    evaluates on the SECOND HALF (holdout), and reports final on test.
    
    This three-way evaluation ensures no double-dipping.
    """
    # Collect all val probs
    val_probs, val_true = collect_probs(model, val_loader, device, use_amp)
    
    n_half = len(val_probs) // 2
    tune_probs = val_probs[:n_half]
    tune_true  = val_true[:n_half]
    hold_probs = val_probs[n_half:]
    hold_true  = val_true[n_half:]
    
    # Tune on first half
    candidates = np.linspace(0.1, 0.9, 19)
    tuned = []
    for c in range(val_true.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            f1 = f1_score(tune_true[:, c], (tune_probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        tuned.append(best_t)
    
    # Evaluate on holdout (not used for tuning)
    y_pred_hold  = np.stack([(hold_probs[:, c] >= tuned[c]).astype(int) for c in range(10)], axis=1)
    f1_hold      = f1_score(hold_true, y_pred_hold, average="macro", zero_division=0)
    
    # Final on test (never touched before)
    test_probs, test_true = collect_probs(model, test_loader, device, use_amp)
    y_pred_test  = np.stack([(test_probs[:, c] >= tuned[c]).astype(int) for c in range(10)], axis=1)
    f1_test      = f1_score(test_true, y_pred_test, average="macro", zero_division=0)
    
    return {
        "tuned_thresholds":    tuned,
        "f1_macro_holdout":    f1_hold,    # honest estimate
        "f1_macro_test":       f1_test,    # final deployment number
        "f1_macro_val_tuned":  ...,        # old inflated number, kept for comparison
    }
```

---

## Solution 8 — External Dataset Integration (SmartBugs Wild / SWC Registry / SolidiFI)
**Target:** DoS starvation (243 training positives), OR-labeling distortion, rare-class coverage
**Expected addition:** ~2,000–5,000 new labelled contracts; DoS positives: 243 → ~970 (4×)
**Requires:** New Slither extraction pass on external .sol files → new .pt graphs; merge into dataset
**Effort:** ~3–5 days (download + extraction + dedup + merge + split refresh)
**Priority in revised order:** Apply LAST (after BCCC cleaning is stable)

### The Problem
BCCC has 41,576 contracts but structural defects that cannot be fixed by cleaning alone:
- **DoS starvation:** 243 training positives after label cleaning. At a 29K training split, that is 0.8% prevalence — far below the ~2% floor needed for reliable signal.
- **OR-labeling distortion:** BCCC groups contracts by folder (vulnerability category). A contract in the `reentrancy/` folder gets Reentrancy=1 regardless of whether it also has AccessControl issues. Per-contract Slither output was not used at labeling time.
- **Rare-class gaps:** FrontRunning, TimeManipulation have very few confirmed positives. A model cannot learn rare classes from 40 training examples.

External datasets provide per-contract, per-tool labels derived from static analysis runs — not folder-based OR-labels.

### Dataset Sources

#### SmartBugs Wild (~47K contracts)
- **URL:** https://github.com/smartbugs/smartbugs-wild
- **Contents:** 47,398 real-world Ethereum contracts scraped from Etherscan. Each has been run through Slither, Mythril, Manticore, and Oyente; per-tool per-detector results are available as JSON.
- **Label mapping:** Use Slither detector names to map to SENTINEL's 10-class schema. Only use Slither output (consistent with SENTINEL's existing Slither-based pipeline).
- **Clean negatives:** Contracts with zero Slither findings → safe anchors (Sol-5 complement at scale)

#### SWC Registry (~400 contracts)
- **URL:** https://github.com/SmartContractSecurity/SWC-registry
- **Contents:** ~400 canonical vulnerability examples with SWC IDs. Each contract has a documented ground-truth vulnerability.
- **Label mapping:** SWC ID → SENTINEL class (see mapping table below)
- **Value:** High-confidence ground truth for rare classes. SWC-113 (DoS with failed call), SWC-128 (DoS with block gas limit), SWC-114 (transaction order dependence → FrontRunning).

#### SolidiFI (~16K contracts)
- **URL:** https://github.com/DependableSystemsLab/SolidiFI
- **Contents:** Injected vulnerability dataset — takes clean contracts and injects specific bug patterns programmatically. ~16K variants across 7 vulnerability types.
- **Value:** DoS, IntegerUO, Reentrancy injected examples with exact ground truth. Especially useful for DoS (rare in BCCC) and synthetic negative controls (pre-injection base contracts).
- **Caveat:** Injected code may not match real-world patterns exactly; use for augmentation only, not as primary training signal.

### SWC → SENTINEL Label Mapping

```python
SWC_TO_SENTINEL = {
    # Reentrancy
    "SWC-107": "Reentrancy",
    # IntegerOverflow/Underflow
    "SWC-101": "IntegerOverflow",
    # TimeManipulation / Timestamp Dependence
    "SWC-116": "TimeManipulation",
    # FrontRunning / Transaction Order Dependence
    "SWC-114": "FrontRunning",
    # DoS
    "SWC-113": "DoS",
    "SWC-128": "DoS",
    # AccessControl
    "SWC-105": "AccessControl",
    "SWC-106": "AccessControl",
    # UncheckedCall (low-level return value)
    "SWC-104": "UncheckedCall",
    # ArithmeticVulnerability (general)
    "SWC-129": "ArithmeticVulnerability",
}
```

### Implementation

#### Step 1 — Download and Extract

```bash
#!/usr/bin/env bash
# Download external datasets into ml/data/external/

mkdir -p ml/data/external/{smartbugs_wild,swc_registry,solidifi}

# SmartBugs Wild
git clone --depth=1 https://github.com/smartbugs/smartbugs-wild \
    ml/data/external/smartbugs_wild/repo

# SWC Registry
git clone --depth=1 https://github.com/SmartContractSecurity/SWC-registry \
    ml/data/external/swc_registry/repo

# SolidiFI
git clone --depth=1 https://github.com/DependableSystemsLab/SolidiFI \
    ml/data/external/solidifi/repo
```

#### Step 2 — Build External Label Index

```python
# ml/scripts/build_external_label_index.py
"""
Walk external datasets, assign SENTINEL labels, output CSV with columns:
  sol_path, source, contract_name, Reentrancy, IntegerOverflow, TimeManipulation,
  FrontRunning, DoS, AccessControl, UncheckedCall, ArithmeticVulnerability,
  BadRandomness, FakeNothing
"""
import json
import csv
from pathlib import Path

SWC_TO_SENTINEL = {
    "SWC-107": "Reentrancy",
    "SWC-101": "IntegerOverflow",
    "SWC-116": "TimeManipulation",
    "SWC-114": "FrontRunning",
    "SWC-113": "DoS",
    "SWC-128": "DoS",
    "SWC-105": "AccessControl",
    "SWC-106": "AccessControl",
    "SWC-104": "UncheckedCall",
    "SWC-129": "ArithmeticVulnerability",
}

SENTINEL_CLASSES = [
    "Reentrancy", "IntegerOverflow", "TimeManipulation", "FrontRunning",
    "DoS", "AccessControl", "UncheckedCall", "ArithmeticVulnerability",
    "BadRandomness", "FakeNothing"
]

def build_swc_index(swc_repo: Path) -> list[dict]:
    """Parse SWC registry JSON metadata."""
    rows = []
    for meta_file in swc_repo.glob("entries/SWC-*.json"):
        with open(meta_file) as f:
            meta = json.load(f)
        swc_id = meta_file.stem  # e.g. "SWC-107"
        sentinel_class = SWC_TO_SENTINEL.get(swc_id)
        if not sentinel_class:
            continue
        for sample in meta.get("samples", []):
            sol_path = swc_repo / "entries" / sample["name"]
            if not sol_path.exists():
                continue
            row = {c: 0 for c in SENTINEL_CLASSES}
            row[sentinel_class] = 1
            row["sol_path"] = str(sol_path)
            row["source"] = "swc_registry"
            row["contract_name"] = sol_path.stem
            rows.append(row)
    return rows


def build_smartbugs_index(sb_repo: Path) -> list[dict]:
    """Use Slither JSON results from SmartBugs results/ directory."""
    rows = []
    results_dir = sb_repo / "results" / "slither"
    if not results_dir.exists():
        print(f"[WARN] SmartBugs Slither results not found at {results_dir}")
        return rows

    slither_to_sentinel = {
        "reentrancy-eth": "Reentrancy",
        "reentrancy-no-eth": "Reentrancy",
        "reentrancy-benign": "Reentrancy",
        "integer-overflow": "IntegerOverflow",
        "toctou": "TimeManipulation",
        "timestamp": "TimeManipulation",
        "tx-order-dependence": "FrontRunning",
        "dos": "DoS",
        "suicidal": "AccessControl",
        "unprotected-upgrade": "AccessControl",
        "unchecked-lowlevel": "UncheckedCall",
        "unchecked-send": "UncheckedCall",
        "weak-prng": "BadRandomness",
    }

    for result_file in results_dir.glob("**/*.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
        except json.JSONDecodeError:
            continue

        row = {c: 0 for c in SENTINEL_CLASSES}
        detectors_hit = set()
        for finding in result.get("results", {}).get("detectors", []):
            det = finding.get("check", "")
            cls = slither_to_sentinel.get(det)
            if cls:
                detectors_hit.add(cls)

        for cls in detectors_hit:
            row[cls] = 1

        # Find the .sol source
        sol_path = sb_repo / "contracts" / result_file.relative_to(results_dir).with_suffix(".sol")
        if not sol_path.exists():
            continue

        row["sol_path"] = str(sol_path)
        row["source"] = "smartbugs_wild"
        row["contract_name"] = sol_path.stem
        rows.append(row)
    return rows


def build_solidifi_index(solidifi_repo: Path) -> list[dict]:
    """SolidiFI has injected/ and original/ subdirs per bug type."""
    rows = []
    bug_type_map = {
        "reentrancy": "Reentrancy",
        "integer-overflow": "IntegerOverflow",
        "timestamp": "TimeManipulation",
        "dos": "DoS",
        "unchecked-call": "UncheckedCall",
    }
    for bug_dir in (solidifi_repo / "injected").iterdir():
        cls = bug_type_map.get(bug_dir.name.lower())
        if not cls:
            continue
        for sol_file in bug_dir.glob("*.sol"):
            row = {c: 0 for c in SENTINEL_CLASSES}
            row[cls] = 1
            row["sol_path"] = str(sol_file)
            row["source"] = "solidifi"
            row["contract_name"] = sol_file.stem
            rows.append(row)

    # Original (pre-injection) contracts → clean negatives
    for sol_file in (solidifi_repo / "original").glob("**/*.sol"):
        row = {c: 0 for c in SENTINEL_CLASSES}
        row["sol_path"] = str(sol_file)
        row["source"] = "solidifi_clean"
        row["contract_name"] = sol_file.stem
        rows.append(row)
    return rows


if __name__ == "__main__":
    swc_rows = build_swc_index(Path("ml/data/external/swc_registry/repo"))
    sb_rows = build_smartbugs_index(Path("ml/data/external/smartbugs_wild/repo"))
    sf_rows = build_solidifi_index(Path("ml/data/external/solidifi/repo"))

    all_rows = swc_rows + sb_rows + sf_rows
    print(f"Total external contracts: {len(all_rows)}")
    print(f"  SWC Registry: {len(swc_rows)}")
    print(f"  SmartBugs Wild: {len(sb_rows)}")
    print(f"  SolidiFI: {len(sf_rows)}")

    out_path = Path("ml/data/external/external_label_index.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sol_path", "source", "contract_name"] + SENTINEL_CLASSES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Written: {out_path}")
```

#### Step 3 — Extract Graphs and Tokens

```bash
# Extract graphs from external contracts using existing pipeline
# Note: requires Slither and solc-select (same env as BCCC extraction)
PYTHONPATH=. python ml/scripts/extract_external_graphs.py \
    --index ml/data/external/external_label_index.csv \
    --output-dir ml/data/graphs_external/ \
    --token-dir ml/data/tokens_external/
```

The extraction script mirrors `preprocess.py` but reads from `external_label_index.csv` instead of the BCCC folder structure.

#### Step 4 — Dedup and Merge

```python
# ml/scripts/merge_external_dataset.py
"""
Dedup external graphs against existing BCCC graphs by bytecode hash.
Then append to dataset and refresh splits.
"""
import hashlib
import re
import csv
from pathlib import Path


def bytecode_hash(sol_path: str) -> str:
    """SHA-256 of normalized source (strip comments, whitespace)."""
    with open(sol_path) as f:
        src = f.read()
    src = re.sub(r"//.*|/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"\s+", " ", src).strip()
    return hashlib.sha256(src.encode()).hexdigest()


def merge_datasets(
    existing_index: str = "ml/data/processed/multilabel_index_cleaned.csv",
    external_index: str = "ml/data/external/external_label_index.csv",
    output_index:   str = "ml/data/processed/multilabel_index_merged.csv",
    dedup: bool = True,
):
    existing_hashes: set[str] = set()
    existing_rows: list[dict] = []
    with open(existing_index) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = bytecode_hash(row["sol_path"]) if dedup else None
            if h:
                existing_hashes.add(h)
            existing_rows.append(row)

    external_rows: list[dict] = []
    duplicates = 0
    with open(external_index) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if dedup:
                h = bytecode_hash(row["sol_path"])
                if h in existing_hashes:
                    duplicates += 1
                    continue
                existing_hashes.add(h)
            external_rows.append(row)

    print(f"Existing: {len(existing_rows)}, External added: {len(external_rows)}, Deduped: {duplicates}")

    all_rows = existing_rows + external_rows
    with open(output_index, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Merged index written: {output_index} ({len(all_rows)} total)")


if __name__ == "__main__":
    merge_datasets()
```

#### Step 5 — Refresh Splits with Stratification

After merging, re-run the split generation script with the merged index:

```bash
PYTHONPATH=. python ml/scripts/generate_splits.py \
    --index ml/data/processed/multilabel_index_merged.csv \
    --output-dir ml/data/splits/merged/ \
    --strategy pragma_temporal  # Sol-6 pragma-aware split if active
```

### Expected Impact

| Source | Contracts | New DoS+ | New Reentrancy+ | New clean |
|--------|-----------|----------|-----------------|-----------|
| SmartBugs Wild | ~15K after dedup | ~100 | ~800 | ~8K |
| SWC Registry | ~300 | ~30 | ~60 | 0 |
| SolidiFI | ~3K | ~600 | ~500 | ~500 |
| **Total** | **~18K** | **~730** | **~1360** | **~8.5K** |

DoS positive count: 243 → **~970** (4× increase). Sufficient for reliable signal.
Clean negative count: +8.5K anchors complement Sol-5 injections.

### Caveats and Risks

1. **SolidiFI synthetic patterns** may cause the model to learn injected artifacts rather than real vulnerability patterns. Monitor per-source F1 by tagging graphs with `source` metadata.
2. **SmartBugs Slither results** were generated with older Slither versions. Detector names may differ from current Slither. Validate mapping against `slither --list-detectors` output.
3. **Dedup by source text** is conservative — different compilers produce same bytecode from different source. A SHA-256 of normalized source is safer than bytecode for dedup.
4. **Split contamination** is the main risk. External contracts from SmartBugs are real-world contracts that may overlap with BCCC. The bytecode-hash dedup step (Step 4) prevents this, but only if the dedup runs before split generation.

---

## Sequencing and Expected Cumulative Impact

Apply in this exact order — each step provides cleaner data for the next. The revised priority order (from second audit) differs from the original numbering:

```
Priority 1 (Sol-5 — safe anchors) → 0/3 behavioral test → 3/3 clean FIRST
                                     Apply before training next run; fast win (~2 hrs effort)

Priority 2 (Sol-1 — CEI check)    → ~300 Reentrancy labels removed
                                     Expected Reentrancy F1: 0.291 → 0.30–0.31

Priority 3 (Sol-2 — pragma check)  → ~1,500 IntegerUO labels removed  
                                     Expected IntegerUO F1: 0.699 → 0.70–0.72 (more stable)

Priority 4 (Sol-4 — ensemble      → ~500 cross-class labels flagged, ~350 auto-removed
             audit)                  Distributed across all classes
                                     Expected macro ceiling: 0.2877 → 0.30–0.31

Priority 5 (Sol-3 — Timestamp     → ~150 Timestamp labels removed
             CFG gating)             Combined with v8.0-B cleaning:
                                     Expected Timestamp F1: 0.255 → 0.27–0.29

Priority 6 (Sol-7 — honest F1)    → Reported tuned F1 drops ~0.01–0.02
                                     But now the number means something

Priority 7 (Sol-6 — pragma split)  → Leakage from anachronistic compiler assumptions removed
                                     Split distribution made more realistic to deployment

Priority 8 (Sol-8 — external data) → DoS positives: 243 → ~970 (4×)
                                      +18K contracts, +8.5K clean anchors
                                      Apply LAST — after BCCC cleaning is stable
```

**The honest projection:** After all steps, expect tuned macro F1 (on the held-out test set, not val) of **0.30–0.33** (higher than original 0.29–0.32 estimate due to Sol-8 DoS signal recovery). The headline number will look lower than current val-tuned F1 because it stops being measured on the set it was optimized for. That is the correct outcome.

---

## What Cannot Be Fixed Without External Data

These problems are structural to the BCCC dataset. Solutions 1–7 cannot fix them; Solution 8 addresses #3 directly.

1. **OR-labeling of ERC20 contracts in reentrancy folders** — the only fix is re-labeling those contracts at the per-contract level using per-contract Slither output. Sol-8 (SmartBugs Wild) uses per-contract Slither results, providing correctly labelled alternatives. But the existing BCCC OR-labelled contracts will still need Sol-1/Sol-4 cleaning to remove the worst examples.

2. **Semantic Timestamp mislabeling** (block.timestamp in safe contexts) — even the CFG-path filter (Solution 3) will keep some false positives where timestamp gates a non-security-critical condition. Active learning with human annotation is the only complete fix.

3. **DoS starvation** — 243 training positives after cleaning. **Solution 8 (SolidiFI + SWC) directly addresses this**: target is ~970 DoS positives (4×) after external data merge. This crosses the minimum viable threshold for rare-class learning in a 10-class multi-label problem.
