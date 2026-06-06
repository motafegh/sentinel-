# ADR-0005: BCCC-SCsVul-2024 as primary training dataset

**Status:** Accepted
**Date:** 2026-06-06
**Deciders:** Ali Motafegh, Claude (assistant)
**Supersedes:** None (no prior dataset decision recorded)
**Superseded by:** None (current)

## Context

SENTINEL needs a labeled corpus of Solidity smart contracts to train a multi-label
classifier (see ADR-0002). Four candidate datasets were considered:

1. **BCCC-SCsVul-2024** (Berkley/City College Chicago Smart Contract Vulnerabilities 2024) —
   111,897 contracts across 10 vulnerability categories, sourced from Etherscan with
   manual labeling. ~87.9% of contracts use Solidity <0.8.x.
2. **SmartBugs-curated** — 143 contracts across 10 categories, drawn from real-world
   exploits (DAO, Parity, etc.). Curated by SmartBugs researchers. All manually
   reviewed.
3. **SolidiFI** — legacy benchmark with 50 contracts, 9 bugs injected into a single
   base contract. Useful for false-positive rate analysis but not training.
4. **Ethereum contract repo (Etherscan scrape)** — millions of contracts, but no
   labels. Would require Slither-derived labels (see ADR-0007 deferred) at scale.

The training set needs to be:

- **Large** (≥10K contracts after dedup) to support deep learning
- **Multi-label** (most real contracts have multiple vulnerabilities)
- **Realistic** (real-world patterns, not synthetic injections)
- **Diverse** (multiple Solidity versions, multiple bug types)
- **Honest** (labels are ground truth, not heuristic)

BCCC is the only candidate that meets all five at sufficient scale.

## Decision

**BCCC-SCsVul-2024 is the primary training dataset.** After dedup
(`ml/scripts/archive/dedup_multilabel_index.py:64` and the v10 pipeline), we use:

- **41,576 contracts** in the v10/deduped corpus
- **2.5 GB cached dataset** (`ml/data/cached_dataset_v10.pkl` for v8, replaced by
  `cached_dataset_v9.pkl` 2.6 GB for v9)
- **Splits:** train=29,103 / val=6,236 / test=6,237 (0 overlap, stratified by class)
- **Label distribution:** ~60% of contracts have ≥2 positives; mean positive count
  is 1.7 per contract

**SmartBugs-curated is the primary OOD benchmark**, not a training source:

- 143 contracts across 10 categories
- Used for per-class precision/recall reporting in `manual_test_smartbugs.py`
  (forthcoming, Fix #7 in `docs/pre-run9-fixes/06-bonus-fixes.md`)
- NOT used for training because 143 contracts cannot move a 41K-corpus model

**SolidiFI is reserved for false-positive rate analysis only.** Its 50 contracts
are too small for training and too synthetic for OOD reporting.

**Ethereum contract repo scraping is not done.** It would require solving
auto-labeling at scale (Slither invocation on 1M+ contracts) which is a separate
project.

### Mapping

The 10 SENTINEL classes map to BCCC tags as follows (full mapping in
`ml/scripts/build_multilabel_index.py`):

| SENTINEL class | BCCC column(s) | Notes |
|----------------|----------------|-------|
| Reentrancy | `reentrancy` | direct |
| Timestamp | `timestamp` | relabeled via `--relabel-timestamp` (Fix #1) |
| IntegerUO | `integer-overflow` | Slither-verified in v9 |
| MishandledException | `unchecked-send`, `unchecked-transfer` | disambiguated |
| DenialOfService | `dos-gas-limit`, `dos-uniswap` | 98.6% co-occurs with Reentrancy |
| ExternalBug | `arbitrary-send-eth`, `controlled-delegatecall` | merged |
| UnusedReturn | `unchecked-return`, `unchecked-lowlevel` | disambiguated from MishandledException |
| GasException | `gas-exception` | BCCC-specific |
| CallToUnknown | `low-level-calls` | BCCC-specific |

The "safe" class is implicit: contracts with all-zero labels.

## Consequences

### Positive

- **Scale is sufficient.** 41K contracts after dedup is enough for a 124M-param
  CodeBERT + ~1M-param GNN to learn without severe overfitting (provided LoRA
  r=16, dropout 0.1, and gradient clipping at 1.0 are applied).
- **Real-world patterns.** BCCC tags are based on Etherscan-sourced contracts with
  manual review, so the model learns actual exploit patterns not synthetic
  injections.
- **Multi-label density is realistic.** ~60% of contracts have multiple positives,
  which matches what we'd see in production.
- **Audit provenance.** BCCC's labels have been externally reviewed (paper
  accepted at academic venue), giving us defensible provenance for the SENTINEL
  product.

### Negative

- **87.9% pre-0.8 Solidity.** This is a hard constraint on what the model can
  learn:
  - `unchecked{}` block (introduced in Solidity 0.8.0) does not exist in 87.9% of
    training data, so `in_unchecked_block` (feat[11]) is structurally forced to 0
    for those contracts. The v9 fix for `feat[11]` only fires on the 12.1% of
    contracts that use 0.8+. This is documented in
    `ml/src/preprocessing/graph_extractor.py:393-403`.
  - The model under-learns the difference between "no overflow check because
    0.4.x" (implicit) and "no overflow check because the developer forgot" (a
    bug). Run 8's IntegerUO F1=0.595 plateaued because the model treats both
    as the same.
  - SmartBugs contracts are 100% 0.8.x, creating a train/OOD distribution shift
    that the model has to generalize across.
- **DoS/Reentrancy inseparability.** 12,381/12,395 DoS contracts are also labeled
  Reentrancy. The model has only 14 DoS-only examples to learn from. Per-class F1
  for DoS plateaus at 0.15–0.30 across runs.
- **Class imbalance.** Reentrancy is the dominant class (most positives);
  UnusedReturn, GasException, ExternalBug are sparse (<5% positives each).
  ASL + WeightedRandomSampler mitigates this but does not eliminate it.
- **BCCC versioning.** The "2024" in the name refers to release date, not contract
  age. Some contracts in the dataset are from 2018–2023, others from 2024. This
  creates a temporal heterogeneity that's not modeled explicitly.

### Neutral

- Smart contract source code is licensed under various open-source terms; using
  it for ML training is fair use under standard interpretations but should be
  flagged in any product-level documentation.
- BCCC's Etherscan scrape is not 100% de-duplicated against itself; we apply
  our own SHA256-based dedup (`ml/scripts/archive/dedup_multilabel_index.py`)
  which removed ~3,665 duplicate rows in v10.

## Alternatives Considered

**1. SmartBugs-curated as primary training source.**

Rejected. 143 contracts is 3 orders of magnitude too small. Even with heavy
augmentation (synthetic variants), the model would overfit in 1-2 epochs and
fail to generalize.

**2. SolidiFI as primary training source.**

Rejected. SolidiFI is 50 contracts, all derived from a single base contract with
injected bugs. The patterns learned are "what does an injected bug look like",
not "what do real-world bugs look like". Useful for FP rate, useless for TP
rate.

**3. Etherscan scrape with auto-labeling (Slither detector output as labels).**

Considered. The advantage is unlimited scale. The disadvantage is that
Slither-derived labels have known FPs (e.g. `integer-overflow` fires on
`unchecked{}` blocks that are *intended* to suppress checks; `unchecked-send` is
not always a bug). Fix #5 in `docs/pre-run9-fixes/05-slither-derived-labels.md`
is a future direction for this. Not adopted for Run 9 because: (a) BCCC is
already proven, (b) the auto-labeling quality question is open, (c) 41K
contracts is enough for the model to learn.

**4. Combine BCCC + SmartBugs into a single training set.**

Considered in v6 design. Rejected because: (a) SmartBugs has overlap with BCCC
(curated from similar real-world exploits), (b) the 143 contracts would be
<0.4% of the training set, well within noise, (c) keeping SmartBugs as a
held-out OOD benchmark gives us an honest test signal.

**5. Train on a balanced subset (equal examples per class).**

Considered. Rejected. Under-sampling the majority class throws away data, and
over-sampling the minority class inflates noise. ASL + WeightedRandomSampler
achieves the same balance effect at training time without losing data.

## References

- Source: `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` (111,897 rows, 10 label columns)
- Source: `ml/scripts/build_multilabel_index.py` (BCCC → SENTINEL class mapping)
- Source: `ml/scripts/archive/dedup_multilabel_index.py:64` (SHA256 dedup, parents[3]
  fix committed)
- Source: `ml/data/processed/multilabel_index_deduped.csv` (cleaned 41,576 rows)
- Source: `ml/data/splits/deduped/` (train/val/test splits, 0 overlap)
- Audit: `project_data_pipeline_audit.md` (BCCC co-occurrence table, 60% ≥2 positives)
- Audit: `docs/pre-run9-fixes/00-overview.md` (Finding I: 87.9% pre-0.8, Finding J:
  safe contracts fire 0.30-0.45)
- Calibration: `ml/calibration/temperatures_run7.json` (per-class thresholds tuned on
  BCCC test split)
- Related: ADR-0002 (multi-label formulation, the 10 classes are BCCC-mapped)
- Related: ADR-0006 (loss formulation, ASL handles the class imbalance from BCCC)
