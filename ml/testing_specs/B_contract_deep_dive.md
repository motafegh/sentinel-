# B_contract_deep_dive — Per-Contract Diagnosis

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.

---

## When This File Applies

- A specific contract is mispredicted (false positive or false negative)
- A drift alert names specific contracts as outliers
- A benchmark result is worse than expected and a per-contract breakdown is needed
- Investigating whether a model failure is data-side or model-side

Always load alongside: `C_diagnostic_checks.md` (model behaviour checks)
and `B_data_pipeline.md` if graph re-extraction may be needed.

---

## BD.1 — Step Order (Always Follow This Order)

Do not jump to model-side diagnosis before data-side integrity is confirmed.
A corrupt or mislabelled graph produces meaningless model output — fixing
the model cannot fix a data problem.

```
1. Dataset-level integrity (BD.2)
2. Per-contract graph check (BD.3)
3. AST / source integrity (BD.4)  — only if graph suspect
4. Model-side per-eye breakdown (BD.5)
5. Source reading (BD.6)
6. Classify and record (BD.7)
```

---

## BD.2 — Dataset-Level Integrity First

Before touching the contract, run `validate_graph_dataset.py` on the full
graph directory or the affected split:

```bash
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
    --graphs-dir ml/data/graphs/
```

Read the script before running — it checks: node feature dim consistency
(against `graph_schema.NODE_FEATURE_DIM`), edge type range (values in
`[0, NUM_EDGE_TYPES)`), NaN/inf in features, and feature value ranges.
Optional flags (`--check-contains-edges`, `--check-control-flow`,
`--check-all`) add connectivity and edge-subtype checks. If the dataset-level
check reports failures, fix those before proceeding to per-contract diagnosis.
A dataset-level failure means multiple contracts are affected — per-contract
work before this step is premature.

---

## BD.3 — Per-Contract Graph Integrity

For the specific contract under investigation:

1. Locate the `.pt` file: read `MEMORY.md` for the graphs directory path;
   the filename stem is the MD5 of the contract path (read `B_data_pipeline.md`
   B.1 for the hash system)
2. Load and inspect:
   ```python
   import torch
   g = torch.load('<path>.pt')
   print(g.x.shape)          # expect (num_nodes, NODE_FEATURE_DIM)
   print(g.edge_index.shape) # expect (2, num_edges)
   print(g.edge_attr.shape)  # expect (num_edges, NUM_EDGE_TYPES)
   print(g.contract_path)    # must resolve to an existing .sol file
   print(g.y)                # multi-hot label vector
   ```
3. Confirm `g.x.shape[1]` matches `graph_schema.NODE_FEATURE_DIM` (read
   from `graph_schema.py` — do not assume the value)
4. Confirm `g.edge_attr.shape[1]` matches `graph_schema.NUM_EDGE_TYPES`
5. If shape mismatches: the graph was built with a different schema version.
   This requires re-extraction before model diagnosis is meaningful.

---

## BD.4 — AST / Source Integrity (If Graph Is Suspect)

Only run this step if BD.3 found shape mismatches, unexpected node counts,
or missing edges that don't match the source contract size.

**Inspect what the extractor would produce for this contract** (no file write):

```python
# Run from project root with `source ml/.venv/bin/activate`
import sys; sys.path.insert(0, ".")
from ml.src.data_extraction.ast_extractor import ASTExtractorV4
extractor = ASTExtractorV4(verbose=True)
g = extractor.contract_to_pyg("<path-to-.sol>")
if g is not None:
    print("nodes:", g.x.shape[0], "  edges:", g.edge_index.shape[1])
    print("x.shape:", g.x.shape, "  edge_attr.shape:", g.edge_attr.shape)
```

Read `ASTExtractorV4.contract_to_pyg()` docstring in
`ml/src/data_extraction/ast_extractor.py` before running — the method
returns `None` if Slither fails (log line will show the reason).

If the node count differs significantly from `g.x.shape[0]` in the existing
`.pt`, the stored graph is stale and was built from a different source or
schema version. Re-extract this contract:

```bash
# reextract_graphs.py reads target MD5 stems from a CSV.
# To re-extract one contract, create a minimal CSV with its md5_stem row,
# then run:
PYTHONPATH=. python ml/scripts/reextract_graphs.py \
    --multilabel-csv <path-to-single-row-csv> \
    --graphs-dir ml/data/graphs/
```

Read `reextract_graphs.py` docstring before running — it **overwrites the
existing `.pt` file** for any MD5 stem listed in the CSV. The single-row
CSV must have a header (`md5_stem,...`) and one data row containing the
MD5 stem of the suspect contract (same stem as its `.pt` filename).
After re-extraction, re-run BD.3 before proceeding.

---

## BD.5 — Per-Eye Model Breakdown

This step diagnoses which model component is responsible for the misprediction.
Read `ml/src/models/sentinel_model.py` before running — understand the
eye-to-output relationship before interpreting the numbers.

```python
from ml.src.inference.predictor import Predictor
predictor = Predictor('<checkpoint path from MEMORY.md>')
result = predictor.predict('<path-to-.sol>')   # combined output
print(result)
```

For per-eye logit breakdown, `Predictor.predict()` does not expose `return_aux`.
Call the model directly — reference `ml/scripts/diag_per_eye_solidifi.py`
as the canonical implementation. Read that script before writing any per-eye call.

Read `sentinel_model.py` forward method to understand:
- Which output indices correspond to which vulnerability class
- How GNN and CodeBERT eye outputs are fused in the final layer
- What `aux_phase2` contributes vs. the primary head

Reference `ml/scripts/diag_per_eye_solidifi.py` as the
canonical implementation of per-eye breakdown — read it for the expected
output format and the interpretation pattern used in prior diagnoses.

---

## BD.6 — Contract Source Reading

After the model-side breakdown, read the contract source directly:

1. Open `g.contract_path` (from BD.3 step 2)
2. Check:
   - Solidity pragma version — older versions may have features underrepresented
     in training data
   - Comment density — high comment density increases token sequence length;
     if tokens are truncated, the CodeBERT eye loses coverage of vulnerable code
   - Interface/abstract body injection — if the contract imports interfaces,
     check whether `ast_extractor.py` injected abstract function bodies;
     this can add phantom nodes with no real code behind them
   - Library verbosity — heavily-inherited contracts may have a large node
     count dominated by library code rather than the contract's own logic
   - Vulnerability location — if the known vulnerability is in a function that
     appears after the tokenization window cutoff, CodeBERT cannot see it;
     check window coverage (read `windowed_tokenizer.WINDOW_SIZE` and `STRIDE`
     from `ml/src/data_extraction/windowed_tokenizer.py`)

---

## BD.7 — Classify and Record

After BD.2–BD.6, classify the failure mode:

| Category | Condition | Action |
|---|---|---|
| **Data-side: stale graph** | Shape mismatch, contract_path stale | Re-extract; re-run BD.3 |
| **Data-side: label error** | Ground truth is wrong for this contract | File a finding; note in audit doc |
| **Data-side: token window miss** | Vulnerable code outside window coverage | Document window gap; consider stride change |
| **Model-side: class imbalance** | Positive count for this class is very low | Check `pos_weight` in trainer config |
| **Model-side: GNN miss** | GNN eye gives low confidence, CodeBERT correct | GNN structural features may not capture this pattern |
| **Model-side: known failure mode** | Matches an existing `BUG-` or `FIND-` entry | Link to existing entry; add this contract as another instance |
| **New failure mode** | None of the above | Create a new `FIND-<ID>` entry (see `H_issue_triage.md` H.5) |

Write the finding immediately (Rule 3 — no floating findings):
- New failure mode → `ml/audit_docs/` via `H_issue_triage.md` H.5
- Existing failure mode → add this contract as an additional instance to
  the existing entry
- Data-side error → note in the run analysis doc and flag for data rebuild

---

## BD.8 — Completion Attestation

After completing this procedure, append to the relevant audit or run doc:

```
## Procedure Attestation — B_contract_deep_dive — <ISO date>
Contract investigated: <md5_stem or contract_path>
Prediction: model said <X>, ground truth <Y>
Steps completed:
  BD.2 dataset-level integrity:         PASS/FAIL/UNVERIFIED
  BD.3 per-contract graph integrity:
    x.shape[1] == NODE_FEATURE_DIM:     YES/NO (actual: N, expected: N)
    edge_attr shape correct:            YES/NO
    contract_path resolves:             YES/NO
  BD.4 AST/source re-extraction:        DONE/NOT NEEDED
  BD.5 per-eye breakdown run:           YES/NO
    GNN eye output:                     <value or N/A>
    CodeBERT eye output:                <value or N/A>
    Fused output:                       <value>
  BD.6 source reading done:             YES/NO
Failure mode classified as:             <category from BD.7>
Finding created:                        FIND-<ID> / BUG-<ID> / existing <ID> / none
Steps skipped:     [any skipped + explicit reason]
Written to:        [path of this attestation]
```
