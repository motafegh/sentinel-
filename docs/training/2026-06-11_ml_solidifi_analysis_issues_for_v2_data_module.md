# SolidiFI Analysis — Issues for v2 Data Module Fix
**Compiled:** 2026-06-11  
**Source:** Step-by-step wrong-contract analysis on 199/341 failed SolidiFI predictions  
**Current state:** v2 Data module at Stage 2 (Representation Extraction)  
**Binding stage plan:** `docs/proposal/Data_Module_Proposals/actionable_plans/`

Each issue has: what it is, evidence from this analysis, which Data module stage fixes it,
and the specific file/function to change.

---

## CATEGORY A — Representation Extraction  
*Fixable in Stage 2 (`sentinel_data/representation/`)*

---

### A-1 — Comments and docstrings are never stripped before tokenization

**What it is:**  
`_tokenize()` and `_tokenize_sliding_window()` pass the raw source string directly to
GraphCodeBERT with zero preprocessing. Every comment, every `// bug` annotation, every
verbose OpenZeppelin SafeMath docstring is tokenized.

**Evidence:**  
`buggy_29.sol` (Re-entrancy, rank 5 miss) — 34-node graph, 469-line contract. The
OpenZeppelin SafeMath library fills all 4 token windows with:
```
"SafeMath: addition overflow"
"SafeMath: subtraction overflow"  
"SafeMath: multiplication overflow"
"SafeMath: modulo by zero"
"Arithmetic operations in Solidity wrap on overflow..."
```
Result: TF eye = 0.8031 IntegerUO, Fused eye = 0.8882 IntegerUO. Reentrancy ranked 5th
despite 8+ injected reentrancy functions. The docstrings entirely determined the prediction.

Confirmed in code — `ml/src/inference/preprocess.py:509`:
```python
encoded = self.tokenizer(source_code, ...)  # raw source, nothing removed
```
Training tokenizer (`ml/scripts/retokenize_windowed.py`, `ml/src/data_extraction/tokenizer.py`)
also has zero comment stripping — confirmed by grep.

**Impact:** Any contract with verbose library docstrings (SafeMath, OpenZeppelin, etc.)
gets its token windows dominated by the docstring vocabulary rather than the actual
vulnerability code. This is a systematic bias across the entire dataset.

**Fix — Stage 2, `sentinel_data/representation/tokenizer.py`:**  
Strip single-line (`//`) and multi-line (`/* */`) comments from source before tokenization.
Do NOT strip NatSpec tags (`@param`, `@notice`) — only free-form comment text.
Preserve code structure (blank lines, indentation) so token positions remain meaningful.

```python
import re

def _strip_comments(source: str) -> str:
    # Remove /* ... */ blocks (including multi-line)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    # Remove // ... to end of line
    source = re.sub(r'//[^\n]*', '', source)
    return source
```

Add a `strip_comments: bool = True` flag to the tokenizer so it can be disabled for
debugging. **Regenerate all token files when this is applied — cache invalidation required.**

**Stage 2 task ref:** Task 2.5 (port tokenizer.py) — add stripping as part of the port.  
**Risk:** Medium — changes token distributions, requires full re-tokenization of corpus.

---

### A-2 — RETURN_TO edges (type 9) are absent despite CALL_ENTRY edges (type 8) being present

**What it is:**  
`graph_extractor.py` builds CALL_ENTRY edges (calling CFG_NODE → callee ENTRYPOINT) but
fails to build the paired RETURN_TO edges (callee terminal node → call-site successor).
Without RETURN_TO, GNN Phase 2 can propagate information INTO a called function but cannot
route it BACK to the caller. The ICFG is one-directional across function boundaries.

**Evidence:**  
`buggy_4.sol` (184 nodes, 507 edges): 10 CALL_ENTRY edges, **0 RETURN_TO edges**.  
`buggy_42.sol` (196 nodes, 440 edges): similar pattern.  
Stage 2 test A18 is supposed to assert CALL_ENTRY/RETURN_TO pairs exist — it would fail.

**Impact:**  
GNN Phase 2 (Layers 3–5, CFG + ICFG subgraph) cannot learn cross-function return paths.
Inter-procedural patterns like reentrancy (call out → attacker re-enters → state inconsistency)
require the return path to be visible in the graph. This directly contributes to GNN eye
underperformance on Reentrancy.

**Fix — Stage 2, `sentinel_data/representation/graph_extractor.py`:**  
In `_build_icfg_edges()` (or equivalent), after adding CALL_ENTRY for each internal call,
find the callee's terminal nodes and the call-site's successor node and add RETURN_TO edges.

```python
# For each internal call site:
# CALL_ENTRY: calling_cfg_node → callee.entry_point
# RETURN_TO:  callee.nodes (END_IF/RETURN/last node) → successor_of_call_site
for callee_exit in _get_terminal_nodes(callee_func):
    for call_successor in _get_successors(calling_cfg_node):
        add_edge(callee_exit, call_successor, EDGE_TYPES["RETURN_TO"])
```

**Stage 2 task ref:** Task 2.6 (regression test) — test A18 will catch this.  
**Already noted in:** `ml/src/preprocessing/graph_extractor.py` open-bug list (3 still-open bugs, Stage 7 originally, but moved to Stage 2 as part of the port).

---

### A-3 — Vulnerability code injected into interface bodies is invisible to Slither

**What it is:**  
SolidiFI injects vulnerability functions inside interface/abstract contract declarations.
Slither cannot build a CFG for functions that have no implementation body (interfaces).
The resulting graph is built from only the concrete implementation contract, missing
all injected vulnerability code.

**Evidence:**  
`buggy_29.sol` (469 lines, `RaffleTokenExchange`):
```solidity
contract ERC20Interface {
    function transferFrom(...) public returns (bool success);
    // ← REENTRANCY BUG INJECTED HERE inside interface
    uint256 counter_re_ent7 = 0;
    function callme_re_ent7() public { msg.sender.send(10 ether); }
}
```
Graph: only 34 nodes from the main contract. All injected reentrancy functions
inside `ERC20Interface` and `IERC20Interface` produce zero CFG nodes.  
Model result: Reentrancy ranked 5th (0.4177), IntegerUO wins (0.7342).

This is not just a SolidiFI problem — any real-world contract with vulnerability-bearing
helper contracts that lack full implementations will hit this.

**Impact:**  
Any contract where the injected/vulnerable code lives in non-concrete contracts produces
a near-empty graph. The model sees a stub and defaults to IntegerUO (dominant class).

**Fix — Stage 2, `sentinel_data/representation/graph_extractor.py`:**  
When processing a `.sol` file, detect and process ALL contracts with implemented function
bodies, not just the "main" contract (the most-derived or last-defined one). Merge their
graphs or at minimum build CFG nodes for all concrete function implementations.

This requires Slither to be invoked with `--json` or programmatically to enumerate all
contracts in the compilation unit and filter to those with `len(func.nodes) > 0` per function.

```python
# Instead of: contract = slither.contracts[-1]  (main contract only)
# Do:
concrete_contracts = [
    c for c in slither.contracts
    if any(len(f.nodes) > 0 for f in c.functions)
]
# Build graph nodes from all concrete_contracts
```

**Stage 2 task ref:** Task 2.2 (port graph_extractor.py) — add concrete-contract enumeration.  
**Risk:** Medium — changes graph structure, regression test must be updated to reflect this.

---

### A-4 — `in_unchecked` (feat[11]) is a Solidity-era proxy, not a per-statement feature

**What it is:**  
Slither 0.10 sets `scope.is_checked = False` for ALL nodes in pre-0.8 Solidity (no
`unchecked{}` syntax exists). `_node_in_unchecked()` returns 1.0 for every node in
any pre-0.8 contract. The feature becomes a binary era flag (pre/post 0.8) rather
than a per-statement arithmetic-safety flag.

**Evidence:**  
`buggy_4.sol` (Solidity 0.5): feat[11] = 1.0 for **153/184 nodes (83%)**. Universal
firing means zero discriminative power within the contract or across Solidity 0.5 contracts.
SolidiFI = 341 contracts, all `>=0.4.22 <0.6.0` → feat[11] fires identically on all of them.

**Impact:**  
The feature provides no per-contract signal for the entire SolidiFI benchmark. In the
training data, 87.9% of contracts are pre-0.8 → feat[11] = 1.0 for most of them, making
it a weak class signal at best.

**Fix — Stage 2, schema redesign (additive, schema becomes v10):**  
Split into two features:
- `solidity_era` [new]: `1.0` if compiled with `<0.8`, `0.0` if `>=0.8`. Contract-level.
  Computed from pragma version, not Slither scope. Placed on FUNCTION/CONTRACT nodes.
- `in_unchecked_block` [feat[11] replacement]: `1.0` ONLY inside explicit `unchecked{}`
  blocks in Solidity 0.8+ contracts. `0.0` for pre-0.8 (since the construct doesn't exist).

**Stage 2 task ref:** Task 2.1 (port graph_schema.py) — add schema v10 note as future work.  
**Priority:** Lower — this is a v2.1 schema change. Document as known limitation for Run 11.

---

### A-5 — `call_target_typed` (feat[8]) = 1.0 universally on `.transfer()`-only contracts

**What it is:**  
`_node_call_target_typed()` returns `1.0` (default: "typed, not applicable") for any node
that does not contain a `LowLevelCall` or raw-address `HighLevelCall`. Solidity `.transfer()`
is a built-in, not a `LowLevelCall`, so all nodes in a `.transfer()`-only contract score 1.0.

**Evidence:**  
`buggy_4.sol`: feat[8] = 1.0 for **184/184 nodes**. The feature carries zero information
about whether this contract uses safe vs unsafe call patterns.

**Impact:**  
Low direct impact on model accuracy (the model can still learn from other features), but
it means call-safety information is invisible for the majority of ERC-20 contracts in SolidiFI.
The CallToUnknown category (which targets raw `.call()` contracts) cannot be detected
structurally — only via text patterns.

**Fix — Stage 2, `sentinel_data/representation/graph_extractor.py`:**  
Add a companion boolean feature `uses_transfer` to CFG_NODE_CALL nodes:
`1.0` if the node calls `.transfer()` or `.send()` (safe, reverts on failure),
`0.0` for raw `.call()`, `-1.0` for no external call.
This creates discriminative signal between "safe built-in" and "raw low-level call" without
replacing `call_target_typed`.

**Stage 2 task ref:** Task 2.2 (port graph_extractor.py) — document as additive improvement.  
**Priority:** Low — v2.1 enhancement. Run 11 can proceed without this.

---

## CATEGORY B — Label Quality and Class Definitions  
*Fixable in Stage 4 (Verification) and Stage 3 (Labeling)*

---

### B-1 — Noisy labels: Reentrancy ~89% FP, CallToUnknown ~87% FP in training data

**What it is:**  
Run 9 was trained on `multilabel_index_deduped.csv` — original BCCC labels, not Phase 5
cleaned. Phase 5 retrospectively estimated ~89% FP rate for Reentrancy and ~87% for
CallToUnknown in the full raw BCCC corpus. The model learned from mostly incorrect labels
for these two classes.

**Evidence from logs:**  
Reentrancy F1 at ep1 = 0.19, peaks at 0.31 after 52 epochs — never converges.  
IntegerUO F1 at ep1 = 0.49, peaks at 0.68 — converges strongly.  
The gap between IntegerUO and Reentrancy learning speed directly reflects label quality:
IntegerUO labels were clean (100% verified in Phase 5), Reentrancy was 89% FP.

**Evidence from SolidiFI:**  
22/50 Re-entrancy contracts lose to IntegerUO. The GNN eye correctly votes Reentrancy
in cases like `buggy_1.sol` (GNN=0.4954 for RE vs 0.3625 for IUO) but the Transformer
and Fused eyes override it, because the Transformer learned "ERC-20 token contract" = IntegerUO
from the noisy training distribution.

**Fix — Stage 4 (`sentinel_data/labeling/verifier.py`):**  
Use Phase 5 verified labels (`contracts_clean_v1.4.csv`) as the ground truth for all
BCCC-sourced contracts. For new sources (DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated,
Web3Bugs), apply the Phase 5 verification pipeline before finalizing labels.  
Minimum: any contract labeled Reentrancy must have a demonstrable call-before-state-update
pattern confirmed by at minimum one Slither detector (`reentrancy-eth`, `reentrancy-no-eth`,
`reentrancy-benign`).

**Stage plan ref:** `04_stage_3_labeling.md` and `05_stage_4_verification.md`.

---

### B-2 — Training imbalance: IntegerUO 14× more than Timestamp, 3× more than Reentrancy

**What it is:**  
Run 9 training split label counts:
- IntegerUO: 9,486 positives
- Reentrancy: 3,100 positives  
- GasException: 3,392 positives
- Timestamp: **678 positives** (smallest class with any learning)
- CallToUnknown: 2,237 positives

The decision boundary for any class vs IntegerUO is heavily biased toward IntegerUO.
Timestamp has so few examples that the Transformer barely learned it (peak F1 = 0.23).

**Evidence from logs:**  
Timestamp F1: ep1=0.04, ep10=0.16, ep52=0.23. After 52 epochs, Timestamp F1 is still
barely above random. This is not architecture failure — it's volume starvation.

**Fix — Stage 3 (Labeling) + Stage 6 (Splitting):**  
Apply per-class caps during corpus assembly. For Run 11, target a maximum ratio of
5:1 between the largest and smallest positive class. Add minimum positive counts per
class as a Go/No-Go gate (already in config.yaml `pipeline.min_viable_corpus`).  
For Timestamp specifically: SolidiFI (50 contracts), SmartBugs Curated, and DIVE are
the primary sources. Ensure all are ingested and labeled before Run 11.

**Stage plan ref:** `04_stage_3_labeling.md` §class distribution, `06_stage_5_splitting.md`.

---

### B-3 — Syntax era mismatch: SolidiFI 0.5 `.call.value()` vs training 0.8 `.call{value:}("")`

**What it is:**  
All 341 SolidiFI contracts use `pragma solidity >=0.4.22 <0.6.0`. They use Solidity 0.5
syntax: `.call.value(amount)("")`, `now`, `address payable`, `msg.sender.send()`.  
The BCCC training corpus is predominantly Solidity 0.8 which uses entirely different
syntax: `.call{value: amount}("")`, `block.timestamp`, `address payable` cast differently.

Different syntax = different AST structure = different Slither IR = different CFG nodes
= different graph topology. The model trained on 0.8 AST patterns cannot recognize 0.5 AST
patterns for the same vulnerability.

**Evidence:**  
MishandledException: all four eyes below 0.33 on SolidiFI despite 2,874 training examples.
The 0.5 `.call.value()` syntax produces a different IR than 0.8 `.call{value:}("")`.
The model learned 0.8 IR patterns but sees 0.5 IR patterns at inference time.

**Fix — Stage 1 (Ingestion):**  
Ensure the v2 corpus includes a representative number of Solidity 0.5 contracts with
confirmed vulnerability labels. SolidiFI itself (50 per class × 7 classes) is a natural
source of pre-0.8 labeled contracts for this purpose. DIVE and DeFiHackLabs also contain
pre-0.8 contracts. During ingestion, track `solc_version_bucket` (already planned in
Stage 1 `meta.json`) and enforce a minimum count per era bucket during Stage 3 assembly.

**Stage plan ref:** `02_stage_1_ingest_preprocess.md` §version-bucketing.

---

### B-4 — Category mismatch: SolidiFI "Unchecked-Send" targets `.transfer()`, SENTINEL CallToUnknown targets raw `.call()`

**What it is:**  
SolidiFI "Unchecked-Send" injects `.transfer(amount)` calls with no subsequent state
update — exploitable reentrancy via `.transfer()`. SENTINEL's `CallToUnknown` class
was defined to detect raw low-level `.call()` to untyped/unverified addresses — a
different vulnerability concept (unchecked low-level call return value).

**Evidence:**  
SolidiFI Unchecked-Send: 0% Top-1, 0% Top-2, 5% Top-3.  
`call_target_typed` = 1.0 for ALL nodes (correctly — `.transfer()` IS a typed call).  
The model cannot find any CallToUnknown signal because there is none — the injected bug
uses a different mechanism than what CallToUnknown was trained on.

**Fix — Stage 4 (Verification), label definition:**  
Clearly separate `CallToUnknown` (raw low-level `.call()` to unverified address, return
unchecked) from `UnhandledTransfer` (`.transfer()` or `.send()` reentrancy).
Either add a separate class or broaden the CallToUnknown definition to include both patterns.
Update the Slither detector mapping in `project_agents.md`:
- `CallToUnknown`: `low-level-calls`, `unchecked-lowlevel`, `arbitrary-send-eth`
- Consider adding: `unchecked-transfer`, `reentrancy-eth` for the `.transfer()` pattern

**Stage plan ref:** `04_stage_3_labeling.md` §CLASS_TO_DETECTORS.

---

### B-5 — TOD (Transaction Order Dependence) requires multi-transaction reasoning — single-contract CFG cannot represent it

**What it is:**  
TOD vulnerability requires reasoning about two transactions happening in a specific order.
The GNN operates on a single-contract CFG. No single-contract graph representation can
encode the inter-transaction state dependency that defines TOD.

**Evidence:**  
TOD results: 0% Top-1, 0% Top-2, 2% Top-3.  
All four eyes below 0.18 for TransactionOrderDependence on every TOD contract.  
This is a structural ceiling, not a data quality issue.

**Fix — Stage 2 / v3.1:**  
For Run 11, accept TOD as a known non-learnable class with current architecture.
Document in the Go/No-Go gate: if TransactionOrderDependence F1 < 0.05 on the test set,
it is expected and does not block Run 11 launch.
For v3: model TOD via a multi-contract or sequential transaction graph representation
(not addressable in the v2 data module alone — requires architecture changes).

**Stage plan ref:** `09_stage_8_run11_launch.md` §Go/No-Go gate — add TOD exception.

---

## CATEGORY C — Inference Pipeline  
*Fixable in `ml/src/inference/` — not Data module, but needed before Run 11 evaluation*

---

### C-1 — Thresholds JSON at wrong path: Predictor silently uses uniform 0.5

**What it is:**  
`Predictor.__init__` looks for thresholds at:
`ml/checkpoints/GCB-P1-Run9-v11-20260606_best_thresholds.json` (does NOT exist)  
Actual file: `ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json`

Falls back silently to uniform 0.5. All SolidiFI benchmark output used wrong thresholds.
Note: raw probabilities and rankings are unaffected — only confirmed/suspicious bucket splits.

**Immediate fix (one command):**
```bash
cp ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json \
   ml/checkpoints/GCB-P1-Run9-v11-20260606_best_thresholds.json
```

**Long-term fix:** `ml/src/inference/predictor.py:301` — add calibration dir as fallback
search path, or accept explicit `thresholds_path` constructor argument.

---

### C-2 — `process_source()` truncates to 512 tokens — not usable for inference

**What it is:**  
`ContractPreprocessor.process_source()` returns a single 512-token tensor (truncating
all tokens beyond 512). The model was trained on `process_source_windowed()` which
creates 4 × 512 sliding windows. Using `process_source()` for a forward pass gives
degraded predictions.

For `buggy_4.sol` (2864 tokens): `process_source()` loses 82% of the source text.

**Fix:** Add a deprecation warning or docstring note to `process_source()`:
```python
# WARNING: single 512-token window only. For inference use process_source_windowed().
```
This is a documentation fix, not a code fix — the method is used correctly internally.

---

## CATEGORY D — Architecture / Training  
*Not fixable in Data module — noted for Run 11 training decisions*

---

### D-1 — Combined output can reverse unanimous 4-eye vote when margins are small

**What it is:**  
In `buggy_5.sol` (Re-entrancy): all four eyes voted Reentrancy > CallToUnknown  
(GNN +0.09, TF +0.09, Fused +0.01, CFG +0.04) but combined output gave  
CallToUnknown=0.6710 > Reentrancy=0.6681.

The final `Linear(512→256) → Linear(256→10)` classifier head has 133,898 parameters and
can learn cross-eye interactions that override per-eye consensus when margins are small.
This makes the model less interpretable than the per-eye view suggests.

**Not a Data module fix.** Mitigations for Run 11:
- Increase auxiliary loss weight to force per-eye heads to be more consistent with the
  combined output
- Consider simple ensemble voting as a post-processing step for close predictions
- Add a consistency check: flag predictions where combined output contradicts all eyes

---

### D-2 — GNN JK attention weight entropy stays constant (~1.09) throughout training

**What it is:**  
`jk_weight_entropy` logs show ~1.09–1.10 (near maximum for 3-way uniform distribution)
throughout all 64 logged epochs. The JK attention mechanism never learned to sharply
prefer any specific GNN layer. The GNN produces a near-uniform average of all 8 layers
rather than a learned depth-specialized representation.

`gnn_share` correctly drops from 0.94 (ep1) to 0.4–0.7 (ep3+) during warmup — the GNN
is functioning. But within the GNN itself, layer aggregation is not specializing.

With noisy Reentrancy labels, the GNN cannot learn a consistent structural pattern for
reentrancy anyway — so sharpening the JK distribution would not help much until label
quality improves (B-1). This issue is secondary to label quality.

**Not a Data module fix.** For Run 11 after v2 clean labels: consider increasing
JK entropy regularization coefficient to force more decisive layer attention.

---

## Summary: What the v2 Data Module Fixes vs What It Doesn't

| ID | Issue | Data module fix? | Stage | Priority |
|----|-------|-----------------|-------|----------|
| A-1 | Comments/docstrings not stripped before tokenization | ✅ Yes | Stage 2, `tokenizer.py` | HIGH |
| A-2 | RETURN_TO edges absent | ✅ Yes | Stage 2, `graph_extractor.py` | HIGH |
| A-3 | Interface injection — vulnerability code in interface bodies invisible | ✅ Yes | Stage 2, `graph_extractor.py` | HIGH |
| A-4 | `in_unchecked` era proxy, not per-statement | Partial (doc now, fix v2.1) | Stage 2 note | LOW |
| A-5 | `call_target_typed` zero signal on `.transfer()` contracts | Partial (additive feat v2.1) | Stage 2 note | LOW |
| B-1 | Noisy labels Reentrancy 89% FP / CallToUnknown 87% FP | ✅ Yes | Stage 4 verification | CRITICAL |
| B-2 | Training imbalance 14× IUO vs Timestamp | ✅ Yes | Stage 3 labeling / Stage 6 | HIGH |
| B-3 | Syntax era mismatch 0.5 vs 0.8 | ✅ Yes | Stage 1 source selection | HIGH |
| B-4 | Category mismatch: Unchecked-Send vs CallToUnknown | ✅ Yes | Stage 4 label definitions | MEDIUM |
| B-5 | TOD requires multi-transaction reasoning | ❌ No (architecture limit) | v3 | LOW |
| C-1 | Thresholds JSON wrong path | ❌ Not Data module (ml/ fix) | Immediate cp command | HIGH |
| C-2 | `process_source()` truncation undocumented | ❌ Not Data module (ml/ fix) | Next PR | LOW |
| D-1 | Classifier head reverses unanimous eye vote | ❌ Not Data module (architecture) | Run 11 training | MEDIUM |
| D-2 | GNN JK entropy constant, not sharpening | ❌ Not Data module (training) | After B-1 fixed | LOW |

**Data module owns 8 of 14 issues (A-1, A-2, A-3, B-1, B-2, B-3, B-4 are high/critical).**  
**A-1 (comment stripping) + A-3 (interface injection) together explain ~40% of the worst misses.**  
**B-1 (noisy labels) is the root cause behind most of the Reentrancy and CallToUnknown failures.**

---

## Stage 2 Specific Actions (current sprint)

The following must be added to the Stage 2 plan (`03_stage_2_representation.md`):

1. **Task 2.2 amendment** — `graph_extractor.py` port must include:
   - Fix A-2: RETURN_TO edge construction paired with CALL_ENTRY
   - Fix A-3: process all concrete contracts in file, not just main contract

2. **Task 2.5 amendment** — `tokenizer.py` port must include:
   - Fix A-1: comment stripping before tokenization with `strip_comments=True` default

3. **New test cases for 2.6:**
   - `test_no_comments_in_tokens`: tokenize a contract with known verbose comments,
     assert the comment text tokens do not appear in `input_ids`
   - `test_interface_injection`: contract with vulnerability in interface body,
     assert CFG nodes for that function ARE extracted
   - `test_return_to_edges`: contract with internal calls, assert RETURN_TO count > 0

4. **Exit criteria amendment:** add "interface injection test passes" and
   "comment stripping regression test passes" to Stage 2 exit criteria.

---

*Related docs:*  
- [`solidifi_preprocessing_issues.md`](solidifi_preprocessing_issues.md) — detailed per-issue technical notes  
- [`benchmark_run9_solidifi_2026-06-10.md`](benchmark_run9_solidifi_2026-06-10.md) — full benchmark results  
- [`solidifi_analysis_live.md`](solidifi_analysis_live.md) — full analysis with per-category findings
