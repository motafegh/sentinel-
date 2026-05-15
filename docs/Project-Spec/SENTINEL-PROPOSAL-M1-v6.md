# SENTINEL — M1 ML Architecture Proposal: v6

**Status:** Draft — for review before v5.2 behavioral gates pass
**Author:** SENTINEL Engineering
**Date:** 2026-05-16
**Depends on:** v5.2 training completing + behavioral test pass + all per-class floors exceeded

---

## Context & Motivation

v5.2 represents the current production-target architecture: three-eye GNN+CodeBERT fusion with
JK attention aggregation over three GAT phases, CrossAttentionFusion locked at 128-dim output,
LoRA r=16 α=32 on all 12 CodeBERT Q+V projections. Training is active on the deduped 44,420-row
dataset. Measured progress: epoch 21 val F1=0.3130 on eval_threshold=0.35, patience=20.

v5.2 is architecturally sound for single-contract, static analysis. Five structural limitations
remain that cannot be addressed by hyperparameter tuning or additional epochs:

**1. Single-contract analysis.** Protocol-scale audits (e.g., Uniswap v4 — 50+ interacting
contracts) are analyzed one contract at a time. Cross-contract reentrancy, flash-loan-mediated
attacks, and privilege escalation via proxy delegation are entirely invisible to the current
model. The contract boundary is a hard analysis horizon.

**2. DoS class data starvation.** DenialOfService has 377 total samples, train≈257. pos_weight
is set to 67.75 to partially compensate, but 257 training examples cannot teach a 124M-parameter
model meaningful DoS signal. The class will likely remain near-zero F1 unless the training
distribution changes.

**3. No uncertainty quantification.** model.forward() returns a single point estimate per class.
A prediction of 0.82 for Reentrancy and a prediction of 0.82 for Timestamp look identical to
downstream consumers, even if the former is stable across forward passes and the latter collapses
to 0.45 under mild stochasticity. Security engineers cannot triage accordingly.

**4. Explainability gap.** The output "Reentrancy: 0.82" gives an auditor no pointer to which
function, code path, or call chain triggered it. GATConv already computes per-edge attention
weights during Phase 2 and Phase 3 — this signal is computed but discarded.

**5. Static graph schema.** NODE_FEATURE_DIM=12 is locked. Adding a new node feature (e.g.,
`is_upgradeable`, `has_delegatecall`) requires rebuilding all 44,420 graphs and fully retraining.
Architectural extensibility was not designed in from the start.

**6. Flat function-level attention.** GNNEncoder Phase 2 (CONTROL_FLOW) operates on CFG nodes
within individual functions, but the inter-function call graph is only modeled via CALLS edges
in Phase 1. Multi-hop reentrancy paths (A calls B calls A) depend on the CFG diameter being
large enough — many compilers flatten this, making the pattern invisible at the CFG level.

The proposals below address each weakness with concrete design decisions, implementation paths,
file locations, and explicit tradeoffs. They are ordered by expected ROI × implementation cost.

---

## V6-1: Protocol-Level Multi-Contract GNN

### Problem

Cross-contract attacks require analyzing multiple contracts and their relationships simultaneously.
Reentrancy-across-contracts, flash-loan-mediated attacks (contract A borrows from B, calls
vulnerable callback in C, repays B), and privilege escalation via proxy delegation (attacker
upgrades proxy to point to malicious implementation) all require seeing at least two contract
graphs at once. The v5.2 GNNEncoder produces a single embedding per contract and has no
mechanism to ingest cross-contract context.

Concrete example: in the 2022 Beanstalk Farms exploit (~$182M), the attack contract called
`vote()` on the governance contract in the same transaction as `emergencyCommit()`. Neither
contract is individually vulnerable; the vulnerability exists in their protocol-level interaction.
v5.2 cannot detect this.

### Design

**Level 1 — contract-level (unchanged):** the existing v5.2 three-phase GNNEncoder runs per
contract and produces a contract embedding of shape [128]. For a protocol with C contracts,
this yields [C, 128].

**Level 2 — protocol-level ProtocolGNN:** each contract's 128-dim embedding is treated as a
node in a protocol graph. Cross-contract edges are typed:

```
CALLS_CONTRACT   = 0   # contract A has an external call to contract B
INHERITS_CONTRACT = 1  # contract A inherits from B (multi-file inheritance)
PROXY_OF         = 2   # contract A is a proxy whose implementation is B
```

The protocol graph is a 2-layer GAT (heads=4, hidden=128) operating over this small graph
(typically 2–50 nodes). This is cheap: protocol graphs are tiny compared to per-contract CFGs.

**Fusion:** per-contract output from Level 1 [C, 128] is concatenated with the protocol-context
embedding from Level 2 [C, 128] → [C, 256] → Linear(256, 128) → per-contract vulnerability
prediction at [C, 10].

**Dataset requirement:** DeFiHackLabs (github.com/SunWeb3Sec/DeFiHackLabs) provides labelled
multi-contract exploit PoCs. Approximately 200+ protocol-level incidents are documented with
contracts. This is the primary source for cross-contract supervision signal.

**New files:**
- `ml/src/models/protocol_gnn.py` — ProtocolGNN (2-layer GAT over protocol graph)
- `ml/src/preprocessing/protocol_builder.py` — builds protocol graph from multi-contract extraction output; resolves import paths via solc's `--combined-json` AST
- `GraphExtractionConfig.multi_contract_policy = "all"` (currently `"primary_only"`)

**Tradeoffs:**

| Consideration | Detail |
|---|---|
| Preprocessing complexity | 2× — each audit now requires a protocol graph pass in addition to per-contract extraction |
| Training data scarcity | Cross-contract labelled exploits number in the hundreds, not tens of thousands |
| Protocol graph construction | Requires resolving import paths at preprocessing time; npm/hardhat dependencies add complexity |
| Inference latency | +10–50ms per protocol (Level 2 GAT over small graph); acceptable |
| Blocked on | Move 9 (multi-contract parsing policy) must be implemented first |

**Recommendation:** implement Level 2 as an optional module gated by `--multi-contract` flag.
v5.2 single-contract behavior is preserved as the default. Run ablation on DeFiHackLabs subset
before committing to full protocol-graph training.

---

## V6-2: Synthetic Minority Augmentation for DoS

### Problem

DenialOfService: 377 total samples, train≈257. The current pos_weight=67.75 gradient signal is
numerically unstable at scale — extreme weights interact badly with gradient accumulation steps=4
and mixed-precision training. More fundamentally, 257 examples is insufficient for a model of
this capacity to learn discriminative DoS patterns, regardless of loss weighting.

The v4 fallback DoS F1 is 0.384. The v5.2 per-class floor for DoS is 0.334 (floor = v4 F1 − 0.05).
Current trajectory suggests this floor will not be met without addressing the data imbalance
directly.

### Design

Three approaches are evaluated, in increasing risk order:

**Approach A — AST-level code mutation (recommended, low risk):**

Take the 377 existing DoS-labelled contracts. Apply AST-level mutations that preserve or
amplify DoS patterns:

- Replace bounded `for` loops over fixed-size arrays with unbounded loops over dynamic mappings
- Remove loop guard conditions (e.g., `require(users.length < 100)`)
- Replace `transfer(gas_limit)` patterns with `call{value: ...}("")` without gas cap
- Add re-entrant state writes inside loops

Each mutated contract is verified by re-running Slither with the DoS detector. Only mutations
that Slither still classifies as DoS are kept. Target: 5× augmentation → ~1,885 DoS samples.

Implementation: `ml/scripts/augment_dos.py` using `py-solc-ast` for AST traversal.

**Approach B — LLM-based synthesis (recommended, medium risk):**

Use the locally hosted `qwen2.5-coder-7b` (via LM Studio) to generate novel DoS-pattern
contracts given 5-shot examples from the existing DoS corpus. Prompt template:

```
Generate a Solidity contract with a Denial-of-Service vulnerability.
The vulnerability should involve [unbounded-loop | gas-exhaustion | block-stuffing].
Reference examples: [3 shortest DoS contracts from corpus].
Output only valid Solidity. No explanation.
```

Filter: (1) must compile with solc; (2) Slither DoS detector must fire; (3) human spot-check
of 10% random sample. Target: 3× augmentation → ~1,131 additional DoS samples.

**Approach C — SMOTE in embedding space (not recommended for v6.0):**

Generate synthetic graph embeddings by linear interpolation between real DoS node embeddings
in the 128-dim GNN latent space. Requires training a conditional GAN to ensure the synthetic
embeddings correspond to realistic graph structures. Risk: mode collapse; generated embeddings
may not correspond to any valid contract graph topology. Defer to v6.1.

**Recommended path for v6.0:** Approach A + B combined. Expected total DoS samples after
augmentation: ~2,500–3,000. This brings DoS to parity with TOD (3,391) and ExternalBug (3,404).

**Gate before adding to training set:**
1. All augmented contracts pass `python -m slither <contract> --detect dos-gas-limit`
2. 10% random sample reviewed by human auditor
3. No augmented contract appears in val or test split (content-hash check)

**Impact on pos_weight:** with 2,500+ DoS samples, pos_weight drops from 67.75 to ~12.5.
This directly stabilises DoS gradients and allows the model to learn actual DoS patterns
rather than amplifying noise.

---

## V6-3: Uncertainty Quantification (MC Dropout)

### Problem

`model.forward(graph, tokens)` returns a tensor of shape [B, 10] — a single point estimate.
Security engineers receive "Reentrancy: 0.82" with no indication of whether this prediction is
stable or whether the model is effectively guessing near the threshold. This matters for triage:
a finding with uncertainty std=0.03 deserves a different review priority than one with std=0.18.

### Design

**Monte Carlo Dropout at inference time:**

The model already uses Dropout layers (rate=0.1) throughout GNNEncoder and CrossAttentionFusion.
During standard inference, PyTorch's `model.eval()` disables them. MC Dropout re-enables
Dropout during inference and runs N=20 stochastic forward passes.

```python
# In ml/src/inference/predictor.py
def _score_mc(self, source: str, n_samples: int = 20) -> dict:
    self.model.train()   # re-enables Dropout
    with torch.no_grad():
        logits_samples = torch.stack([
            self.model(graph, tokens) for _ in range(n_samples)
        ])  # [N, B, 10]
    self.model.eval()
    probs = torch.sigmoid(logits_samples)  # [N, B, 10]
    mean = probs.mean(0)    # [B, 10]
    std  = probs.std(0)     # [B, 10]
    return {"probability": mean, "std": std}
```

The existing `_score()` method is unchanged and used by default for latency-sensitive paths.

**API contract** (POST /predict with `"uncertainty": true`):

```json
{
  "Reentrancy": {
    "probability": 0.82,
    "uncertainty": 0.03,
    "confidence_interval_95": [0.76, 0.88],
    "triggered": true
  },
  "DenialOfService": {
    "probability": 0.51,
    "uncertainty": 0.19,
    "confidence_interval_95": [0.13, 0.89],
    "triggered": false
  }
}
```

**Threshold adjustment under high uncertainty:**

```python
effective_threshold = base_threshold + (0.05 if std < 0.10 else 0.0) - (0.05 if std > 0.15 else 0.0)
```

When uncertainty is high (std > 0.15), the effective detection threshold lowers — the auditor
sees the finding flagged as "uncertain" rather than silently suppressed. This is preferable to
a false negative in a security context.

**Performance:** 20 forward passes at ~50ms each = ~1s additional latency per request. This
is acceptable for the high-stakes, non-interactive audit workflow. Fast path (`"uncertainty":
false`, the default) is unaffected.

**Implementation scope:**
- `ml/src/inference/predictor.py` — add `_score_mc()` and `uncertainty` field to response schema
- `ml/src/inference/api.py` — add `uncertainty: bool = False` to `PredictRequest`
- No model architecture changes; no retraining required

---

## V6-4: Attention-Based Explainability

### Problem

The output "Reentrancy: 0.82" is a scalar with no attribution. An auditor needs to know *which
function's control flow* raised the Reentrancy signal, not just the contract-level probability.
GATConv already computes per-edge attention weights α_ij during every forward pass. These
weights are computed but discarded. CodeBERT LoRA-adapted layers similarly produce per-token
attention matrices.

### Design

**GNN attention extraction:**

PyG's `GATConv` supports `return_attention_weights=True`. Phase 2 (CONTROL_FLOW, directed
edges, layer 3) and Phase 3 (REVERSE_CONTAINS type-7, layer 4) carry the most vulnerability-
relevant signal. During explainability inference:

```python
out, (edge_index, alpha) = self.gat_phase2(x, edge_index, return_attention_weights=True)
```

Node importance scores are computed as the sum of incoming attention weights per node,
normalised over FUNCTION/FALLBACK/RECEIVE/CONSTRUCTOR nodes only (consistent with v5.1
three-eye pooling fix).

**Top-K flagged nodes per vulnerability class:**

For each predicted vulnerability class with `triggered=True`, return the top-3 FUNCTION nodes
by attention score. Node metadata (function name, line_start, line_end) is stored in
`graph.node_metadata` at extraction time (already present in v5 schema for the three-eye fix).

**CodeBERT token attribution:**

Extract the last-layer attention matrix from the LoRA-adapted CodeBERT. Aggregate across
heads (mean). Map attention-dominant token positions back to source line ranges using the
tokenizer's offset mapping (`return_offsets_mapping=True`). Return top-5 line ranges.

**Output schema** (POST /predict with `"explain": true`):

```json
{
  "explanations": [
    {
      "vulnerability_class": "Reentrancy",
      "probability": 0.82,
      "flagged_functions": [
        {"name": "withdraw", "line_start": 45, "line_end": 67, "attention_score": 0.82},
        {"name": "_transfer", "line_start": 112, "line_end": 134, "attention_score": 0.51}
      ],
      "flagged_lines_codebert": [48, 52, 58]
    }
  ]
}
```

**Implementation files:**
- `ml/src/inference/explainer.py` — `GNNExplainer` class, `CodeBERTExplainer` class
- `ml/src/inference/predictor.py` — call explainer when `explain=True`
- `ml/src/inference/api.py` — add `explain: bool = False` to `PredictRequest`

**Tradeoffs:**

| Consideration | Detail |
|---|---|
| GATConv attention overhead | Storing α_ij doubles peak memory during Phase 2/3 forward pass; acceptable |
| Attention ≠ causation | Attention weights are a proxy, not a formal proof; must be documented clearly |
| node_metadata availability | Must verify all 44,420 graphs have node_metadata populated; 280 stale graphs may not |
| LoRA attention extraction | Requires accessing attention weights through PEFT wrapper; test adapter hooks |

---

## V6-5: Dynamic Feature Schema (Extensible Without Re-extraction)

### Problem

`NODE_FEATURE_DIM=12` is a locked constant. Adding a single new feature (e.g., `is_upgradeable:
bool` for UUPS proxy functions, `has_delegatecall: bool` for functions containing `delegatecall`)
requires: (1) bumping `FEATURE_SCHEMA_VERSION`, (2) re-running graph extraction on all 44,420
contracts, (3) invalidating the RAM cache, (4) full retraining from scratch. This has already
happened once (v2→v3, 2026-05-12). The cost is high enough to suppress useful feature additions.

### Design

**Extension module architecture:**

The base schema ("v3", 12-dim) remains frozen. New features are packaged as "extension modules"
in `ml/src/preprocessing/feature_extensions/`. Each extension module:

1. Takes as input: the existing `torch_geometric.data.Data` graph (base features) + raw Slither
   JSON output for the contract
2. Computes [N, k] additional features where k is module-specific
3. Returns an augmented feature tensor

Extension modules are not stored in graph `.pt` files — they are computed at inference time
(and optionally cached separately).

**Projection layer (schema adapter):**

Before the base [N, 12] tensor enters GNNEncoder, it passes through a learned `SchemaAdapter`:

```python
class SchemaAdapter(nn.Module):
    def __init__(self, base_dim=12, ext_dim=0, out_dim=12):
        # base_dim + ext_dim → out_dim via Linear + LayerNorm
        # when ext_dim=0, reduces to identity (no overhead in base case)
```

`GNNEncoder` input signature stays `[N, 12]` always — the SchemaAdapter projects the
concatenated base+extension features back to 12-dim before the first GAT layer. This preserves
the locked `NODE_FEATURE_DIM=12` invariant.

**Training strategy:**

The SchemaAdapter's projection weight is learned jointly during fine-tuning on extension-
augmented data. Base model checkpoint is frozen; only the SchemaAdapter + GNNEncoder layer-0
weights are updated. This avoids full retraining when adding a new feature extension.

**Example extension: `IsUpgradeable`**

```python
# ml/src/preprocessing/feature_extensions/is_upgradeable.py
def compute(graph, slither_json) -> torch.Tensor:  # [N, 1]
    # Returns 1.0 for FUNCTION nodes in contracts that inherit from
    # OpenZeppelin's UUPSUpgradeable or TransparentUpgradeableProxy
```

**Tradeoffs:**

| Consideration | Detail |
|---|---|
| SchemaAdapter adds a projection at inference time | +0.1ms; negligible |
| Extension features derived from Slither | Slither must be available at inference time (it already is) |
| Feature interaction via projection | Linear projection may not capture all feature interactions; consider deeper adapter for complex extensions |
| Training cost for new extensions | Fine-tuning SchemaAdapter only: ~2–4 epochs on a 1K-sample probing set |
| Risk of adapter drift | If base checkpoint is retrained (v7), SchemaAdapter must be re-aligned |

**Note:** this is a design-for-extensibility feature, not a performance feature. Implement
after v6-2 and v6-3 are stable. Design carefully — a poorly specified SchemaAdapter interface
will create technical debt equivalent to the problem it solves.

---

## V6-6: Hierarchical Function-Level Attention

### Problem

GNNEncoder Phase 2 (CONTROL_FLOW, directed GAT, layer 3) models control flow within individual
function CFGs. Phase 1 includes CALLS edges (type=0) but these are aggregated with all other
structural edges in a single message-passing step. Multi-hop reentrancy paths — A calls B which
calls back into A — require the model to propagate information across at least two CALLS hops.
In practice, many Solidity compilers flatten the CFG representation, placing `call` sites as
single CFG_NODE_CALL nodes without connecting them to the callee's CFG. This makes multi-hop
reentrancy invisible at the CFG level.

### Design

**Two-level attention hierarchy:**

- Level 1 (intra-function): current Phase 2 CONTROL_FLOW (unchanged)
- Level 2 (inter-function, new Phase 2b): CALLS-only edges connecting FUNCTION nodes

Phase 2b operates on FUNCTION-level embeddings (not CFG nodes). After Phase 2 pools CFG nodes
into their parent FUNCTION node embedding, Phase 2b runs one additional GAT hop over FUNCTION
nodes connected by CALLS edges only.

```python
# In GNNEncoder.forward(), after Phase 2:
func_embs = self._pool_cfgnodes_to_functions(x2, graph)  # [F, 128] where F = num functions
x2b = self.gat_phase2b(func_embs, calls_edge_index)       # [F, 128]
# expand back to [N, 128] — CFG nodes inherit their function's Phase 2b embedding
x2_augmented = self._broadcast_function_to_cfgnodes(x2b, graph)  # [N, 128]
```

**Architecture impact:**

- `gnn_layers=5` (was 4); new layer is Phase 2b
- Phase 2b GAT: `heads=1, add_self_loops=False, edge_dim=32` (CALLS edge type)
- JK aggregation: updated to include Phase 2b output as a 4th phase input (or merge into Phase 2 output — ablate both)
- Per-phase LayerNorm: add `LayerNorm(128)` after Phase 2b

**Benefit:** multi-hop reentrancy A→B→A becomes detectable in 2 Phase 2b hops instead of
requiring CFG diameter ≥ 4. This is directly relevant to cross-function reentrancy patterns
that currently require large CFG graphs to be visible.

**Prerequisite:** run ablation study first. Train `gnn_layers=5` on the v5.2 deduped dataset
with Phase 2b disabled (identity pass) to establish that adding a layer doesn't hurt base
metrics. Then enable Phase 2b and measure Reentrancy F1 delta on the held-out test split.

---

## Implementation Priority Order

| Priority | Proposal | Rationale | Prerequisite |
|---|---|---|---|
| 1 | V6-2 DoS Augmentation | Highest ROI; no architecture changes; unblocks DoS floor | v5.2 behavioral gate pass |
| 2 | V6-3 MC Dropout UQ | Low implementation cost; high auditor trust value; no retraining | v5.2 checkpoint |
| 3 | V6-4 Explainability | Medium cost; directly requested by security teams; GATConv already computes attention | v5.2 checkpoint + node_metadata audit |
| 4 | V6-1 Multi-Contract GNN | High value, high cost; cross-contract attack coverage | Move 9 (multi-contract parsing) |
| 5 | V6-6 Hierarchical Attention | Ablation first; gnn_layers=5 smoke run; may not improve over v5.2 | v5.2 baseline metrics |
| 6 | V6-5 Dynamic Schema | Design-for-extensibility; long payoff horizon | V6-2 stable, no new features actively needed |

**Do not start V6 design work before:** v5.2 training completes, behavioral tests pass
(`ml/scripts/manual_test.py`), and all per-class F1 floors are exceeded (v4 fallback F1 − 0.05).
The v4 fallback remains active until those gates pass.

---

## Constraints & Non-Negotiables

The following v5 constants remain locked across all v6 proposals unless explicitly justified
and reviewed:

```
fusion_output_dim   = 128     ZKML proxy MLP Linear(128→64→32→10) depends on it
MAX_TOKEN_LENGTH    = 512     CodeBERT positional embedding limit
NUM_CLASSES         = 10      append-only; bump requires label migration
NODE_FEATURE_DIM    = 12      base schema; SchemaAdapter (V6-5) projects back to 12 before GNN
ARCHITECTURE base   = "v5_three_eye" (GNNEncoder + CrossAttentionFusion + three-eye classifier)
```

Any proposal that changes `fusion_output_dim=128` must include a ZKML proxy MLP retrain plan
and updated EZKL circuit setup. This is a high-cost breaking change.
