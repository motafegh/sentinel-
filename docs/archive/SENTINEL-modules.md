# SENTINEL — Module Technical Specifications

## How to Use This File

Architecture, design decisions, key constraints, and flows — not implementations. When implementing, generate code from these specs. Active module is set by the session handover — read that first.

For dependency map, phase order, interview stories → SENTINEL-architecture.md
For file paths, code standards, templates → code-standards.md

---

## Module Navigation

| Module                                                            | Phase            | Path                                         |
| ----------------------------------------------------------------- | ---------------- | -------------------------------------------- |
| [Module 1: ML Core](#module-1-deep-learning-intelligence-core)    | Phase 1          | `/home/motafeq/projects/sentinel/ml`         |
| [Module 2: ZKML](#module-2-zkml--zero-knowledge-machine-learning) | Phase 2          | `/home/motafeq/projects/sentinel/zkml/`      |
| [Module 3: MLOps](#module-3-mlops--model-operations)              | Phase 4          | `/home/motafeq/projects/sentinel/mlops/`     |
| [Module 4: Agents](#module-4-multi-agent-audit-system)            | Phase 3          | `/home/motafeq/projects/sentinel/agents/`    |
| [Module 5: Solidity](#module-5-solidity-contracts--tokenomics)    | Phase 1 parallel | `/home/motafeq/projects/sentinel/contracts/` |
| [Module 6: Integration](#module-6-system-integration--deployment) | Phase 4          | root                                         |

---

## Module 1: Deep Learning Intelligence Core

*No dependencies — builds first. Feeds Module 2 (distillation) and Module 4 (inference API).*

**Status: Complete — M3.4 inference API done. Production checkpoint: `run-alpha-tune_best.pt`.**

> **Note on original plan vs actual implementation:** This module was originally designed for
> multi-label classification (13 vulnerability classes, Gated Multimodal Unit fusion, DR-GCN
> layers, 79-dim node features). The implemented system differs in every architectural detail.
> The sections below document **what was actually built and is running in production**.

### Skills (As Implemented)
| Skill | Tool | Depth |
|---|---|---|
| Deep Learning | PyTorch | Advanced |
| Transformers | HuggingFace Transformers | Intermediate |
| Graph Neural Networks | PyTorch Geometric (GAT) | Intermediate |
| Multi-modal Fusion | MLP concat fusion | Intermediate |
| Experiment Tracking | MLflow | Intermediate |

### Data Source

**Primary dataset:** BCCC-SCsVul-2024 — 111,798 Solidity contracts across 12 vulnerability
categories. The dataset is genuinely multi-label: 41.2% of contracts appear in multiple
vulnerability folders simultaneously.

**Training data as processed:**
- 68,555 paired samples (graph `.pt` + token `.pt` files)
- Binary labels only: `1 = vulnerable`, `0 = safe`
- Multi-label signal from BCCC **was not preserved** — preprocessing picked one label per unique
  contract hash and collapsed to binary
- Distribution: 64.33% vulnerable / 35.67% safe

See [ML_DATASET_PIPELINE.md](ML_DATASET_PIPELINE.md) for the full explanation of why binary
was chosen and the upgrade path to multi-class.

### Data Pipeline Flow (Actual)

```
BCCC-SCsVul-2024/SourceCodes/<VulnType>/*.sol
      │
      ├──── Slither (AST extraction) ────► PyG Data(x=[N,8], edge_index=[2,E], y=[0or1])
      │                                          │
      │                                    GNNEncoder
      │                                    3×GAT layers
      │                                    global_mean_pool
      │                                          │ [B, 64]
      │                                          ▼
      └──── CodeBERT tokenizer ──────► [B, 512] input_ids
                                                │
                                         TransformerEncoder
                                         (frozen CodeBERT)
                                         CLS token [B, 768]
                                                │
                                           FusionLayer
                                     concat(832) → MLP(256→64)
                                                │ [B, 64]
                                                ▼
                                           Classifier
                                      Linear(64→1) → Sigmoid
                                                │ [B] ∈ [0,1]
                                                ▼
                                    threshold 0.50 → "vulnerable" / "safe"
```

### Model Architecture (Actual — LOCKED)

The architecture is frozen post-training. Changing any layer invalidates the EZKL circuit.

```
GNNEncoder:
  Input: [N, 8] — 8 float features per AST node
  GATConv(8→64, heads=8, concat=True)   → [N, 512]
  GATConv(512→64, heads=8, concat=True) → [N, 512]  (note: intermediate dim is 64*8=512)
  GATConv(512→64, heads=1, concat=False)→ [N, 64]
  global_mean_pool → [B, 64]
  Dropout p=0.2 between layers

TransformerEncoder:
  microsoft/codebert-base (RoBERTa-base, 12 layers, 125M params)
  Fully FROZEN — no gradient through CodeBERT
  CLS token from last hidden state → [B, 768]

FusionLayer:
  concat([B,64], [B,768]) → [B, 832]
  Linear(832→256) → ReLU → Dropout(0.3)
  Linear(256→64) → ReLU
  Output: [B, 64]

Classifier:
  Linear(64→1) → Sigmoid → squeeze(1) → [B] float ∈ [0,1]
```

**NOT what was originally planned:** The original spec called for DR-GCN (not GAT), a Gated
Multimodal Unit fusion (not MLP concat), 79-dim node features (not 8-dim), and a 13-class
multi-label output head (not binary). The current implementation is the result of iterative
simplification during development.

### Parameter Counts

| Component | Trainable | Frozen |
|---|---|---|
| GNNEncoder | 124,928 | 0 |
| TransformerEncoder | 0 | 124,645,632 |
| FusionLayer | 111,168 | 0 |
| Classifier | 65 | 0 |
| **Total** | **239,041** | **124,645,632** |

### Node Feature Vector (8-dim, float32)

| Index | Feature | Values |
|---|---|---|
| 0 | `type_id` | 0=STATE_VAR, 1=FUNCTION, 2=MODIFIER, 3=EVENT, 4=FALLBACK, 5=RECEIVE, 6=CONSTRUCTOR, 7=CONTRACT |
| 1 | `visibility` | 0=public/external, 1=internal, 2=private |
| 2 | `pure` | 1.0 if pure function |
| 3 | `view` | 1.0 if view function |
| 4 | `payable` | 1.0 if payable |
| 5 | `reentrant` | Slither's `is_reentrant` flag |
| 6 | `complexity` | `float(len(func.nodes))` — CFG node count |
| 7 | `loc` | `float(len(source_mapping.lines))` — lines of code |

This exact 8-dim vector was used to build all 68,555 training `.pt` files. **Any change requires
rebuilding the full dataset and retraining from scratch.**

### Classification Output

**Binary, not multi-label.** Output is a single sigmoid probability per contract.
- `>= 0.50` → "vulnerable"
- `< 0.50` → "safe"

The model cannot tell you *which* vulnerability type is present — only *whether* any exists.

### Loss Function: Binary Focal Loss

```python
focal_loss = alpha_t × (1 - pt)^gamma × BCE
  gamma = 2.0   # focus on hard examples
  alpha = 0.25  # weight for label=1 (vulnerable, majority); 0.75 for safe (minority)
```

`alpha=0.25` is confirmed correct — vulnerable is the majority class (64.33%) and must be
down-weighted. Do not change it.

### Training Results

| Run | Status | Val F1-macro | Threshold | Checkpoint |
|---|---|---|---|---|
| `baseline` | complete | 0.6515 | — | `sentinel_best.pt` (ep 16) |
| `run-alpha-tune` | **complete** | **0.6686** | **0.50** ✓ | `run-alpha-tune_best.pt` |
| `run-more-epochs` | killed ep25/40 | 0.6584 | pending | `run-more-epochs_best.pt` (ep 22) |
| `run-lr-lower` | never ran | — | — | — |

**Production checkpoint:** `run-alpha-tune_best.pt`

Threshold sweep on val set (10,283 samples):
```
Threshold |  F1-vuln | Precision |   Recall | F1-macro
---------------------------------------------------------
     0.45 |   0.7936 |    0.7160 |   0.8901 |   0.6294
     0.50 |   0.7458 |    0.7797 |   0.7147 |   0.6686  ← production
     0.55 |   0.6446 |    0.8543 |   0.5176 |   0.6325
```

### Pending Work (M3.4 completion)

1. Sweep `run-more-epochs_best.pt` — compare vs run-alpha-tune (0.6686)
2. Run final **test-set** evaluation on winner (10,284 never-touched samples)
3. Smoke test predictor on a real contract from `contracts/`

### Inference Pipeline

```python
predictor = Predictor(checkpoint="ml/checkpoints/run-alpha-tune_best.pt", threshold=0.50)
result = predictor.predict("contracts/MyContract.sol")
# {"score": 0.823, "label": "vulnerable", "threshold": 0.50, "truncated": False}
```

Or via FastAPI:
```bash
poetry run uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict -d '{"source_code": "..."}'
```

### Known Limitations

| Limitation | Impact |
|---|---|
| Binary output only | No per-vulnerability-type prediction |
| 41.2% of BCCC is multi-label but treated as single-label | Some labels may be wrong relative to ground truth |
| ~24K contracts excluded (Slither failures) | Training set biased toward parseable contracts |
| 512-token truncation | Long contracts partially analysed |
| F1-macro 0.6686 | ~28% false negative rate on validation set |
| Test set evaluation not yet run | Generalisation performance unconfirmed |

### Future Enhancements

| Enhancement | Effort | Benefit |
|---|---|---|
| 12-class classification | High (full dataset rebuild + new EZKL circuit) | Can identify which vulnerability type |
| SHAP explainability | Medium | Highlight which contract lines contributed to the score |
| CodeBERT fine-tuning (unfreeze) | High (10× memory + training time) | Better Solidity-specific semantics |
| Sliding window for long contracts | Medium | Reduce truncation impact |
| WeightedRandomSampler | Low | Better handling of rare vulnerability types in multi-class |

---

## Module 2: ZKML — Zero-Knowledge Machine Learning

*Depends on Module 1 trained model. Feeds Module 5 (ZKMLVerifier.sol).*

### Core Constraint
```
Full model: ~100K params → NOT ZK-compatible (limit: <10K params)

Solution: Knowledge distillation → proxy model
  Architecture: Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)
  Size: ~3K parameters
  Input: 64 compressed features from full model
  Output: risk_score (0-100)
  Target: proxy agrees with full model ≥95% of the time
  Loss: MSE(proxy_output, full_model_output)
```

### ONNX Export — Key Parameters
```
opset_version: 11          ← EZKL compatibility requirement, do not change
dynamic_axes: batch dim    ← required for variable batch sizes
do_constant_folding: True  ← reduces circuit size
dummy_input shape: (1, 64) ← must match exact inference shape
```

### EZKL Pipeline — 6 Steps
```
Step 1: gen_settings(model.onnx)
  → settings.json (accuracy vs speed tradeoff)

Step 2: calibrate_settings(model.onnx, calibration_data.json)
  → witness sizes + scale factors

Step 3: compile_model(model.onnx, settings.json)
  → model.compiled (ONNX → R1CS arithmetic circuit)

Step 4: setup()   ← ONE TIME, ~minutes, expensive
  → proving_key.pk   (~10MB, PRIVATE — prover only)
  → verification_key.vk  (~1KB, PUBLIC — embed in Solidity verifier)

Step 5: prove()   ← PER AUDIT, ~30-60 seconds
  inputs:  contract features (public) + model weights (witness, private) + pk
  outputs: proof π (~2KB) + publicSignals[risk_score]

Step 6: verify()  ← ON-CHAIN or OFF-CHAIN
  inputs:  proof π + publicSignals + vk
  output:  true / false
  gas:     ~250K on-chain | ~50K batch verification
```

### Solidity Verifier
- Auto-generated by EZKL from verification_key.vk
- Embeds BN254 curve parameters as constants
- Exposes: `verify(bytes proof, uint256[] publicSignals) → bool`
- Deploy this contract → reference address from AuditRegistry

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Proof system | Groth16 via EZKL | + PLONK (faster proving) |
| Proxy size | ~3K params | + up to 10K params |
| Verification | Off-chain + on-chain stub | + Full on-chain |
| Batching | Single proof | + Batch (~50K gas) |

### Fallbacks
| Problem | Action |
|---|---|
| Circuit too large | Reduce proxy → quantize int8 → use EZKL accuracy mode |
| Proving > 5 min | GPU acceleration → reduce calibration data → pre-generate + cache |
| On-chain gas > 500K | Batch verify → optimistic + challenge period → hash on-chain only |
| EZKL version conflict | Pin last working version → custom ZK circuit |

---

## Module 3: MLOps — Model Operations

*Depends on Module 1. Runs parallel to Module 1 training.*

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Experiment Tracking | MLflow | Intermediate |
| Data Versioning | DVC | Intermediate |
| Drift Detection | Evidently AI | Intermediate |
| Pipeline Orchestration | Dagster | Stretch |

### MLflow — What to Track
Every training run logs:
- **Params:** model_type, lr, batch_size, epochs, focal_gamma, class_weights
- **Metrics per epoch:** train_loss, val_f1_macro, val_f1_per_class
- **Artifacts:** model checkpoint, config YAML, model source file
- **Registry:** promote run to model registry when val_f1_macro > 0.85

### DVC Pipeline — 3 Stages
```
preprocess → train → evaluate

preprocess:
  in:  data/raw/contracts/
  out: data/processed/{train,val,test}.pkl

train:
  in:  data/processed/train.pkl + val.pkl + configs/model_config.yaml
  out: data/models/sentinel_model.pt
  metrics: reports/metrics.json

evaluate:
  in:  data/processed/test.pkl + data/models/sentinel_model.pt
  metrics: reports/evaluation.json
```

### Drift Detection Logic
- Tool: Evidently AI DataDriftPreset + ClassificationPreset
- Trigger: drift_share > 0.10 on incoming production contracts
- Action: log warning → call retraining pipeline
- Schedule: monthly or on accumulating 1K new contracts

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Tracking | MLflow local server | + Remote + model registry |
| Versioning | DVC local | + DVC remote (S3/GCS) |
| Drift | Manual Evidently reports | + Automated monthly |
| Orchestration | Manual scripts | + Dagster pipeline |

### Fallbacks
| Problem | Action |
|---|---|
| MLflow won't start | File-based tracking (./mlruns) → SQLite backend |
| DVC remote fails | Local remote (/tmp/dvc-storage) → metadata-only in git |
| Dagster too complex | Shell script + cron → GitHub Actions scheduled → manual |

---

## Module 4: Multi-Agent Audit System

*Depends on Module 1 inference API + FAISS index. Feeds Module 6.*

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Agent Framework | CrewAI | Intermediate |
| State Machines | LangGraph | Intermediate |
| RAG | LangChain + FAISS | Intermediate |
| LLM | Ollama (local) | Intermediate |

### 5-Agent Architecture
```
Input: .sol contract source
  ↓
Agent 1: Static Analyzer
  Tools: Slither, Mythril
  Output: {issues, severity_map}

Agent 2: ML Intelligence
  Tools: SentinelInferenceTool → calls Module 1 API
  Output: {vulnerability_probs, risk_score}

Agent 3: RAG Researcher
  Tools: FAISSRetrieverTool → 1000+ historical exploits
  Output: {similar_exploits, patterns}

Agent 4: Code Logic Reviewer
  Tools: ASTAnalysisTool, ControlFlowTool
  Output: {logic_issues, access_control_findings}

Agent 5: Synthesizer
  Input: all agent outputs
  Output: structured AuditReport
```

### LangGraph Routing
```
static_analysis → ml_assessment → rag_research
                                      ↓
                          risk_score > 70?
                          YES → logic_review → synthesis
                          NO  → synthesis → END
```

### RAG Design
- Index: FAISS over exploit write-ups (Rekt.news, Cyfrin, Immunefi)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Chunk size: 512, overlap: 64
- Retrieval: similarity_search_with_score, k=5
- Index location: `data/processed/exploit_index`

### LLM Configuration
- Local: Ollama llama3 (free, no API cost during dev)
- Port: 11434
- Fallback quality check: Claude API (~$0.01/audit, demo only)

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Agents | Single LangChain agent + 3 tools | + Full 5-agent CrewAI |
| RAG | 100 exploit examples | + 1000+ monthly updates |
| Routing | Linear pipeline | + LangGraph conditional |
| LLM | Ollama llama3 | + Quality gate with Claude API |

### Fallbacks
| Problem | Action |
|---|---|
| Ollama quality too low | Prompt engineering → Mistral/CodeLlama → Claude API for demo |
| CrewAI agents looping | max_iterations limit → linear pipeline → single agent all tools |
| FAISS memory error | Batch loading → ChromaDB (disk-based) → 100-example MVP |
| Slither/Mythril missing | pip install → Docker eth-security-toolbox → mock tool |

---

## Module 5: Solidity Contracts & Tokenomics

*Depends on Module 2 ZKMLVerifier.sol. Feeds Module 6.*

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Solidity Advanced | Foundry | Advanced |
| Proxy Patterns | OpenZeppelin UUPS | Advanced |
| Tokenomics | Custom staking/slashing | Intermediate |
| ZK Verification | EZKL-generated verifier | Intermediate |
| Testing | Foundry fuzz + invariant | Advanced |

### Contract Architecture
```
SentinelToken.sol (ERC-20 + staking)
  stake(amount)         → lock tokens, become eligible agent
  unstake(amount)       → unlock after lock period
  slash(agent, amount)  → penalize dishonest agent
  stakedBalance(agent)  → view staked amount

AuditRegistry.sol (UUPS upgradeable)
  submitAudit(contractAddress, score, zkProof, publicSignals)
    requires: stakedBalance(msg.sender) ≥ MIN_STAKE (1000 SENTINEL)
    requires: ZKMLVerifier.verify(proof, signals) == true
    stores:   AuditResult{score, proofHash, timestamp, agent, verified}
    emits:    AuditSubmitted(contractAddress, proofHash, agent, score)
  getLatestAudit(contractAddress) → AuditResult
  getAuditHistory(contractAddress) → AuditResult[]

ZKMLVerifier.sol (auto-generated by EZKL)
  verify(proof, publicSignals) → bool

CrossChainOracle.sol (stretch — Chainlink CCIP)
  syncAuditResult(destChain, contractAddress, auditId)
```

### Key Design Decisions
- **Storage:** store `keccak256(zkProof)` on-chain, not full proof bytes (gas)
- **Staking:** 1000 SENTINEL tokens minimum — economic security, not ETH
- **Upgradeability:** UUPS chosen over transparent proxy — gas efficient, Ali knows the pattern
- **Verifier:** reference ZKMLVerifier by interface (IZKMLVerifier) — swappable if EZKL updates
- **Initialization:** `_disableInitializers()` in constructor — standard UUPS safety

### Foundry Testing Strategy
```
tests/
├── unit/         happy path + error cases per contract
├── integration/  full flow: stake → submitAudit → verify
├── fuzz/         testFuzz_score(uint8 score) — catches score=101
│                 testFuzz_stake(uint256 amount)
├── invariant/    invariant_totalStakedConsistent() — always holds
│                 invariant_allAuditsVerified() — no audit without proof
└── fork/ (stretch) real CCIP router interaction

Gas targets:
  submitAudit < 100K gas
  ZK verify   ~ 250K gas (hard to optimize — ZK cost)
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Contracts | AuditRegistry + SentinelToken + ZKMLVerifier | + CrossChainOracle + Paymaster + Governance |
| Testing | Unit + fuzz | + Invariant + fork |
| Deployment | Sepolia testnet | + Polygon + Arbitrum |
| Patterns | UUPS | + ERC-4337 |

### Fallbacks
| Problem | Action |
|---|---|
| UUPS fails on testnet | Transparent proxy → non-upgradeable first → Hardhat scripts |
| CCIP unavailable | Mock cross-chain (events + relayer) → LayerZero → single-chain |
| ERC-4337 gas issues | Standard paymaster → user pays gas → Biconomy/Pimlico |
| Fuzz finds reverts | This is good — add require bounds + vm.assume() |

---

## Module 6: System Integration & Deployment

*Depends on all modules. Final phase.*

### Skills
| Skill | Tool | Depth |
|---|---|---|
| API Design | FastAPI + Celery | Intermediate |
| Containerization | Docker Compose | Intermediate |
| CI/CD | GitHub Actions | Intermediate |
| Frontend | Streamlit MVP / Next.js stretch | Basic |
| Monitoring | Prometheus + Grafana | Basic |

### API Routes
```
POST /v1/audit
  body: {contract_code: str} or {contract_address: str}
  returns: {job_id: str}
  behavior: creates Celery task, returns immediately

GET /v1/audit/{job_id}
  returns: {status: pending|processing|complete, result?: AuditResult}

GET /v1/proof/{job_id}
  returns: {proof: bytes, public_signals: list, on_chain_tx?: str}

GET /health
  returns: {status: ok, services: {ml: ok, agents: ok, db: ok}}

WS /ws/audit-stream/{job_id}   (stretch)
  streams real-time audit progress
```

### Docker Compose Services
```
api          → FastAPI, port 8000, depends: postgres + redis + ml-server
ml-server    → ML inference, port 8001
agents       → CrewAI, port 8002, depends: ollama + ml-server
ollama       → local LLM, port 11434
celery-worker → async task processing
postgres     → port 5432
redis        → port 6379
mlflow       → experiment tracking, port 5000
prometheus   → metrics, port 9090
grafana      → dashboards, port 3000, depends: prometheus
```

### CI/CD Pipeline Structure
```
On push to main/develop or PR to main:

test-ml:
  python 3.11 + poetry install
  pytest + coverage check (threshold: 80%)
  evaluate.py threshold check

test-contracts:
  foundry-toolchain
  forge test -vvv --gas-report
  slither static analysis

deploy (main only, after both tests pass):
  forge script Deploy.s.sol
  --rpc-url SEPOLIA_RPC --broadcast --verify
  secrets: DEPLOYER_PRIVATE_KEY, ETHERSCAN_API_KEY
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Containers | Core services (Docker Compose) | + GPU support + all services |
| API | REST + Celery async | + GraphQL + WebSocket |
| Frontend | Streamlit | + Next.js + RainbowKit |
| CI/CD | Test + build | + Auto-deploy + staging |
| Deployment | Local + Sepolia | + Railway/Fly.io |

### Fallbacks
| Problem | Action |
|---|---|
| Containers won't communicate | docker network inspect → network_mode: host → tmux local |
| API timeouts | increase timeouts (30s→120s) → Redis cache → async polling |
| CI failing | run locally first → split workflows → disable @pytest.mark.slow |
| Deployment fails | check logs → test image locally → DigitalOcean → ngrok tunnel |