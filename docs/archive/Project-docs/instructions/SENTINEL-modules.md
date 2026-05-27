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

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Deep Learning | PyTorch | Advanced |
| Transformers | HuggingFace | Intermediate |
| Graph Neural Networks | PyTorch Geometric | Intermediate → Advanced |
| Multi-modal Fusion | Custom GMU | Advanced |
| Continual Learning | PyCIL | Stretch |

### Data Pipeline Flow
```
Input: .sol file
  ↓ solc validation + version detection
  ↓ AST extraction (py-solidity-ast)

  GNN PATH                          TRANSFORMER PATH
  ─────────────────────             ──────────────────────
  Graph nodes: Functions,           CodeBERT tokenizer
  StateVars, Modifiers, Events      Max 512 tokens
                                    >512: sliding window
  Node features (79-dim):           (stride 256, max-pool
  Type one-hot (8)                  over windows)
  Visibility (4)
  Mutability (3)
  Code embedding (64)

  Output: PyG Data object           Output: (batch, 512) token IDs
```

### Model Architecture
```
GNN PATH (128-dim out)        TRANSFORMER PATH (768-dim out)
DR-GCN ×2 + ReLU + Dropout   CodeBERT [CLS] token
Global Mean Pool              Frozen or fine-tuned

              ↓                         ↓
       Gated Multimodal Unit (GMU)
       gate = σ(W · concat[128, 768])
       fused = gate⊙gnn + (1-gate)⊙transformer
       output: 256-dim
              ↓
       Multi-Label Classifier
       256 → 128 → 64 → 13 (sigmoid)
       One output per vulnerability class
```

### Class Imbalance Strategy
Three techniques stacked in order:
1. **Focal Loss** — down-weights easy safe-contract examples (gamma=2.0)
2. **Class weights** — `weight[c] = total / (num_classes × count[c])` — reentrancy ~12x safe
3. **WeightedRandomSampler** — rare classes appear in every batch
4. **Inference thresholds** — class-specific, tuned on val set (rare: ~0.3, common: ~0.6), stored in config

### 13 Vulnerability Classes
| # | Class | Severity | Dataset % |
|---|---|---|---|
| 1 | Reentrancy | Critical | ~5% |
| 2 | Integer Overflow/Underflow | High | ~12% |
| 3 | Access Control | High | ~8% |
| 4 | Unchecked Return Values | Medium | ~6% |
| 5 | Front-running | Medium | ~4% |
| 6 | Timestamp Dependence | Low-Medium | ~3% |
| 7 | Denial of Service | Medium | ~3% |
| 8 | Logic Errors | High | ~5% |
| 9 | Uninitialized Storage | High | ~2% |
| 10 | Delegatecall Issues | Critical | ~2% |
| 11 | Flash Loan Attacks | Critical | ~2% |
| 12 | Oracle Manipulation | High | ~2% |
| 13 | Safe (no vulnerability) | — | ~60% |

### Data Sources
| Source | Size | Use |
|---|---|---|
| Kaggle Smart Contract Dataset | 35K | Base training |
| HuggingFace Contracts | 20K | Validation |
| Cyfrin Solodit | Ongoing | Continual learning |
| CodeBERT (microsoft/codebert-base) | — | Transfer learning |

### Continual Learning Design (stretch)
- Monthly new data from: Cyfrin Solodit, Immunefi, Rekt.news
- Drift trigger: Evidently >10% feature drift → retraining
- Anti-forgetting: EWC (weight penalty) + Experience Replay (500 exemplars/class, 30% old / 70% new)

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Model | CodeBERT fine-tuning only | + GNN path + GMU fusion |
| Training | Multi-label focal loss | + Continual learning (PyCIL) |
| Output | 13 vulnerability probabilities | + SHAP explanations |
| Tracking | Local checkpoints | + MLflow |

### Fallbacks
| Problem | Action |
|---|---|
| GNN loss > 2.0 after 10 epochs | GCN → Transformer-only → Frozen CodeBERT + classifier head |
| VRAM overflow | gradient checkpointing → smaller batch → separate paths → mixed precision |
| Accuracy < 70% | Confusion matrix analysis → check data leakage → SolidiFI synthetics → binary first |
| Training > 24h/epoch | Colab Pro A100 → 10K subset → DistilCodeBERT → pre-cache embeddings |

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