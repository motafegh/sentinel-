# SENTINEL — Module Technical Specifications

## How to Use This File

Each module section contains the full technical spec needed to implement it. Read only the section relevant to your current phase. Cross-references to SENTINEL-architecture.md for dependencies and interview stories.

---

## Module 1: Deep Learning Intelligence Core

### Goal
Master PyTorch, GNNs, and Transformers by building a smart contract vulnerability detector.

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Deep Learning | PyTorch | Advanced |
| Transformers | HuggingFace | Intermediate |
| Graph Neural Networks | PyTorch Geometric | Intermediate → Advanced |
| Multi-modal Fusion | Custom GMU | Advanced |
| Continual Learning | PyCIL | Intermediate (stretch) |

### Data Pipeline
```
Input: Solidity .sol file
  ▼ Validation: solc syntax check + version detection
  ▼ AST Extraction: py-solidity-ast

  ┌─── GNN PATH ──────────────────────────────────────┐
  │ Graph Builder                                      │
  │ Nodes: Functions, StateVars, Modifiers, Events     │
  │ Edges: CALLS, READS, WRITES, INHERITS, MODIFIES    │
  │ Node features (79-dim):                            │
  │   Type one-hot (8) + Visibility (4) +              │
  │   Mutability (3) + Code embedding (64)             │
  │ Output: PyTorch Geometric Data object              │
  └────────────────────────────────────────────────────┘

  ┌─── TRANSFORMER PATH ──────────────────────────────┐
  │ CodeBERT Tokenizer                                 │
  │ Max 512 tokens, truncate from END                  │
  │ >512 tokens: sliding window (512, stride 256)      │
  │   → run CodeBERT on each window                    │
  │   → max-pool over windows                          │
  │ Output: (batch, 512) token IDs                     │
  └────────────────────────────────────────────────────┘
```

### Model Architecture
```
GNN PATH                       TRANSFORMER PATH
DR-GCN Layer 1: 79 → 256      CodeBERT (12 layers, 768-dim)
ReLU + Dropout(0.3)            frozen or fine-tuned
DR-GCN Layer 2: 256 → 128     [CLS] token → 768-dim vector
ReLU + Dropout(0.3)
Global Mean Pool → 128-dim

              ↓                         ↓
         ┌────────────────────────────────────┐
         │  Gated Multimodal Unit (GMU)        │
         │  gate = σ(W · concat[128, 768])     │
         │  fused = gate⊙gnn + (1-gate)⊙trans  │
         │  output: 256-dim                    │
         └────────────────────────────────────┘
                          ↓
              Multi-Label Classifier
              256 → 128 → 64 → 13 (sigmoid)
              One output per vulnerability class
```

### Class Imbalance Stack
```python
# 1. Focal Loss (down-weights easy safe-contract examples)
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # class weights tensor shape (13,)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        return (focal_weight * bce).mean()

# 2. Class weights: weight[c] = total / (num_classes × count[c])
# safe: ~0.8 | reentrancy: ~12.0 | access_control: ~7.5

# 3. Stratified sampling
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset))

# 4. Class-specific thresholds at inference (tuned on val set)
# Rare classes: 0.3 threshold | Common classes: 0.6 threshold
# Store in configs/model_config.yaml, load at inference
```

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

### Continual Learning (stretch)
```
Monthly new data: Cyfrin Solodit + Immunefi + Rekt.news
  ↓ Evidently drift detection (>10% → trigger retraining)

Anti-forgetting:
  EWC: penalize changes to weights important for old tasks
       Loss = L_new + λ · Σ F_i · (θ_i - θ*_i)²
  Experience Replay: 500 exemplars per class (herding selection)
                     30% old + 70% new during retraining
```

### Data Sources
| Source | Size | Use |
|---|---|---|
| Kaggle Smart Contract Dataset | 35K | Base training |
| HuggingFace Contracts | 20K | Validation |
| Cyfrin Solodit | Ongoing | Continual learning |
| CodeBERT (microsoft/codebert-base) | — | Transfer learning |

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Model | CodeBERT fine-tuning only | + GNN path + GMU fusion |
| Training | Multi-label focal loss | + Continual learning (PyCIL) |
| Output | 13 vulnerability probabilities | + SHAP explanations |
| Tracking | Local checkpoints | + MLflow |

### Module 1 Fallbacks
```
GNN diverges (loss > 2.0 after 10 epochs):
  F1: GCN instead of DR-GCN (fewer hyperparameters)
  F2: Transformer-only path (80% of learning value)
  F3: Frozen CodeBERT + train classifier head only

VRAM overflow:
  F1: torch.utils.checkpoint.checkpoint()
  F2: Reduce batch size: 16 → 8 → 4
  F3: Train paths separately, freeze one
  F4: torch.cuda.amp.autocast() mixed precision

Accuracy < 70% after full training:
  S1: Confusion matrix — which classes are confused?
  S2: Check train/test data leakage
  S3: Add SolidiFI synthetic vulnerabilities
  S4: Binary classification first (vulnerable vs safe)

Training > 24 hours/epoch:
  F1: Google Colab Pro + A100 (~$50/mo, 8x faster)
  F2: 10K contracts instead of 35K
  F3: DistilCodeBERT (40% faster, 95% performance)
  F4: Pre-compute + cache CodeBERT embeddings to disk
```

---

## Module 2: ZKML — Zero-Knowledge Machine Learning

### Goal
Generate cryptographic proofs that ML predictions are correct, enabling trustless AI audits.

### Critical Constraint
```
Full SENTINEL model: ~100K params → NOT ZK-compatible
ZK-friendly limit:   <10K params

Solution: PROXY MODEL via knowledge distillation
  Full model (teacher) → Tiny proxy (student)
  Proxy architecture: Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)
  ~3K parameters | Input: 64 compressed features | Output: risk_score (0-100)
  Target: proxy agrees with full model ≥95% of the time
  Distillation loss: MSE(proxy_output, full_model_output)
```

### ONNX Export
```python
dummy_input = torch.randn(1, 64)  # must match exact inference shape
torch.onnx.export(
    proxy_model, dummy_input,
    "zkml/artifacts/proxy_model.onnx",
    input_names=["features"], output_names=["risk_score"],
    dynamic_axes={"features": {0: "batch"}, "risk_score": {0: "batch"}},
    opset_version=11,        # EZKL compatibility requirement
    do_constant_folding=True,
)
# Verify: onnx.checker.check_model(onnx.load("proxy_model.onnx"))
```

### EZKL Pipeline (6 Steps)
```
Step 1: ezkl.gen_settings(model.onnx)
  → settings.json (accuracy vs speed tradeoff config)

Step 2: ezkl.calibrate_settings(model.onnx, calibration_data.json)
  → calibrated settings (witness sizes + scale factors)

Step 3: ezkl.compile_model(model.onnx, settings.json)
  → model.compiled (ONNX → R1CS arithmetic circuit)

Step 4: ezkl.setup()   ← ONE TIME, EXPENSIVE (~minutes)
  → proving_key.pk  (~10MB, PRIVATE — prover only)
  → verification_key.vk  (~1KB, PUBLIC — verifier)

Step 5: ezkl.prove()   ← PER AUDIT (~30-60 seconds)
  inputs:  contract features (public) + model weights (witness, private) + pk
  outputs: proof π (~2KB) + publicSignals[risk_score]

Step 6: ezkl.verify()  ← ON-CHAIN or OFF-CHAIN
  inputs:  proof π + publicSignals + vk
  output:  true / false
  gas:     ~250K on-chain | ~50K with batch verification
```

### Solidity Verifier (auto-generated by EZKL)
```solidity
// ZKMLVerifier.sol — generated from verification_key.vk
// Deploy this contract, then reference from AuditRegistry
contract ZKMLVerifier {
    // Verification key embedded as constants (BN254 curve parameters)
    uint256 constant VK_ALPHA = 0x...;

    function verify(
        bytes calldata proof,
        uint256[] calldata publicSignals  // [0] = risk_score (0-100, scaled)
    ) public view returns (bool) {
        // Groth16 pairing check — assembly optimized
    }
}
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Proxy | 3-layer MLP, ~3K params | + knowledge distillation |
| EZKL | Local proof generation | + gas-optimized circuit |
| Verification | Off-chain only | + on-chain ZKMLVerifier |
| Performance | <2 min proof | + batch verification |

### Module 2 Fallbacks
```
EZKL install fails:
  F1: Official EZKL Docker image
  F2: Pre-built binaries from GitHub releases
  F3: Mock verifier — demonstrate concept, note limitation

Proof > 5 minutes:
  F1: Reduce proxy: 5K → 1K params, 3 → 2 layers
  F2: Lower precision (accept small accuracy loss)
  F3: Hash contract to 32 bytes instead of full features
  F4: CPU-only generation (slower but more stable)

On-chain gas > 1M:
  F1: Tune EZKL circuit parameters
  F2: Off-chain verify — store proof hash on-chain only
  F3: Batch verification (aggregate proofs)
  F4: Deploy on L2 (Arbitrum/Optimism — cheaper)

Proxy accuracy << full model:
  60-70% agreement acceptable for demonstration
  F1: Knowledge distillation (formal)
  F2: Ensemble of 3-5 tiny models voting
  F3: Better input feature selection
```

---

## Module 3: MLOps — Production ML Infrastructure

### Goal
Build production-grade ML infrastructure: experiment tracking, data versioning, pipeline orchestration, monitoring.

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Experiment Tracking | MLflow | Intermediate |
| Data Versioning | DVC | Intermediate |
| Pipeline Orchestration | Dagster | Basic |
| Drift Detection | Evidently AI | Intermediate |
| Feature Store | Feast | Basic (stretch) |
| Monitoring | Prometheus + Grafana | Basic |

### MLflow Integration
```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentinel-vulnerability-detection")

with mlflow.start_run(run_name="codebert-focal-v2"):
    mlflow.log_params({
        "model_type": "codebert", "learning_rate": 1e-4,
        "batch_size": 32, "gamma": 2.0, "epochs": 50
    })
    for epoch in range(epochs):
        mlflow.log_metrics({"train_loss": loss, "val_f1_macro": f1}, step=epoch)

    # Register to model registry — promote when val_f1 > 0.85
    mlflow.pytorch.log_model(model, "sentinel-model",
                             registered_model_name="SentinelVulnDetector")
```

### DVC Pipeline (dvc.yaml)
```yaml
stages:
  preprocess:
    cmd: python ml/src/data/preprocessing.py
    deps: [data/raw/, ml/src/data/preprocessing.py]
    outs: [data/processed/]

  train:
    cmd: python mlops/pipelines/scripts/train.py
    deps: [data/processed/, ml/configs/model_config.yaml]
    outs: [data/models/sentinel_v1.pt]
    metrics: [metrics/train_metrics.json]

  evaluate:
    cmd: python mlops/pipelines/scripts/evaluate.py
    deps: [data/models/sentinel_v1.pt, data/processed/]
    metrics: [metrics/eval_metrics.json]
```

### Drift Detection
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def check_drift(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    results = report.as_dict()
    drift_share = results["metrics"][0]["result"]["drift_share"]
    if drift_share > 0.10:   # >10% features drifted
        trigger_retraining()
        logger.warning(f"Drift detected: {drift_share:.1%} — retraining triggered")
    return drift_share
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Tracking | MLflow local | + Remote + model registry promotion |
| Versioning | DVC local storage | + Remote (S3/GCS) |
| Orchestration | Manual scripts + cron | + Dagster pipeline |
| Monitoring | MLflow dashboard | + Prometheus + Grafana |
| Feature Store | None | + Feast (online + offline) |

### Module 3 Fallbacks
```
MLflow port conflict: mlflow server --port 5001
Dagster too complex: simple train.py + evaluate.py scripts with cron
DVC remote costs: Google Drive (15GB free) or git LFS for <2GB
Feast not working: skip — it's stretch; use Redis directly
Evidently false positives: increase threshold 10% → 20%
```

---

## Module 4: AI Agents — LangChain, CrewAI, LangGraph

### Goal
Build a multi-agent system orchestrating static analysis, ML inference, RAG research, and report synthesis.

### Skills
| Skill | Tool | Depth |
|---|---|---|
| LLM Orchestration | LangChain | Intermediate |
| Multi-Agent Coordination | CrewAI | Intermediate |
| State Machines | LangGraph | Intermediate |
| RAG | FAISS + HuggingFace | Intermediate |
| Tool Integration | Custom tools | Advanced |

### Agent System Design
```
CrewAI Crew:
  Agent 1: Static Analyzer
    Tools: Slither, Mythril
    Task:  Run static analysis, find known patterns

  Agent 2: ML Intelligence
    Tools: SENTINEL inference API (Module 1)
    Task:  Get GNN+CodeBERT vulnerability probabilities

  Agent 3: Researcher (RAG)
    Tools: FAISS vector search
    Index: 1000+ exploits (Rekt.news + Immunefi + Solodit)
    Task:  Find similar past vulnerabilities for context

  Agent 4: Code Reviewer
    Tools: LLM reasoning
    Task:  Analyze business logic, access control, state management

  Agent 5: Synthesizer
    Input: Outputs from Agents 1-4
    Output: Structured audit report
           {risk_score, vulnerabilities[], severity[], remediation[]}
```

### RAG Pipeline
```
INDEXING (one-time setup):
  Sources → chunk (512 tokens, 20% overlap)
          → embed (code-aware embedding model)
          → store in FAISS index
          → save to agents/knowledge_base/faiss_index/

RETRIEVAL (per audit):
  Query: contract code + detected vulnerability type
  → dense retrieval: cosine similarity top-k
  → (stretch) BM25 keyword match
  → (stretch) cross-encoder reranker on top results
  → return 3-5 most relevant exploit examples
```

### LangGraph State Machine (stretch)
```python
from langgraph.graph import StateGraph

workflow = StateGraph(AuditState)
workflow.add_node("static", run_static_analysis)
workflow.add_node("ml", run_ml_inference)
workflow.add_node("rag", run_rag_research)
workflow.add_node("synthesis", synthesize_report)

# Conditional: high risk score → deeper review before synthesis
workflow.add_conditional_edges(
    "rag",
    lambda s: "deep_review" if s["risk_score"] > 70 else "synthesis"
)
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| LLM | Ollama + Llama 3 8B (free, local) | + GPT-4 / Claude for quality |
| Agent system | Single LangChain agent + 3 tools | + CrewAI 5-agent crew |
| RAG | FAISS + dense retrieval | + BM25 hybrid + reranker |
| Orchestration | LangChain LCEL | + LangGraph state machine |
| Observability | Logs | + LangSmith tracing |

### Module 4 Fallbacks
```
API costs too high: Ollama + Llama 3 8B (local, free)
RAG irrelevant results:
  F1: Semantic chunking (not fixed size)
  F2: BM25 + dense hybrid
  F3: Cross-encoder reranker
  F4: 50 curated examples > 10K noisy ones
CrewAI coordination fails:
  F1: Sequential pipeline (Agent1 → 2 → 3)
  F2: LangGraph (more control)
  F3: Single agent with all tools
FAISS memory overflow:
  F1: IndexIVFFlat (disk-based)
  F2: PCA: 768 → 384 dimensions
  F3: Shard by vulnerability category
  F4: Pinecone or Weaviate (managed, free tier)
```

---

## Module 5: Advanced Solidity & Foundry

### Goal
Master production Solidity: upgradeable contracts, tokenomics, ZK integration, and advanced testing.

### Skills
| Skill | Tool | Depth |
|---|---|---|
| Upgradeable Contracts | OpenZeppelin UUPS | Advanced |
| Token Design | ERC-20 + staking/slashing | Intermediate |
| ZK Verification | EZKL-generated verifier | Intermediate |
| Advanced Testing | Foundry fuzz + invariant | Advanced |
| Gas Optimization | Storage packing | Intermediate |
| Cross-Chain | Chainlink CCIP | Intermediate (stretch) |
| Account Abstraction | ERC-4337 | Intermediate (stretch) |

### Contract System
```
AuditRegistry.sol (UUPS upgradeable)           ← core contract
  • Registry: contractAddress → AuditResult
  • AuditResult: {score, zkProofHash, timestamp, verified}
  • Requires ZK proof verification before storing
  • Emits: AuditSubmitted(contractAddress, hash, agent, score)

SentinelToken.sol (ERC-20 + staking)
  • Agents stake tokens to participate (economic security)
  • Slashing: dishonest agents lose stake
  • Delegation: voting power for governance

ZKMLVerifier.sol (auto-generated by EZKL from vk)
  • Verification key embedded as Solidity constants
  • verify(proof, publicSignals) → bool
  • ~250K gas per verification

CrossChainOracle.sol — Chainlink CCIP (stretch)
AccountAbstractionPaymaster.sol — ERC-4337 (stretch)
GovernanceExecutor.sol — Timelock + Governor (stretch)
```

### AuditRegistry Core Pattern
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

contract AuditRegistry is UUPSUpgradeable, OwnableUpgradeable {
    struct AuditResult {
        uint8  vulnerabilityScore;   // 0-100
        bytes32 zkProofHash;         // IPFS hash of full proof
        uint256 timestamp;
        bool    verified;            // ZK proof passed
    }

    mapping(address => AuditResult) private _audits;
    mapping(address => uint256) private _agentStakes;
    uint256 public constant MIN_STAKE = 1 ether;

    event AuditSubmitted(address indexed contract_, bytes32 proofHash,
                         address indexed agent, uint8 score);

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() { _disableInitializers(); }

    function initialize() public initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
    }

    function submitAudit(
        address contractAddress,
        uint8 score,
        bytes calldata zkProof
    ) external {
        require(_agentStakes[msg.sender] >= MIN_STAKE, "Insufficient stake");
        require(score <= 100, "Score must be 0-100");
        require(IZKMLVerifier(verifierAddress).verify(zkProof, score), "Invalid proof");

        _audits[contractAddress] = AuditResult(score, keccak256(zkProof), block.timestamp, true);
        emit AuditSubmitted(contractAddress, keccak256(zkProof), msg.sender, score);
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}
}
```

### Foundry Testing Strategy
```
test/
├── unit/
│   AuditRegistry.t.sol:
│     test_submitAudit_validProof()
│     test_submitAudit_invalidProof_reverts()
│     test_submitAudit_insufficientStake_reverts()
│     test_getLatestAudit_returnsCorrectData()
│   SentinelToken.t.sol:
│     test_stake_minAmount() | test_unstake_afterLockPeriod()
│     test_slash_onInvalidAudit()
│   ZKMLVerifier.t.sol:
│     test_verify_validProof() | test_verify_invalidProof_returnsFalse()
│
├── integration/
│   FullFlow.t.sol:
│     test_fullAuditFlow_stakeSubmitVerify()
│     test_challengeFlow_stakeChallengeresolve()
│
├── fuzz/
│   FuzzAuditRegistry.t.sol:
│     testFuzz_submitAudit_anyScore(uint8 score)  ← catches score=101 etc.
│     testFuzz_stake_anyAmount(uint256 amount)
│
├── invariant/
│   InvariantStaking.t.sol:
│     invariant_totalStakedConsistent()  ← total == sum of all stakes, ALWAYS
│     invariant_allAuditsVerified()      ← no audit stored without valid proof
│
└── fork/ (stretch)
    test_interactWithRealCCIPRouter()

Gas profiling:
  forge snapshot           # baseline
  forge snapshot --diff    # before/after optimization
  forge test --gas-report  # per-function breakdown

Targets: submitAudit <100K | verify ~250K (ZK — hard to optimize)
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Contracts | AuditRegistry + SentinelToken + ZKMLVerifier | + CrossChainOracle + Paymaster + Governance |
| Testing | Unit + fuzz | + Invariant + fork |
| Deployment | Sepolia testnet | + Polygon + Arbitrum |
| Patterns | UUPS upgradeable | + ERC-4337 |

### Module 5 Fallbacks
```
UUPS upgrade fails on testnet:
  F1: Transparent proxy (simpler mechanism)
  F2: Non-upgradeable first, add upgradeability later
  F3: Hardhat instead of Foundry scripts

Chainlink CCIP unavailable on testnet:
  F1: Mock cross-chain (events + off-chain relayer)
  F2: LayerZero
  F3: Single-chain MVP (skip cross-chain entirely)

ERC-4337 paymaster gas issues:
  F1: Standard paymaster (simple gas sponsorship)
  F2: User pays gas normally — skip gasless for MVP
  F3: Biconomy or Pimlico (third-party)

Fuzz tests finding reverts:
  This is GOOD — fix the contracts!
  F1: Add require bounds validation
  F2: Bound fuzz inputs with vm.assume()
```

---

## Module 6: System Integration & Deployment

### Goal
Connect all modules into a running system: Docker Compose, CI/CD, API, and frontend demo.

### Skills
| Skill | Tool | Depth |
|---|---|---|
| API Design | FastAPI + Celery | Intermediate |
| Containerization | Docker Compose | Intermediate |
| CI/CD | GitHub Actions | Intermediate |
| Frontend | Streamlit (MVP) / Next.js (stretch) | Basic |
| Monitoring | Prometheus + Grafana | Basic |
| Deployment | Railway / Fly.io | Basic |

### Docker Compose (core services)
```yaml
services:
  api:
    build: ./api
    ports: ["8000:8000"]
    depends_on: [postgres, redis, ml-server]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

  ml-server:
    build: ./ml
    ports: ["8001:8001"]
    volumes: ["./ml/models:/app/models"]
    # Add GPU reservation when available

  agents:
    build: ./agents
    ports: ["8002:8002"]
    depends_on: [ollama, ml-server]

  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]

  celery-worker:
    build: ./api
    command: celery -A app.worker worker --loglevel=info
    depends_on: [redis, postgres]

  postgres:
    image: postgres:16
    environment: {POSTGRES_USER: sentinel, POSTGRES_PASSWORD: password, POSTGRES_DB: sentinel}

  redis:
    image: redis:7

  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]
```

### GitHub Actions CI/CD
```yaml
# .github/workflows/ci.yml
on:
  push: {branches: [main, develop]}
  pull_request: {branches: [main]}

jobs:
  test-ml:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: cd ml && pip install poetry && poetry install
      - run: cd ml && poetry run pytest tests/ -v --cov=src
      - run: cd ml && poetry run python scripts/evaluate.py --threshold 0.80

  test-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: {submodules: recursive}
      - uses: foundry-rs/foundry-toolchain@v1
      - run: cd contracts && forge test -vvv --gas-report
      - uses: crytic/slither-action@v0.3.0
        with: {target: contracts/}

  deploy:
    needs: [test-ml, test-contracts]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: {submodules: recursive}
      - uses: foundry-rs/foundry-toolchain@v1
      - run: |
          cd contracts
          forge script script/Deploy.s.sol \
            --rpc-url ${{ secrets.SEPOLIA_RPC }} --broadcast --verify
        env:
          PRIVATE_KEY: ${{ secrets.DEPLOYER_PRIVATE_KEY }}
          ETHERSCAN_API_KEY: ${{ secrets.ETHERSCAN_API_KEY }}
```

### API Route Design
```
POST /v1/audit
  Body: {contract_code: str} or {contract_address: str}
  Returns: {job_id: str}
  Behavior: Creates Celery task, returns immediately

GET /v1/audit/{job_id}
  Returns: {status: "pending|processing|complete", result?: AuditResult}

GET /v1/proof/{job_id}
  Returns: {proof: bytes, public_signals: list, on_chain_tx?: str}

GET /health
  Returns: {status: "ok", services: {ml: ok, agents: ok, db: ok}}

WS /ws/audit-stream/{job_id}    (stretch)
  Streams real-time progress updates during audit
```

### MVP vs Stretch
| Component | MVP | Stretch |
|---|---|---|
| Containers | Docker Compose core services | + GPU support + all services |
| API | REST + Celery async | + GraphQL + WebSocket |
| Frontend | Streamlit demo | + Next.js + RainbowKit |
| CI/CD | Test + build | + Auto-deploy + staging |
| Deployment | Local + Sepolia | + Railway/Fly.io production |

### Module 6 Fallbacks
```
Docker containers won't communicate:
  F1: docker network inspect sentinel_default
  F2: network_mode: host
  F3: Run services locally with tmux

API endpoints timing out:
  F1: Increase timeouts (30s → 120s for ML)
  F2: Redis caching for repeated contracts
  F3: Async endpoints with job polling

GitHub Actions CI failing:
  F1: Run tests locally first
  F2: Split into separate workflows per module
  F3: Disable @pytest.mark.slow in CI

Deployment fails:
  F1: Check logs — usually port or env var
  F2: Test Docker image locally first
  F3: DigitalOcean instead of PaaS
  F4: ngrok tunnel for local demo
```
