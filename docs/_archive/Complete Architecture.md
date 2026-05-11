
## Complete Architecture & Implementation Guide 

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Philosophy](#project-philosophy)
3. [Project Structure](#project-structure)
4. [System Architecture Overview](#system-architecture-overview)
5. [Module 1: Deep Learning Intelligence Core](#module-1-deep-learning-intelligence-core)
6. [Module 2: ZKML - Zero-Knowledge Machine Learning](#module-2-zkml---zero-knowledge-machine-learning)
7. [Module 3: MLOps - Production ML Infrastructure](#module-3-mlops---production-ml-infrastructure)
8. [Module 4: AI Agents - LangChain, CrewAI, LangGraph](#module-4-ai-agents---langchain-crewai-langgraph)
9. [Module 5: Advanced Solidity & Foundry](#module-5-advanced-solidity--foundry)
10. [Module 6: System Integration & Deployment](#module-6-system-integration--deployment)
11. [Learning Phases & Milestones](#learning-phases--milestones)
12. [Fallback Plans](#fallback-plans)
13. [Skill Coverage Matrix](#skill-coverage-matrix)

---

## Executive Summary

**SENTINEL** is a decentralized platform where AI agents autonomously audit smart contracts, generate zero-knowledge proofs of their analysis, coordinate via multi-agent systems, and serve results through blockchain oracles—all while continuously learning from new exploits.

### Core Value Proposition

|Stakeholder|Problem Solved|
|---|---|
|**DeFi Protocols**|Trustless, verifiable security audits without revealing audit methodology|
|**Developers**|Instant vulnerability detection before deployment|
|**Insurance Providers**|Cryptographic proof that contract was audited by certified AI model|
|**Security Researchers**|Platform for contributing to and learning from exploit database|

### Key Differentiators

1. **Verifiable AI**: Zero-knowledge proofs ensure model predictions are correct without revealing model weights
2. **Multi-Modal Analysis**: GNN (structure) + Transformer (semantics) dual-path architecture
3. **Continuous Learning**: Model updates as new vulnerabilities emerge, with drift detection
4. **Cross-Chain**: Audit results synced across Ethereum, Polygon, Arbitrum via CCIP
5. **Gasless UX**: ERC-4337 Account Abstraction enables audits without ETH

---

## Project Philosophy

### "The Swiss Army Knife Approach"

Instead of one monolithic project, SENTINEL is designed as **one integrated system with 6 learning modules**. Each module is a standalone skill cluster that feeds into the final system.

**Think of it like building Iron Man's suit**: Each component (arc reactor, repulsor, AI system, flight stabilizers) can be built separately, tested independently, then integrated.

### MVP + Stretch Goals Structure

Each module has two tiers:

|Tier|Purpose|Completion Criteria|
|---|---|---|
|**MVP**|Core learning, demo-able output|Must complete to move forward|
|**Stretch**|Advanced features, production polish|Complete if time allows|

### Learning Principles

1. **Exposure > Perfection**: See and understand concepts even if implementation isn't perfect
2. **Parallel Progress**: Work on complementary modules simultaneously (ML training + Solidity contracts)
3. **Fallback Safety**: Every risky component has a simpler alternative
4. **AI-Assisted Development**: Leverage Claude/AI tools for boilerplate, focus human effort on architecture decisions

---

## Project Structure

```
sentinel/
│
├── README.md                           # Project overview & quick start
├── ARCHITECTURE.md                     # This document
├── docker-compose.yml                  # Full system orchestration
├── docker-compose.dev.yml              # Development environment
├── .env.example                        # Environment variables template
├── Makefile                            # Common commands
│
├── ml/                                 # MODULE 1: Deep Learning
│   ├── README.md
│   ├── pyproject.toml                  # Python dependencies (Poetry)
│   ├── data/
│   │   ├── raw/                        # Original datasets (DVC tracked)
│   │   ├── processed/                  # Preprocessed data
│   │   └── embeddings/                 # Pre-computed embeddings
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py              # PyTorch Dataset classes
│   │   │   ├── preprocessing.py        # Contract preprocessing
│   │   │   ├── graph_builder.py        # AST → Graph conversion
│   │   │   └── tokenizer.py            # CodeBERT tokenization
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gnn.py                  # Graph Neural Network
│   │   │   ├── transformer.py          # CodeBERT wrapper
│   │   │   ├── fusion.py               # Gated Multimodal Unit
│   │   │   └── classifier.py           # Final classification head
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py              # Training loop
│   │   │   ├── losses.py               # Focal loss, class weights
│   │   │   └── metrics.py              # Multi-label metrics
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py            # Single contract prediction
│   │   │   └── batch_predictor.py      # Batch inference
│   │   └── continual/
│   │       ├── __init__.py
│   │       ├── ewc.py                  # Elastic Weight Consolidation
│   │       └── replay.py               # Experience replay buffer
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_model_experiments.ipynb
│   │   └── 03_error_analysis.ipynb
│   ├── tests/
│   │   ├── test_preprocessing.py
│   │   ├── test_models.py
│   │   └── test_training.py
│   └── configs/
│       ├── model_config.yaml
│       └── training_config.yaml
│
├── zkml/                               # MODULE 2: Zero-Knowledge ML
│   ├── README.md
│   ├── pyproject.toml
│   ├── models/
│   │   ├── proxy_model.py              # Tiny MLP for ZK proofs
│   │   └── export_onnx.py              # PyTorch → ONNX conversion
│   ├── ezkl/
│   │   ├── generate_settings.py        # EZKL configuration
│   │   ├── compile_circuit.py          # ONNX → ZK circuit
│   │   ├── generate_proof.py           # Proof generation
│   │   └── verify_proof.py             # Local verification
│   ├── artifacts/                      # Generated keys, proofs
│   │   ├── proving_key.pk
│   │   ├── verification_key.vk
│   │   └── proofs/
│   └── tests/
│       └── test_zkml_pipeline.py
│
├── mlops/                              # MODULE 3: MLOps Infrastructure
│   ├── README.md
│   ├── pipelines/
│   │   ├── dagster/
│   │   │   ├── __init__.py
│   │   │   ├── assets.py               # Dagster assets
│   │   │   ├── jobs.py                 # Training jobs
│   │   │   └── schedules.py            # Automated retraining
│   │   └── scripts/
│   │       ├── train.py                # Manual training script
│   │       └── evaluate.py             # Evaluation script
│   ├── feature_store/
│   │   ├── feast/
│   │   │   ├── feature_store.yaml      # Feast configuration
│   │   │   └── features.py             # Feature definitions
│   │   └── redis/
│   │       └── docker-compose.yml      # Redis for online store
│   ├── monitoring/
│   │   ├── evidently/
│   │   │   ├── drift_detection.py      # Data/concept drift
│   │   │   └── reports/                # Generated reports
│   │   ├── prometheus/
│   │   │   └── prometheus.yml          # Metrics configuration
│   │   └── grafana/
│   │       └── dashboards/             # Grafana dashboard JSON
│   ├── mlflow/
│   │   ├── Dockerfile
│   │   └── mlflow_config.py
│   └── dvc/
│       ├── .dvc/                       # DVC configuration
│       └── dvc.yaml                    # DVC pipeline stages
│
├── agents/                             # MODULE 4: AI Agents
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── static_analyzer.py      # Agent 1: Slither/Mythril
│   │   │   ├── ml_intelligence.py      # Agent 2: ML model
│   │   │   ├── researcher.py           # Agent 3: Historical search
│   │   │   ├── code_reviewer.py        # Agent 4: Logic review
│   │   │   └── synthesizer.py          # Agent 5: Report generation
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── slither_tool.py         # Slither integration
│   │   │   ├── ml_tool.py              # ML model inference
│   │   │   ├── vector_search.py        # FAISS similarity search
│   │   │   └── web_search.py           # Tavily API
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── indexer.py              # Document indexing
│   │   │   ├── retriever.py            # Hybrid retrieval
│   │   │   └── reranker.py             # Cross-encoder reranking
│   │   ├── orchestration/
│   │   │   ├── __init__.py
│   │   │   ├── crew.py                 # CrewAI setup
│   │   │   └── graph.py                # LangGraph state machine
│   │   └── prompts/
│   │       ├── system_prompts.py       # Agent system prompts
│   │       └── few_shot_examples.py    # Few-shot examples
│   ├── knowledge_base/
│   │   ├── exploits/                   # Rekt.news, Immunefi data
│   │   ├── best_practices/             # OpenZeppelin docs
│   │   └── faiss_index/                # Vector store
│   └── tests/
│       ├── test_tools.py
│       └── test_agents.py
│
├── contracts/                          # MODULE 5: Solidity
│   ├── README.md
│   ├── foundry.toml                    # Foundry configuration
│   ├── remappings.txt                  # Import remappings
│   ├── src/
│   │   ├── AuditRegistry.sol           # Main registry (UUPS)
│   │   ├── SentinelToken.sol           # Governance token (ERC-20)
│   │   ├── ZKMLVerifier.sol            # EZKL-generated verifier
│   │   ├── CrossChainOracle.sol        # Chainlink CCIP
│   │   ├── AccountAbstractionPaymaster.sol  # ERC-4337
│   │   ├── GovernanceExecutor.sol      # Timelock + Governance
│   │   └── interfaces/
│   │       ├── IAuditRegistry.sol
│   │       └── ISentinelToken.sol
│   ├── test/
│   │   ├── unit/
│   │   │   ├── AuditRegistry.t.sol
│   │   │   ├── SentinelToken.t.sol
│   │   │   └── ZKMLVerifier.t.sol
│   │   ├── integration/
│   │   │   └── FullFlow.t.sol
│   │   ├── fuzz/
│   │   │   └── FuzzAuditRegistry.t.sol
│   │   └── invariant/
│   │       └── InvariantStaking.t.sol
│   ├── script/
│   │   ├── Deploy.s.sol                # Deployment script
│   │   ├── Upgrade.s.sol               # Upgrade script
│   │   └── HelperConfig.s.sol          # Network configs
│   └── lib/                            # Dependencies (forge install)
│
├── api/                                # MODULE 6: Backend API
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── audit.py                # /v1/audit endpoints
│   │   │   ├── proof.py                # /v1/proof endpoints
│   │   │   └── health.py               # Health checks
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── audit_service.py        # Audit orchestration
│   │   │   ├── ml_service.py           # ML model calls
│   │   │   └── blockchain_service.py   # Web3 interactions
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── request.py              # Pydantic request models
│   │   │   └── response.py             # Pydantic response models
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                 # JWT authentication
│   │   │   └── rate_limit.py           # Rate limiting
│   │   └── config/
│   │       └── settings.py             # Configuration
│   ├── tests/
│   │   ├── test_routes.py
│   │   └── test_services.py
│   └── Dockerfile
│
├── frontend/                           # MODULE 6: Frontend
│   ├── README.md
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx                # Landing page
│   │   │   ├── audit/
│   │   │   │   └── page.tsx            # Audit interface
│   │   │   ├── verify/
│   │   │   │   └── page.tsx            # ZK proof verification
│   │   │   └── governance/
│   │   │       └── page.tsx            # DAO voting
│   │   ├── components/
│   │   │   ├── WalletConnect.tsx       # RainbowKit
│   │   │   ├── AuditForm.tsx           # Contract upload
│   │   │   ├── VulnerabilityCard.tsx   # Result display
│   │   │   └── ProofVerifier.tsx       # ZK verification
│   │   ├── hooks/
│   │   │   ├── useAudit.ts
│   │   │   └── useContract.ts
│   │   └── lib/
│   │       ├── wagmi.ts                # Wagmi config
│   │       └── api.ts                  # API client
│   └── Dockerfile
│
├── infra/                              # Infrastructure
│   ├── kubernetes/
│   │   ├── deployments/
│   │   ├── services/
│   │   └── configmaps/
│   └── terraform/                      # Cloud provisioning
│
├── scripts/                            # Utility scripts
│   ├── setup_dev.sh                    # Dev environment setup
│   ├── download_datasets.sh            # Dataset download
│   └── run_tests.sh                    # Full test suite
│
├── docs/                               # Documentation
│   ├── api/                            # API documentation
│   ├── architecture/                   # Architecture diagrams
│   └── guides/                         # User guides
│
└── .github/
    └── workflows/
        ├── ml-tests.yml                # ML pipeline CI
        ├── contract-tests.yml          # Foundry CI
        ├── agent-tests.yml             # Agent tests CI
        └── deploy.yml                  # Deployment CD
```

---

## System Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  1. User uploads Solidity contract (.sol file or on-chain address)      ││
│  │  2. User connects wallet (MetaMask/WalletConnect via RainbowKit)        ││
│  │  3. User requests audit (pays in SENTINEL tokens or gasless via AA)     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Wallet      │  │ Contract    │  │ Results     │  │ Governance          │ │
│  │ Connection  │  │ Upload      │  │ Dashboard   │  │ (Vote/Propose)      │ │
│  │ RainbowKit  │  │ Form        │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ HTTPS (REST/GraphQL/WebSocket)
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY (FastAPI)                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ POST /v1/audit  │  │ GET /v1/proof   │  │ WebSocket /ws/audit-stream  │  │
│  │ Submit contract │  │ Get ZK proof    │  │ Real-time audit progress    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
┌───────────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐
│   ASYNC WORKERS       │  │   AI AGENT SYSTEM │  │   ML INFERENCE ENGINE     │
│   (Celery + Redis)    │  │   (CrewAI)        │  │                           │
│                       │  │                   │  │  ┌─────────────────────┐  │
│ • Batch audit jobs    │  │ • Multi-agent     │  │  │ Preprocessing       │  │
│ • Report generation   │  │   orchestration   │  │  │ • Tokenization      │  │
│ • Scheduled retrain   │  │ • RAG retrieval   │  │  │ • Graph building    │  │
│                       │  │ • Tool calling    │  │  └─────────────────────┘  │
│                       │  │                   │  │           │               │
│                       │  │  Agents:          │  │           ▼               │
│                       │  │  1. Static Anal.  │  │  ┌─────────────────────┐  │
│                       │  │  2. ML Intel.     │  │  │ Dual-Path Model     │  │
│                       │  │  3. Researcher    │  │  │ ┌───────┐ ┌───────┐ │  │
│                       │  │  4. Code Review   │  │  │ │ GNN   │ │CodeBERT│ │  │
│                       │  │  5. Synthesizer   │  │  │ └───┬───┘ └───┬───┘ │  │
│                       │  │                   │  │  │     └────┬────┘     │  │
│                       │  │                   │  │  │          ▼         │  │
│                       │  │                   │  │  │   ┌───────────┐    │  │
│                       │  │                   │  │  │   │GMU Fusion │    │  │
│                       │  │                   │  │  │   └─────┬─────┘    │  │
│                       │  │                   │  │  │         ▼         │  │
│                       │  │                   │  │  │   ┌───────────┐    │  │
│                       │  │                   │  │  │   │Classifier │    │  │
│                       │  │                   │  │  │   │(13 vulns) │    │  │
│                       │  │                   │  │  │   └───────────┘    │  │
│                       │  │                   │  │  └─────────────────────┘  │
└───────────────────────┘  └───────────────────┘  └───────────────────────────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ZKML PROVER (EZKL)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  1. Take model prediction + inputs                                       ││
│  │  2. Run inference in ZK circuit (tiny proxy model)                      ││
│  │  3. Generate ZK-SNARK proof (π)                                          ││
│  │  4. Output: Proof + Public signals (vulnerability scores)               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BLOCKCHAIN LAYER                                   │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ AuditRegistry   │  │ SentinelToken   │  │ ZKMLVerifier                │  │
│  │ (UUPS Proxy)    │  │ (ERC-20 + Votes)│  │ (EZKL-generated)            │  │
│  │                 │  │                 │  │                             │  │
│  │ • Store audits  │  │ • Governance    │  │ • verify(proof, signals)    │  │
│  │ • Risk scores   │  │ • Staking       │  │ • Returns: true/false       │  │
│  │ • ZK proof hash │  │ • Rewards       │  │ • ~250K gas                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ CrossChainOracle│  │ Paymaster       │  │ GovernanceExecutor          │  │
│  │ (Chainlink CCIP)│  │ (ERC-4337)      │  │ (Timelock)                  │  │
│  │                 │  │                 │  │                             │  │
│  │ • Sync audits   │  │ • Gasless UX    │  │ • Proposal voting           │  │
│  │ • Multi-chain   │  │ • Sponsored tx  │  │ • Upgrades                  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                              │
│  Deployed on: Ethereum Sepolia, Polygon Mumbai, Arbitrum Sepolia            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MLOps LAYER                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ MLflow      │  │ DVC         │  │ Dagster     │  │ Evidently AI        │ │
│  │             │  │             │  │             │  │                     │ │
│  │ • Experiment│  │ • Data      │  │ • Pipeline  │  │ • Data drift        │ │
│  │   tracking  │  │   versioning│  │   orchestr. │  │ • Concept drift     │ │
│  │ • Model     │  │ • Git for   │  │ • Scheduled │  │ • Prediction drift  │ │
│  │   registry  │  │   data      │  │   retraining│  │ • Alert triggers    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐                                           │
│  │ Feast       │  │ Prometheus  │                                           │
│  │             │  │ + Grafana   │                                           │
│  │ • Feature   │  │             │                                           │
│  │   store     │  │ • Metrics   │                                           │
│  │ • Online/   │  │ • Dashboards│                                           │
│  │   Offline   │  │ • Alerts    │                                           │
│  └─────────────┘  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### End-to-End Audit Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SENTINEL AUDIT FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Step 1: CONTRACT SUBMISSION
═══════════════════════════
User uploads contract.sol
         │
         ▼
┌─────────────────────────┐
│ Preprocessing Pipeline  │
│ ├─ Parse with solc      │
│ ├─ Extract AST          │
│ ├─ Build control flow   │
│ └─ Validate syntax      │
└───────────┬─────────────┘
            │
            ▼
Step 2: DUAL-PATH ANALYSIS
══════════════════════════
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌─────────┐   ┌─────────────┐
│  GNN    │   │ Transformer │
│  Path   │   │    Path     │
├─────────┤   ├─────────────┤
│ AST →   │   │ Code →      │
│ Graph   │   │ Tokens      │
│   │     │   │     │       │
│   ▼     │   │     ▼       │
│ DR-GCN  │   │ CodeBERT    │
│   │     │   │     │       │
│   ▼     │   │     ▼       │
│ 128-dim │   │ 768-dim     │
│ embed   │   │ embed       │
└────┬────┘   └──────┬──────┘
     │               │
     └───────┬───────┘
             │
             ▼
Step 3: GATED FUSION
════════════════════
┌─────────────────────────────┐
│   Gated Multimodal Unit     │
│                             │
│   gate = σ(W·[gnn;trans])   │
│   fused = g·gnn + (1-g)·trans│
│                             │
│   Output: 256-dim vector    │
└───────────────┬─────────────┘
                │
                ▼
Step 4: CLASSIFICATION
══════════════════════
┌─────────────────────────────┐
│   Multi-Label Classifier    │
│                             │
│   256 → 64 → 13 (sigmoid)   │
│                             │
│   Focal Loss (γ=2)          │
│   Class-specific thresholds │
│                             │
│   Output: 13 probabilities  │
│   [reentrancy: 0.87,        │
│    overflow: 0.12,          │
│    access_control: 0.45,    │
│    ...]                     │
└───────────────┬─────────────┘
                │
                ▼
Step 5: AGENT ANALYSIS (Parallel)
═════════════════════════════════
┌─────────────────────────────────────────────────────────────────────┐
│                        CrewAI Orchestration                          │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent 4  │            │
│  │ Static   │  │ ML       │  │ Research │  │ Code     │            │
│  │ Analyzer │  │ Intel.   │  │ (RAG)    │  │ Reviewer │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       ▼             ▼             ▼             ▼                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Slither  │  │ GNN+BERT │  │ FAISS    │  │ Logic    │            │
│  │ Mythril  │  │ Inference│  │ Search   │  │ Analysis │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       └─────────────┴──────┬──────┴─────────────┘                   │
│                            │                                         │
│                            ▼                                         │
│                    ┌──────────────┐                                  │
│                    │  Agent 5     │                                  │
│                    │  Synthesizer │                                  │
│                    │              │                                  │
│                    │  Combine all │                                  │
│                    │  findings    │                                  │
│                    └──────┬───────┘                                  │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                            ▼
Step 6: ZK PROOF GENERATION
═══════════════════════════
┌─────────────────────────────┐
│   EZKL Pipeline             │
│                             │
│   1. Tiny proxy model       │
│      (3-layer MLP, <5K p.)  │
│                             │
│   2. Input: Contract feats  │
│                             │
│   3. ZK Circuit execution   │
│                             │
│   4. Output:                │
│      • Proof π              │
│      • Public: risk_score   │
│                             │
│   Time: ~30-60 seconds      │
└───────────────┬─────────────┘
                │
                ▼
Step 7: ON-CHAIN SUBMISSION
═══════════════════════════
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  1. ZKMLVerifier.verify(proof, publicSignals) → true                │
│                                                                      │
│  2. AuditRegistry.submitAudit({                                     │
│       contractAddress: 0x...,                                       │
│       riskScore: 87,                                                │
│       vulnerabilities: [REENTRANCY, ACCESS_CONTROL],                │
│       zkProofHash: ipfs://Qm...,                                    │
│       modelVersion: "SENTINEL-v2.1"                                 │
│     })                                                              │
│                                                                      │
│  3. Emit AuditSubmitted event                                       │
│                                                                      │
│  4. (Optional) CrossChainOracle syncs to Polygon/Arbitrum via CCIP  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
Step 8: RESULT DELIVERY
═══════════════════════
┌─────────────────────────────┐
│   User receives:            │
│                             │
│   • Vulnerability report    │
│   • Risk score (0-100)      │
│   • On-chain proof link     │
│   • Remediation suggestions │
│   • Historical comparisons  │
│   • PDF/Markdown export     │
└─────────────────────────────┘
```

---

## Module 1: Deep Learning Intelligence Core

### Goal

Master PyTorch, GNNs, Transformers, and continual learning by building a state-of-the-art smart contract vulnerability detector.

### Skills Covered

|Skill|Tool/Framework|Depth|
|---|---|---|
|Deep Learning|PyTorch|Advanced|
|Graph Neural Networks|PyTorch Geometric|Intermediate→Advanced|
|Transformers|Hugging Face Transformers|Intermediate|
|Multi-modal Fusion|Custom (GMU)|Advanced|
|Continual Learning|PyCIL|Intermediate|
|GPU Training|CUDA|Intermediate|

### Technical Architecture

#### Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Input: Solidity Source Code (.sol file)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PREPROCESSING                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Validation                                                    │   │
│  │ • Syntax check (solc compiler)                                        │   │
│  │ • Version detection (pragma solidity ^0.8.0)                          │   │
│  │ • Import resolution                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: AST Extraction (py-solidity-ast)                             │   │
│  │                                                                       │   │
│  │ contract MyContract {           AST:                                  │   │
│  │   uint256 balance;        →     ├─ ContractDefinition                │   │
│  │   function withdraw() {         │  ├─ StateVariableDeclaration      │   │
│  │     msg.sender.call{value:      │  └─ FunctionDefinition            │   │
│  │       balance}("");             │     ├─ MemberAccess               │   │
│  │   }                             │     └─ FunctionCall               │   │
│  │ }                               │                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│              ┌───────────────┴───────────────┐                              │
│              │                               │                              │
│              ▼                               ▼                              │
│  ┌─────────────────────────┐   ┌─────────────────────────────────────┐     │
│  │  GNN PATH               │   │  TRANSFORMER PATH                    │     │
│  │                         │   │                                      │     │
│  │  Graph Builder:         │   │  Tokenizer:                         │     │
│  │                         │   │                                      │     │
│  │  Nodes:                 │   │  CodeBERT Tokenizer                 │     │
│  │  • Functions            │   │  • Max length: 512 tokens           │     │
│  │  • State variables      │   │  • Truncation: from END             │     │
│  │  • Modifiers            │   │  • Special tokens: [CLS], [SEP]     │     │
│  │  • Events               │   │                                      │     │
│  │                         │   │  Variable-Length Strategy:          │     │
│  │  Edges:                 │   │  ┌─────────────────────────────┐    │     │
│  │  • CALLS (func→func)    │   │  │ Contract > 512 tokens?      │    │     │
│  │  • READS (func→var)     │   │  │                             │    │     │
│  │  • WRITES (func→var)    │   │  │ YES → Sliding Window        │    │     │
│  │  • INHERITS (contract)  │   │  │   • Window: 512, Stride: 256│    │     │
│  │  • MODIFIES (modifier)  │   │  │   • Run CodeBERT on each    │    │     │
│  │                         │   │  │   • Max-pool over windows   │    │     │
│  │  Node Features:         │   │  │                             │    │     │
│  │  • Type one-hot (8 dim) │   │  │ NO → Standard tokenization  │    │     │
│  │  • Visibility (4 dim)   │   │  │   • Pad to 512 tokens       │    │     │
│  │  • Mutability (3 dim)   │   │  │   • Attention mask          │    │     │
│  │  • Code embedding (64)  │   │  └─────────────────────────────┘    │     │
│  │                         │   │                                      │     │
│  │  Output: PyG Data obj   │   │  Output: (batch, 512) token IDs     │     │
│  └─────────────────────────┘   └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DUAL-PATH MODEL                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         CONTRACT INPUT               │
                    │  • Source code                       │
                    │  • AST                               │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
                    ▼                                      ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────────┐
│       PATH 1: GNN               │    │       PATH 2: TRANSFORMER            │
│                                 │    │                                      │
│  Input: Graph (nodes, edges)    │    │  Input: Token sequence (512)         │
│                                 │    │                                      │
│  ┌───────────────────────────┐  │    │  ┌─────────────────────────────────┐│
│  │ DR-GCN Layer 1            │  │    │  │ CodeBERT (frozen/fine-tuned)    ││
│  │ • Degree-free convolution │  │    │  │ • 12 transformer layers         ││
│  │ • Input: 79-dim features  │  │    │  │ • Hidden: 768-dim               ││
│  │ • Output: 256-dim         │  │    │  │ • Pretrained on code            ││
│  │ • ReLU + Dropout(0.3)     │  │    │  └─────────────────────────────────┘│
│  └───────────────────────────┘  │    │                 │                    │
│              │                  │    │                 ▼                    │
│              ▼                  │    │  ┌─────────────────────────────────┐│
│  ┌───────────────────────────┐  │    │  │ [CLS] Token Extraction          ││
│  │ DR-GCN Layer 2            │  │    │  │ • Take first token embedding    ││
│  │ • Input: 256-dim          │  │    │  │ • Output: 768-dim               ││
│  │ • Output: 128-dim         │  │    │  └─────────────────────────────────┘│
│  │ • ReLU + Dropout(0.3)     │  │    │                 │                    │
│  └───────────────────────────┘  │    │                 ▼                    │
│              │                  │    │  ┌─────────────────────────────────┐│
│              ▼                  │    │  │ Projection Layer (optional)     ││
│  ┌───────────────────────────┐  │    │  │ • 768 → 768 (identity or MLP)   ││
│  │ Global Mean Pooling       │  │    │  └─────────────────────────────────┘│
│  │ • Aggregate all nodes     │  │    │                                      │
│  │ • Output: 128-dim vector  │  │    │  Output: 768-dim embedding          │
│  └───────────────────────────┘  │    │                                      │
│                                 │    │                                      │
│  Output: 128-dim embedding     │    │                                      │
└────────────────┬────────────────┘    └──────────────────┬──────────────────┘
                 │                                         │
                 └─────────────────┬───────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GATED MULTIMODAL UNIT (GMU)                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   GNN embedding (128-dim) ────────┐                                  │   │
│  │                                    │                                  │   │
│  │                                    ▼                                  │   │
│  │                            ┌──────────────┐                           │   │
│  │                            │   Concat     │                           │   │
│  │                            │  [128; 768]  │                           │   │
│  │                            │   = 896-dim  │                           │   │
│  │                            └──────┬───────┘                           │   │
│  │                                   │                                   │   │
│  │   Trans embedding (768-dim) ──────┤                                  │   │
│  │                                   │                                   │   │
│  │                    ┌──────────────┼──────────────┐                   │   │
│  │                    │              │              │                   │   │
│  │                    ▼              ▼              ▼                   │   │
│  │             ┌───────────┐  ┌───────────┐  ┌───────────┐              │   │
│  │             │ W_gnn     │  │ W_trans   │  │ W_gate    │              │   │
│  │             │ 128→256   │  │ 768→256   │  │ 896→256   │              │   │
│  │             └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │   │
│  │                   │              │              │                    │   │
│  │                   ▼              ▼              ▼                    │   │
│  │              gnn_proj       trans_proj      sigmoid                 │   │
│  │              (256-dim)      (256-dim)       (256-dim)               │   │
│  │                   │              │              │                    │   │
│  │                   │              │              │                    │   │
│  │                   │              │         gate (g)                  │   │
│  │                   │              │              │                    │   │
│  │                   ▼              ▼              ▼                    │   │
│  │              ┌─────────────────────────────────────┐                │   │
│  │              │                                     │                │   │
│  │              │  fused = g * gnn_proj               │                │   │
│  │              │        + (1-g) * trans_proj         │                │   │
│  │              │                                     │                │   │
│  │              │  Output: 256-dim fused embedding    │                │   │
│  │              │                                     │                │   │
│  │              │  Interpretation:                    │                │   │
│  │              │  • g≈1: Trust GNN (structure)       │                │   │
│  │              │  • g≈0: Trust Transformer (code)    │                │   │
│  │              │  • g≈0.5: Equal contribution        │                │   │
│  │              │                                     │                │   │
│  │              └─────────────────────────────────────┘                │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LABEL CLASSIFIER                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Input: 256-dim fused embedding                                      │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌─────────────┐                                                     │   │
│  │   │ Linear      │  256 → 64                                          │   │
│  │   │ + ReLU      │                                                     │   │
│  │   │ + Dropout   │  p=0.3                                              │   │
│  │   └──────┬──────┘                                                     │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌─────────────┐                                                     │   │
│  │   │ Linear      │  64 → 13 (vulnerability classes)                   │   │
│  │   │ + Sigmoid   │  (NOT softmax - multi-label!)                      │   │
│  │   └──────┬──────┘                                                     │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   Output: 13 independent probabilities [0, 1]                         │   │
│  │                                                                       │   │
│  │   Classes:                                                            │   │
│  │   ┌───────────────────────────────────────────────────────────────┐  │   │
│  │   │  0. reentrancy           7. denial_of_service                 │  │   │
│  │   │  1. integer_overflow     8. front_running                     │  │   │
│  │   │  2. integer_underflow    9. time_manipulation                 │  │   │
│  │   │  3. unchecked_return    10. tx_origin                         │  │   │
│  │   │  4. access_control      11. weak_randomness                   │  │   │
│  │   │  5. bad_randomness      12. signature_replay                  │  │   │
│  │   │  6. delegatecall_injection                                    │  │   │
│  │   └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Class Imbalance Strategy

````
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CLASS IMBALANCE HANDLING                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Problem: Highly imbalanced vulnerability distribution
─────────────────────────────────────────────────────

Typical Distribution (Kaggle 35K dataset):
├─ Safe contracts:      ~60%  ████████████████████████████████████
├─ Arithmetic:          ~12%  ████████
├─ Access Control:      ~8%   █████
├─ Unchecked Return:    ~6%   ████
├─ Reentrancy:          ~5%   ███
├─ Other vulnerabilities: ~9% ██████

Solution Stack (Applied in order):
──────────────────────────────────

1. MULTI-LABEL ARCHITECTURE
   ├─ One contract can have MULTIPLE vulnerabilities
   ├─ Output: 13 independent sigmoid outputs (not mutually exclusive)
   ├─ Each output: P(vulnerability_i | contract)
   └─ Threshold per class (default 0.5, tuned on validation set)

2. FOCAL LOSS (Primary loss function)
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │   Standard BCE:  L = -y·log(p) - (1-y)·log(1-p)                    │
   │                                                                     │
   │   Focal Loss:    FL = -α·(1-p)^γ·y·log(p)                          │
   │                     - (1-α)·p^γ·(1-y)·log(1-p)                     │
   │                                                                     │
   │   Where:                                                            │
   │   • γ (gamma) = 2 (focusing parameter)                             │
   │   • α = class weight (higher for rare classes)                     │
   │   • (1-p)^γ down-weights easy examples (safe contracts)            │
   │                                                                     │
   │   Effect: Model focuses on hard-to-classify examples               │
   │                                                                     │
   └────────────────────────────────────────────────────────────────────┘

3. CLASS WEIGHTS
   ├─ Formula: weight[c] = total_samples / (num_classes × class_count[c])
   ├─ Example weights (approximate):
   │   ├─ safe:            0.8  (most common, lowest weight)
   │   ├─ reentrancy:     12.0  (rare, high weight)
   │   ├─ access_control:  7.5  
   │   ├─ arithmetic:      5.0
   │   └─ ...
   └─ Applied as α in focal loss

4. STRATIFIED SAMPLING
   ├─ Each batch contains proportional representation
   ├─ PyTorch: WeightedRandomSampler
   ├─ Ensures rare classes appear in every batch
   └─ Stabilizes training gradients

5. CLASS-SPECIFIC THRESHOLDS (Inference only)
   ├─ Default threshold: 0.5
   ├─ Tune on validation set to maximize F1 per class
   ├─ Rare classes: lower threshold (0.3) to catch more
   ├─ Common classes: higher threshold (0.6) to reduce FP
   └─ Store thresholds in config, load at inference

Implementation (PyTorch):
─────────────────────────
```python
````

class FocalLoss(nn.Module): def **init**(self, alpha=None, gamma=2.0): super().**init**() self.alpha = alpha # Class weights tensor self.gamma = gamma

```
def forward(self, inputs, targets):
    BCE_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none'
    )
    pt = torch.exp(-BCE_loss)
    focal_weight = (1 - pt) ** self.gamma



    if self.alpha is not None:
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight

    return (focal_weight * BCE_loss).mean()

```



# Usage

```
class_weights = compute_class_weights(train_labels) # Shape: (13,) criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

#### Continual Learning System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CONTINUAL LEARNING SYSTEM                                │
└─────────────────────────────────────────────────────────────────────────────┘

Purpose: Update model with new vulnerabilities without forgetting old ones
─────────────────────────────────────────────────────────────────────────

Timeline:
─────────
v1.0 (Initial)     v1.1 (Month 1)      v1.2 (Month 2)      ...
├─ 35K contracts   ├─ +500 new         ├─ +500 new
├─ 13 vuln types   │   exploits        │   exploits
└─ Base accuracy   ├─ New vuln type?   ├─ Drift detected?
                   └─ Incremental      └─ Retrain


Architecture:
─────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ New Data        │    │ Drift Detection │    │ Training Decision       │ │
│  │ (Monthly)       │───▶│ (Evidently AI)  │───▶│                         │ │
│  │                 │    │                 │    │ IF drift > 10%:         │ │
│  │ • Cyfrin Solodit│    │ • Data drift    │    │   → Trigger retraining  │ │
│  │ • Immunefi      │    │ • Concept drift │    │                         │ │
│  │ • Rekt.news     │    │ • Pred. drift   │    │ IF new vuln type:       │ │
│  └─────────────────┘    └─────────────────┘    │   → Expand classifier   │ │
│                                                 └───────────┬─────────────┘ │
│                                                             │               │
│                                                             ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ANTI-FORGETTING MECHANISMS                        │   │
│  │                                                                       │   │
│  │  1. Elastic Weight Consolidation (EWC)                               │   │
│  │     ┌─────────────────────────────────────────────────────────────┐ │   │
│  │     │ • Compute Fisher Information matrix on old data              │ │   │
│  │     │ • Penalize changes to "important" weights                    │ │   │
│  │     │ • Loss = L_new + λ·Σ F_i·(θ_i - θ*_i)²                       │ │   │
│  │     │                                                               │ │   │
│  │     │ Effect: Weights critical for old tasks stay stable           │ │   │
│  │     └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  │  2. Experience Replay                                                │   │
│  │     ┌─────────────────────────────────────────────────────────────┐ │   │
│  │     │ • Store 500 exemplars per vulnerability class                │ │   │
│  │     │ • Selection: Herding (most representative samples)           │ │   │
│  │     │ • During retraining: Mix old exemplars + new data            │ │   │
│  │     │ • Ratio: 30% old, 70% new                                    │ │   │
│  │     │                                                               │ │   │
│  │     │ Effect: Model rehearses old examples while learning new      │ │   │
│  │     └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Training Schedule:                                                          │
│  ─────────────────                                                          │
│  • Weekly: Drift check (automated, Evidently AI)                            │
│  • Monthly: Incremental update (if drift detected)                          │
│  • Quarterly: Full retraining (refresh base model)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Sources

|Source|Size|Purpose|Access|
|---|---|---|---|
|Kaggle Smart Contract Dataset|35K contracts|Base training|Free download|
|Hugging Face Contracts|20K contracts|Validation|Free API|
|Cyfrin Solodit|Ongoing|Continual learning (new exploits)|Free|
|Pre-trained CodeBERT|-|Transfer learning|Hugging Face Hub|

### MVP vs Stretch Goals

|Component|MVP (Must Complete)|Stretch (If Time)|
|---|---|---|
|**Data**|Kaggle 35K, basic preprocessing|+ HuggingFace, + Solodit continual|
|**Model**|CodeBERT fine-tuning only|+ GNN path + GMU fusion|
|**Training**|Multi-label focal loss|+ Continual learning (PyCIL)|
|**Output**|13 vulnerability probabilities|+ SHAP explanations|
|**Tracking**|Local checkpoints|+ MLflow experiment tracking|

### Learning Outcomes

- [ ]  Train GNN from scratch in PyTorch Geometric
- [ ]  Fine-tune transformer models (Hugging Face Transformers)
- [ ]  Implement multi-label classification with class imbalance handling
- [ ]  Build attention-based fusion (Gated Multimodal Unit)
- [ ]  Handle multi-modal data (graph + sequence)
- [ ]  Implement continual learning with PyCIL (stretch)

---

## Module 2: ZKML - Zero-Knowledge Machine Learning

### Goal

Generate cryptographic proofs that ML predictions are correct, enabling trustless AI audits.

### Skills Covered

|Skill|Tool/Framework|Depth|
|---|---|---|
|ZKML|EZKL|Intermediate|
|ZK-SNARKs|Groth16, PLONK|Conceptual|
|ONNX Export|PyTorch → ONNX|Intermediate|
|Model Quantization|INT8|Intermediate|
|Solidity Verifier|EZKL-generated|Basic|

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ZKML PIPELINE                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────────────┐
                         │   CRITICAL CONSTRAINT       │
                         │                             │
                         │   Full model: ~100K params  │
                         │   ZK-friendly: <10K params  │
                         │                             │
                         │   Solution: PROXY MODEL     │
                         └─────────────────────────────┘

STAGE 1: PROXY MODEL TRAINING
═════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Full SENTINEL Model                    Tiny Proxy Model                    │
│   (Not ZK-compatible)                    (ZK-compatible)                     │
│                                                                              │
│   ┌─────────────────────┐                ┌─────────────────────┐            │
│   │ GNN (128-dim)       │                │ Input: 64-dim       │            │
│   │ + Transformer (768) │                │ (compressed feats)  │            │
│   │ + GMU Fusion        │                │         │           │            │
│   │ + MLP Classifier    │                │         ▼           │            │
│   │                     │                │ Linear (64→32)      │            │
│   │ ~100K parameters    │                │ + ReLU              │            │
│   │                     │                │         │           │            │
│   │ Output: 13 probs    │       ────▶    │         ▼           │            │
│   └─────────────────────┘   distill      │ Linear (32→16)      │            │
│                                          │ + ReLU              │            │
│   Teacher predictions                    │         │           │            │
│   used to train proxy                    │         ▼           │            │
│                                          │ Linear (16→1)       │            │
│                                          │ (risk score 0-100)  │            │
│                                          │                     │            │
│                                          │ ~3K parameters      │            │
│                                          └─────────────────────┘            │
│                                                                              │
│   Knowledge Distillation:                                                    │
│   • Input: Contract features (64 selected from full feature set)            │
│   • Label: Risk score from full model (not ground truth)                    │
│   • Loss: MSE(proxy_output, full_model_score)                               │
│   • Goal: Proxy agrees with full model ≥95% of time                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 2: ONNX EXPORT
════════════════════
```python


# Export proxy model to ONNX

import torch.onnx

dummy_input = torch.randn(1, 64) # Batch=1, Features=64 torch.onnx.export( proxy_model, dummy_input, "proxy_model.onnx", input_names=["features"], output_names=["risk_score"], dynamic_axes={"features": {0: "batch"}, "risk_score": {0: "batch"}} )


# Quantize to INT8 (ZK circuits work on integers)

# EZKL handles quantization internally via calibration


STAGE 3: EZKL WORKFLOW
══════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Step 1: gen_settings()                                                      │
│  ───────────────────────                                                    │
│  • Configure proof parameters (accuracy vs speed tradeoff)                   │
│  • Output: settings.json                                                     │
│                                                                              │
│  Step 2: calibrate_settings()                                                │
│  ────────────────────────────                                               │
│  • Run sample inputs through model                                           │
│  • Determine witness sizes and scale factors                                 │
│  • Output: calibrated settings                                               │
│                                                                              │
│  Step 3: compile_model()                                                     │
│  ───────────────────────                                                    │
│  • Convert ONNX → Arithmetic circuit (R1CS)                                 │
│  • Each layer becomes constraint equations                                   │
│  • Output: model.compiled                                                    │
│                                                                              │
│  Step 4: setup() - ONE TIME, EXPENSIVE                                       │
│  ─────────────────────────────────────                                      │
│  • Trusted setup ceremony (or use universal SRS)                             │
│  • Generate:                                                                 │
│    ├─ Proving Key (pk): ~10MB, used by prover, PRIVATE                      │
│    └─ Verification Key (vk): ~1KB, used by verifier, PUBLIC                 │
│  • Output: pk.key, vk.key                                                    │
│                                                                              │
│  Step 5: prove() - PER AUDIT                                                │
│  ──────────────────────────                                                 │
│  • Input:                                                                    │
│    ├─ Contract features (public input)                                      │
│    ├─ Model weights (witness, private)                                      │
│    └─ Proving key                                                            │
│  • Computation:                                                              │
│    ├─ Run inference inside ZK circuit                                       │
│    └─ Generate proof π                                                      │
│  • Output:                                                                   │
│    ├─ Proof π (~2KB)                                                        │
│    └─ Public outputs (risk_score)                                           │
│  • Time: ~30-60 seconds (CPU-bound)                                         │
│                                                                              │
│  Step 6: verify() - ON-CHAIN OR OFF-CHAIN                                   │
│  ───────────────────────────────────────                                    │
│  • Input: Proof π, public outputs, verification key                          │
│  • Output: true/false (proof valid?)                                         │
│  • Gas cost: ~250K gas (on-chain)                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 4: ON-CHAIN INTEGRATION
═════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ZKMLVerifier.sol (Auto-generated by EZKL)                                  │
│  ─────────────────────────────────────────                                  │
│                                                                              │
│  // SPDX-License-Identifier: MIT                                            │
│  contract ZKMLVerifier {                                                     │
│      // Verification key embedded as constants                               │
│      uint256 constant VK_ALPHA = 0x...;                                     │
│      uint256 constant VK_BETA = 0x...;                                      │
│      // ... more constants                                                   │
│                                                                              │
│      function verify(                                                        │
│          bytes calldata proof,                                              │
│          uint256[] calldata publicSignals                                   │
│      ) public view returns (bool) {                                         │
│          // Pairing check (BN254 curve)                                     │
│          // Returns true if proof is valid                                  │
│      }                                                                       │
│  }                                                                           │
│                                                                              │
│  Integration with AuditRegistry:                                             │
│  ───────────────────────────────                                            │
│                                                                              │
│  function submitAudit(                                                       │
│      address contractAddress,                                               │
│      uint256 riskScore,                                                     │
│      bytes calldata zkProof,                                                │
│      uint256[] calldata publicSignals                                       │
│  ) external {                                                                │
│      // 1. Verify ZK proof                                                  │
│      require(                                                                │
│          zkmlVerifier.verify(zkProof, publicSignals),                       │
│          "Invalid ZK proof"                                                 │
│      );                                                                      │
│                                                                              │
│      // 2. Check public signals match claimed risk score                    │
│      require(publicSignals[0] == riskScore, "Score mismatch");              │
│                                                                              │
│      // 3. Store audit result                                               │
│      audits[contractAddress].push(AuditResult({                             │
│          riskScore: riskScore,                                              │
│          zkProofHash: keccak256(zkProof),                                   │
│          timestamp: block.timestamp,                                        │
│          auditor: msg.sender                                                │
│      }));                                                                    │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
### Use Cases

|Use Case|Description|Benefit|
|---|---|---|
|**Trustless Audits**|Users verify model ran correctly without seeing weights|No need to trust auditor|
|**Model IP Protection**|Prove accuracy without revealing architecture|Auditors protect proprietary models|
|**Decentralized Registry**|On-chain model leaderboard with verified claims|Compare auditors fairly|
|**Insurance Integration**|Prove contract was audited by certified model|Automated claims processing|

### MVP vs Stretch Goals

|Component|MVP (Must Complete)|Stretch (If Time)|
|---|---|---|
|**Model**|Tiny 3-layer MLP (<5K params)|Attempt quantized CodeBERT|
|**EZKL**|Full workflow locally|Proof aggregation (batching)|
|**On-chain**|Deploy Verifier on Sepolia|Gas optimization|
|**Integration**|Manual proof generation|API endpoint for proofs|

### Learning Outcomes

- [ ]  Master EZKL Python library
- [ ]  Understand ZK-SNARK proof systems conceptually
- [ ]  Export PyTorch models to ONNX
- [ ]  Deploy and interact with verifier contracts
- [ ]  Debug ZK circuit constraints and witness issues

---


## Module 3: MLOps - Production ML Infrastructure

### Goal

Build enterprise-grade ML operations stack for continuous training, monitoring, and deployment.

### Skills Covered

|Skill|Tool/Framework|Depth|
|---|---|---|
|Experiment Tracking|MLflow|Intermediate|
|Data Versioning|DVC|Intermediate|
|Pipeline Orchestration|Dagster|Basic→Intermediate|
|Feature Store|Feast|Basic|
|Drift Detection|Evidently AI|Intermediate|
|Model Serving|FastAPI + ONNX|Intermediate|
|Containerization|Docker|Intermediate|
|CI/CD|GitHub Actions|Intermediate|


### Technical Architecture

```

┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps STACK                                        │
└─────────────────────────────────────────────────────────────────────────────┘

LAYER 1: DATA MANAGEMENT
════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  DVC (Data Version Control)                                                  │
│  ──────────────────────────                                                 │
│                                                                              │
│  datasets/                                                                   │
│  ├─ v1.0.0/                   # Original 35K contracts                      │
│  │   ├─ train.parquet                                                       │
│  │   ├─ val.parquet                                                         │
│  │   └─ test.parquet                                                        │
│  ├─ v1.1.0/                   # Added 5K new exploits                       │
│  └─ v2.0.0/                   # Rebalanced with focal loss tuning           │
│                                                                              │
│  Commands:                                                                   │
│  • dvc add data/raw/contracts.parquet                                       │
│  • dvc push (to remote: S3/GCS/local)                                       │
│  • dvc checkout v1.0.0 (switch versions)                                    │
│  • dvc diff (compare versions)                                              │
│                                                                              │
│  Benefits:                                                                   │
│  • Git-like versioning for large datasets                                   │
│  • Reproducible experiments                                                 │
│  • Team collaboration on data                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Feast Feature Store (Stretch Goal)                                          │
│  ──────────────────────────────────                                         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │   Offline Store (Training)          Online Store (Inference)        │    │
│  │   ─────────────────────────          ───────────────────────        │    │
│  │   Parquet files on disk              Redis (<10ms latency)          │    │
│  │                                                                      │    │
│  │   Features:                          Features (same, materialized): │    │
│  │   ├─ contract_features (93)          Fetched by contract_address    │    │
│  │   │   ├─ ast_node_count                                             │    │
│  │   │   ├─ function_count                                             │    │
│  │   │   ├─ state_var_count                                            │    │
│  │   │   ├─ external_call_count                                        │    │
│  │   │   ├─ modifier_count                                             │    │
│  │   │   └─ ...                                                        │    │
│  │   │                                                                  │    │
│  │   ├─ graph_features (20)                                            │    │
│  │   │   ├─ node_count                                                 │    │
│  │   │   ├─ edge_count                                                 │    │
│  │   │   ├─ avg_degree                                                 │    │
│  │   │   └─ ...                                                        │    │
│  │   │                                                                  │    │
│  │   └─ semantic_features (768)                                        │    │
│  │       └─ codebert_embedding (precomputed)                           │    │
│  │                                                                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Point-in-Time Correctness:                                                  │
│  • Features have timestamps                                                  │
│  • Training uses features available AT training time                         │
│  • Prevents data leakage (future info in past predictions)                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


LAYER 2: EXPERIMENT TRACKING
════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  MLflow                                                                      │
│  ──────                                                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     MLflow Tracking UI                               │   │
│  │  ───────────────────────────────────────────────────────────────────│   │
│  │                                                                       │   │
│  │  Experiment: sentinel-vuln-detection                                 │   │
│  │                                                                       │   │
│  │  Run ID    │ Model      │ F1-Macro │ Params              │ Tags      │   │
│  │  ──────────┼────────────┼──────────┼─────────────────────┼───────────│   │
│  │  abc123    │ CodeBERT   │ 0.82     │ lr=1e-5, bs=16      │ baseline  │   │
│  │  def456    │ GNN+BERT   │ 0.87     │ lr=1e-5, fusion=gmu │ best      │   │
│  │  ghi789    │ GNN only   │ 0.71     │ lr=1e-3, layers=3   │           │   │
│  │                                                                       │   │
│  │  Artifacts:                                                           │   │
│  │  ├─ model.pt (trained weights)                                       │   │
│  │  ├─ model.onnx (exported)                                            │   │
│  │  ├─ confusion_matrix.png                                             │   │
│  │  ├─ roc_curves.png                                                   │   │
│  │  └─ requirements.txt                                                 │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Code Integration:                                                           │
│  ```python                                                                   │
│  import mlflow                                                               │
│                                                                              │
│  mlflow.set_experiment("sentinel-vuln-detection")                           │
│                                                                              │
│  with mlflow.start_run(run_name="gnn-bert-fusion"):                         │
│      # Log parameters                                                        │
│      mlflow.log_params({                                                     │
│          "learning_rate": 1e-5,                                             │
│          "batch_size": 16,                                                  │
│          "fusion_type": "gmu"                                               │
│      })                                                                      │
│                                                                              │
│      # Train model...                                                        │
│                                                                              │
│      # Log metrics                                                           │
│      mlflow.log_metrics({                                                    │
│          "f1_macro": 0.87,                                                  │
│          "accuracy": 0.91,                                                  │
│          "reentrancy_f1": 0.82                                              │
│      })                                                                      │
│                                                                              │
│      # Log model                                                             │
│      mlflow.pytorch.log_model(model, "model")                               │
│  ```                                                                         │
│                                                                              │
│  Model Registry:                                                             │
│  • staging → Champion model candidate                                       │
│  • production → Live model serving traffic                                  │
│  • archived → Previous versions                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


LAYER 3: PIPELINE ORCHESTRATION (Stretch)
═════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Dagster                                                                     │
│  ───────                                                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Pipeline DAG                                    │   │
│  │                                                                       │   │
│  │   ┌──────────────┐                                                   │   │
│  │   │ raw_contracts│ ◄─── Kaggle API / DVC pull                       │   │
│  │   └──────┬───────┘                                                   │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌──────────────────┐                                               │   │
│  │   │extracted_features│ ◄─── Slither + py-solidity-ast               │   │
│  │   └──────┬───────────┘                                               │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌──────────────────┐                                               │   │
│  │   │ training_dataset │ ◄─── Train/val/test split (stratified)       │   │
│  │   └──────┬───────────┘                                               │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌──────────────────┐                                               │   │
│  │   │  trained_model   │ ◄─── PyTorch training + MLflow logging       │   │
│  │   └──────┬───────────┘                                               │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌──────────────────┐                                               │   │
│  │   │model_evaluation  │ ◄─── Test metrics, SHAP, adversarial tests   │   │
│  │   └──────┬───────────┘                                               │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌──────────────────┐                                               │   │
│  │   │  deploy_model    │ ◄─── If metrics pass → MLflow registry       │   │
│  │   └──────────────────┘                                               │   │
│  │                                                                       │   │
│  │   Schedule: Weekly (Sunday 2 AM) or on drift detection               │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


LAYER 4: MONITORING
═══════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Evidently AI (Drift Detection)                                              │
│  ──────────────────────────────                                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  1. DATA DRIFT                                                        │   │
│  │     ────────────                                                      │   │
│  │     Method: Kolmogorov-Smirnov test per feature                      │   │
│  │     Alert: If >30% features show drift (p < 0.05)                    │   │
│  │     Example: "external_call_count" distribution shifted              │   │
│  │     Action: Investigate data source, possible retraining             │   │
│  │                                                                       │   │
│  │  2. CONCEPT DRIFT                                                     │   │
│  │     ──────────────                                                    │   │
│  │     Method: PSI (Population Stability Index) on predictions          │   │
│  │     Alert: If PSI > 0.2 (significant shift)                          │   │
│  │     Example: Model predicting "safe" more often than baseline        │   │
│  │     Action: Manual review, check for new attack patterns             │   │
│  │                                                                       │   │
│  │  3. PREDICTION DRIFT                                                  │   │
│  │     ──────────────────                                                │   │
│  │     Track: Vulnerability type distribution over time                 │   │
│  │     Expected: Reentrancy 30%, Overflow 20%, ...                      │   │
│  │     Alert: If distribution changes >15%                              │   │
│  │     Action: Correlate with real-world exploit trends                 │   │
│  │                                                                       │   │
│  │  4. MODEL PERFORMANCE (if labels available)                          │   │
│  │     ────────────────────────────────────                             │   │
│  │     Track: Accuracy, Precision, Recall, F1 (rolling 7-day)           │   │
│  │     Alert: If accuracy drops >5% from baseline                       │   │
│  │     Action: Trigger retraining pipeline                              │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Prometheus + Grafana (System Metrics)                                       │
│  ─────────────────────────────────────                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Metrics:                                                             │   │
│  │  ├─ sentinel_inference_latency_seconds (histogram)                   │   │
│  │  │   Target: p50 < 100ms, p99 < 500ms                                │   │
│  │  │                                                                    │   │
│  │  ├─ sentinel_requests_total (counter)                                │   │
│  │  │   Labels: endpoint, status_code                                   │   │
│  │  │                                                                    │   │
│  │  ├─ sentinel_model_prediction (histogram)                            │   │
│  │  │   Labels: vulnerability_type                                      │   │
│  │  │                                                                    │   │
│  │  ├─ sentinel_gpu_memory_bytes (gauge)                                │   │
│  │  │                                                                    │   │
│  │  └─ sentinel_proof_generation_seconds (histogram)                    │   │
│  │      Target: < 120 seconds                                           │   │
│  │                                                                       │   │
│  │  Alerts:                                                              │   │
│  │  ├─ High latency (p99 > 1s for 5min) → PagerDuty                    │   │
│  │  ├─ Error rate > 1% → Slack                                         │   │
│  │  └─ Drift detected → Trigger retraining job                         │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘



LAYER 5: SERVING
════════════════
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  FastAPI + ONNX Runtime                                                      │
│  ──────────────────────                                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Inference Optimization:                                              │   │
│  │  ├─ ONNX Runtime: 3x speedup vs PyTorch eager                        │   │
│  │  ├─ Quantization: INT8 models (4x smaller, 2x faster)               │   │
│  │  ├─ Batching: Group requests (throughput 10x)                        │   │
│  │  └─ GPU: CUDA inference for large batches                            │   │
│  │                                                                       │   │
│  │  Model Router (Stretch):                                              │   │
│  │  ├─ Production model: 90% traffic                                    │   │
│  │  ├─ Canary model: 10% traffic (new version testing)                  │   │
│  │  ├─ Shadow mode: Run both, compare (no user impact)                  │   │
│  │  └─ Fallback: If new model errors → route to stable                  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```



### MVP vs Stretch Goals

|Component|MVP (Must Complete)|Stretch (If Time)|
|---|---|---|
|**Tracking**|MLflow (experiments, models)|+ Weights & Biases|
|**Data**|DVC for dataset versioning|+ Feast feature store|
|**Pipeline**|Manual Python scripts|+ Dagster orchestration|
|**Monitoring**|Basic accuracy logging|+ Evidently drift detection|
|**Serving**|FastAPI + ONNX|+ Canary deployments|
|**Infra**|Docker Compose|+ Kubernetes|

### Learning Outcomes

- [ ]  Build end-to-end MLOps pipelines (not just training scripts!)
- [ ]  Version datasets with DVC
- [ ]  Track experiments with MLflow
- [ ]  Implement drift detection with Evidently AI
- [ ]  Deploy models with monitoring
- [ ]  Master feature stores (Feast) for production ML (stretch)

---

## Module 4: AI Agents - LangChain, CrewAI, LangGraph

### Goal

Build multi-agent systems for intelligent, comprehensive contract auditing.

### Skills Covered

|Skill|Tool/Framework|Depth|
|---|---|---|
|LLM Framework|LangChain|Intermediate|
|Multi-Agent|CrewAI|Intermediate|
|State Machines|LangGraph|Basic|
|RAG|FAISS, Sentence Transformers|Intermediate|
|Tool Calling|Custom tools|Intermediate|
|Prompt Engineering|System prompts, few-shot|Intermediate|
|LLMOps|LangSmith|Basic|



### Technical Architecture

```

┌─────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT AUDIT SYSTEM                                 │
└─────────────────────────────────────────────────────────────────────────────┘

LLM STRATEGY
════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  MVP: Single LLM for all agents (Ollama + Llama 3.2 8B)                     │
│  ───────────────────────────────────────────────────────                    │
│                                                                              │
│  Why:                                                                        │
│  • Free (runs locally on RTX 3070)                                          │
│  • Consistent behavior across agents                                         │
│  • Easier debugging (one model to understand)                                │
│  • ~5GB VRAM, leaves room for ML models                                     │
│                                                                              │
│  Stretch: Specialized LLMs per agent                                         │
│  ─────────────────────────────────────                                      │
│  • Agent 1 (Static): Llama 3.2 (fast, tool-calling)                         │
│  • Agent 2 (ML): GPT-4o-mini (reasoning about SHAP)                         │
│  • Agent 3 (Research): Mistral 7B (retrieval tasks)                         │
│  • Agent 4 (Code): GPT-4 (complex reasoning)                                │
│  • Agent 5 (Report): Claude 3 Sonnet (long-form writing)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


AGENT ARCHITECTURE
══════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  AGENT 1: Static Analyzer (Senior Security Engineer)                        │
│  ────────────────────────────────────────────────────                       │
│                                                                              │
│  Role: Run formal analysis tools and extract structured findings            │
│                                                                              │
│  Tools:                                                                      │
│  ├─ run_slither(contract_path) → JSON findings                             │
│  │   • Detectors: reentrancy, shadowing, unused-state, etc.                │
│  │   • Output: [{detector, impact, confidence, description}]               │
│  │                                                                          │
│  ├─ run_mythril(contract_path) → JSON findings                             │
│  │   • Symbolic execution for path coverage                                 │
│  │   • Output: [{swc_id, severity, tx_sequence}]                           │
│  │                                                                          │
│  └─ parse_compiler_warnings(contract_path) → warnings list                 │
│      • Solc warnings often indicate issues                                  │
│                                                                              │
│  Output: List of potential issues with line numbers, severity               │
│  LLM: Llama 3.2 (via Ollama)                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  AGENT 2: ML Intelligence (AI Research Scientist)                           │
│  ─────────────────────────────────────────────────                          │
│                                                                              │
│  Role: Run deep learning models and interpret results                       │
│                                                                              │
│  Tools:                                                                      │
│  ├─ run_sentinel_model(contract_path) → predictions                        │
│  │   • Calls ML inference API                                              │
│  │   • Output: {vulnerability: probability} for 13 classes                 │
│  │                                                                          │
│  ├─ generate_shap_explanation(contract, prediction) → explanation          │
│  │   • Feature importance for top vulnerabilities                          │
│  │   • Output: "High risk due to: external_calls (0.4), state_writes (0.3)"│
│  │                                                                          │
│  └─ run_adversarial_test(contract) → robustness_score                      │
│      • Small perturbations shouldn't flip prediction                        │
│                                                                              │
│  Output: Vulnerability probabilities + human-readable explanations          │
│  LLM: Llama 3.2 (via Ollama)                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  AGENT 3: Historical Researcher (Blockchain Archaeologist)                  │
│  ──────────────────────────────────────────────────────                     │
│                                                                              │
│  Role: Find similar past exploits and provide context                       │
│                                                                              │
│  Tools:                                                                      │
│  ├─ search_vector_db(query, k=5) → similar_exploits                        │
│  │   • FAISS index of 1000+ past exploits                                  │
│  │   • Embeddings: all-MiniLM-L6-v2                                        │
│  │   • Sources: Rekt.news, Immunefi, Cyfrin Solodit                        │
│  │                                                                          │
│  ├─ search_etherscan(contract_address) → deployment_info                   │
│  │   • Check if contract already deployed                                   │
│  │   • Check for past exploits on this address                             │
│  │                                                                          │
│  └─ search_web(query) → recent_articles                                    │
│      • Tavily API for latest vulnerability research                        │
│                                                                              │
│  Output: Similar historical cases with references and lessons learned       │
│  LLM: Llama 3.2 (via Ollama)                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  AGENT 4: Code Reviewer (Smart Contract Expert)                             │
│  ───────────────────────────────────────────────                            │
│                                                                              │
│  Role: Manual code review for logic bugs tools can't catch                  │
│                                                                              │
│  Tools:                                                                      │
│  ├─ analyze_business_logic(contract) → logic_issues                        │
│  │   • Check economic invariants                                           │
│  │   • "withdraw() can drain more than user deposited"                     │
│  │                                                                          │
│  ├─ check_best_practices(contract) → compliance_report                     │
│  │   • OpenZeppelin patterns adherence                                     │
│  │   • CEI pattern, checks-effects-interactions                           │
│  │                                                                          │
│  ├─ review_access_control(contract) → authorization_matrix                 │
│  │   • Who can call which functions                                        │
│  │   • Missing modifiers, incorrect permissions                            │
│  │                                                                          │
│  └─ python_repl(code) → execution_result                                   │
│      • Run custom analysis scripts                                         │
│                                                                              │
│  Output: Logic vulnerabilities not detectable by automated tools            │
│  LLM: Llama 3.2 (via Ollama), upgrade to GPT-4 for complex cases           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  AGENT 5: Report Synthesizer (Technical Writer)                             │
│  ───────────────────────────────────────────────                            │
│                                                                              │
│  Role: Combine all findings into coherent audit report                      │
│                                                                              │
│  Inputs: Outputs from Agents 1-4                                            │
│                                                                              │
│  Tools:                                                                      │
│  ├─ rank_by_severity(findings) → prioritized_list                          │
│  │   • CVSS scoring + ML confidence weighting                              │
│  │   • Critical > High > Medium > Low > Informational                      │
│  │                                                                          │
│  ├─ generate_remediation(vulnerability) → fix_suggestion                   │
│  │   • Code snippets for fixes                                             │
│  │   • References to secure implementations                                │
│  │                                                                          │
│  ├─ create_executive_summary(findings) → summary                           │
│  │   • Non-technical overview for stakeholders                             │
│  │   • Risk score, key recommendations                                     │
│  │                                                                          │
│  └─ export_report(content, format) → file_path                             │
│      • Formats: PDF, Markdown, JSON                                        │
│                                                                              │
│  Output: Multi-format audit report (PDF, Markdown, JSON)                    │
│  LLM: Llama 3.2 (via Ollama)                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


ORCHESTRATION (CrewAI)
══════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Sequential Process with Parallel Branches:                                  │
│  ──────────────────────────────────────────                                 │
│                                                                              │
│                      ┌─────────────────────┐                                │
│                      │  Contract Input     │                                │
│                      └──────────┬──────────┘                                │
│                                 │                                            │
│                                 ▼                                            │
│                      ┌─────────────────────┐                                │
│                      │  Agent 1: Static    │                                │
│                      │  Analyzer           │                                │
│                      └──────────┬──────────┘                                │
│                                 │                                            │
│                    ┌────────────┴────────────┐                              │
│                    │                         │                              │
│                    ▼                         ▼                              │
│         ┌─────────────────┐       ┌─────────────────┐                       │
│         │  Agent 2: ML    │       │  Agent 3:       │                       │
│         │  Intelligence   │       │  Researcher     │                       │
│         │  (PARALLEL)     │       │  (PARALLEL)     │                       │
│         └────────┬────────┘       └────────┬────────┘                       │
│                  │                         │                                │
│                  └────────────┬────────────┘                                │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │  Agent 4: Code      │                                  │
│                    │  Reviewer           │                                  │
│                    │  (if ML conf < 0.5) │  ◄─── Conditional execution     │
│                    └──────────┬──────────┘                                  │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │  Agent 5: Report    │                                  │
│                    │  Synthesizer        │                                  │
│                    └──────────┬──────────┘                                  │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │  Final Audit Report │                                  │
│                    └─────────────────────┘                                  │
│                                                                              │
│  State Management (LangGraph):                                               │
│  ├─ State: {contract, findings_static, findings_ml, findings_hist, ...}    │
│  ├─ Conditional edges based on confidence scores                            │
│  └─ Human-in-the-loop: Approve report before blockchain submission          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


RAG SYSTEM
══════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Knowledge Base:                                                             │
│  ├─ Vulnerability Database: CVE, CWE, Immunefi reports (~500 docs)         │
│  ├─ Best Practices: OpenZeppelin, ConsenSys, Trail of Bits guides          │
│  ├─ Past Audits: Community audit reports                                   │
│  └─ Solidity Docs: Language reference                                       │
│                                                                              │
│  Vector Store: FAISS                                                         │
│  ├─ Embeddings: all-MiniLM-L6-v2 (384-dim, fast)                           │
│  ├─ Index size: ~100K chunks                                                │
│  └─ Search latency: <50ms                                                   │
│                                                                              │
│  Retrieval Strategy:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  MVP: Dense retrieval only                                            │   │
│  │  ─────────────────────────                                           │   │
│  │  query → embed → FAISS search → top-k docs                           │   │
│  │                                                                       │   │
│  │  Stretch: Hybrid search + reranking                                   │   │
│  │  ──────────────────────────────                                      │   │
│  │  query ─┬─▶ BM25 (keyword) ───────┬─▶ merge ─▶ rerank ─▶ top-k       │   │
│  │         └─▶ Dense (semantic) ─────┘    (cross-encoder)              │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Chunking:                                                                   │
│  ├─ Size: 512 tokens                                                        │
│  ├─ Overlap: 50 tokens                                                      │
│  └─ Metadata: {source, date, vulnerability_type, severity}                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


LLMOps (LangSmith)
══════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Tracing:                                                                    │
│  ├─ Every agent step logged (inputs, outputs, latency, tokens)              │
│  ├─ Visualize agent decision tree                                           │
│  └─ Debug failed runs easily                                                │
│                                                                              │
│  Evaluation:                                                                 │
│  ├─ A/B test prompt versions                                                │
│  ├─ Compare: "You are a security expert..." vs "You are a senior auditor..."│
│  └─ Ground truth comparisons (manual audit samples)                         │
│                                                                              │
│  Cost Tracking:                                                              │
│  ├─ Token usage per agent                                                   │
│  ├─ Target: <$0.50 per audit (with local LLM: ~$0)                         │
│  └─ Alert if costs spike                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MVP vs Stretch Goals

|Component|MVP (Must Complete)|Stretch (If Time)|
|---|---|---|
|**LLM**|Ollama (Llama 3.2) for ALL agents|Specialized LLMs per agent|
|**Framework**|LangChain single agent|CrewAI 5-agent system|
|**RAG**|FAISS + dense retrieval|+ Hybrid search + reranking|
|**Tools**|3 tools (ML, Slither, RAG)|10+ specialized tools|
|**Orchestration**|Sequential pipeline|LangGraph state machine|
|**Ops**|Basic logging|LangSmith full tracing|

### Learning Outcomes

- [ ]  Build LangChain agents with tool calling
- [ ]  Implement RAG with vector stores (FAISS)
- [ ]  Create multi-agent systems with CrewAI
- [ ]  Design complex workflows with LangGraph
- [ ]  Master prompt engineering for specialized agents
- [ ]  Use LLMOps tools (LangSmith) for debugging

---

## Module 5: Advanced Solidity & Foundry

### Goal

Master cutting-edge smart contract patterns for the SENTINEL protocol.

### Skills Covered

|Skill|Tool/Framework|Depth|
|---|---|---|
|Upgradeable Contracts|UUPS, Transparent, Beacon|Advanced|
|Cross-chain|Chainlink CCIP|Intermediate|
|Account Abstraction|ERC-4337|Intermediate|
|Tokenomics|Staking, Slashing|Intermediate|
|Gas Optimization|Assembly, Storage Packing|Intermediate|
|Governance|On-chain voting, Timelock|Intermediate|
|Testing|Fuzz, Invariant, Fork|Advanced|

### Contract Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SMART CONTRACT SYSTEM                                  │
└─────────────────────────────────────────────────────────────────────────────┘


CONTRACT 1: AuditRegistry.sol (UUPS Upgradeable)
════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Purpose: Store all audit results with model provenance and ZK verification │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          Storage Layout                              │   │
│  │                                                                       │  │
│  │  // Packed struct for gas efficiency                                 │   │
│  │  struct AuditResult {                                                │   │
│  │      uint96 riskScore;           // 0-100, fits in 96 bits          │   │
│  │      uint160 auditor;            // address as uint160               │   │
│  │      // --- slot boundary ---                                        │   │
│  │      bytes32 modelVersion;       // keccak256("GNN-v2.1")           │   │
│  │      bytes32 zkProofHash;        // IPFS hash of full proof         │   │
│  │      uint256 timestamp;          // block.timestamp                 │   │
│  │      uint16 vulnerabilities;     // bitmap: 13 vulns fit in 16 bits │   │
│  │  }                                                                   │   │
│  │                                                                       │   │
│  │  mapping(address => AuditResult[]) public audits;                    │   │
│  │  mapping(address => bool) public authorizedAuditors;                 │   │
│  │  mapping(uint256 => Challenge) public challenges;                    │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          Key Functions                               │   │
│  │                                                                       │   │
│  │  function submitAudit(                                               │   │
│  │      address contractAddress,                                        │   │
│  │      uint96 riskScore,                                               │   │
│  │      bytes32 modelVersion,                                           │   │
│  │      bytes calldata zkProof,                                         │   │
│  │      uint256[] calldata publicSignals                                │   │
│  │  ) external onlyAuthorizedAuditor {                                  │   │
│  │      // 1. Verify ZK proof                                           │   │
│  │      require(zkmlVerifier.verify(zkProof, publicSignals));           │   │
│  │                                                                       │   │
│  │      // 2. Check public signals match claimed score                  │   │
│  │      require(publicSignals[0] == riskScore);                         │   │
│  │                                                                       │   │
│  │      // 3. Store audit                                               │   │
│  │      audits[contractAddress].push(AuditResult({...}));               │   │
│  │                                                                       │   │
│  │      emit AuditSubmitted(contractAddress, riskScore, msg.sender);    │   │
│  │  }                                                                   │   │
│  │                                                                       │   │
│  │  function challengeAudit(uint256 auditId) external payable;          │   │
│  │  function resolveChallenge(uint256 challengeId) external;            │   │
│  │  function getLatestAudit(address) external view returns (AuditResult);│   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Upgradeability: UUPS Pattern                                                │
│  ├─ Proxy contract: Minimal, just delegatecalls                             │
│  ├─ Implementation: Contains all logic                                      │
│  ├─ Upgrade: Deploy new impl → call upgradeTo(newImpl)                     │
│  └─ Safety: onlyOwner + 2-day timelock via GovernanceExecutor              │
│                                                                              │
│  Gas Optimizations:                                                          │
│  ├─ Struct packing: 3 slots instead of 6                                   │
│  ├─ Vulnerability bitmap: 13 bools → 2 bytes                               │
│  ├─ Batch submissions: submitBatchAudits() for multiple contracts          │
│  └─ Cold/warm storage: Use memory for intermediate calculations            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


CONTRACT 2: SentinelToken.sol (ERC-20 + Governance)
═══════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Purpose: Governance token with staking and rewards                          │
│                                                                              │
│  Inheritance:                                                                │
│  ├─ ERC20 (OpenZeppelin)                                                    │
│  ├─ ERC20Votes (for delegation)                                            │
│  ├─ ERC20Permit (gasless approvals)                                        │
│  └─ Ownable (initial setup only)                                           │
│                                                                              │
│  Tokenomics:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Total Supply: 100,000,000 SENTINEL                                  │   │
│  │                                                                       │   │
│  │  Distribution:                                                        │   │
│  │  ├─ 40% Community Rewards (audit incentives)  ████████████████       │   │
│  │  ├─ 30% Team (4-year vesting)                 ████████████           │   │
│  │  ├─ 20% Treasury (DAO-controlled)             ████████               │   │
│  │  └─ 10% Initial Liquidity                     ████                   │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Staking Mechanism:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Auditor Flow:                                                        │   │
│  │  1. Stake 1000 SENTINEL to become authorized auditor                 │   │
│  │  2. Submit audits (locked for 30 days after submission)              │   │
│  │  3. If audit challenged and proven wrong → slashed 50%              │   │
│  │  4. If audit unchallenged → earn 10% APY from protocol fees         │   │
│  │                                                                       │   │
│  │  struct StakeInfo {                                                  │   │
│  │      uint256 amount;                                                 │   │
│  │      uint256 lockedUntil;                                            │   │
│  │      uint256 rewardsAccrued;                                         │   │
│  │  }                                                                   │   │
│  │                                                                       │   │
│  │  mapping(address => StakeInfo) public stakes;                        │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Governance:                                                                 │
│  ├─ Propose: Require 100K tokens (0.1% of supply)                          │
│  ├─ Vote: 1 token = 1 vote (delegatable via ERC20Votes)                    │
│  ├─ Quorum: 4% of supply must participate                                  │
│  └─ Execution: 2-day timelock via GovernanceExecutor                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


CONTRACT 3: CrossChainOracle.sol (Chainlink CCIP) [STRETCH]
═══════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Purpose: Sync audit results across Ethereum, Polygon, Arbitrum             │
│                                                                              │
│  Why: DeFi protocols deploy on multiple chains, need unified risk view      │
│                                                                              │
│  Flow:                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Ethereum (Source)                    Polygon (Destination)         │   │
│  │                                                                       │   │
│  │   ┌─────────────────┐                 ┌─────────────────┐            │   │
│  │   │ AuditRegistry   │                 │ AuditRegistry   │            │   │
│  │   │ submitAudit()   │                 │ (receives)      │            │   │
│  │   └────────┬────────┘                 └────────▲────────┘            │   │
│  │            │                                    │                    │   │
│  │            ▼                                    │                    │   │
│  │   ┌─────────────────┐                 ┌─────────────────┐            │   │
│  │   │ CrossChainOracle│    CCIP         │ CrossChainOracle│            │   │
│  │   │ sendMessage()   │ ────────────▶  │ _ccipReceive()  │            │   │
│  │   └─────────────────┘                 └─────────────────┘            │   │
│  │                                                                       │   │
│  │   Message Payload:                                                    │   │
│  │   abi.encode(contractAddress, riskScore, zkProofHash, modelVersion)  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Security:                                                                   │
│  ├─ Whitelist: Only messages from trusted source chains                     │
│  ├─ Replay protection: Nonce tracking per source chain                     │
│  └─ Gas estimation: Overpay gas, excess refunded                           │
│                                                                              │
│  Cost: ~$1-3 per message (testnet free)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


CONTRACT 4: AccountAbstractionPaymaster.sol (ERC-4337) [STRETCH]
════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Purpose: Enable gasless audits (users don't need ETH)                       │
│                                                                              │
│  Flow:                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   1. User creates UserOperation (off-chain):                         │   │
│  │      ├─ sender: SimpleAccount address                                │   │
│  │      ├─ callData: submitAudit(contractAddress, ...)                 │   │
│  │      └─ signature: ECDSA signature                                  │   │
│  │                                                                       │   │
│  │   2. Paymaster sponsors gas:                                          │   │
│  │      ├─ validatePaymasterUserOp(): Check user whitelist             │   │
│  │      └─ Paymaster deposits ETH in EntryPoint                        │   │
│  │                                                                       │   │
│  │   3. Bundler submits UserOp to EntryPoint                            │   │
│  │                                                                       │   │
│  │   4. EntryPoint executes → calls AuditRegistry.submitAudit()        │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Paymaster Logic:                                                            │
│  ├─ Free tier: 10 audits per user per month                                │
│  ├─ Premium: Unlimited (pay in SENTINEL tokens)                             │
│  └─ Security: Rate limiting, whitelist, deposit limits                     │
│                                                                              │
│  Benefits:                                                                   │
│  ├─ Onboard non-crypto users (no ETH needed)                               │
│  ├─ Batch operations (10 audits in one UserOp)                             │
│  └─ Protocol can sponsor audits for key partners                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


CONTRACT 5: GovernanceExecutor.sol (Timelock)
═════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Purpose: Decentralized protocol upgrades with time-delay safety            │
│                                                                              │
│  Proposal Examples:                                                          │
│  ├─ "Increase staking requirement to 2000 SENTINEL"                         │
│  ├─ "Upgrade AuditRegistry to v2"                                           │
│  ├─ "Add new vulnerability type to classification"                          │
│  └─ "Modify slashing percentage from 50% to 30%"                           │
│                                                                              │
│  Voting Flow:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   Create Proposal                                                     │   │
│  │         │                                                             │   │
│  │         ▼                                                             │   │
│  │   ┌─────────────┐                                                    │   │
│  │   │  7-day      │  Voting period                                     │   │
│  │   │  voting     │  • For/Against/Abstain                             │   │
│  │   │  period     │  • Quorum: 4% of supply                           │   │
│  │   └──────┬──────┘                                                    │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   ┌─────────────┐                                                    │   │
│  │   │  2-day      │  Timelock                                          │   │
│  │   │  delay      │  • Community can exit if disagree                 │   │
│  │   │             │  • Multisig can veto (3/5)                        │   │
│  │   └──────┬──────┘                                                    │   │
│  │          │                                                            │   │
│  │          ▼                                                            │   │
│  │   Execute Proposal                                                    │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Safety Mechanisms:                                                          │
│  ├─ Veto power: 3/5 multisig can cancel malicious proposals                │
│  ├─ Timelock: 2-day delay for community response                           │
│  └─ Invariant tests: Ensure no locked funds, proper access control         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


CONTRACT 6: ZKMLVerifier.sol (EZKL-generated)
═════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Purpose: On-chain verification of ML model proofs                          │
│                                                                              │
│  NOTE: This contract is AUTO-GENERATED by EZKL                               │
│  DO NOT write manually!                                                      │
│                                                                              │
│  Generated Code Structure:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  contract ZKMLVerifier {                                             │   │
│  │      // Verification key constants (embedded)                        │   │
│  │      uint256 constant VK_ALPHA_X = 0x...;                           │   │
│  │      uint256 constant VK_ALPHA_Y = 0x...;                           │   │
│  │      uint256 constant VK_BETA_X1 = 0x...;                           │   │
│  │      // ... hundreds of constants                                    │   │
│  │                                                                       │   │
│  │      function verify(                                                │   │
│  │          bytes calldata proof,                                       │   │
│  │          uint256[] calldata publicSignals                           │   │
│  │      ) public view returns (bool) {                                 │   │
│  │          // Groth16 pairing check on BN254 curve                    │   │
│  │          // Assembly-optimized for gas efficiency                    │   │
│  │          // Returns true if proof is valid                          │   │
│  │      }                                                               │   │
│  │  }                                                                   │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Gas Cost: ~250K gas per verification                                        │
│  Optimization: Batch proofs to reduce per-proof cost to ~50K                │
│                                                                              │
│  Public Signals:                                                             │
│  ├─ publicSignals[0]: Risk score (0-100)                                   │
│  └─ publicSignals[1...n]: Contract feature hashes (optional)               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```


### Foundry Testing Suite

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FOUNDRY TESTING STRATEGY                                │
└─────────────────────────────────────────────────────────────────────────────┘

test/
├─ unit/                    # Isolated function tests
│   ├─ AuditRegistry.t.sol
│   │   ├─ test_submitAudit_validProof()
│   │   ├─ test_submitAudit_invalidProof_reverts()
│   │   ├─ test_submitAudit_unauthorizedAuditor_reverts()
│   │   ├─ test_challengeAudit_withinWindow()
│   │   └─ test_getLatestAudit_returnsCorrectData()
│   │
│   ├─ SentinelToken.t.sol
│   │   ├─ test_stake_minAmount()
│   │   ├─ test_unstake_afterLockPeriod()
│   │   ├─ test_slash_onInvalidAudit()
│   │   └─ test_delegate_votingPower()
│   │
│   └─ ZKMLVerifier.t.sol
│       ├─ test_verify_validProof()
│       └─ test_verify_invalidProof_returnsFalse()

├─ integration/             # Multi-contract interactions
│   └─ FullFlow.t.sol
│       ├─ test_fullAuditFlow_stakeSubmitVerify()
│       ├─ test_challengeFlow_stakeChallengeresolve()
│       └─ test_upgradeFlow_proposeVoteExecute()

├─ fuzz/                    # Random input testing
│   └─ FuzzAuditRegistry.t.sol
│       ├─ testFuzz_submitAudit_anyRiskScore(uint96 riskScore)
│       │   // Foundry generates random riskScores
│       │   // Test: 0, 1, 99, 100, 101, MAX_UINT96
│       │   // Catch edge cases humans miss
│       │
│       └─ testFuzz_stake_anyAmount(uint256 amount)

├─ invariant/               # Property-based testing
│   └─ InvariantStaking.t.sol
│       │
│       │   // Invariant: Total staked == sum of all user stakes
│       │   function invariant_totalStakedConsistent() public {
│       │       uint256 sumStakes = 0;
│       │       for (uint i = 0; i < stakers.length; i++) {
│       │           sumStakes += token.stakes(stakers[i]).amount;
│       │       }
│       │       assertEq(token.totalStaked(), sumStakes);
│       │   }
│       │
│       │   // Invariant: No audit without valid ZK proof
│       │   function invariant_allAuditsVerified() public {
│       │       // Check every stored audit passed verification
│       │   }
│       │
│       └─  // Foundry runs random sequences of actions,
│           // checks invariant holds after each action

└─ fork/                    # Mainnet fork testing
    └─ ForkMainnet.t.sol
        ├─ test_interactWithRealCCIPRouter()
        └─ test_upgradeOnMainnetState()

Gas Profiling:
──────────────
$ forge snapshot
# Creates .gas-snapshot file

submitAudit: 95,234 gas
challengeAudit: 45,678 gas
verify: 248,901 gas

Compare before/after optimization:
$ forge snapshot --diff
```


### MVP vs Stretch Goals

|Component|MVP (Must Complete)|Stretch (If Time)|
|---|---|---|
|**Contracts**|AuditRegistry + SentinelToken + ZKMLVerifier|+ CrossChainOracle + Paymaster + Governance|
|**Testing**|Unit tests + fuzz tests|+ Invariant tests + fork tests|
|**Deploy**|Sepolia testnet|+ Multi-chain (Polygon, Arbitrum)|
|**Patterns**|UUPS upgradeable|+ ERC-4337 Account Abstraction|

### Learning Outcomes

- [ ]  Master upgradeable contract patterns (UUPS)
- [ ]  Implement cross-chain messaging (Chainlink CCIP)
- [ ]  Build ERC-4337 account abstraction (cutting-edge!)
- [ ]  Design tokenomics (staking, slashing, governance)
- [ ]  Write advanced Foundry tests (fuzz, invariant, fork)
- [ ]  Gas optimization techniques (storage packing, assembly)

---

## Module 6: System Integration & Deployment

### Goal

Connect all modules into a production-ready system.

### Skills Covered

|Skill|Tool/Framework|Depth|
|---|---|---|
|Containerization|Docker, Docker Compose|Intermediate|
|Orchestration|Kubernetes (stretch)|Basic|
|API Design|FastAPI, REST, GraphQL|Intermediate|
|Frontend|Next.js, RainbowKit|Basic|
|CI/CD|GitHub Actions|Intermediate|
|Monitoring|Prometheus, Grafana|Basic|
|Cloud Deployment|Railway, Fly.io|Basic|

### Docker Compose Configuration

yaml

```yaml
# docker-compose.yml
version: "3.8"

services:
  # ==================== API LAYER ====================
  api:
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - ml-server
    environment:
      - DATABASE_URL=postgresql://sentinel:password@postgres:5432/sentinel
      - REDIS_URL=redis://redis:6379
      - ML_SERVER_URL=http://ml-server:8001
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ==================== ML LAYER ====================
  ml-server:
    build: ./ml
    ports:
      - "8001:8001"
    volumes:
      - ./ml/models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/app/models/sentinel_v1.onnx
      - DEVICE=cuda

  # ==================== AGENT LAYER ====================
  agents:
    build: ./agents
    ports:
      - "8002:8002"
    depends_on:
      - ollama
      - ml-server
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - ML_SERVER_URL=http://ml-server:8001
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ==================== DATABASES ====================
  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=sentinel
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=sentinel
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sentinel"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  # ==================== ASYNC WORKERS ====================
  celery-worker:
    build: ./api
    command: celery -A app.worker worker --loglevel=info
    depends_on:
      - redis
      - postgres
    environment:
      - DATABASE_URL=postgresql://sentinel:password@postgres:5432/sentinel
      - REDIS_URL=redis://redis:6379

  celery-beat:
    build: ./api
    command: celery -A app.worker beat --loglevel=info
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379

  # ==================== MLOps ====================
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts
    volumes:
      - mlflow-data:/mlflow

  # ==================== MONITORING ====================
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    volumes:
      - grafana-data:/var/lib/grafana
      - ./infra/grafana/dashboards:/etc/grafana/provisioning/dashboards

  # ==================== FRONTEND ====================
  frontend:
    build: ./frontend
    ports:
      - "3001:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_CHAIN_ID=11155111  # Sepolia

volumes:
  postgres-data:
  redis-data:
  ollama-data:
  mlflow-data:
  prometheus-data:
  grafana-data:
```

### CI/CD Pipeline

yaml

```yaml
# .github/workflows/ci.yml
name: SENTINEL CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly retraining check

jobs:
  # ==================== ML TESTS ====================
  test-ml:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd ml
          pip install poetry
          poetry install
      
      - name: Run ML tests
        run: |
          cd ml
          poetry run pytest tests/ -v --cov=src
      
      - name: Check model performance
        run: |
          cd ml
          poetry run python scripts/evaluate_model.py --threshold 0.80

  # ==================== CONTRACT TESTS ====================
  test-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Install Foundry
        uses: foundry-rs/foundry-toolchain@v1
      
      - name: Run Foundry tests
        run: |
          cd contracts
          forge test -vvv --gas-report
      
      - name: Check coverage
        run: |
          cd contracts
          forge coverage --report lcov
      
      - name: Run Slither
        uses: crytic/slither-action@v0.3.0
        with:
          target: contracts/

  # ==================== AGENT TESTS ====================
  test-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd agents
          pip install poetry
          poetry install
      
      - name: Run agent tests
        run: |
          cd agents
          poetry run pytest tests/ -v

  # ==================== API TESTS ====================
  test-api:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      
      - name: Run API tests
        run: |
          cd api
          pip install poetry
          poetry install
          poetry run pytest tests/ -v

  # ==================== DEPLOY STAGING ====================
  deploy-staging:
    needs: [test-ml, test-contracts, test-agents, test-api]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker images
        run: docker compose build
      
      - name: Push to Registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker compose push
      
      - name: Deploy to Railway
        run: |
          npm install -g @railway/cli
          railway up --environment staging

  # ==================== DEPLOY CONTRACTS ====================
  deploy-contracts:
    needs: [test-contracts]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Install Foundry
        uses: foundry-rs/foundry-toolchain@v1
      
      - name: Deploy to Sepolia
        run: |
          cd contracts
          forge script script/Deploy.s.sol --rpc-url ${{ secrets.SEPOLIA_RPC }} --broadcast --verify
        env:
          PRIVATE_KEY: ${{ secrets.DEPLOYER_PRIVATE_KEY }}
          ETHERSCAN_API_KEY: ${{ secrets.ETHERSCAN_API_KEY }}
```

### MVP vs Stretch Goals

|Component|MVP (Must Complete)|Stretch (If Time)|
|---|---|---|
|**Containers**|Docker Compose (core services)|+ All services + GPU support|
|**API**|REST endpoints|+ GraphQL + WebSocket|
|**Frontend**|Streamlit demo|+ Next.js + RainbowKit|
|**CI/CD**|GitHub Actions (test + build)|+ Auto-deploy + staging env|
|**Monitoring**|Basic health checks|+ Prometheus + Grafana|
|**Deployment**|Local + Sepolia|+ Railway/Fly.io production|

### Learning Outcomes

- [ ]  Orchestrate complex multi-service architecture with Docker Compose
- [ ]  Build production REST APIs with FastAPI
- [ ]  Implement full CI/CD pipeline with GitHub Actions
- [ ]  Deploy smart contracts with Foundry scripts
- [ ]  Set up monitoring dashboards (stretch)
- [ ]  Deploy end-to-end system to cloud (stretch)

---

## Learning Phases & Milestones

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SENTINEL LEARNING PHASES                                │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: FOUNDATION
═══════════════════
Modules: 1 (ML MVP) + 5 (Contracts Core)
Why together: You already know Solidity - quick wins while ML trains

Timeline: Flexible (no pressure)

Morning Routine:
├─ Start ML training job (GPU-intensive)
├─ Training runs in background (2-4 hours)
└─ Monitor via MLflow UI

Evening Routine:
├─ Write Solidity contracts (CPU, different context)
├─ Run Foundry tests
└─ Context switch keeps ADHD engaged

Deliverables:
├─ ✅ CodeBERT model predicting vulnerabilities (80%+ F1)
├─ ✅ AuditRegistry.sol deployed on Sepolia
├─ ✅ SentinelToken.sol with staking
└─ ✅ ZKMLVerifier.sol (placeholder, real in Phase 2)

Milestone Demo:
"Upload contract → get vulnerability score → store on-chain"

───────────────────────────────────────────────────────────────────────────────

PHASE 2: VERIFICATION LAYER
═══════════════════════════
Module: 2 (ZKML)
Depends on: Phase 1 (need trained model)

Focus: This is novel territory - expect debugging time

Deliverables:
├─ ✅ Tiny proxy model (<5K params)
├─ ✅ EZKL pipeline working locally
├─ ✅ ZK proofs generated in <2 minutes
└─ ✅ On-chain verification passing

Milestone Demo:
"Generate proof → verify on Sepolia → trustless audit"

───────────────────────────────────────────────────────────────────────────────

PHASE 3: INTELLIGENCE LAYER
═══════════════════════════
Module: 4 (Agents MVP)
Can run parallel with Phase 2

Focus: RAG + single agent, then expand

Deliverables:
├─ ✅ FAISS index built (1000+ exploits)
├─ ✅ Single LangChain agent working
├─ ✅ 3 tools integrated (ML, Slither, RAG)
└─ ✅ Basic audit report generation

Milestone Demo:
"Chat: 'Analyze this contract' → structured report"

───────────────────────────────────────────────────────────────────────────────

PHASE 4: PRODUCTION LAYER
═════════════════════════
Modules: 3 (MLOps) + 6 (Integration)
Depends on: Phases 1, 2, 3 working

Focus: Connecting everything, monitoring

Deliverables:
├─ ✅ MLflow tracking all experiments
├─ ✅ DVC versioning datasets
├─ ✅ Docker Compose running all services
├─ ✅ GitHub Actions CI/CD
└─ ✅ Basic API endpoints

Milestone Demo:
"docker compose up → full system running"

───────────────────────────────────────────────────────────────────────────────

PHASE 5: STRETCH GOALS
══════════════════════
Add features based on interest/time

Options:
├─ Add GNN path to Module 1 (dual-path fusion)
├─ Multi-agent CrewAI system (5 agents)
├─ Hybrid RAG (BM25 + dense + reranking)
├─ Chainlink CCIP cross-chain
├─ ERC-4337 gasless audits
├─ Dagster pipeline orchestration
├─ Evidently drift detection
├─ Feast feature store
├─ Kubernetes deployment
└─ Next.js frontend with RainbowKit

Pick based on:
├─ What excites you most (ADHD dopamine)
├─ What's most marketable (job interviews)
└─ What fills skill gaps
```

### Checkpoint Validation

After each phase, validate with these questions:

|Phase|Validation Questions|
|---|---|
|**Phase 1**|Can you explain how CodeBERT tokenizes Solidity? Can you write a fuzz test in Foundry?|
|**Phase 2**|Can you explain what a ZK-SNARK proves? Can you debug an EZKL constraint failure?|
|**Phase 3**|Can you explain RAG retrieval strategies? Can you design a prompt for tool-calling?|
|**Phase 4**|Can you explain MLflow's model registry? Can you debug Docker networking issues?|

---

## Fallback Plans

### Module-Specific Fallbacks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FALLBACK DECISION TREE                               │
└─────────────────────────────────────────────────────────────────────────────┘

MODULE 1: ML CORE
═════════════════

Problem: GNN training diverges (loss > 2.0 after 10 epochs)
├─ Fallback 1: Simplify architecture
│   └─ Use GCN instead of DR-GCN (fewer hyperparameters)
├─ Fallback 2: Reduce scope
│   └─ Train Transformer-only path (still 80% of learning value)
└─ Fallback 3: Use pre-trained
    └─ Load CodeBERT embeddings, train only classifier head

Problem: VRAM overflow during dual-path training
├─ Fallback 1: Gradient checkpointing
│   └─ torch.utils.checkpoint.checkpoint()
├─ Fallback 2: Reduce batch size
│   └─ From 16 → 8 → 4
├─ Fallback 3: Train paths separately
│   └─ Freeze one while training fusion layer
└─ Fallback 4: Mixed precision
    └─ torch.cuda.amp.autocast()

Problem: Accuracy < 70% after full training
├─ Step 1: Analyze confusion matrix
│   └─ Which vulnerability classes are confused?
├─ Step 2: Check for data leakage
│   └─ Are train/test splits contaminated?
├─ Step 3: Expand dataset
│   └─ Add SolidiFI synthetic vulnerabilities
└─ Step 4: Simplify target
    └─ Binary classification (vulnerable vs. safe) first

Problem: Training takes > 24 hours per epoch
├─ Fallback 1: Use Google Colab Pro with A100
│   └─ Cost: $50/month, 8x faster
├─ Fallback 2: Reduce dataset size
│   └─ Sample 10K contracts instead of 35K
├─ Fallback 3: Use smaller CodeBERT
│   └─ DistilCodeBERT (40% faster, 95% performance)
└─ Fallback 4: Cache embeddings
    └─ Pre-compute CodeBERT features, save to disk

───────────────────────────────────────────────────────────────────────────────

MODULE 2: ZKML
══════════════

Problem: EZKL installation fails
├─ Fallback 1: Use Docker container
│   └─ Official EZKL Docker image
├─ Fallback 2: Use pre-built binaries
│   └─ Download from GitHub releases
└─ Fallback 3: Simplified mock verifier
    └─ Still demonstrate concept, note limitation

Problem: Proof generation takes > 5 minutes
├─ Fallback 1: Reduce model size
│   ├─ 5K params → 1K params
│   └─ 3 layers → 2 layers
├─ Fallback 2: Lower precision
│   ├─ Float32 → Float16
│   └─ Accept some accuracy loss
├─ Fallback 3: Reduce input size
│   └─ Hash contract to 32 bytes instead of full features
└─ Fallback 4: Use CPU-only proof generation
    └─ Slower but more stable

Problem: On-chain verification gas > 1M
├─ Fallback 1: Use optimized EZKL settings
│   └─ Tune circuit parameters for gas
├─ Fallback 2: Off-chain verification only
│   └─ Store proof hash on-chain, verify API-side
├─ Fallback 3: Batch verification
│   └─ Aggregate multiple proofs into one
└─ Fallback 4: Layer 2 deployment
    └─ Deploy on Arbitrum/Optimism (cheaper gas)

Problem: Proxy model accuracy << full model
├─ Acceptable: 60-70% is OK for demonstration
├─ Fallback 1: Knowledge distillation
│   └─ Train proxy to mimic full model outputs
├─ Fallback 2: Ensemble of tiny models
│   └─ 3-5 small models voting
└─ Fallback 3: Feature engineering
    └─ Better input features to proxy

───────────────────────────────────────────────────────────────────────────────

MODULE 3: MLOps
═══════════════

Problem: MLflow won't start (port conflicts)
├─ Fallback 1: Change port
│   └─ mlflow server --port 5001
├─ Fallback 2: Use SQLite backend
│   └─ Simpler than PostgreSQL
└─ Fallback 3: File-based tracking only
    └─ Log to local files, skip UI

Problem: DVC remote storage costs
├─ Fallback 1: Use local storage only
│   └─ External HDD for datasets
├─ Fallback 2: Use free tiers
│   ├─ Google Drive (15GB free)
│   └─ AWS S3 (5GB free first year)
└─ Fallback 3: Git LFS
    └─ For datasets < 2GB

Problem: Dagster pipelines too complex
├─ Fallback 1: Use simple Python scripts
│   └─ train.py, evaluate.py with cron
├─ Fallback 2: GitHub Actions for scheduling
│   └─ Workflow dispatch triggers
└─ Fallback 3: Manual orchestration
    └─ Run jobs manually, focus on other modules

Problem: Feast integration not working
├─ This is stretch goal - skip if time-constrained
├─ Fallback 1: Redis only for online features
├─ Fallback 2: Direct database queries
└─ Fallback 3: In-memory feature cache

Problem: Evidently drift detection false positives
├─ Fallback 1: Tune thresholds
│   └─ Increase from 10% to 20% drift tolerance
├─ Fallback 2: Manual inspection
│   └─ Review reports monthly, don't auto-trigger
└─ Fallback 3: Simple statistical tests
    └─ KS-test on prediction distributions

───────────────────────────────────────────────────────────────────────────────

MODULE 4: AI AGENTS
═══════════════════

Problem: LangChain API costs too high
├─ Fallback 1: Use local LLMs
│   ├─ Ollama with Llama 3 8B
│   └─ Free but slower
├─ Fallback 2: Reduce context window
│   └─ Chunk contracts, summarize results
├─ Fallback 3: Single-agent system
│   └─ One agent with all tools
└─ Fallback 4: Use Claude Sonnet 4 (cheaper than GPT-4)
    └─ $3/$15 per million tokens

Problem: RAG retrieval irrelevant results
├─ Fallback 1: Better chunking strategy
│   ├─ Semantic chunking vs. fixed size
│   └─ Overlap chunks by 20%
├─ Fallback 2: Hybrid search
│   └─ BM25 + dense retrieval
├─ Fallback 3: Add reranker
│   └─ Cross-encoder for top-k results
└─ Fallback 4: Manual curated examples
    └─ 50 high-quality examples vs. 10K noisy

Problem: CrewAI agents don't coordinate
├─ Fallback 1: Simplify to sequential
│   └─ Agent1 → Agent2 → Agent3 pipeline
├─ Fallback 2: Use LangGraph instead
│   └─ More control over state machine
├─ Fallback 3: Single-agent with tools
│   └─ One intelligent agent calling multiple tools
└─ Fallback 4: Hard-coded orchestration
    └─ Python script coordinating agents

Problem: FAISS index too large for memory
├─ Fallback 1: Use disk-based index
│   └─ IndexIVFFlat with on-disk storage
├─ Fallback 2: Reduce embedding dimensions
│   ├─ 768 → 384 with PCA
│   └─ Or use smaller embedding model
├─ Fallback 3: Shard index by category
│   └─ Separate indices per vulnerability type
└─ Fallback 4: Use Pinecone/Weaviate
    └─ Managed vector DB (has free tier)

Problem: Slither/Mythril not finding vulnerabilities
├─ Expected - these tools have false negatives
├─ Fallback 1: Add more static analysis tools
│   ├─ Semgrep rules
│   └─ Custom regex patterns
├─ Fallback 2: Focus on ML path
│   └─ Static analysis as supplementary
└─ Fallback 3: Manual pattern matching
    └─ Known vulnerability signatures

───────────────────────────────────────────────────────────────────────────────

MODULE 5: ADVANCED SOLIDITY
═══════════════════════════

Problem: UUPS upgrade fails on testnet
├─ Fallback 1: Use transparent proxy
│   └─ Simpler upgrade mechanism
├─ Fallback 2: Deploy non-upgradeable first
│   └─ Add upgradeability later
└─ Fallback 3: Use Hardhat instead
    └─ Foundry upgrade scripts can be tricky

Problem: Chainlink CCIP not available on testnet
├─ Fallback 1: Mock cross-chain messaging
│   └─ Simulate with events + off-chain relayer
├─ Fallback 2: Use LayerZero
│   └─ Alternative cross-chain protocol
├─ Fallback 3: Single-chain deployment
│   └─ Skip cross-chain for MVP
└─ Fallback 4: Manual message passing
    └─ Operator-triggered cross-chain updates

Problem: ERC-4337 paymaster gas issues
├─ Fallback 1: Standard paymaster
│   └─ Simple gas sponsorship
├─ Fallback 2: User pays gas normally
│   └─ Skip gasless UX for MVP
├─ Fallback 3: Use Biconomy/Pimlico
│   └─ Third-party paymaster services
└─ Fallback 4: Testnet only with faucets
    └─ Demonstrate concept, note production needs work

Problem: Fuzz tests finding unexpected reverts
├─ This is GOOD - fix the contracts!
├─ Fallback 1: Add input validation
│   └─ Require statements for bounds
├─ Fallback 2: Bound fuzz inputs
│   └─ Reasonable ranges only
└─ Fallback 3: Accept and document
    └─ Known limitations in comments

Problem: Gas optimization < 50% savings
├─ Acceptable for learning project
├─ Fallback 1: Use storage packing analyzer
│   └─ Foundry gas reports
├─ Fallback 2: Focus on critical paths
│   └─ Optimize hot functions only
└─ Fallback 3: Document trade-offs
    └─ Readability vs. gas savings

───────────────────────────────────────────────────────────────────────────────

MODULE 6: SYSTEM INTEGRATION
════════════════════════════

Problem: Docker containers won't communicate
├─ Fallback 1: Check Docker network
│   └─ docker network inspect sentinel_default
├─ Fallback 2: Use host networking
│   └─ network_mode: host
├─ Fallback 3: Run services locally
│   └─ Skip Docker, use tmux/screen
└─ Fallback 4: Debug with docker compose logs
    └─ Identify failing service

Problem: API endpoints timing out
├─ Fallback 1: Increase timeouts
│   └─ 30s → 120s for ML inference
├─ Fallback 2: Add caching layer
│   └─ Redis for repeated contracts
├─ Fallback 3: Async endpoints
│   └─ Return job ID, poll for results
└─ Fallback 4: Optimize bottlenecks
    └─ Profile with py-spy

Problem: Frontend won't connect to contracts
├─ Fallback 1: Check contract addresses
│   └─ Update .env with deployed addresses
├─ Fallback 2: Verify ABI exports
│   └─ foundry's out/ folder → frontend/
├─ Fallback 3: Use Streamlit instead
│   └─ Much simpler than Next.js
└─ Fallback 4: CLI-only demo
    └─ Skip frontend for technical demo

Problem: GitHub Actions CI failing
├─ Fallback 1: Run tests locally first
│   └─ make test-all
├─ Fallback 2: Split into separate workflows
│   └─ ML, contracts, agents separate
├─ Fallback 3: Disable slow tests in CI
│   └─ Mark with @pytest.mark.slow
└─ Fallback 4: Skip CI, test manually
    └─ Not ideal but acceptable for learning

Problem: Deployment to Railway/Fly.io fails
├─ Fallback 1: Check logs carefully
│   └─ Often port/env var issues
├─ Fallback 2: Test Docker image locally
│   └─ docker run -p 8000:8000 ...
├─ Fallback 3: Deploy to DigitalOcean
│   └─ More straightforward than PaaS
└─ Fallback 4: Local deployment only
    └─ Demo via ngrok tunnel

───────────────────────────────────────────────────────────────────────────────

GENERAL DECISION HEURISTICS
═══════════════════════════

When to pivot vs. persist:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Persist (debug for 1-2 more days) if:                                 │
│  ✓ Error messages are clear and Googleable                             │
│  ✓ Solution seems close (80% working)                                  │
│  ✓ Learning value is high (core technology)                            │
│  ✓ Issue is in your code, not external tool                            │
│                                                                         │
│  Pivot (use fallback) if:                                               │
│  ✗ No clear error messages (mystery bugs)                              │
│  ✗ Issue is in external tool/library                                   │
│  ✗ Blocking other module progress                                      │
│  ✗ Alternative achieves 80% of learning goal                           │
│  ✗ Time spent > 2 days with no progress                                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘

Priority matrix for modules:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MUST HAVE (Core learning):                                            │
│  • Module 1 - Transformer path (skip GNN if necessary)                 │
│  • Module 5 - Basic contracts (skip upgradeability if necessary)       │
│                                                                         │
│  SHOULD HAVE (Impressive but optional):                                │
│  • Module 2 - ZKML (even if slow/inefficient)                          │
│  • Module 4 - Basic RAG agent (skip multi-agent if necessary)          │
│                                                                         │
│  NICE TO HAVE (Resume boosters):                                       │
│  • Module 3 - MLflow/DVC (can be basic)                                │
│  • Module 6 - Docker Compose (local is fine)                           │
│                                                                         │
│  STRETCH (If time allows):                                             │
│  • GNN path, multi-agent CrewAI, Kubernetes, Next.js frontend          │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```



## Skill Coverage Matrix

```

┌─────────────────────────────────────────────────────────────────────────────┐
│                       SENTINEL SKILL COVERAGE MATRIX                         │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
━━━━━━
🟢 Beginner      (0-3 months experience)
🟡 Intermediate  (3-12 months experience)
🔴 Advanced      (12+ months mastery)
⭐ Expert        (Could teach/architect with this)

📈 Market Demand: [1-5 stars] Job posting frequency
🎯 Project Impact: [Core/High/Medium/Low] Importance to SENTINEL
⏰ Time Investment: Estimated hours to reach proficiency
🔗 Prerequisites: Required prior knowledge



MACHINE LEARNING & AI
══════════════════════

┌─────────────────────┬─────────┬────────┬──────────┬──────┬──────────────────┐
│ Skill               │ Entry   │ Target │ Market   │Impact│ Time (hrs)       │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ PyTorch             │ 🟢      │ 🔴     │ ⭐⭐⭐⭐⭐  │ Core │ 80-120           │
│                     │         │        │          │      │ Pre: Python      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Transformers (Hugging│ 🟢     │ 🟡     │ ⭐⭐⭐⭐⭐  │ Core │ 40-60            │
│ Face)               │         │        │          │      │ Pre: PyTorch     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Graph Neural Networks│ 🟢     │ 🟡     │ ⭐⭐⭐⭐   │ High │ 60-80            │
│ (PyG)               │         │        │          │      │ Pre: PyTorch     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Multi-Label Class.  │ 🟢      │ 🔴     │ ⭐⭐⭐⭐   │ Core │ 30-40            │
│                     │         │        │          │      │ Pre: ML basics   │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Continual Learning  │ 🟢      │ 🟡     │ ⭐⭐⭐    │ Med  │ 40-60            │
│                     │         │        │          │      │ Pre: PyTorch     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ LangChain           │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ High │ 30-50            │
│                     │         │        │          │      │ Pre: Python, LLMs│
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ RAG (Retrieval      │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ High │ 40-60            │
│ Augmented Gen.)     │         │        │          │      │ Pre: Embeddings  │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Vector Databases    │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ High │ 20-30            │
│ (FAISS)             │         │        │          │      │ Pre: Embeddings  │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Multi-Agent Systems │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Med  │ 40-60            │
│ (CrewAI)            │         │        │          │      │ Pre: LangChain   │
└─────────────────────┴─────────┴────────┴──────────┴──────┴──────────────────┘


MLOPS & INFRASTRUCTURE
══════════════════════

┌─────────────────────┬─────────┬────────┬──────────┬──────┬──────────────────┐
│ Skill               │ Entry   │ Target │ Market   │Impact│ Time (hrs)       │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ MLflow              │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ High │ 20-30            │
│                     │         │        │          │      │ Pre: Python      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ DVC (Data Version   │ 🟢      │ 🟡     │ ⭐⭐⭐    │ Med  │ 15-25            │
│ Control)            │         │        │          │      │ Pre: Git         │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Dagster             │ 🟢      │ 🟡     │ ⭐⭐⭐    │ Low  │ 30-40            │
│                     │         │        │          │      │ Pre: Python      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Docker              │ 🟡      │ 🔴     │ ⭐⭐⭐⭐⭐  │ Core │ 40-60            │
│                     │         │        │          │      │ Pre: Linux basics│
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Docker Compose      │ 🟢      │ 🔴     │ ⭐⭐⭐⭐   │ High │ 20-30            │
│                     │         │        │          │      │ Pre: Docker      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Kubernetes          │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ Low  │ 60-100           │
│                     │         │        │          │      │ Pre: Docker      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Prometheus/Grafana  │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Med  │ 25-35            │
│                     │         │        │          │      │ Pre: Metrics     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Evidently AI        │ 🟢      │ 🟡     │ ⭐⭐⭐    │ Med  │ 20-30            │
│                     │         │        │          │      │ Pre: ML basics   │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Feast (Feature Store)│ 🟢     │ 🟡     │ ⭐⭐     │ Low  │ 30-40            │
│                     │         │        │          │      │ Pre: Redis       │
└─────────────────────┴─────────┴────────┴──────────┴──────┴──────────────────┘


BLOCKCHAIN & WEB3
═════════════════

┌─────────────────────┬─────────┬────────┬──────────┬──────┬──────────────────┐
│ Skill               │ Entry   │ Target │ Market   │Impact│ Time (hrs)       │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Solidity            │ 🟡      │ ⭐     │ ⭐⭐⭐⭐⭐  │ Core │ 80-120           │
│                     │         │        │          │      │ Pre: Programming │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Foundry             │ 🟢      │ 🔴     │ ⭐⭐⭐⭐⭐  │ Core │ 40-60            │
│                     │         │        │          │      │ Pre: Solidity    │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ UUPS Proxies        │ 🟢      │ 🔴     │ ⭐⭐⭐⭐   │ High │ 30-40            │
│                     │         │        │          │      │ Pre: Solidity    │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ ERC-4337 (Account   │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ Med  │ 40-60            │
│ Abstraction)        │         │        │          │      │ Pre: Solidity    │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Chainlink CCIP      │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Med  │ 30-50            │
│                     │         │        │          │      │ Pre: Oracles     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ OpenZeppelin        │ 🟢      │ 🔴     │ ⭐⭐⭐⭐⭐  │ High │ 20-30            │
│ Contracts           │         │        │          │      │ Pre: Solidity    │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Smart Contract      │ 🟢      │ 🔴     │ ⭐⭐⭐⭐⭐  │ Core │ 60-80            │
│ Security            │         │        │          │      │ Pre: Solidity    │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Fuzz Testing        │ 🟢      │ 🔴     │ ⭐⭐⭐⭐   │ High │ 30-40            │
│ (Foundry)           │         │        │          │      │ Pre: Testing     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Gas Optimization    │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Med  │ 40-60            │
│                     │         │        │          │      │ Pre: EVM basics  │
└─────────────────────┴─────────┴────────┴──────────┴──────┴──────────────────┘


CRYPTOGRAPHY & ZERO-KNOWLEDGE
══════════════════════════════

┌─────────────────────┬─────────┬────────┬──────────┬──────┬──────────────────┐
│ Skill               │ Entry   │ Target │ Market   │Impact│ Time (hrs)       │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ ZK-SNARKs (Theory)  │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ High │ 40-60            │
│                     │         │        │          │      │ Pre: Math basics │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ EZKL                │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Core │ 30-50            │
│                     │         │        │          │      │ Pre: ZK basics   │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ ONNX                │ 🟢      │ 🟡     │ ⭐⭐⭐    │ High │ 15-25            │
│                     │         │        │          │      │ Pre: PyTorch     │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ zkML Architectures  │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ Core │ 40-60            │
│                     │         │        │          │      │ Pre: ML + ZK     │
└─────────────────────┴─────────┴────────┴──────────┴──────┴──────────────────┘


SOFTWARE ENGINEERING
════════════════════

┌─────────────────────┬─────────┬────────┬──────────┬──────┬──────────────────┐
│ Skill               │ Entry   │ Target │ Market   │Impact│ Time (hrs)       │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Python (Advanced)   │ 🟡      │ ⭐     │ ⭐⭐⭐⭐⭐  │ Core │ 100-150          │
│                     │         │        │          │      │ Pre: Basic Python│
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ FastAPI             │ 🟢      │ 🔴     │ ⭐⭐⭐⭐⭐  │ High │ 30-40            │
│                     │         │        │          │      │ Pre: Python      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Async Python        │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Med  │ 30-40            │
│                     │         │        │          │      │ Pre: Python      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ TypeScript          │ 🟡      │ 🟡     │ ⭐⭐⭐⭐⭐  │ Med  │ 40-60            │
│                     │         │        │          │      │ Pre: JavaScript  │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Next.js/React       │ 🟢      │ 🟡     │ ⭐⭐⭐⭐⭐  │ Low  │ 60-80            │
│                     │         │        │          │      │ Pre: React basics│
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ Git (Advanced)      │ 🟡      │ 🔴     │ ⭐⭐⭐⭐⭐  │ High │ 20-30            │
│                     │         │        │          │      │ Pre: Basic Git   │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ GitHub Actions      │ 🟢      │ 🟡     │ ⭐⭐⭐⭐   │ Med  │ 20-30            │
│                     │         │        │          │      │ Pre: CI/CD basics│
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ pytest              │ 🟢      │ 🔴     │ ⭐⭐⭐⭐   │ High │ 20-30            │
│                     │         │        │          │      │ Pre: Python      │
├─────────────────────┼─────────┼────────┼──────────┼──────┼──────────────────┤
│ System Architecture │ 🟢      │ 🔴     │ ⭐⭐⭐⭐⭐  │ Core │ 80-120           │
│                     │         │        │          │      │ Pre: Many        │
└─────────────────────┴─────────┴────────┴──────────┴──────┴──────────────────┘


TOTAL SKILL HOURS ESTIMATE
═══════════════════════════

MVP (Must-Have Skills):           600-900 hours  (~4-6 months full-time)
MVP + Stretch (All Skills):       1200-1800 hours (~8-12 months full-time)

Realistic Timeline:
├─ Part-time (20 hrs/week):  30-45 weeks (MVP)
├─ Full-time (40 hrs/week):  15-23 weeks (MVP)
└─ Intensive (60 hrs/week):  10-15 weeks (MVP)


SKILL DEPENDENCIES (Learning Order)
════════════════════════════════════

Level 1 (Start Here):
├─ Python (Advanced)
├─ Git (Advanced)
└─ Docker basics

Level 2 (Build On Level 1):
├─ PyTorch
├─ Solidity + Foundry
└─ FastAPI

Level 3 (Parallel Tracks):
├─ Track A: ML → Transformers → GNN → Continual Learning
├─ Track B: Blockchain → UUPS → Security → ERC-4337
└─ Track C: Agents → LangChain → RAG → Multi-Agent

Level 4 (Integration):
├─ MLOps tools (MLflow, DVC)
├─ Docker Compose
└─ CI/CD

Level 5 (Advanced):
├─ ZKML (EZKL + ZK theory)
├─ System Architecture
└─ Production deployment


HIGHEST ROI SKILLS (Job Market + Project)
═══════════════════════════════════════════

Top 10 Skills to Master:
1. ⭐ PyTorch                   - Foundation for ML work
2. ⭐ Solidity + Foundry        - Core blockchain skills
3. ⭐ LangChain/RAG              - Hottest AI skill right now
4. ⭐ Docker/Kubernetes          - Essential DevOps
5. ⭐ Smart Contract Security    - High-demand specialization
6. ⭐ FastAPI                    - Modern Python backend
7. ⭐ Transformers (HF)          - NLP state-of-the-art
8. ⭐ ZK-SNARKs/zkML             - Cutting-edge, differentiator
9. ⭐ System Architecture        - Senior-level skill
10. ⭐ ERC-4337                  - Future of Web3 UX

---
```

