# SENTINEL Quick Start Guide

## Getting Started in 30 Minutes

This guide will help you set up your development environment and start working on SENTINEL immediately.

---

## Prerequisites

### Required Software
- **Python 3.10+** (for ML and agents)
- **Node.js 18+** (for frontend, if building)
- **Git** (version control)
- **CUDA-capable GPU** (recommended for ML, not required)
- **Docker & Docker Compose** (for deployment)

### Required Accounts
- **GitHub** (for version control)
- **Hugging Face** (for pre-trained models) - free account
- **Alchemy/Infura** (for Ethereum RPC) - free tier
- **OpenAI/Anthropic** (for AI agents) - API key

---

## Day 1: Environment Setup

### Step 1: Clone and Setup Project Structure

```bash
# Create project directory
mkdir -p ~/projects/sentinel
cd ~/projects/sentinel

# Initialize git
git init
git branch -M main

# Create directory structure (following architecture doc)
mkdir -p ml/{data/{raw,processed,embeddings},src/{data,models,training,inference,continual},notebooks,tests,configs}
mkdir -p zkml/{models,ezkl,artifacts,tests}
mkdir -p mlops/{pipelines/{dagster,scripts},feature_store/{feast,redis},monitoring/{evidently,prometheus,grafana},mlflow,dvc}
mkdir -p agents/{src/{agents,tools,rag,orchestration,prompts},knowledge_base/{exploits,best_practices,faiss_index},tests}
mkdir -p contracts/{src,test/{unit,integration,fuzz,invariant},script,lib}
mkdir -p api/{src/{routes,services,models,middleware,config},tests}
mkdir -p frontend/{src/{app,components,hooks,lib}}
mkdir -p infra/{kubernetes/{deployments,services,configmaps},terraform}
mkdir -p scripts docs .github/workflows

# Create README files
touch README.md
touch ml/README.md zkml/README.md mlops/README.md agents/README.md contracts/README.md api/README.md frontend/README.md
```

### Step 2: Setup Python Environment (Module 1 - ML)

```bash
cd ml

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch with CUDA (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
# pip install torch torchvision torchaudio

# Install core ML dependencies
pip install \
    torch-geometric \
    transformers[torch] \
    datasets \
    accelerate \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    mlflow \
    dvc \
    evidently \
    py-solidity-ast \
    solc-select

# Install development tools
pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Save dependencies
pip freeze > requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name=sentinel-ml
```

### Step 3: Install Solidity Tools (Module 5 - Contracts)

```bash
cd ../contracts

# Install Foundry
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Initialize Foundry project
forge init --force

# Install OpenZeppelin contracts
forge install OpenZeppelin/openzeppelin-contracts
forge install OpenZeppelin/openzeppelin-contracts-upgradeable

# Create remappings
echo "@openzeppelin/=lib/openzeppelin-contracts/" > remappings.txt
echo "@openzeppelin-upgradeable/=lib/openzeppelin-contracts-upgradeable/" >> remappings.txt

# Setup solc for Python (for ML preprocessing)
cd ../ml
source .venv/bin/activate
solc-select install 0.8.20
solc-select use 0.8.20
```

### Step 4: Setup MLflow & DVC (Module 3 - MLOps)

```bash
cd ../ml

# Initialize DVC
dvc init

# Setup DVC remote (local for now, can switch to S3 later)
mkdir -p ../data_storage
dvc remote add -d local_remote ../data_storage

# Start MLflow server (in separate terminal)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# Access MLflow UI at http://localhost:5000
```

### Step 5: Install Agent Dependencies (Module 4 - Agents)

```bash
cd ../agents

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install LangChain and related
pip install \
    langchain \
    langchain-openai \
    langchain-anthropic \
    langchain-community \
    crewai \
    faiss-cpu \
    sentence-transformers \
    beautifulsoup4 \
    requests

# Install static analysis tools
pip install slither-analyzer mythril

# Save dependencies
pip freeze > requirements.txt
```

---

## Week 1: First Working Prototype

### Monday: Data Acquisition

```bash
cd ~/projects/sentinel/ml

# Create data download script
cat > scripts/download_data.sh << 'EOF'
#!/bin/bash

# Create data directories
mkdir -p data/raw data/processed

# Download Kaggle dataset (requires Kaggle API setup)
# First: pip install kaggle
# Then: setup ~/.kaggle/kaggle.json with your API key

kaggle datasets download -d xenogearcap/smart-contract-vulnerabilities
unzip smart-contract-vulnerabilities.zip -d data/raw/

# Alternative: Manual download
echo "If Kaggle API fails, manually download from:"
echo "https://www.kaggle.com/datasets/xenogearcap/smart-contract-vulnerabilities"
echo "Place in: ml/data/raw/"
EOF

chmod +x scripts/download_data.sh
./scripts/download_data.sh

# Track data with DVC
dvc add data/raw/contracts.csv
git add data/raw/contracts.csv.dvc .gitignore
git commit -m "Add raw dataset"
```

### Tuesday-Wednesday: First Jupyter Notebook (EDA)

```bash
cd ~/projects/sentinel/ml
source .venv/bin/activate
jupyter notebook
```

Create `notebooks/01_exploratory_data_analysis.ipynb`:

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load data
df = pd.read_csv('../data/raw/contracts.csv')
print(f"Dataset shape: {df.shape}")
df.head()

# Cell 3: Check for nulls
df.isnull().sum()

# Cell 4: Vulnerability distribution
vuln_cols = [col for col in df.columns if 'vulnerability' in col.lower()]
vuln_counts = df[vuln_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
vuln_counts.plot(kind='bar')
plt.title('Vulnerability Type Distribution')
plt.xlabel('Vulnerability Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Cell 5: Class imbalance analysis
total_contracts = len(df)
for vuln in vuln_cols:
    count = df[vuln].sum()
    percentage = (count / total_contracts) * 100
    print(f"{vuln}: {count} ({percentage:.2f}%)")

# Cell 6: Multi-label statistics
vulnerabilities_per_contract = df[vuln_cols].sum(axis=1)
print(f"Average vulnerabilities per contract: {vulnerabilities_per_contract.mean():.2f}")
print(f"Max vulnerabilities in one contract: {vulnerabilities_per_contract.max()}")

plt.figure(figsize=(10, 5))
vulnerabilities_per_contract.hist(bins=20)
plt.title('Distribution of Vulnerabilities per Contract')
plt.xlabel('Number of Vulnerabilities')
plt.ylabel('Count')
plt.show()
```

### Thursday-Friday: First Model (Baseline)

Create `ml/src/models/baseline.py`:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BaselineVulnDetector(nn.Module):
    """Simple CodeBERT-based vulnerability detector (MVP)"""

    def __init__(self, num_classes=13, dropout=0.3):
        super().__init__()

        # Load pre-trained CodeBERT
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")

        # Freeze encoder initially (fine-tune later)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Get [CLS] token embedding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # Classify
        logits = self.classifier(cls_embedding)  # [batch, 13]
        return logits

# Usage example
if __name__ == "__main__":
    model = BaselineVulnDetector(num_classes=13)

    # Test forward pass
    batch_size = 4
    seq_len = 512
    dummy_input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)

    logits = model(dummy_input_ids, dummy_attention_mask)
    print(f"Output shape: {logits.shape}")  # Should be [4, 13]

    # Test with sigmoid (multi-label)
    probs = torch.sigmoid(logits)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample prediction: {probs[0]}")
```

### Weekend: First Training Run

Create `ml/src/training/train_baseline.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import mlflow
from tqdm import tqdm

from models.baseline import BaselineVulnDetector
from data.dataset import VulnDataset  # You'll create this

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # [batch, 13]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MLflow tracking
    mlflow.set_experiment("sentinel-baseline")

    with mlflow.start_run(run_name="codebert-baseline-v1"):
        # Log parameters
        mlflow.log_params({
            "model": "CodeBERT",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        })

        # Load data (you'll implement this)
        # train_dataset = VulnDataset('data/processed/train.csv')
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = BaselineVulnDetector(num_classes=13).to(device)

        # Loss function (Binary Cross Entropy for multi-label)
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            # avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            # mlflow.log_metric("train_loss", avg_loss, step=epoch)
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            pass  # Remove this when you have data

        # Save model
        torch.save(model.state_dict(), 'models/baseline_v1.pt')
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
```

---

## Week 1: First Smart Contract

Create `contracts/src/AuditRegistry.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title AuditRegistry
 * @notice Stores smart contract audit results on-chain
 * @dev MVP version - UUPS upgradeability comes later
 */
contract AuditRegistry is Ownable, ReentrancyGuard {

    struct AuditResult {
        uint256 riskScore;           // 0-100
        uint256 timestamp;
        address auditor;
        bytes32 zkProofHash;         // IPFS hash of ZK proof
        string modelVersion;         // e.g., "SENTINEL-v1.0"
        uint8[] vulnerabilities;     // Array of vulnerability IDs
    }

    // Contract address => array of audits
    mapping(address => AuditResult[]) public audits;

    // Events
    event AuditSubmitted(
        address indexed contractAddress,
        uint256 riskScore,
        address indexed auditor,
        uint256 timestamp
    );

    constructor() Ownable(msg.sender) {}

    /**
     * @notice Submit a new audit result
     * @param _contractAddress Address of audited contract
     * @param _riskScore Risk score (0-100)
     * @param _zkProofHash Hash of ZK proof
     * @param _modelVersion Model version string
     * @param _vulnerabilities Array of detected vulnerability IDs
     */
    function submitAudit(
        address _contractAddress,
        uint256 _riskScore,
        bytes32 _zkProofHash,
        string memory _modelVersion,
        uint8[] memory _vulnerabilities
    ) external nonReentrant {
        require(_contractAddress != address(0), "Invalid contract address");
        require(_riskScore <= 100, "Risk score must be <= 100");

        AuditResult memory newAudit = AuditResult({
            riskScore: _riskScore,
            timestamp: block.timestamp,
            auditor: msg.sender,
            zkProofHash: _zkProofHash,
            modelVersion: _modelVersion,
            vulnerabilities: _vulnerabilities
        });

        audits[_contractAddress].push(newAudit);

        emit AuditSubmitted(
            _contractAddress,
            _riskScore,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Get latest audit for a contract
     * @param _contractAddress Address to query
     * @return Latest audit result
     */
    function getLatestAudit(address _contractAddress)
        external
        view
        returns (AuditResult memory)
    {
        require(audits[_contractAddress].length > 0, "No audits found");
        return audits[_contractAddress][audits[_contractAddress].length - 1];
    }

    /**
     * @notice Get number of audits for a contract
     * @param _contractAddress Address to query
     * @return Number of audits
     */
    function getAuditCount(address _contractAddress)
        external
        view
        returns (uint256)
    {
        return audits[_contractAddress].length;
    }
}
```

Create test file `contracts/test/unit/AuditRegistry.t.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../../src/AuditRegistry.sol";

contract AuditRegistryTest is Test {
    AuditRegistry public registry;

    address constant MOCK_CONTRACT = address(0x1234);
    address constant AUDITOR = address(0x5678);

    function setUp() public {
        registry = new AuditRegistry();
    }

    function testSubmitAudit() public {
        uint8[] memory vulns = new uint8[](2);
        vulns[0] = 0;  // Reentrancy
        vulns[1] = 4;  // Access control

        vm.prank(AUDITOR);
        registry.submitAudit(
            MOCK_CONTRACT,
            85,
            bytes32(uint256(12345)),
            "SENTINEL-v1.0",
            vulns
        );

        AuditRegistry.AuditResult memory result = registry.getLatestAudit(MOCK_CONTRACT);

        assertEq(result.riskScore, 85);
        assertEq(result.auditor, AUDITOR);
        assertEq(result.vulnerabilities.length, 2);
    }

    function testCannotSubmitInvalidRiskScore() public {
        uint8[] memory vulns = new uint8[](0);

        vm.expectRevert("Risk score must be <= 100");
        registry.submitAudit(
            MOCK_CONTRACT,
            150,
            bytes32(0),
            "SENTINEL-v1.0",
            vulns
        );
    }

    function testFuzzRiskScore(uint256 score) public {
        vm.assume(score <= 100);

        uint8[] memory vulns = new uint8[](0);

        registry.submitAudit(
            MOCK_CONTRACT,
            score,
            bytes32(0),
            "SENTINEL-v1.0",
            vulns
        );

        AuditRegistry.AuditResult memory result = registry.getLatestAudit(MOCK_CONTRACT);
        assertEq(result.riskScore, score);
    }
}
```

Run tests:

```bash
cd contracts
forge test -vv
forge coverage
```

---

## Essential Commands Cheat Sheet

### ML Development
```bash
# Activate environment
cd ml && source .venv/bin/activate

# Start Jupyter
jupyter notebook

# Start MLflow UI
mlflow ui --port 5000

# Train model
python src/training/train_baseline.py

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
```

### Solidity Development
```bash
cd contracts

# Compile contracts
forge build

# Run tests
forge test -vv

# Run specific test
forge test --match-test testSubmitAudit -vvv

# Run fuzz tests
forge test --fuzz-runs 1000

# Check coverage
forge coverage

# Deploy to Sepolia
forge script script/Deploy.s.sol --rpc-url $SEPOLIA_RPC_URL --broadcast --verify
```

### Git Workflow
```bash
# Daily commits
git add .
git commit -m "feat: implement baseline model"
git push origin main

# Track data with DVC
dvc add data/raw/contracts.csv
git add data/.gitignore data/raw/contracts.csv.dvc
git commit -m "data: add raw dataset"
dvc push
```

---

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use gradient accumulation
```python
# In training script
batch_size = 8  # Reduce from 16
gradient_accumulation_steps = 2  # Accumulate gradients
```

### Issue: Foundry Install Fails
**Solution**: Use Docker
```bash
docker pull ghcr.io/foundry-rs/foundry:latest
alias forge='docker run --rm -v $(pwd):/app -w /app ghcr.io/foundry-rs/foundry:latest forge'
```

### Issue: Solc Version Mismatch
**Solution**: Use solc-select
```bash
solc-select install 0.8.20
solc-select use 0.8.20
```

### Issue: MLflow UI Won't Start
**Solution**: Check port and use SQLite
```bash
# Kill any process on port 5000
lsof -ti:5000 | xargs kill -9

# Start with SQLite backend
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Next Steps

After completing Week 1:
1. Review [ROADMAP.md](ROADMAP.md) for Phase 1 detailed plan
2. Read [Complete Architecture.md](Complete%20Architecture.md) for system design
3. Join relevant Discord/Slack communities for support
4. Set up weekly progress tracking (see ROADMAP.md)
5. Start Phase 1, Week 2: ML Model Foundation

---

## Support & Resources

### Getting Help
- **ML Questions**: PyTorch Forums, r/MachineLearning
- **Solidity Questions**: r/ethdev, Foundry Telegram
- **ZKML Questions**: EZKL Discord
- **General**: Stack Overflow, GitHub Discussions

### Key Documentation
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Foundry Book](https://book.getfoundry.sh/)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)

---

**You're now ready to start building SENTINEL! 🚀**

Focus on small wins, document everything, and don't hesitate to use the fallback plans in ROADMAP.md when stuck.

Good luck!
