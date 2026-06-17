# Setup Plan — Environment Configuration

**Duration:** 2-3 hours (one-time setup)  
**Goal:** Get all components running (LM Studio, ML API, 4 MCP servers) and verified working

---

## Prerequisites Verification (15 minutes)

Run these checks first:

```bash
# Check Python environment
python --version  # Should be 3.10+
poetry --version  # Should be 1.0+

# Check agents venv
cd ~/projects/sentinel/agents && poetry env info
# Should show: Using Python 3.12.x

# Check ML checkpoint exists
ls -lh ~/projects/sentinel/ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt
# Should exist, ~280MB

# Check RAG index exists
ls -lh ~/projects/sentinel/agents/data/index/
# Should show faiss.index + bm25.pickle

# Check Slither + Aderyn installed
cd ~/projects/sentinel/agents && poetry run python -c "from slither.slither import Slither; from aderyn import Aderyn; print('OK')"
# Should print OK
```

---

## Step 1: LM Studio Setup (30 minutes)

**Option A: Local LM Studio**

1. Download LM Studio from https://lmstudio.ai/
2. Install on your machine
3. Launch LM Studio
4. Download a model (recommended: Qwen2.5-7B or Mistral-7B)
   - Start menu → Model Library → Search for model → Download
   - Expect: 15-20 min download + load time
5. Verify it's running at `http://localhost:1234`
   ```bash
   curl http://localhost:1234/v1/models
   # Should return list of loaded models
   ```

**Option B: OpenAI API**

1. Get your OpenAI API key
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Update `agents/src/llm/client.py` to use OpenAI instead of LM Studio

**Recommendation:** Use LM Studio (no API costs, full control)

---

## Step 2: ML Inference API (30 minutes)

**Terminal 1: Start ML API**

```bash
cd ~/projects/sentinel/ml

# Activate venv
source .venv/bin/activate

# Start API on port 8001 (MLOps standard, matches agents inference_server expectations)
uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001

# Expected output:
# INFO: Loaded checkpoint: GCB-P1-Run12-v3dospatched-20260613_FINAL.pt
# INFO: Model on device: cuda
# INFO:     Application startup complete
# Uvicorn running on http://127.0.0.1:8001
```

**Verify it's working:**

```bash
# In another terminal, test the API
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code":"contract Test { function test() public {} }"}'

# Should return:
# {
#   "label": "safe",
#   "probabilities": {...},
#   ...
# }
```

**Keep this terminal running in the background.**

---

## Step 3: Launch MCP Servers (4 servers, 45 minutes)

**Terminal 2: Inference MCP Server (:8010)**

```bash
cd ~/projects/sentinel/agents
poetry run python src/mcp/servers/inference_server.py

# Expected output:
# INFO: Starting inference_server MCP on port 8010
# INFO: Ready for SSE connections
```

**Terminal 3: RAG MCP Server (:8011)**

```bash
cd ~/projects/sentinel/agents
poetry run python src/mcp/servers/rag_server.py

# Expected output:
# INFO: Starting rag_server MCP on port 8011
# INFO: RAG index loaded (726 documents)
```

**Terminal 4: Audit MCP Server (:8012)**

```bash
cd ~/projects/sentinel/agents
poetry run python src/mcp/servers/audit_server.py

# Expected output:
# INFO: Starting audit_server MCP on port 8012
# INFO: Ready for on-chain queries
```

**Terminal 5: Graph Inspector MCP Server (:8013)**

```bash
cd ~/projects/sentinel/agents
poetry run python src/mcp/servers/graph_inspector_server.py

# Expected output:
# INFO: Starting graph_inspector_server MCP on port 8013
# INFO: Ready for graph analysis
```

**Verify all 4 servers are running:**

```bash
# In a new terminal, check all ports are open
lsof -i :8010 && echo "✓ inference" || echo "✗ inference"
lsof -i :8011 && echo "✓ rag" || echo "✗ rag"
lsof -i :8012 && echo "✓ audit" || echo "✗ audit"
lsof -i :8013 && echo "✓ graph" || echo "✗ graph"

# Should show all 4 ✓
```

---

## Step 4: Connectivity Check (15 minutes)

**Create test script:** `agents/scripts/test_connectivity.py`

```python
import aiohttp
import asyncio

async def check_all():
    async with aiohttp.ClientSession() as session:
        services = {
            'ml': 'http://localhost:8001/health',
            'inference': 'http://localhost:8010/health',
            'rag': 'http://localhost:8011/health',
            'audit': 'http://localhost:8012/health',
            'graph': 'http://localhost:8013/health',
        }
        
        for name, url in services.items():
            try:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        print(f"✓ {name:12} OK")
                    else:
                        print(f"✗ {name:12} HTTP {resp.status}")
            except Exception as e:
                print(f"✗ {name:12} {str(e)}")

asyncio.run(check_all())
```

**Run it:**

```bash
cd ~/projects/sentinel/agents
poetry run python scripts/test_connectivity.py

# Expected output:
# ✓ ml           OK
# ✓ inference    OK
# ✓ rag          OK
# ✓ audit        OK
# ✓ graph        OK
```

**If any fail:**
- Check the corresponding terminal for error messages
- Common issues:
  - Port already in use → find `lsof -i :PORT` and kill old process
  - ML API crashed → check checkpoint path and disk space
  - RAG index missing → run `agents/src/rag/build_index.py`

---

## Step 5: Get Test Contracts (10 minutes)

Create test contracts to audit:

**Contract 1: Simple Safe (ERC20 baseline)**

Create `agents/test_contracts/erc20_safe.sol`:

```solidity
pragma solidity ^0.8.0;

contract SafeToken {
    string public name = "Safe Token";
    uint256 public totalSupply;
    mapping(address => uint256) public balances;
    
    function mint(address to, uint256 amount) public {
        balances[to] += amount;
        totalSupply += amount;
    }
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}
```

**Contract 2: Intentionally Vulnerable**

Create `agents/test_contracts/vulnerable_reentrant.sol`:

```solidity
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;  // REENTRANCY BUG!
    }
}
```

**Contract 3: Real contract (optional)**

Download from Etherscan or use a well-known contract (Uniswap V2 Router, etc.)

---

## Setup Verification Checklist

- [ ] **LM Studio**
  - [ ] Running at :1234
  - [ ] Model loaded
  - [ ] Responds to API calls

- [ ] **ML API**
  - [ ] Running at :8001 (MLOps standard, matches agents inference_server)
  - [ ] Checkpoint loaded: Run 12 FINAL
  - [ ] /predict endpoint responds
  - [ ] /health endpoint responds

- [ ] **MCP Servers**
  - [ ] inference_server running at :8010
  - [ ] rag_server running at :8011
  - [ ] audit_server running at :8012
  - [ ] graph_inspector_server running at :8013

- [ ] **Connectivity**
  - [ ] All 5 services responding to /health

- [ ] **Test Contracts**
  - [ ] erc20_safe.sol exists
  - [ ] vulnerable_reentrant.sol exists
  - [ ] (optional) Real contract downloaded

- [ ] **Monitoring Setup**
  - [ ] Can monitor memory with `top` or `htop`
  - [ ] Can see logs from each service

---

## Monitoring Commands

Keep these handy during testing:

```bash
# Monitor memory + CPU
top -b -n 1 | head -20

# Check specific process memory
ps aux | grep python | grep inference

# Monitor ports in real-time
watch -n 1 'lsof -i :8001,:8010,:8011,:8012,:8013'

# Monitor logs (if services log to files)
tail -f ~/projects/sentinel/ml/logs/run12.log
tail -f ~/projects/sentinel/agents/logs/*.log
```

---

## Common Issues & Solutions

| Issue | Symptom | Fix |
|-------|---------|-----|
| LM Studio not responding | `curl :1234` → refused | Restart LM Studio app |
| Port in use | `Address already in use` | `lsof -i :PORT \| kill` |
| ML API OOM | Out of memory during inference | Reduce batch size or restart |
| RAG index missing | `FileNotFoundError: index/` | Run `build_index.py` |
| MCP server crash | Process exits | Check logs for error, restart |
| Timeout in LLM | cross_validator hangs | Increase timeout or reduce model size |
| No GPU memory | Slow inference | Use CPU mode (slower but works) |

---

## Next Steps

When all 5 services are running and verified:

1. ✓ Setup complete
2. → Open `02_EXECUTION_PLAN.md`
3. → Run first audit on test contract

---

## Duration Check

- ✓ LM Studio setup: 30 min
- ✓ ML API startup: 10 min
- ✓ MCP servers: 15 min (parallel, so 15 total)
- ✓ Connectivity check: 10 min
- ✓ Test contracts: 10 min
- **Total: ~75 minutes (1.25 hours)**

If everything goes smoothly, you can be ready for execution in 1-2 hours.

---

**Setup complete when all 5 services show ✓ in connectivity check.**

**Next:** `02_EXECUTION_PLAN.md` →
