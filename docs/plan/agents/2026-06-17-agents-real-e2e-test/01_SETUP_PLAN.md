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

# Check Slither + Aderyn installed (use poetry run, not system Python)
cd ~/projects/sentinel/agents
poetry run slither --version    # expect 0.11.x
poetry run aderyn --version     # expect ≥ 0.4.21

# Check .env has required vars
grep -E '^(LM_STUDIO_BASE_URL|AUDIT_MOCK|LM_STUDIO_API_KEY)' .env
# Expect all 3 lines present
```

---

## Step 1: LM Studio Setup (30 minutes)

**Port is fluid — never hardcode it.** LM Studio's desktop app auto-picks a port to avoid
collisions (was :1234, then :4567, currently :1256 on Ali's machine). Always read from
`LM_STUDIO_BASE_URL` env var.

**Option A: Local LM Studio**

1. Download LM Studio from https://lmstudio.ai/
2. Install on your machine
3. Launch LM Studio
4. Open the **Local Server** tab. Note the **port** it picked (e.g. :1256).
5. Download the 4 required models (see `00_MASTER_TEST_PLAN.md` §LLM Model Selection):
   - `gemma-4-e2b-it` (FAST)
   - `qwen3.5-9b-ud` (STRONG — used by cross_validator)
   - `qwen2.5-coder-7b-instruct` (CODER)
   - `text-embedding-nomic-embed-text-v1.5` (EMBED)
   - Expect: 15-20 min download per large model
6. Add to `agents/.env`:
   ```bash
   LM_STUDIO_BASE_URL="http://127.0.0.1:<THE_PORT_YOU_SAW>/v1"
   LM_STUDIO_API_KEY="lm-studio"
   AUDIT_MOCK=true
   ```
7. Verify LM Studio is responding:
   ```bash
   curl -s $LM_STUDIO_BASE_URL/models | jq '.data[].id'
   # Expect: gemma-4-e2b-it, qwen3.5-9b-ud, qwen2.5-coder-7b-instruct, text-embedding-nomic-embed-text-v1.5
   ```

**On WSL2 + Windows host:** if LM Studio runs as a Windows app (not WSL), replace
`127.0.0.1` with the WSL2 gateway IP:
```bash
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# Use that IP, e.g. LM_STUDIO_BASE_URL="http://172.21.16.1:1256/v1"
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
  - [ ] Running (port written to `LM_STUDIO_BASE_URL` in `agents/.env`)
  - [ ] All 4 model IDs loaded (see `00_MASTER` §LLM Model Selection)
  - [ ] Responds to `/v1/models`

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
| LM Studio not responding | `curl $LM_STUDIO_BASE_URL/models` → refused | Restart LM Studio app, recheck port |
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

---

## Learning Outcomes (Plan Onboarding)

→ You now know: The original plan hardcoded `http://localhost:1234` for LM Studio — but the actual code (`client.py:60`) reads `LM_STUDIO_BASE_URL` env var with a stale WSL2 gateway IP as fallback. The desktop app also picks a fluid port (Ali is currently on :1256). Using env vars + verifying with `/v1/models` is the only stable way.

→ You now know: `AUDIT_MOCK=true` is critical — `audit_server.py` defaults to real Sepolia RPC mode, which would hang at startup without a `SEPOLIA_RPC_URL`. Mock mode returns canned responses so the graph can complete without blockchain access.

→ You now know: `slither --version` and `aderyn --version` are added to prereqs because Run 12 eval found that pre-0.4.21 Slither versions produced 6,782 errors across 47K contracts (not model bugs — tool bugs). Catching this BEFORE running E2E saves debugging time.
