# D1.5 — Operations Doc

**Doc target:** `docs/handbook/10_operations.md`
**Estimated time:** 1h
**Rule:** Every command must be tested live and produce expected output before writing.

---

## Source files to read + live testing (5 files + live verification)

1. `agents/.env` — all env vars
   - Extract: every variable name and its current value
   - Verify: SEPOLIA_RPC_URL, AUDIT_REGISTRY_ADDRESS, SENTINEL_OPERATOR_KEY, MCP_AUDIT_PORT, etc.

2. `ml/src/inference/api.py:74-80` — CHECKPOINT env var
   - 3-level precedence: SENTINEL_CHECKPOINT env > mlops_config.json > hardcoded default
   - Default: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt`

3. `contracts/foundry.toml` — compiler config
   - solc = "0.8.22"
   - libs = ["lib"]
   - remappings: @openzeppelin

4. `zkml/ezkl/settings.json` — EZKL circuit parameters
   - model_instance_shapes: [[1,128],[1,10]]
   - input_scale: 13, param_scale: 13
   - logrows: 15

5. Python environments (3 venvs):
   - `ml/.venv/` — ML + ZKML (torch, transformers, ezkl, torch_geometric)
   - `agents/.venv/` — AGENTS (langgraph, mcp, web3, faiss)
   - `data_module/.venv/` — DATA (sentinel_data, slither)

---

## Sections to write

**1. TL;DR** (4 lines)
```
What: How to start, stop, and debug every service in SENTINEL
Services: ML API (8001), MCP inference (8010), MCP audit (8012), Anvil (8545)
Venvs: ml/.venv, agents/.venv (Poetry), data_module/.venv
Quickstart: see "Starting all services" below
```

**2. Prerequisites** (~0.5 page)
- System requirements:
  - Python 3.12
  - Poetry (for agents)
  - Foundry (forge, anvil, cast) — `curl -L https://foundry.paradigm.xyz | bash`
  - solc-select (for Solidity version management)
  - CUDA GPU with 8GB+ VRAM (for ML inference)
- Python packages (verify installed versions):
  - torch 2.x, torch_geometric, transformers
  - ezkl 23.0.5
  - web3.py, eth_account, eth_abi
  - langgraph, mcp
- Verify commands (must all succeed):
  ```bash
  python --version  # 3.12.x
  which forge && forge --version
  which anvil
  which cast
  source ml/.venv/bin/activate && python -c "import ezkl; print(ezkl.__version__)"  # 23.0.5
  ```

**3. Starting all services** (~1 page)
- ML inference server (port 8001):
  ```bash
  cd ~/projects/sentinel
  source ml/.venv/bin/activate
  TRANSFORMERS_OFFLINE=1 nohup uvicorn ml.src.inference.api:app \
    --host 0.0.0.0 --port 8001 --log-level warning \
    > /tmp/ml_inference.log 2>&1 &
  ```
  Verify: `curl http://localhost:8001/health` returns JSON with `status: ok`

- Anvil local chain (port 8545):
  ```bash
  nohup anvil --port 8545 --host 0.0.0.0 --block-time 1 \
    > /tmp/anvil.log 2>&1 &
  ```
  Verify: `cast block-number --rpc-url http://localhost:8545` returns a number

- MCP inference proxy (port 8010):
  ```bash
  cd ~/projects/sentinel/agents
  nohup poetry run python -m src.mcp.servers.inference_server \
    > /tmp/mcp_inference.log 2>&1 &
  ```
  Verify: `curl http://localhost:8010/health`

- MCP audit server (port 8012):
  ```bash
  cd ~/projects/sentinel/agents
  nohup poetry run python -m src.mcp.servers.audit._server \
    > /tmp/mcp_audit.log 2>&1 &
  ```
  Verify: `curl http://localhost:8012/health`

- Order matters: ML server must start before MCP inference proxy. Anvil must start before MCP audit server.

**4. Local testing with Anvil** (~1 page)
- Deploy contracts (Python script with web3.py — copy-pasteable):
  - Deploy MockZKMLVerifier (for testing) or real ZKMLVerifier (for production)
  - Deploy SentinelToken
  - Deploy AuditRegistry impl
  - Deploy ERC1967Proxy(impl, initialize(verifier, token))
  - Verify: registry.sentinelToken() == token address, registry.zkmlVerifier() == verifier address

- Stake operator:
  ```bash
  # Approve token transfer
  cast send $TOKEN "approve(address,uint256)" $TOKEN 1000000000000000000000 \
    --private-key $KEY --rpc-url http://localhost:8545
  # Stake
  cast send $TOKEN "stake(uint256)" 1000000000000000000000 \
    --private-key $KEY --rpc-url http://localhost:8545
  ```
  Verify: `cast call $TOKEN "stakedBalance(address)" $OPERATOR_ADDR` returns >= 1000e18

- Run an audit (via gateway):
  ```bash
  curl -X POST http://localhost:8011/audit \
    -H "Content-Type: application/json" \
    -d '{"source_code": "...solidity source...", "contract_address": "0x..."}'
  ```
  Or via direct graph invocation (see 06_agents_module.md)

- Verify on-chain:
  ```bash
  cast call $REGISTRY "getLatestAuditV2(address)" $CONTRACT_ADDR \
    --rpc-url http://localhost:8545
  ```

**5. Sepolia deployment** (~0.5 page)
- What changes from Anvil:
  - RPC URL: real Sepolia endpoint (Infura/Alchemy)
  - ETH: real testnet ETH needed for gas
  - ZKMLVerifier: deploy the REAL Halo2Verifier (not Mock)
  - Operator key: funded Sepolia account with staked SNTL
- Deploy order: ZKMLVerifier → SentinelToken → AuditRegistry → ERC1967Proxy
- Gas estimates:
  - ZKMLVerifier: ~5M gas (large contract, 1426 lines)
  - SentinelToken: ~2M gas
  - AuditRegistry: ~2M gas
  - ERC1967Proxy: ~1M gas
  - Total: ~10M gas (~0.1 ETH at 10 gwei)
- Post-deploy: update `agents/.env` with real addresses + Sepolia RPC

**6. Debugging** (~0.5 page)
- Common failures and symptoms:

| Symptom | Cause | Fix |
|---|---|---|
| ML server returns 503 | Model not loaded | Check checkpoint path, TRANSFORMERS_OFFLINE=1 |
| ML server returns 413 | Contract too large | Check MAX_SOURCE_BYTES limit |
| EZKL proof fails | Artifacts stale or missing | Re-run setup_circuit.py |
| EZKL proof fails | Wrong proxy architecture | Re-run distillation + setup |
| forge build fails | solc version mismatch | Check foundry.toml solc, OZ lib version |
| MCP server won't start | Missing .env vars | Check SEPOLIA_RPC_URL, AUDIT_REGISTRY_ADDRESS |
| submitAuditV2 reverts | Operator not staked | Stake MIN_STAKE before submitting |
| submitAuditV2 reverts | Score mismatch | Verify class_score_felts from proof, not PyTorch |
| submitAuditV2 reverts | publicSignals too short | Check proof.json has 138 instances |

- Log file locations:
  - `/tmp/ml_inference.log` — ML API
  - `/tmp/anvil.log` — Anvil chain
  - `/tmp/mcp_inference.log` — MCP inference proxy
  - `/tmp/mcp_audit.log` — MCP audit server
  - `/tmp/distill_full.log` — distillation training

**7. Test commands** (~0.5 page, all copy-pasteable)
```bash
# Agents (634 tests)
cd ~/projects/sentinel/agents && poetry run pytest -q

# ML (214 tests)
cd ~/projects/sentinel && source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 SENTINEL_CHECKPOINT=ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt \
  python -m pytest ml/tests/ -q

# ZKML (37 tests)
cd ~/projects/sentinel && source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 python -m pytest zkml/tests/ -q

# Data (569 tests)
cd ~/projects/sentinel/data_module && .venv/bin/python -m pytest -q

# Contracts (66 tests)
cd ~/projects/sentinel/contracts && forge test
```

---

## Verification checklist
- [ ] Every command in this doc is tested live and produces the expected output
- [ ] Every port number matches `.env` or source
- [ ] Prerequisites list matches actual installed versions (run `--version` for each)
- [ ] Deploy script works on a fresh Anvil instance
- [ ] Stake command results in stakedBalance >= MIN_STAKE
- [ ] All 5 test commands produce their stated pass counts
- [ ] Debugging table: every "fix" actually fixes the symptom
- [ ] Gas estimates are reasonable (cross-check with `forge test --gas-report`)
