# D1.3 — Cross-Module Integration Doc

**Doc target:** `docs/handbook/07_cross_module.md`
**Estimated time:** 1h
**Rule:** Every claim verified against source code. This is the most important doc — no other doc covers the integration.

---

## Source files to read before writing (12 files — spans all modules)

### Boundary 1: ML → ZKML (3 files)
1. `ml/src/models/sentinel_model.py:586-593` — `fusion_embedding` key in aux_dict
   - This is where the 128-dim vector is born: `fused_eye = self.fusion(node_embs, batch, token_embs, flat_mask)`
   - Added to aux_dict at line 592: `"fusion_embedding": fused_eye`

2. `ml/src/inference/api.py:235-240,467-510` — `/fusion-embedding` endpoint
   - FusionEmbeddingResponse: fusion_embedding (128 floats), num_nodes, num_edges, model_hash, windows_used
   - Calls predictor.predict_fusion_embedding(source_code)

3. `ml/src/inference/predictor.py:830-910` — `predict_fusion_embedding()` method
   - Runs model.forward(return_aux=True)
   - Extracts aux["fusion_embedding"] → [128] float list
   - Returns dict with model_hash for provenance

### Boundary 2: ZKML → Contracts (4 files)
4. `zkml/src/distillation/proxy_model.py:104-108` — frozen dims
   - FROZEN_INPUT_DIM=128 (must match fusion output)
   - FROZEN_NUM_CLASSES=10 (must match contract NUM_CLASSES)

5. `zkml/src/ezkl/run_proof.py:117-200` — proof generation
   - Takes fusion embedding as input (128 floats)
   - gen_witness → prove → verify
   - Decodes 10 output field elements
   - Parses proof.json → 138 publicSignals

6. `zkml/src/ezkl/extract_calldata.py:104` — signal layout
   - TOTAL_SIGNALS = 138 (128 + 10)
   - publicSignals[0..127] = fusion features
   - publicSignals[128..137] = class scores

7. `contracts/src/AuditRegistry.sol:140-175` — submitAuditV2 receiver
   - publicSignals.length >= 138 check (line 158-161, BEFORE verifyProof)
   - verifyProof(proof, publicSignals) (line 159-161)
   - Guard 3: loop publicSignals[128+i] == classScores[i] for i in 0..9 (line 163-173)

### Boundary 3: AGENTS → on-chain (3 files)
8. `agents/src/mcp/servers/audit/_submit.py:80-216` — full submit pipeline
   - Step 1: GET /fusion-embedding → 128 floats + model_hash
   - Step 2: Proxy model local inference → 10 class scores
   - Step 3: EZKL gen_witness → prove → verify (inline, not subprocess)
   - Step 3b: Build provenance manifest
   - Step 4: web3.py transact submitAuditV2
   - Critical fix: class_score_felts overwritten from proof's publicSignals[128:] (line ~197)

9. `agents/src/mcp/servers/audit/_submit.py:219-265` — provenance manifest
   - build_provenance_manifest(): teacher_model_hash, proxy_checkpoint_hash, fusion_embedding_hash, class_scores, timestamp, operator_address, signature
   - EIP-191 signed with operator key
   - Bridges the gap: ZK proof proves proxy, manifest claims the fusion came from teacher

10. `agents/src/orchestration/nodes/synthesizer.py:372-386` — on_chain section
    - final_report["on_chain"]: submitted, tx_hash, proof_hash, class_scores, class_score_felts, model_hash, provenance

### Supporting (2 files)
11. `agents/src/orchestration/nodes/ml_assessment.py:118` — model_hash propagation
    - Extracts model_hash from ML API response → state["model_hash"]
    - Flows through to synthesizer → final_report["model_provenance"]["model_hash"]

12. `agents/.env` — operator config
    - SENTINEL_OPERATOR_KEY, AUDIT_REGISTRY_ADDRESS, SEPOLIA_RPC_URL

---

## Sections to write

**1. TL;DR** (3 lines)
```
What: The full flow — Solidity source → ML fusion(128) → proxy(10) → EZKL proof → AuditRegistry.submitAuditV2
Key insight: ZK proof proves the PROXY, not the teacher. Provenance manifest bridges the gap off-chain.
Read this: if you only read one handbook doc, read this one.
```

**2. The full pipeline diagram** (~1 page)
- ASCII art showing every step with:
  - Module name (DATA/ML/ZKML/CONTRACTS/AGENTS)
  - Function name
  - Data shape at each boundary
- Flow:
  ```
  [Solidity source]
       │ ml_assessment / predict_fusion_embedding
       ▼
  [128-dim fusion embedding] ← ML module (CrossAttentionFusion output)
       │ _submit.py Step 2: proxy(fusion)
       ▼
  [10 class logits] ← ZKML module (ProxyModel forward)
       │ _submit.py Step 3: ezkl.gen_witness → prove → verify
       ▼
  [proof.json: hex_proof + 138 publicSignals] ← ZKML module (EZKL)
       │ _submit.py Step 4: web3.py transact
       ▼
  [AuditRegistry.submitAuditV2] ← CONTRACTS module (on-chain)
       │
       ▼
  [tx_hash → final_report["on_chain"]] ← AGENTS module
  ```

**3. Boundary 1: ML → ZKML** (~1 page)
- What crosses: 128-dim float vector (CrossAttentionFusion output)
- Where it's born: `sentinel_model.py:592` — `fused_eye = self.fusion(...)` → aux_dict["fusion_embedding"]
- How it crosses: `predictor.py:predict_fusion_embedding()` → HTTP `/fusion-embedding` → `_submit.py` reads JSON
- What's NOT proved: that this vector came from this specific teacher checkpoint
- How provenance bridges: manifest records teacher_model_hash + fusion_embedding_hash, signed by operator

**4. Boundary 2: ZKML → Contracts** (~1 page)
- What crosses: proof.json → hex_proof (bytes) + public_signals (uint256[138])
- Where: `run_proof.py` generates → `_submit.py` decodes → `submitAuditV2(classScores[10], proof, publicSignals[138], modelHash)`
- The binding guarantee: `class_score_felts = all_public_signals[128:]` — scores come FROM the proof, not from PyTorch
- Why this matters: EZKL's fixed-point sigmoid (lookup table, scale=13) can differ from PyTorch's float32 sigmoid after rounding. Using proof's values guarantees Guard 3 passes.
- Guard 3 on-chain: `publicSignals[128+i] == classScores[i]` for i in 0..9
- Length check: `publicSignals.length >= 138` BEFORE verifyProof (fail fast)

**5. Boundary 3: AGENTS → on-chain** (~1 page)
- What crosses: signed transaction from operator key
- Where: `_submit.py:170-210` — web3.py `account.sign_transaction` + `send_raw_transaction`
- Pre-conditions:
  - SENTINEL_OPERATOR_KEY set in .env
  - Operator has Sepolia ETH for gas
  - Operator staked >= MIN_STAKE (1000 SNTL) on AuditRegistry
- Result: tx_hash → `final_report["on_chain"]["tx_hash"]`
- Structured degraded return (Rule 5C): if any step fails, returns {status, failed_step, reason}

**6. Cross-module constants** (table)
- Verify each against source:

| Constant | Value | Source location |
|---|---|---|
| INPUT_OFFSET | 128 | `AuditRegistry.sol:40` |
| NUM_CLASSES | 10 | `AuditRegistry.sol:39` |
| TOTAL_SIGNALS | 138 | `extract_calldata.py:44` |
| SCALE | 8192 | `extract_calldata.py:46` |
| EZKL_PARAM_LIMIT | 12000 | `proxy_model.py:73` |
| CIRCUIT_VERSION | "v2.0" | `proxy_model.py:68` |
| FROZEN_INPUT_DIM | 128 | `proxy_model.py:105` |
| FROZEN_NUM_CLASSES | 10 | `proxy_model.py:108` |
| MIN_STAKE | 1000e18 | `SentinelToken.sol:14` |
| SCALE_FACTOR | 8192 | `_config.py:76` |

**7. Common failure modes** (~0.5 page)
- Score mismatch: EZKL sigmoid ≠ PyTorch sigmoid after quantization
  - Fixed: class_score_felts overwritten from proof's publicSignals
- publicSignals too short: contract reverts
  - Fixed: length check before verifyProof with clear error message
- Teacher/proxy disagreement: proxy agreement < 100% per class
  - Mitigation: provenance manifest records disagreement, retry with different contract
- Operator not staked: Guard 1 reverts
  - Fix: stake before submitting (see 10_operations.md)
- ML server down: /fusion-embedding returns connection error
  - Fix: start ML server (see 10_operations.md)

---

## Verification checklist
- [ ] Every boundary step corresponds to a real function call in source (cite file:line)
- [ ] Every constant in the table matches its source location
- [ ] The pipeline diagram has no gaps (every arrow has a function name + data shape)
- [ ] The binding guarantee (class_score_felts from proof, not PyTorch) is clearly explained
- [ ] Provenance manifest purpose is explained (bridges the ZK gap)
- [ ] All 3 boundaries have: what crosses, where, what's NOT proved, how gaps are bridged
