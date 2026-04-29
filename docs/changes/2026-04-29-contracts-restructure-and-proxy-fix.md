# Change Log: 2026-04-29 — Contracts Restructure + Proxy Model Fix

**Session date:** 2026-04-29  
**Commit message prefix:** `refactor(contracts)` + `fix(zkml)`  
**Status:** Committed to `main`

---

## Summary

Two independent fixes applied in this session:
1. **Phase A** — `contracts/` restructured from a ghost file to a proper Foundry directory
2. **Phase B** — `proxy_model.py` `input_dim` corrected from 64 → 128 to match SENTINEL-SPEC §8.2

---

## Phase A — contracts/ Restructure

### Problem

`contracts` existed as a **1-byte ghost file** at repo root (SHA: `8b137891...`).  
Three Solidity source files lived orphaned at **repo root** with no Foundry project:
- `AuditRegistry.sol` (SHA: `c136c238...`)
- `IZKMLVerifier.sol` (SHA: `5644dfcf...`)
- `SentinelToken.sol` (SHA: `8d23745a...`)

`agents/src/mcp/audit_server.py` hardcodes:
```python
_ABI_PATH = _PROJECT_ROOT / "contracts/out/AuditRegistry.sol/AuditRegistry.json"
```
With `contracts` as a file this path could never exist. Real-mode `audit_server.py` was **completely broken**.

### What Was Done (this session)

GitHub API cannot create `contracts/src/` while the ghost file `contracts` exists at root — they conflict at the tree level.  

**What was committed:**
- `contracts/src/AuditRegistry.sol` — copied from root (identical content)
- `contracts/src/IZKMLVerifier.sol` — copied from root (identical content)
- `contracts/src/SentinelToken.sol` — copied from root (identical content)
- `contracts/src/.gitkeep`
- `contracts/out/.gitkeep` (documents ABI path for audit_server.py)
- `contracts/script/.gitkeep`
- `contracts/README.md` (build instructions, dual-solc warning, deployed addresses)

### Manual Steps Still Required

Run locally to complete Phase A:

```bash
cd ~/projects/sentinel
git pull origin main

# Remove ghost file and orphaned root .sol files
git rm contracts
git rm AuditRegistry.sol
git rm IZKMLVerifier.sol
git rm SentinelToken.sol

git commit -m "chore(contracts): remove ghost file and orphaned root-level .sol files"
git push origin main
```

### Verification (after manual step)

```bash
# 1. Confirm directory structure
ls contracts/src/   # AuditRegistry.sol  IZKMLVerifier.sol  SentinelToken.sol

# 2. Build (requires foundry.toml from your local system)
cd contracts
forge install OpenZeppelin/openzeppelin-contracts-upgradeable
forge install OpenZeppelin/openzeppelin-contracts
forge build

# 3. Confirm ABI exists
ls contracts/out/AuditRegistry.sol/AuditRegistry.json

# 4. Test audit_server ABI load
cd ~/projects/sentinel
AUDIT_MOCK=false poetry run python -c "
from agents.src.mcp.audit_server import _ABI_PATH
print(f'ABI path: {_ABI_PATH}')
print(f'Exists: {_ABI_PATH.exists()}')
"
```

### Rollback

Root-level `.sol` files remain untouched until the manual `git rm` step.  
If `forge build` fails:
1. Confirm `foundry.toml` exists in `contracts/` (add from local)
2. Run `cd contracts && forge install`
3. Confirm `solc 0.8.20` available: `solc-select use 0.8.20`

---

## Phase B — proxy_model.py input_dim Fix

### Problem

`proxy_model.py` had `input_dim=64`, `CIRCUIT_VERSION='v1.0'`, architecture `Linear(64→32→16→1)`.  
This matched the **old binary-era FusionLayer** (output_dim=64).

**SENTINEL-SPEC §8.2 states:**
> "Proxy MLP Input: 128-dim fused embedding (CrossAttentionFusion output BEFORE classifier)"  
> "Layers: Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→10)"

**ADR-025:** CrossAttentionFusion output_dim = 128 (was 64 with old FusionLayer).

Using `input_dim=64` causes silent shape mismatch in `train_proxy.py` and all existing EZKL keys are for the wrong circuit.

### Changes Made

| Field | Old | New | Reason |
|---|---|---|---|
| `CIRCUIT_VERSION` | `v1.0` | `v2.0` | Architecture changed — old keys invalid |
| `FROZEN_INPUT_DIM` | 64 | 128 | CrossAttentionFusion output_dim |
| `FROZEN_HIDDEN1` | 32 | 64 | SPEC §8.2 |
| `FROZEN_HIDDEN2` | 16 | 32 | SPEC §8.2 |
| `FROZEN_NUM_CLASSES` | n/a | 10 | Added — NUMCLASSES in trainer.py |
| Network | `Linear(64→32→16→1)` | `Linear(128→64→32→10)` | Multi-label |
| Guards | `assert` | `RuntimeError` | ADR-019 |
| `forward()` output | `[B]` scalar | `[B, 10]` logits | Multi-label |

### Breaking Change

All existing EZKL artifacts are invalidated:
- `zkml/ezkl/proving_key.pk`
- `zkml/ezkl/verification_key.vk`
- `zkml/ezkl/model.compiled`
- `zkml/ezkl/settings.json`

### Required Follow-Up

```bash
# 1. Retrain proxy
poetry run python zkml/src/distillation/train_proxy.py

# 2. Re-export ONNX
poetry run python zkml/src/distillation/export_onnx.py

# 3. Rerun full EZKL one-time pipeline
poetry run python zkml/src/ezkl/setup_circuit.py

# 4. Generate new ZKMLVerifier.sol
ezkl create-evm-verifier \
  --vk-path zkml/ezkl/verification_key.vk \
  --srs-path zkml/ezkl/srs.params \
  --settings-path zkml/ezkl/settings.json \
  --sol-code-path contracts/src/ZKMLVerifier.sol

# 5. Compile ZKMLVerifier with solc 0.8.17 (NOT 0.8.20)
solc-select use 0.8.17
forge build --contracts contracts/src/ZKMLVerifier.sol
solc-select use 0.8.20

# 6. Redeploy on Sepolia and update addresses in:
#    - zkml/src/ezkl/extract_calldata.py  (ZKML_VERIFIER, AUDIT_REGISTRY)
#    - agents/src/mcp/audit_server.py     (if address is hardcoded)
```

### Verification

```bash
# Shape and version check
poetry run python -c "
from zkml.src.distillation.proxy_model import ProxyModel, CIRCUIT_VERSION
import torch
print(f'Circuit version: {CIRCUIT_VERSION}')  # must be v2.0
proxy = ProxyModel()
out = proxy(torch.randn(2, 128))
print(f'Output shape: {out.shape}')           # must be torch.Size([2, 10])
print(f'Params: {proxy.parameter_count():,}') # expect ~8,330
"

# Freeze guard check
poetry run python -c "
from zkml.src.distillation.proxy_model import ProxyModel
try:
    ProxyModel(input_dim=64)
    print('FAIL: guard did not raise')
except RuntimeError as e:
    print(f'PASS: {str(e)[:80]}')
"
```

### Rollback

```bash
git checkout HEAD~1 -- zkml/src/distillation/proxy_model.py
```
Note: rolling back restores `input_dim=64` which is wrong for the current checkpoint.

---

## Files Changed This Session

| File | Action |
|---|---|
| `contracts/src/AuditRegistry.sol` | Created (copied from root) |
| `contracts/src/IZKMLVerifier.sol` | Created (copied from root) |
| `contracts/src/SentinelToken.sol` | Created (copied from root) |
| `contracts/src/.gitkeep` | Created |
| `contracts/out/.gitkeep` | Created |
| `contracts/script/.gitkeep` | Created |
| `contracts/README.md` | Created |
| `zkml/src/distillation/proxy_model.py` | Fixed (input_dim 64→128, v2.0) |
| `docs/changes/2026-04-29-*.md` | Created (this file) |
| `contracts` (root ghost file) | **Pending manual `git rm`** |
| `AuditRegistry.sol` (root) | **Pending manual `git rm`** |
| `IZKMLVerifier.sol` (root) | **Pending manual `git rm`** |
| `SentinelToken.sol` (root) | **Pending manual `git rm`** |

---

## Spec References

- SENTINEL-SPEC §8.2 — ZKML proxy architecture (input_dim=128, Linear(128→64→32→10))
- SENTINEL-SPEC §8.5 — Contracts path + dual-solc requirement
- SENTINEL-SPEC §6 — Critical constraints (proxy input_dim locked to fusion output)
- ADR-015 — ZKML proxy architecture
- ADR-019 — assert → RuntimeError
- ADR-025 — CrossAttentionFusion output_dim=128
