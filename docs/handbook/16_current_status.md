# 16 — Current status and gap ledger

**Read this when:** you need to know what is passing, failing, local-only, degraded, or disconnected today.

**Skip this if:** never; every learning path should end here before operational decisions.

**Estimated reading time:** 8 minutes.

## 30-second summary

This page is the only canonical home for volatile test counts. The baseline below was measured on 2026-07-13 at commit `ac78c057b`. Several suites have real failures, some required artifacts are not available in a fresh clone, and the off-chain gateway flow is not connected to the MCP ZK/on-chain submission flow. Those facts are documentation targets, not defects fixed by D1.

## Just-enough mental model

“Implemented” means executable code exists. “Verified” means the named check passed at the bound commit. “Available in a fresh clone” means Git supplies the artifact. “Live-ready” additionally requires external dependencies, secrets, services, or chain state. These labels are deliberately separate.

## Actual runtime/source walkthrough

The baseline was captured by invoking each module’s native suite with WSL-safe temporary variables:

| Module | Result | Interpretation |
|---|---:|---|
| AGENTS | 631 passed, 3 failed | implemented; module suite not green |
| ML | 198 passed, 19 failed | implemented; module suite not green |
| DATA | 569 passed, 9 failed, 47 skipped | implemented; module suite not green and optional coverage skipped |
| ZKML | 37 passed | suite green in the measured environment |
| Contracts | 66 passed | current-checkout Foundry suite green; includes ignored local V2 coverage |

The failing AGENTS cases are mock-audit tests. D1 does not reinterpret or suppress them. Consult raw test output when repairing the product; do not lower documentation validation to hide product failures.

### D1 v3 documentation implementation verification

The documentation-only v3 worktree was checked on the same 2026-07-13 source commit:

| Check | Result | Meaning |
|---|---:|---|
| Handbook static contract | 285 passed | chapters, guides, labs, links, symbols, 429 active production source files, secrets, ownership, artifacts, and critical facts agree |
| Validator unit tests | 10 passed | broken-link, missing-symbol/section, secret, volatile-count, and artifact-classification checks behave as intended |
| Safe lab preflights | 7 passed | every lab marked safe has tracked source/test prerequisites |
| Focused DATA command | 34 passed | documented CLI/representation command is executable |
| Focused ZKML command | 21 passed | documented proxy/layout/endian command is executable |
| Focused AGENTS command | 119 passed | documented fusion/routing/service/gateway/security command is executable |
| Focused tracked registry command | 15 passed | documented `AuditRegistry.t.sol` command is executable without relying on ignored V2 test |
| Focused ML model/predictor/API command | 43 passed, 1 failed | command is executable; `test_forward_return_aux_shapes` expects three auxiliary keys while current source returns the additional `phase2` head |
| L02 dataset-seam command | 16 passed | local export, schema/hash gates, shard loading, and collate command is executable |
| L04 trainer/predictor/promotion command | 23 passed, 2 failed | promotion tests expect generated behavioral-probe JSON that their current fixture does not create |

The focused ML failures are product-test/source or fixture expectation mismatches already inside the non-green ML baseline. D1 v3 records them and does not alter model or test behavior.

The live availability probe at documentation finalization found ML on port 8001 healthy and Anvil on 8545 reachable. Gateway 8000 and all five MCP health endpoints on 8010–8014 refused connections because those services were not running. The live command therefore failed as designed; it did not convert absent services into skipped or passing checks.

## Interfaces, data shapes, and configuration

### Environment note

On this WSL/Desktop setup, inherited `TMP`, `TEMP`, or `TMPDIR` values can point pytest at a Windows temporary directory and cause capture failures unrelated to the test target. Prefix pytest commands with:

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
```

### Availability ledger

| Item | State at baseline |
|---|---|
| Source, configs, proxy model, compiled circuit, verification key | tracked |
| Run 12 teacher checkpoint and its `.dvc` pointers | present locally; ignored/untracked in this checkout |
| DATA exports and splits | present locally; ignored/untracked |
| EZKL proving key and SRS | ignored/private prerequisites |
| RAG indexes, gateway DB, LangGraph checkpoint DB, inference caches | regenerated/local runtime state |
| `contracts/test/AuditRegistryV2.t.sol` | local but ignored by root `test` rule; absent from fresh-clone coverage |

The 66-test result was measured from this working checkout, where Foundry discovers the ignored `AuditRegistryV2.t.sol` file and its 14 tests. A fresh clone does not contain that file, so 66 must not be presented as reproducible fresh-clone contract coverage until the file/ignore rule is fixed.

## Failure modes and current limitations

### Submission boundary

- `POST /audit` runs the off-chain graph and persists a report; it does not invoke MCP `submit_audit`.
- `final_report["on_chain"]` starts as an unsubmitted placeholder.
- Direct ZK/on-chain submission is a separately invoked MCP tool path.
- The submission response reports a SHA-256 proof-file hash, while the registry records `keccak256(proof)`.
- The input `model_hash` is sent to the contract; an ML-fetched hash is reported separately and is not automatically substituted.
- Proof generation uses shared `proof_input.json`, `witness.json`, and `proof.json` paths, so concurrent submissions can race.

### Trust boundary

- `verdict_provable` is deterministic evidence fusion, but the EZKL proof verifies proxy logits from the fusion embedding—not that verdict.
- Provenance is an off-chain operator assertion. It can be unsigned in degraded operation and does not prove that the teacher generated the supplied embedding.
- `check_mode="UNSAFE"` requires explicit review before any production security claim.

### Integration maturity

- Live behavior depends on local model/data/proving artifacts and external toolchains.
- `graph_inspector_server.py` currently defaults its ML upstream URL to port 8000 instead of the ML service’s port 8001; operators must override `SENTINEL_ML_API_URL` for real GNN hotspots.
- Some paths intentionally return structured degraded status; callers must not treat “did not run” as “ran clean.”
- The ignored V2 contract test and ignored DVC pointers must be tracked/fixed in a separate product/repository task before fresh-clone claims change.

## Common change recipe

To update this status page:

1. Record the date and exact commit.
2. Set WSL-safe temporary variables.
3. Run the suites without `|| true`, filtered output, or hidden skips.
4. Record pass/fail/skip outcomes only here.
5. Run handbook static/inventory validation.
6. If artifact availability changed, update `_meta/handbook.toml` and the reference matrix too.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents && poetry run pytest -q
cd .. && ml/.venv/bin/python -m pytest ml/tests -q
data_module/.venv/bin/python -m pytest data_module/tests -q
ml/.venv/bin/python -m pytest zkml/tests -q
cd contracts && forge test
```

These are module checks. GPU inference, real EZKL proof generation, and live chain submission are separate live checks in [operations](14_operations.md).

## Optional deep references

- [`handbook.toml`](_meta/handbook.toml) — machine-readable baseline contract
- [Security and trust](12_security_and_trust.md)
- [Operations](14_operations.md)
- [Runtime flows](02_runtime_flows.md)

## Technical mastery layer

### Prerequisite knowledge

Know commit binding, tracked versus ignored/DVC artifacts, test maturity, mocked versus live evidence, and gap-ledger discipline.

### Source map and reading order

Read `handbook.toml` verified commit/artifact classifications, this page’s dated evidence, then inventory output and owning technical guide. Never infer current readiness from an undated learning page.

### Execution trace and worked example

A status claim is acceptable only as commit + date + command/environment + artifact availability + result + limitation. For example, local contract totals that include ignored `AuditRegistryV2.t.sol` are local-checkout evidence and cannot become fresh-clone coverage.

### Implementation practice

Use four maturity labels: implemented-and-verified, implemented-with-known-failure, degraded/optional, and planned/disconnected. Update only this page when volatile results change; module chapters link here. [L10](labs/10_end_to_end_capstone.md) distinguishes preflight availability from live success.

### Review and ownership check

Can you tell which claims survive a fresh clone and which depend on local/private/live state?
