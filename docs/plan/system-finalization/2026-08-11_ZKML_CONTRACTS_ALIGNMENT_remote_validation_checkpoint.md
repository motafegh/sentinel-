# SENTINEL ZKML + Contracts Alignment — Remote Validation Checkpoint

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**PR:** #62 (draft)  
**State:** remote implementation substantially validated; branch remains unmerged pending final seam disposition.

## What remote execution has actually proved

GitHub Actions workflow `SENTINEL system alignment` completed all four jobs green on commit `847911a4b55e55902e2e14526cb4041bd9e62f64`. The subsequent README-only commits do not change executable behavior.

Validated lanes:

1. **ZKML boundary tests — PASS**
   - dependency-light proxy/proof/bundle tests;
   - tracked V2 artifact-bundle structural/protocol validation.

2. **V3 policy protocol — PASS**
   - Python EIP-712 request construction and policy invariants;
   - context/proof/signal/score identity binding behavior.

3. **Contracts / Foundry — PASS**
   - deterministic dependency bootstrap from locked revisions;
   - compiler/toolchain identification;
   - Python↔Solidity golden V3 digest parity;
   - registry build;
   - full tracked Foundry tests, including V1/V2 history, V3 protocol, replay/substitution, and upgrade/storage preservation.

4. **Canonical generated verifier + real tracked proof — PASS**
   - Solidity fixture generated from tracked `zkml/ezkl/proof.json`;
   - actual tracked `contracts/src/ZKMLVerifier.sol` compiled and executed;
   - canonical proof/public signals verify;
   - mutated public output is rejected fail-closed. The generated verifier may reject invalid data by returning `false` or by reverting; both are accepted rejection semantics, while successful `true` is forbidden.

## Alignment changes now established

- Legacy raw-key/direct `cast send` audit helper removed from ZKML runtime helper surface.
- V2 proof helper is read-only and explicitly policy-ineligible.
- Proxy score semantics are explicit: direct student output regresses teacher probabilities; no second sigmoid in current ZKML proof helper.
- Proxy training and calibration require an explicitly selected DATA export and bind lineage metadata.
- Tracked ZKML artifact bundle has machine-readable integrity/protocol validation.
- Duplicate generated verifier authority removed; `contracts/src/ZKMLVerifier.sol` is canonical.
- V3 registry state is append-only relative to historical UUPS storage.
- V3 uses dual attestation: proxy ZK proof + EIP-712 policy-signed audit context.
- V3 request binds target runtime bytecode, agent, chain/registry domain, round, teacher/proxy/DATA/schema identities, proof, public signals, scores, and expiry.
- Request digest replay is rejected.
- V1/V2 writes become permanently disabled when V3 is initialized; historical reads/storage remain available.
- Fresh deployment now activates V3 before the deployment script exits.
- Existing registry upgrade path performs `upgradeToAndCall(...initializeV3...)` and preserves historical state in tracked tests.
- Contract dependency bootstrap is deterministic from `foundry.lock` revisions.
- ZKML and contracts READMEs now describe executable V2/V3 reality instead of the obsolete 64/65-signal direct-write design.

## Remaining current defect

### Legacy V2 AGENTS score transform

`agents/src/mcp/servers/audit/_submit.py` still contains the historical second sigmoid around the proxy output. This changes the V2 score statement relative to the score directly proved by ONNX/EZKL.

Risk is currently contained because the entire V2 submission path is classified `legacy_proxy_only_unbound` and is policy-rejected for finality. It should still be corrected or explicitly retired before the alignment PR is merged as a clean system baseline.

The file also owns the hardened R0 transaction-state machinery; any edit must be surgical and regression-tested rather than a broad rewrite.

## Intentionally deferred — not a reason to falsify current status

### New proxy / circuit / verifier regeneration

Do **not** retrain the proxy or regenerate a production-candidate circuit from an arbitrary historical DATA export. R4 is actively repairing DATA/label truth and has not yet promoted the future training lineage.

After R4 promotes the intended DATA/ML bundle, regenerate and validate:

1. proxy checkpoint from the explicitly promoted export;
2. ONNX + parity/lineage manifest;
3. calibration bundle from the same lineage;
4. EZKL settings/compiled circuit/SRS/PK/VK;
5. canonical generated Solidity verifier;
6. fresh proof + public-signal fixture;
7. bundle identity/protocol report;
8. Foundry/generated-verifier integration;
9. V3 registry root configuration/deployment candidate.

### Live network / isolated signer deployment

The branch validates the protocol and registry behavior but does not claim a production KMS/HSM-backed signer or a live chain deployment. Network deployment identities must be established from the actual future artifact bundle and intended chain.

## Merge gate for this alignment branch

Before converting PR #62 from draft to mergeable baseline:

- resolve or explicitly retire the remaining `_submit.py` V2 double-sigmoid seam;
- rerun the system-alignment workflow on the resulting executable head;
- review the final branch diff for accidental DATA/ML policy changes;
- keep future R4-dependent artifact regeneration documented as deferred, not silently performed against stale data.

A merge of this branch would mean **the source/protocol baseline is aligned and fail-closed**. It would not mean a newly retrained ZK artifact bundle has been promoted or deployed.
