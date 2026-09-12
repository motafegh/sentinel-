# P7 License-Sensitive Content Inventory

**Date:** 2026-09-12  
**Scope:** bounded tracked-content review to support the repository-level license decision  
**Status:** COMPLETE FOR DECISION SUPPORT — not a legal audit

## Why this check exists

A root `LICENSE` file can be read as applying to repository content that does not carry a separate license notice. Sentinel also contains submodules, generated proof artifacts, retained model artifacts, and research data/provenance material. The project should therefore choose a license only after distinguishing original project material from separately governed or provenance-sensitive content.

This inventory is engineering decision support, not legal advice.

## Findings

### 1. No root repository license exists today

GitHub reports `license: null` for `motafegh/sentinel-`. No root `LICENSE` file is present in the current tracked tree.

### 2. Local Solidity source is consistently MIT-marked

The current `contracts/src/` surface contains four Solidity files:

- `AuditRegistry.sol` — `SPDX-License-Identifier: MIT`;
- `IZKMLVerifier.sol` — `SPDX-License-Identifier: MIT`;
- `SentinelToken.sol` — `SPDX-License-Identifier: MIT`;
- `ZKMLVerifier.sol` — `SPDX-License-Identifier: MIT`.

`ZKMLVerifier.sol` is the retained/generated Halo2/EZKL verifier and already carries the MIT SPDX identifier.

This is the strongest repository-local signal for choosing MIT rather than introducing a different default license for the project-owned source layer.

### 3. Contracts dependencies are submodules with their own licenses

`contracts/.gitmodules` declares:

- `foundry-rs/forge-std`;
- `OpenZeppelin/openzeppelin-contracts`;
- `OpenZeppelin/openzeppelin-contracts-upgradeable`.

The current `contracts/foundry.lock` pins:

- forge-std `v1.12.0` / `7117c90...`;
- OpenZeppelin Contracts `v5.6.0-rc.0` / `f910b26...`;
- OpenZeppelin Contracts Upgradeable `v5.6.0-rc.0` / `37d55aa...`.

Current upstream repository metadata reports:

- forge-std: **Apache-2.0**;
- OpenZeppelin Contracts: **MIT**;
- OpenZeppelin Contracts Upgradeable: **MIT**.

These are separate submodule repositories and must retain their upstream license terms; a Sentinel root license must not be described as relicensing them.

### 4. Retained ZKML model/proof artifacts are tracked

The repository tracks retained artifacts including:

- `zkml/models/proxy.onnx`;
- `zkml/models/proxy.onnx.data`;
- `zkml/models/proxy_best.pt`;
- EZKL compiled/proof/settings/witness/verification-key artifacts under `zkml/ezkl/`.

These artifacts are part of Sentinel's historical/reproducibility story. A root source-code license should not be used to imply a stronger provenance or downstream-use grant for upstream tools/models/data than the project can establish.

### 5. DATA lineage includes external research sources/provenance

Sentinel's DATA/R4 work incorporates or derives evidence from external datasets/sources such as SmartBugs, SolidiFI, and DIVE, with large/local/DVC-managed material not necessarily present in every fresh clone.

This bounded review did **not** establish one uniform license grant covering every upstream dataset/source artifact. Therefore a repository-level license should explicitly preserve upstream dataset/source terms rather than purport to relicense third-party material.

### 6. Python/package dependencies do not need to share Sentinel's root license

Dependency declarations/lockfiles reference many separately licensed packages. Ordinary dependency use does not require Sentinel to adopt the same license, but notices or bundled/vendor material would need separate treatment when applicable.

No root-level vendored Python dependency tree was identified as part of this bounded review.

## Recommendation

For original Sentinel code and documentation, **MIT is now the preferred repository-level candidate**.

Why MIT fits this repository better than the preliminary Apache-2.0 suggestion:

- all current local Solidity source already declares MIT;
- the generated verifier declares MIT;
- OpenZeppelin dependencies are MIT;
- MIT is simple and familiar for a public portfolio/research repository;
- forge-std remains separately Apache-2.0 as a submodule and does not need to be relicensed by Sentinel.

The recommendation is still subject to the owner's license choice.

## Required companion boundary if MIT is selected

Do not add only a bare root license and stop. Pair it with a concise `THIRD_PARTY_NOTICES.md` or equivalent statement making clear that:

- third-party submodules retain their upstream licenses;
- external datasets/source corpora retain their applicable upstream terms;
- upstream model/tool artifacts retain their applicable upstream terms;
- per-file SPDX notices remain authoritative where present;
- the Sentinel license applies to original project material to the extent the repository owner has licensing authority.

Do not attempt to rewrite or normalize third-party SPDX headers to match the root license.

## Decision boundary

P7 can now present a more informed owner choice:

1. **MIT + third-party notice boundary — recommended**;
2. Apache-2.0 + third-party notice boundary — technically possible but less aligned with the existing local Solidity licensing;
3. intentionally remain unlicensed — legally restrictive for reuse and weaker for an open portfolio, but still a valid owner choice.

No license file should be committed until the owner chooses the posture explicitly.
