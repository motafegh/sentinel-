# Third-Party and Artifact Notices

The root `LICENSE` applies to Sentinel's original project code and documentation except where a file or component states a different license or where the material is third-party, generated from third-party tooling, imported, or retained as research/data/model/proof evidence with separate rights.

This file is a licensing-scope notice, not a complete software-bill-of-materials.

## Solidity dependencies

The Foundry project uses Git submodules rather than vendoring those dependency trees into Sentinel:

- `foundry-rs/forge-std` — Apache License 2.0 upstream.
- `OpenZeppelin/openzeppelin-contracts` — MIT License upstream.
- `OpenZeppelin/openzeppelin-contracts-upgradeable` — MIT License upstream.

Use and redistribution of those dependencies remain subject to their upstream licenses.

## Generated verifier

`contracts/src/ZKMLVerifier.sol` is generated verifier material and carries its own SPDX identifier:

```text
SPDX-License-Identifier: MIT
```

The root license does not erase generated provenance; retain the file-level SPDX notice when copying or modifying that file.

## Models, proof artifacts, data, and research evidence

Tracked or referenced model binaries, ONNX files, EZKL compiled/proof/witness/key artifacts, datasets, source corpora, reports, exports, and other research evidence may originate from or encode information derived from external sources, tools, models, datasets, or services.

The root MIT license does **not** grant rights that the Sentinel repository owner does not possess in third-party source material, datasets, model weights, generated artifacts, trademarks, or externally sourced content. Users are responsible for checking the provenance and applicable terms of any such artifact before reuse or redistribution.

In particular, the repository's retention of an artifact for reproducibility, auditability, or research evidence must not be interpreted as a representation that the artifact is independently relicensable under Sentinel's root MIT license.

## Package dependencies

Python, JavaScript/Solidity, and other package dependencies are not relicensed by Sentinel. Their own package metadata and upstream licenses control their use.

## Trademark / project-name note

The MIT license grants copyright permissions; it does not grant trademark rights in third-party names, logos, or project identities referenced by Sentinel.

If a third-party attribution or licensing issue is discovered, report it through the repository's `SECURITY.md` / maintainer contact path so the notice or repository contents can be corrected.
