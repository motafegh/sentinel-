# 03 — Source Authority Matrix

- **Phase:** R4 Phase 2 — Label-Corruption Mechanism Reconstruction
- **Basis:** executable source on `r4/phase2-label-corruption-reconstruction` plus Phase-1 recovered evidence
- **Rule:** source authority is class-specific. A source positive for one class is not automatically a confirmed negative for every other class.

## Active/deferred source reality

| Source | Config state | Acquisition executable now? | Label parser/crosswalk executable now? | Historical/current role recovered | Positive authority | Negative authority |
|---|---|---:|---:|---|---|---|
| SolidiFI | enabled | YES | YES | 283 preprocessed/generated labels; active historical export source | **T0** for the injected class only | **NONE** for non-injected classes; injection corpus has no negative controls |
| DIVE | enabled | manual/local staging | YES | 22,073 preprocessed/labeled records; active historical export source | T2 folder/CSV assertion; EB/RE precision evidence is extremely poor | **NONE by default**; CSV unknown/absence and dropped categories collapse to zero |
| SmartBugs Curated | enabled | manual/local staging | YES | 143 hand-labeled contracts available; later `sentinel-v3-smartbugs` export includes the source | T1 hand-labeled category assertion | Explicit row-level NonVulnerable evidence exists for 4 recovered contracts; other cross-class zeros are not proven negatives |
| Web3Bugs | enabled | NO usable material recovered | NO — configured crosswalk path is absent; no parser | Declared active but unavailable; no recovered contribution to historical target rows | NONE | NONE |
| DISL | enabled, `non_vulnerable_only` | **NO** — Etherscan connector raises `NotImplementedError` | no positive parser/crosswalk | Intended negative pool; older export audit records it as skipped/unavailable | NONE | NONE; unlabeled does not equal confirmed safe |
| DeFiHackLabs | disabled | partial historical acquisition only | crosswalk exists | excluded from active export due Foundry/forge-std compile barrier | historical exploit evidence only | NONE |
| BCCC | deferred | historical corpus/evidence exists | not an active current parser path | v1.4 verified/provisional labels recovered but not loaded as active source | class-dependent historical evidence | class-dependent; not imported as active negative authority |

## Canonical class authority by active source

Legend:

- `P` = source can emit a positive assertion for this class;
- `U` = class unsupported by the source; emitted zero therefore means unsupported/absence, not negative;
- `NV` = source category can map to all-zero/NonVulnerable;
- `—` = no executable source contribution.

| Canonical class | SolidiFI | DIVE | SmartBugs Curated | Web3Bugs | DISL |
|---|---|---|---|---|---|
| CallToUnknown | P (`Unchecked-Send`) | U | P (`unchecked_low_level_calls`) | — | — |
| DenialOfService | U | P (`DoS`) | P | — | — |
| ExternalBug | P (`tx.origin`) | P (`Access Control`) | P (`access_control`) | — | — |
| GasException | U | U | no direct recovered category in current crosswalk | — | — |
| IntegerUO | P (`Overflow-Underflow`) | P (`Arithmetic`) | P (`arithmetic`) | — | — |
| MishandledException | P (`Unhandled-Exceptions`) | U | no direct folder mapping | — | — |
| Reentrancy | P | P | P | — | — |
| Timestamp | P (`Timestamp-Dependency`) | P (`Time manipulation`) | P; also receives lossy `bad_randomness` | — | — |
| TransactionOrderDependence | P (`TOD`) | P (`Front Running`) | P (`front_running`) | — | — |
| UnusedReturn | U | P (`Unchecked Return Values`) | no direct folder mapping | — | — |

SmartBugs additionally maps `short_addresses` and `other` to `NonVulnerable`, producing all-zero vectors. DIVE drops `Bad Randomness`; a Bad-Randomness-only record therefore becomes all-zero after crosswalk loss.

## Authority conclusions

1. **There is almost no class-specific negative authority in the historical active pipeline.** The parsers are predominantly positive-label generators that fill every non-positive class with `0`.
2. **SolidiFI's T0 guarantee is positive-only.** Programmatic injection proves the injected vulnerability exists; it does not prove the other nine classes are absent.
3. **DIVE zero semantics are especially unsafe.** Folderization preserves only positive symlinks even though its CSV reader distinguishes an empty cell as `unknown`; the parser later converts no folder membership to zero.
4. **SmartBugs has real hand-label authority, but the parser still turns one category into a ten-cell binary vector.** Only explicit safe/non-vulnerable examples can support row-level negative claims; cross-class zeros are not independently adjudicated.
5. **Web3Bugs and DISL must not be counted as evidence simply because config says `enabled: true`.** Neither has a complete current executable route to historical labels.
6. **BCCC evidence is historical/deferred, not an active-source authority.** Its v1.4 artifact may inform later decisions, but importing it is a future policy choice, not Phase-2 reconstruction.

## Reproducibility seam

The lower-level parsers and merger exist, but the current top-level `sentinel-data label` CLI handler prints `NOT IMPLEMENTED — implement in Stage 3`, while `data_module/dvc.yaml` still declares that command as the label stage. Therefore the historical label build cannot be reproduced end-to-end through the nominal DVC/CLI seam from current source alone.

This is a reproducibility defect, not permission to rewrite labels in Phase 2.
