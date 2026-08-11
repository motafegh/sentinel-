# 03 — Crosswalk Effect Table

This table records how source-native categories are transformed into the locked SENTINEL ten-class target space. It describes semantic effects, not whether a future DATA vNext policy should keep them.

## DIVE

| Source-native category/state | SENTINEL result | Effect type | Historical zero consequence |
|---|---|---|---|
| Reentrancy | Reentrancy | direct positive | all other cells become 0 |
| DoS | DenialOfService | direct positive | all other cells become 0 |
| Arithmetic | IntegerUO | direct positive | all other cells become 0 |
| Time manipulation | Timestamp | direct positive | all other cells become 0 |
| Front Running | TransactionOrderDependence | direct positive | all other cells become 0 |
| Access Control | ExternalBug | semantic compression | all other cells become 0; EB meaning broadened |
| Unchecked Return Values | UnusedReturn | semantic compression | CallToUnknown/MishandledException remain 0 |
| Bad Randomness | **DROPPED** | dropped source-native category | if this is the only positive category, target becomes all-zero |
| CSV empty/unknown cell | no folder symlink | unknown erased before parser | parser later emits 0 |
| no mapped folder membership | all-zero | absence/default→NonVulnerable | all ten cells 0 |
| CallToUnknown coverage | unsupported | source absence | 0 for every DIVE-only row |
| GasException coverage | unsupported | source absence | 0 for every DIVE-only row |
| MishandledException coverage | unsupported | source absence | 0 for every DIVE-only row |

Recovered counts relevant to these effects:

- 22,073 DIVE records were historically labeled after preprocessing.
- 2,686 source files were reported with no vulnerability-folder membership before accounting for Bad-Randomness-only loss.
- 634 files are reported in Bad Randomness; the exact exclusive-only subset is not retained in tracked remote evidence.
- Because CallToUnknown, GasException and MishandledException are unsupported, the historical parser emitted at least `22,073 × 3 = 66,219` unsupported-class zero cells across DIVE source records before any multi-source override.

## SmartBugs Curated

| Source-native category | SENTINEL result | Effect type | Historical zero consequence |
|---|---|---|---|
| reentrancy | Reentrancy | direct | other 9 cells 0 |
| arithmetic | IntegerUO | direct | other 9 cells 0 |
| denial_of_service | DenialOfService | direct | other 9 cells 0 |
| time_manipulation | Timestamp | direct | other 9 cells 0 |
| unchecked_low_level_calls | CallToUnknown | direct/near-direct | other 9 cells 0 |
| access_control | ExternalBug | semantic compression | other 9 cells 0 |
| bad_randomness | Timestamp | **lossy many-to-one** | BadRandomness distinction disappears |
| front_running | TransactionOrderDependence | direct/near-direct | other 9 cells 0 |
| short_addresses | NonVulnerable | mapped-to-NonVulnerable | all ten cells 0 |
| other | NonVulnerable | mapped-to-NonVulnerable | all ten cells 0 |

The parser enforces one mapped category per row. On the recovered 143-row corpus with four NonVulnerable examples, this representation emits 1,291 zero cells (`139×9 + 4×10`). This is a serialization count, **not 1,291 confirmed-negative judgments**.

## SolidiFI

| Source-native injection | SENTINEL result | Effect type | Historical zero consequence |
|---|---|---|---|
| Re-entrancy | Reentrancy | direct T0 positive | other 9 cells 0 |
| Timestamp-Dependency | Timestamp | direct T0 positive | other 9 cells 0 |
| Unhandled-Exceptions | MishandledException | direct T0 positive | other 9 cells 0 |
| TOD | TransactionOrderDependence | direct T0 positive | other 9 cells 0 |
| Overflow-Underflow | IntegerUO | direct T0 positive | other 9 cells 0 |
| Unchecked-Send | CallToUnknown | semantic compression | other 9 cells 0 |
| tx.origin | ExternalBug | semantic compression | other 9 cells 0 |
| DenialOfService | unsupported | source absence | 0 across SolidiFI rows |
| GasException | unsupported | source absence | 0 across SolidiFI rows |
| UnusedReturn | unsupported | source absence | 0 across SolidiFI rows |

Recovered corpus count is 283 generated labels. Therefore the single-positive parser emits `283×9 = 2,547` zero cells. Of these, `283×3 = 849` are for classes the source does not support at all. None should be promoted to confirmed-negative status solely from SolidiFI injection semantics.

## Web3Bugs

Configured crosswalk path: `sentinel_data/labeling/crosswalks/web3bugs.yaml`.  
Tracked reality: the file does not exist and no parser is present.

**Effect:** no class transformation can be reconstructed because no active executable label path exists. The correct historical state is source `UNAVAILABLE`, not ten zero cells.

## DISL

Configured role: `non_vulnerable_only`; no crosswalk. Acquisition connector is a stub.

**Effect:** the design intended unlabeled contracts to enter a negative pool, but the tracked executable path cannot currently produce those rows. “Unlabeled” itself is not a ten-class negative assertion.

## Out-of-band historical target mutation

The June 13 data audit records a direct post-export mutation:

- condition: `DenialOfService=1 AND Reentrancy=1`;
- action: set `DenialOfService` from `1` to `0` in `labels.parquet`;
- rows affected: **2,655**;
- DoS count reported before/after: **3,750 → 1,095**.

This was not a source crosswalk. It is classified in Phase 2 as **other: direct post-export target mutation**. Its zero means “suppressed by historical DoS/Reentrancy patch,” not “source confirmed absence of DoS.”

## Crosswalk conclusion

The historical ten-class target space conflates at least five distinct semantic states into the same numeric zero:

1. class was explicitly not asserted by a source row;
2. class was unsupported by that source;
3. source-native category was dropped;
4. source-native category was mapped to NonVulnerable;
5. a positive was deliberately zeroed by a later patch.

These states must be separated in DATA vNext; Phase 2 only reconstructs them.
