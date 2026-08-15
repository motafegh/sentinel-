# SENTINEL DATA Module Working Instructions

These instructions refine root `CLAUDE.md` for current DATA work. Root project authority and the committed R4 machine-readable policy/manifests remain higher authority.

## Current DATA state

The historical `sentinel-r4-vnext-v1` / v9 / `v2.1-windowed-gcb` lineage remains immutable evidence of the G7-passed 2026-08-12 baseline. The 2026-08-14 physical real-data audit later found material preprocessing and representation defects, so that historical lineage is **not authorized as the input for the Phase-8 full retrain**.

Repository repair now defines a new candidate lineage:

- preprocessing artifact: `sentinel-preprocessed-r4-v2`;
- preprocessing metadata schema: `2`;
- provenance schema: `r4-provenance-v1`;
- evidence ledger identity: `evidence-ledger-r4-v2`;
- leakage grouping: `r4-leakage-groups-v2`;
- role partition identity: `r4-vnext-roles-v2`;
- DATA publication: `sentinel-r4-vnext-v2`;
- representation extractor: `v2.2-r4-repaired`;
- graph feature schema: still `v9`;
- token tensor contract: still `[4, 512]`.

The Git-ignored repaired physical corpus was rebuilt and accepted locally on
2026-08-15 at source commit
`fb31326da4420c2289822c2a6db8a022ac25876a`: 22,540 contracts and 67,620
graph/token/sidecar files bind with zero missing/invalid payloads. This local
evidence is not present in a fresh clone and does not claim model quality.

## Mandatory repair semantics

1. **Never overwrite historical DATA artifacts.** Repaired outputs go to new versioned roots.
2. **Unknown is not negative.** No target `0` may be generated without class-specific confirmed-negative evidence. Current policy v1 has none.
3. **Source record != contract identity != leakage group.** Preserve source-record provenance, aggregate exact content identity deterministically, and assign roles only after final leakage grouping.
4. **Ethereum address coincidence is not duplicate identity.** Same-source shared addresses may be conservative family evidence for leakage prevention; they never delete a content-distinct contract or create label truth.
5. **Compile the exact promoted normalized source.** Regex-only comment stripping and compile-before-normalize are not permitted in repaired preprocessing.
6. **File-level graph selection preserves label scope.** An explicit provenance target is authoritative. Otherwise represent every unrelated application inheritance leaf in one disconnected file graph; inheritance parents arrive through their leaves. Library-only files may retain executable libraries, while interfaces alone are not implementation targets. Never guess one unrelated contract or silently fall back.
7. **Long-contract truncation is visible.** Preserve `[4,512]` for this architecture-frozen tranche, but record pre-subsampling coverage and do not claim adequacy from shape validity.
8. **Weak evidence stays weak.** DIVE Front Running→TransactionOrderDependence remains WEAK training-only under `data-vnext-policy-v1`; other DIVE positives remain masked unless policy changes with evidence.
9. **SmartBugs native category is provenance.** Direct `time_manipulation→Timestamp` is authorized strong evidence; `bad_randomness→Timestamp` is superseded/no-target. The distinction must be bound before training, not guessed by the ML consumer.
10. **Every failure is explicit.** Drops, target ambiguity, compile failure, representation failure, and binding mismatch must be recorded or raised; no silent skip/collision behavior.

## Supported repaired execution seam

Use:

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py --help
```

and the durable local-execution handoff referenced by the R4 status matrix once repository validation is complete.

Do not use historical `data/preprocessed` or `data/representations` as mutable repair destinations.

## Validation

Repository-safe repair tests live primarily under:

- `data_module/tests/test_preprocessing/test_r4_repair.py`
- `data_module/tests/test_preprocessing/test_r4_data_gates.py`
- `data_module/tests/test_preprocessing/test_r4_grouping.py`
- `data_module/tests/test_representation/test_r4_target_selector.py`
- `data_module/tests/test_vnext/test_r4_source_claims.py`
- `data_module/tests/test_vnext/test_r4_builder.py`
- `data_module/tests/test_vnext/test_r4_binding.py`

Physical acceptance additionally requires the protected raw manifests/corpus, DIVE label CSV, historical solc binaries, generated repaired representations/parquet outputs, and the bounded local GPU smoke. Repository-only CI cannot substitute for those gates.

## Training boundary

Physical repaired-v2 DATA acceptance has completed through steps 1-5 below;
step 6 remains false:

1. raw-manifest byte verification passes;
2. repaired preprocessing/claims/grouping/representations/publication complete;
3. physical representation binding passes;
4. attrition/recovery and token-coverage evidence are reviewed against the 2026-08-14 audit;
5. a bounded repaired-data GPU smoke passes;
6. R4 governance explicitly re-authorizes the full Phase-8 run.

The physical dataset is accepted for bounded research, but **100-epoch training
is not authorized** because all 899 effective loss cells are positive-only and
the long-contract selector is not yet promoted.
