# R4 Decision Register

| Decision ID | Date | Status | Scope | Decision | Evidence/ADR | Migration | Rollback | Owner |
|---|---|---|---|---|---|---|---|---|
| R4-D-001 | 2026-08-12 | ACCEPTED | label state/schema | Separate source claim, canonical outcome, nullable training target, training signal, and STRONG/WEAK/NONE strength; target zero requires confirmed negative | ADR-R4-001; Phase-2 reconstruction; Phase-3 ledger | New v2 semantic row schema; historical v1 unchanged | select prior hash-bound v1 bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-002 | 2026-08-12 | ACCEPTED | source/class roles | SolidiFI injected and approved SmartBugs categories strong positive; DIVE masked except weak TOD; no blanket negative authority; GasException and UnusedReturn supervision-disabled | ADR-R4-002; Phase-1 source recovery; R4-GAP-002 Phase-4 review | Phase-7 source-policy application only after G6 | restore previous policy bundle; never mutate historical labels | delegated technical owner / GPT-5.6 Sol |
| R4-D-003 | 2026-08-12 | ACCEPTED | crosswalk/merger | Preserve source-native claims; no-target actions remain no-target; remove synthetic NonVulnerable and binary positive-precedence-over-zero semantics; aggregate confirmed/weak evidence states explicitly | ADR-R4-003; Phase-2 crosswalk/merger reconstruction | Versioned vNext crosswalk and evidence-state aggregation | select prior versioned crosswalk/policy bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-004 | 2026-08-12 | ACCEPTED | export/ML compatibility | New export format v2 with canonical long contract×class state plus derived per-contract target/strength/mask projection; no silent v1 fallback; numeric weak weight belongs to Phase-8 config | ADR-R4-004; current label_writer/SentinelDataset/collate/loss seam audit | Phase 7 writes v2; Phase 8 adds allowed consumer compatibility | explicit v1 reader + historical artifact selection | delegated technical owner / GPT-5.6 Sol |
| R4-D-005 | 2026-08-12 | ACCEPTED | lineage/versioning | Keep historical export and Phase-3 ledger immutable; bind policy/schema/partition/code/artifact hashes; staged publication; rollback by artifact selection | ADR-R4-005; Phase-3 fail-closed publication pattern | New versioned paths/manifests only | select earlier compatible hashed bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-006 | 2026-08-12 | ACCEPTED | leakage-safe roles/acceptance | Freeze one role per project/dedup/contract group; exclude incomplete-representation groups; split represented strong groups into train/model-selection/internal-audit with class coverage; DIVE TOD weak groups train-weak; remaining represented groups train-unlabeled; threshold/calibration/untouched acceptance remain controlled empty unsupported | ADR-R4-006; Phase-6 role-support inventory; quickstart/manual/Tier-E exposure audit | Phase 7 must consume `r4-vnext-roles-v1` manifests exactly; no implicit rebalancing | select previous hash-bound role manifest; acceptance remains empty until separately versioned corpus exists | delegated technical owner / GPT-5.6 Sol |

| R4-D-007 | 2026-08-12 | ACCEPTED | DATA vNext implementation / G7 | Accept `sentinel-r4-vnext-v1` as the G7-passed v2 semantic/representation-bound lineage and sole Phase-8 training-data authority; historical v1 remains immutable and unsupported evaluation roles remain empty | ADR-R4-007; G7 local binding + final validation; implementation merge `81d9c547d` | Phase 8 consumes exact manifest/roles/masks and binds any training-weight choices to checkpoint config | select prior hash-bound compatible bundle; never rewrite v1/v2 in place | delegated technical owner / GPT-5.6 Sol |

## Decisions requiring ADR

- class definition change;
- source/class KEEP, DROP, MASK, or TRAIN_WEAK decision;
- crosswalk change;
- merger policy change;
- label schema version;
- partition policy;
- DATA vNext acceptance;
- calibration method;
- threshold objective;
- inference verdict naming;
- architecture unfreeze;
- promotion.
