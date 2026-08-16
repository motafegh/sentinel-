# R4 Decision Register

| Decision ID | Date | Status | Scope | Decision | Evidence/ADR | Migration | Rollback | Owner |
|---|---|---|---|---|---|---|---|---|
| R4-D-001 | 2026-08-12 | ACCEPTED | label state/schema | Separate source claim, canonical outcome, nullable training target, training signal, and STRONG/WEAK/NONE strength; target zero requires confirmed negative | ADR-R4-001; Phase-2 reconstruction; Phase-3 ledger | New v2 semantic row schema; historical v1 unchanged | select prior hash-bound v1 bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-002 | 2026-08-12 | ACCEPTED | source/class roles | SolidiFI injected and approved SmartBugs categories strong positive; DIVE masked except weak TOD; no blanket negative authority; GasException and UnusedReturn supervision-disabled | ADR-R4-002; Phase-1 source recovery; R4-GAP-002 Phase-4 review | Phase-7 source-policy application only after G6 | restore previous policy bundle; never mutate historical labels | delegated technical owner / GPT-5.6 Sol |
| R4-D-003 | 2026-08-12 | ACCEPTED | crosswalk/merger | Preserve source-native claims; no-target actions remain no-target; remove synthetic NonVulnerable and binary positive-precedence-over-zero semantics; aggregate confirmed/weak evidence states explicitly | ADR-R4-003; Phase-2 crosswalk/merger reconstruction | Versioned vNext crosswalk and evidence-state aggregation | select prior versioned crosswalk/policy bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-004 | 2026-08-12 | ACCEPTED | export/ML compatibility | New export format v2 with canonical long contract×class state plus derived per-contract target/strength/mask projection; no silent v1 fallback; numeric weak weight belongs to Phase-8 config | ADR-R4-004; current label_writer/SentinelDataset/collate/loss seam audit | Phase 7 writes v2; Phase 8 adds allowed consumer compatibility | explicit v1 reader + historical artifact selection | delegated technical owner / GPT-5.6 Sol |
| R4-D-005 | 2026-08-12 | ACCEPTED | lineage/versioning | Keep historical export and Phase-3 ledger immutable; bind policy/schema/partition/code/artifact hashes; staged publication; rollback by artifact selection | ADR-R4-005; Phase-3 fail-closed publication pattern | New versioned paths/manifests only | select earlier compatible hashed bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-006 | 2026-08-12 | ACCEPTED | leakage-safe roles/acceptance | Freeze one role per project/dedup/contract group; exclude incomplete-representation groups; split represented strong groups into train/model-selection/internal-audit with class coverage; DIVE TOD weak groups train-weak; remaining represented groups train-unlabeled; threshold/calibration/untouched acceptance remain controlled empty unsupported | ADR-R4-006; Phase-6 role-support inventory | Phase 7 must consume `r4-vnext-roles-v1` manifests exactly; no implicit rebalancing | select previous hash-bound role manifest | delegated technical owner / GPT-5.6 Sol |
| R4-D-007 | 2026-08-12 | ACCEPTED | DATA vNext implementation / G7 | Accept `sentinel-r4-vnext-v1` as the G7-passed semantic/representation-bound lineage at the historical G7 boundary; historical artifacts remain immutable | ADR-R4-007; G7 local binding/final validation; merge `81d9c547d` | Historical consumer remains reproducible; later replacements require new versioned decision | select prior compatible bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-008 | 2026-08-15 | ACCEPTED | repaired-v2 physical DATA / Phase-8 launch boundary | Accept repaired-v2 physical lineage after 22,540/22,540 representation validation/binding; keep G8 open and prohibit 100-epoch run because supervision is positive-only and evaluation/selector adequacy unresolved | ADR-R4-008; `runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md`; digest `16dd4a3f...` | Objective/selector changes require new versioned lineage/binding and training-horizon decision | select historical compatible bundle; never mutate accepted repaired-v2 evidence | delegated technical owner / GPT-5.6 Sol |
| R4-D-009 | 2026-08-15 | ACCEPTED | leakage grouping / logical Phase-8 lineage | Remove arbitrary Ethereum address literals from grouping authority; accept `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3` after protected validation proved 22,394 groups, max 7, zero address-authority edges, unchanged semantics and exact repaired-v2 physical digest. Grouping authority is global normalized-code/exact-artifact identity plus **source-namespaced** explicit source family IDs. Post-acceptance research evidence must be manifest/binding/source coherent before durable snapshotting | ADR-R4-009; `runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`; pre-hardening checkpoint retained historically | Use V3 as logical authority; preserve repaired-v2 physical bytes; regenerate hardened research reports before final snapshot; selector/objective/promotion remain separate decisions | preserve V2 as historical evidence; regenerate stale V3-derived reports rather than re-authorizing V2 address grouping | delegated technical owner / GPT-5.6 Sol |

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
