# R4 Decision Register

| Decision ID | Date | Status | Scope | Decision | Evidence/ADR | Migration | Rollback | Owner |
|---|---|---|---|---|---|---|---|---|
| R4-D-001 | 2026-08-12 | ACCEPTED | label state/schema | Separate source claim, canonical outcome, nullable training target, training signal, and STRONG/WEAK/NONE strength; target zero requires confirmed negative | ADR-R4-001; Phase-2 reconstruction; Phase-3 ledger | New v2 semantic row schema; historical v1 unchanged | select prior hash-bound v1 bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-002 | 2026-08-12 | ACCEPTED | source/class roles | SolidiFI injected and approved SmartBugs categories strong positive; DIVE masked except weak TOD; no blanket negative authority; GasException and UnusedReturn supervision-disabled | ADR-R4-002; Phase-1 source recovery; R4-GAP-002 Phase-4 review | Phase-7 source-policy application only after G6 | restore previous policy bundle; never mutate historical labels | delegated technical owner / GPT-5.6 Sol |
| R4-D-003 | 2026-08-12 | ACCEPTED | crosswalk/merger | Preserve source-native claims; no-target actions remain no-target; remove synthetic NonVulnerable and binary positive-precedence-over-zero semantics; aggregate confirmed/weak evidence states explicitly | ADR-R4-003; Phase-2 crosswalk/merger reconstruction | Versioned vNext crosswalk and evidence-state aggregation | select prior versioned crosswalk/policy bundle | delegated technical owner / GPT-5.6 Sol |
| R4-D-004 | 2026-08-12 | ACCEPTED | export/ML compatibility | New export format v2 with canonical long contract×class state plus derived per-contract target/strength/mask projection; no silent v1 fallback; numeric weak weight belongs to Phase-8 config | ADR-R4-004; current label_writer/SentinelDataset/collate/loss seam audit | Phase 7 writes v2; Phase 8 adds allowed consumer compatibility | explicit v1 reader + historical artifact selection | delegated technical owner / GPT-5.6 Sol |
| R4-D-005 | 2026-08-12 | ACCEPTED | lineage/versioning | Keep historical export and Phase-3 ledger immutable; bind policy/schema/partition/code/artifact hashes; staged publication; rollback by artifact selection | ADR-R4-005; Phase-3 fail-closed publication pattern | New versioned paths/manifests only | select earlier compatible hashed bundle | delegated technical owner / GPT-5.6 Sol |

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
