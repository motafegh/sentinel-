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
| R4-D-009 | 2026-08-15 | ACCEPTED | leakage grouping / logical Phase-8 lineage | Remove arbitrary Ethereum address literals from grouping authority; accept `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3` after protected validation proved 22,394 groups, max 7, zero address-authority edges, unchanged semantics and exact repaired-v2 physical digest. Grouping authority is global normalized-code/exact-artifact identity plus **source-namespaced** explicit source family IDs. Post-acceptance research evidence must be manifest/binding/source coherent before durable use | ADR-R4-009; `runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`; durable evidence `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`; snapshot commit `44fbb9c1d...` | Use V3 as logical authority; preserve repaired-v2 physical bytes. Hardened acceptance/sensitivity/selector/queue/GPU reports have been regenerated coherently and snapshotted. Selector/objective/promotion remain separate decisions | preserve V2 as historical evidence; regenerate any future stale/mismatched V3-derived report rather than re-authorizing V2 address grouping | delegated technical owner / GPT-5.6 Sol |
| R4-D-010 | 2026-08-21 | ACCEPTED | graph call-kind semantics / future physical representation lineage | Preserve accepted v9 artifacts as immutable historical evidence but prohibit the new full run from using v9 after the full-population audit proved library-call false edges and systematic Transfer/Send/low-level omissions. Require versioned graph schema v10 / extractor `v2.3-r4-call-semantics` / `representations-r4-v3-candidate`, explicit call-kind semantics, corrected consumers, full local side-by-side acceptance, and no silent token-selector promotion | ADR-R4-010; R4-GAP-008; `runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`; population report SHA-256 `77f90260...` | Implement source/tests remotely if desired; generate and accept the candidate only locally against ignored repaired-v2 data; bind a new physical digest before any later training decision | retain v9 bytes and prior decisions for reproduction only; if v10 fails, keep G8 open and revise a new versioned candidate rather than patching v9 | delegated technical owner / Codex local primary |

## R4-D-010 implementation-status note — 2026-08-27

R4-D-010's table row intentionally preserves the **original accepted decision** and its initial V10 candidate extractor identity `v2.3-r4-call-semantics`. That historical row must not be rewritten as if the later implementation revisions were known on 2026-08-21.

The decision itself explicitly requires versioning and says a failed V10 candidate must be revised as a new versioned candidate rather than patching v9. Subsequent extractor evolution therefore remains inside R4-D-010's accepted semantic boundary:

- `v2.3-r4-call-semantics` — frozen structural-reference diagnostic lineage;
- `v2.4-r4-call-semantics-compat` — versioned compatibility repair that closes the 26-contract parse-only tail and preserves the required 22,539 primary + 1 identity-bound runtime split;
- `v2.5-r4-call-semantics-deterministic-cfg` — current future-candidate extractor, adding only the evidence-backed deterministic persistent-storage CFG WRITE correction while keeping graph schema V10 and the R4-D-010 call-kind vocabulary unchanged.

The V2.5 bounded structural tranche is closed 20/20 as 8 exact node-index-invariant graph-equivalence identities plus 12 deterministic persistent-storage WRITE corrections, with zero unexplained drift. This is implementation/acceptance evidence under R4-D-010, **not a new policy decision and not physical acceptance**.

Current implementation/restart authority:

- `runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`;
- `reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
- `runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`.

A separate physical-acceptance decision record is still required after the fresh staged V2.5 full candidate, complete binding, complete V3 transition audit, and explicit review pass. Stage A has not yet executed.

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
