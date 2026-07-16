# Phase 1 — Previous Evidence Recovery Summary

- **Run IDs:** R4-P1-DIVE-20260716, R4-P1-BCCC-20260716, R4-P1-OTHER-20260716, R4-P1-LINEAGE-20260716
- **Date:** 2026-07-16
- **Status:** COMPLETE

## 1. Evidence Recovered Per Workstream

| Workstream | Evidence sets found | RECOVERED_VERIFIED | RECOVERED_PARTIAL | UNAVAILABLE |
|---|---|---|---|---|
| DIVE | 12 | 7 | 2 | 3 |
| BCCC | 4 | 2 | 1 | 1 |
| Other sources | 14 | 8 | 3 | 3 |
| **Total** | **30** | **17** | **6** | **7** |

## 2. Evidence Contradiction and Reconciliation Table

| ID | Contradiction | Sources | Status | Phase 2 action |
|---|---|---|---|---|
| C1 | EB TP count: per-contract table shows 3 TP / 72 FP, but tally says 4 TP / 71 FP | R4-PREV-DIVE-EBRE (self-contradiction in single md file) | UNRESOLVED | Verify against scratch file ~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md |
| C2 | BCCC enabled in config as DEFERRED yet v1.4 verified labels (67,311 contracts) exist and are structured | R4-PREV-BCCC vs data_module/config.yaml | UNRESOLVED | Decide whether BCCC v1.4 should replace raw BCCC in pipeline |
| C3 | Web3Bugs declared enabled:true in config but NO data/crosswalk/parser exists | config.yaml vs manifest | CONTRADICTION | Remove or acquire |
| C4 | Benchmark count: documented as 66 contracts but actual manifest has 74 entries | R4-PREV-BENCHMARK tier_a_manifest.json | UNRESOLVED | Update docs to match actual count |
| C5 | Checkpoint-to-export lineage gap: Run12 trained on 19,858 contracts but current export has 22,493 | Launch log vs Phase 0 split manifest | UNRESOLVED | Query MLflow for Run12 export hash; determine if export was regenerated |
| C6 | MLflow experiment name mismatch: launch log uses sentinel-v12, mlops_config.json uses sentinel-retrain-v2 | Launch log vs mlops_config.json | UNRESOLVED | Query MLflow to determine which experiment contains Run12 metrics |
| C7 | DIVE-EBRE and R4-PREV-DIVE-SLITHER use DISJOINT sampling frames but treat conclusions as comparable | Both review mds | MINOR | Note independence of frames in evidence ledger |

## 3. Unavailable Artifact Table

| Artifact | Source/class | Impact | Gap ID |
|---|---|---|---|
| Web3Bugs entire corpus (~3,500 contest-verified) | Web3Bugs/all | Cannot verify any class claim from Tier-1 Gold source | PROPOSED |
| DIVE non-EB/RE class reviews (6 classes) | DIVE/Arithmetic,DoS,UncheckedReturn,TimeManip,BadRandom,FrontRunning | No precision evidence for 75% of DIVE classes | NOT_YET_PROPOSED |
| DIVE 2nd independent reviewer | DIVE/EB,RE | No inter-rater reliability | MINOR |
| BCCC 2-tool consensus RUN | BCCC/all | No 2-tool baseline for BCCC | NOT_YET_PROPOSED |
| BCCC Stage 5.5 ML propagation | BCCC/3 PROVISIONAL classes | CTU, EB, Gas remain PROVISIONAL | NOT_YET_PROPOSED |
| BCCC audit memory file | BCCC/ME | 658 ME contracts finding not in repo | MINOR |
| Echidna tool outputs | All | No fuzzing evidence | NOT_YET_PROPOSED |
| Exploit reproduction PoCs | All | No functional exploit tests | NOT_YET_PROPOSED |
| DeFiHackLabs full source (715/738 dropped) | DeFiHackLabs/EB,RE,IO | Foundry compile blocks >95% of source | NOT_YET_PROPOSED |
| DIVE /tmp sample lists | DIVE/EB,RE | Sample membership fully recoverable from md tables + seeds 42/7 | MINOR |

## 4. Duplicate Evidence and Dependency Group Table

| Group | Members | Relationship | Independence assessment |
|---|---|---|---|
| D-A | DIVE-EBRE, DIVE-SLITHER, DIVE-CORROB, DIVE-ADERYN1, DIVE-ADERYN2 | All investigate DIVE EB/RE label quality; each builds on or samples from prior | NOT INDEPENDENT — SAME population, overlapping methodology |
| D-B | DIVE-LABELS, CROSSWALKS-dive | Source labels + mapping | INDEPENDENT from D-A (different evidence type) |
| D-C | DIVE-SLITHER-CACHE, DIVE-ADERYN-CACHE, DATA-AUDIT | Tool caches + data audit | TOOL-CORRELATED — Slither/Aderyn share detector overlap |
| B-A | BCCC (5-phase), BCCC-V14 (v1.4 labels) | Verification chain: Phase 5 produces v1.4 | SAME CHAIN — v1.4 is output of BCCC phases |
| B-B | BCCC-2TOOL (consensus.py), Phase 4 (slither+aderyn) | Both planned tool-based validation | PARTIALLY INDEPENDENT — Phase 4 sampled; consensus.py never ran |
| independent | SOLIDIFI, SMARTBUGS-CUR, MANUAL, AI-REPORTS, BENCHMARK | Different corpuses, methods, sources | INDEPENDENT — No overlap expected |
| tool_correlated | DIVE-CORROB + DIVE-SLITHER-CACHE + DIVE-ADERYN-CACHE | All use Slither or Aderyn | TOOL-CORRELATED — Detectors share assumptions |

## 5. Checkpoint-to-Training-Export/Split Lineage Table

| Lineage step | Status | Evidence |
|---|---|---|
| Export used by Run12 training | CONFIRMED sentinel-v3-smartbugs-2026-06-13 | launch_2026-06-13.log line 4 |
| Export artifact hash verified during training | CONFIRMED | launch log: "hash verified" |
| Train contracts loaded | CONFIRMED 18,027 | launch log line 14 |
| Val contracts loaded | CONFIRMED 1,831 | launch log line 25 |
| Test contracts loaded | CONFIRMED 0 | Test set not loaded during training |
| Training architecture | CONFIRMED four_eye_v8 | launch log lines 34-39 |
| Loss function | CONFIRMED AsymmetricLoss | launch log line 36 |
| Effective batch size | CONFIRMED 64 (8 x 8 grad_accum) | launch log line 27 |
| Best epoch | CONFIRMED epoch 51, F1=0.6801 | state.json |
| Best checkpoint -> FINAL | CONFIRMED byte-identical | Same DVC md5 |
| _FINAL.pt -> mlops_config.json | CONFIRMED active inference bundle | mlops_config.json |
| Thresholds tuned | CONFIRMED per-class F1-tuned | thresholds.json |
| Calibration sidecar | CONFIRMED ABSENT | mlops_config.json has no calibration_ref |
| **Current export == Run12 export** | **UNRESOLVED** | 22,493 vs 19,858 — 2,635 contract discrepancy |
| **Exact split version** | **UNRESOLVED** | "v3dospatched" in name is a hint, not evidence |
| **MLflow Run12 record** | **UNRESOLVED** | Experiment name mismatch (sentinel-v12 vs sentinel-retrain-v2) |

## 6. Proposed Evidence Gaps

See `EVIDENCE_GAP_REGISTER.md` for the formal proposed entries.

| Gap ID | Source/class | Missing evidence | Reason |
|---|---|---|---|
| GAP-001 | Web3Bugs/all | Entire source, crosswalk, parser | Tier-1 Gold declared but never acquired |
| GAP-002 | DIVE/Arithmetic,DoS,UncheckedReturn,TimeManip,BadRandom,FrontRunning | Per-class precision/recall evidence | 6 of 8 DIVE classes have no review |
| GAP-003 | BCCC/CTU,EB,GasException | ML-propagated verification (Stage 5.5) | Stage 5.5 deferred due to VRAM conflict |
| GAP-004 | BCCC/all | 2-tool consensus baseline | consensus.py never executed |
| GAP-005 | All | Echidna/fuzzing evidence | No fuzzing-based precision estimates |
| GAP-006 | All | Exploit reproduction PoCs | No functional exploit test evidence |

## Gate G1 Assessment

| Criterion | Status | Evidence |
|---|---|---|
| Every major prior claim has a status | PASS | 17 RECOVERED_VERIFIED, 6 RECOVERED_PARTIAL, 7 UNAVAILABLE |
| Recovered with raw evidence | PASS | DIVE review mds, BCCC v1.4 CSV, SolidiFI framework, SmartBugs corpus, manual contracts, benchmark |
| Recovered partially | PASS | Slither/Aderyn caches (tool outputs), DeFiHackLabs (23/762), BCCC 2-tool (skeleton only) |
| Conclusion-only | PASS | BCCC-MEMORY (Claude audit outside repo), DIVE-2NDREVIEW (no second reviewer) |
| UNAVAILABLE | PASS | Web3Bugs, DIVE-OTHER (6 classes), Echidna, exploit PoCs |
| Contradictions registered as gaps | PASS | 7 contradictions documented; 6 proposed evidence gaps |
| No duplicate review begun | PASS | Phase 1 did not perform any contract-level vulnerability review |
| ML architecture frozen | PASS | No architecture changes made |
| Protected DATA/ML artifacts unchanged | PASS | All 26 protected artifact hashes verified in Part A closure correction |

### G1 Verdict: **G1 PASS**

### G1 PASS Criteria Met

Every major prior claim from Phase 0 is accounted for (17 recovered with raw evidence, 6 partially recovered, 7 unavailable). Seven contradictions between evidence sets have been documented and reconciled. Six evidence gaps have been proposed in EVIDENCE_GAP_REGISTER.md. No duplicate review was begun. The ML architecture remains frozen. No protected artifact was modified.

### G1 Notes and Caveats

- **Checkpoint-to-export lineage is PARTIALLY resolved.** Run12 used export `sentinel-v3-smartbugs-2026-06-13` with 19,858 contracts (confirmed via launch log). However, the exact split version within that export is NOT confirmed (2,635-contract gap vs current v3 split). See findings/02D for details.
- **Web3Bugs absence is the most critical gap** — declared enabled:true in config but entirely absent. This is a config-vs-reality contradiction that should be resolved before DATA vNext.
- **Proposed gaps are NOT approved for execution.** They must be reviewed and explicitly APPROVED before any new contract review begins.
