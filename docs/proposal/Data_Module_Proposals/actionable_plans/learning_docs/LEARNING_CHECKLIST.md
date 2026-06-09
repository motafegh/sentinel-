# Learning Checklist — Sentinel v2 Data Module Build

**Date:** 2026-06-09
**How to use:** Each row is one concept. Tick `[x]` when you can answer the "Test yourself" question from memory. After all rows in a stage are ticked, that stage is mastered.

---

## Stage 0 — Skeleton + Data/ Restructure

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 0.1 | **Why v2 exists** | Why is F1=0.31 the ceiling, and what part is the data vs the model? | [ ] |
| 0.2 | **BCCC failure pattern** | What does 89% Reentrancy FP + 86.9% CallToUnknown FP tell us about folder-based labeling? | [ ] |
| 0.3 | **The 3 branches considered** | Why was "full package split" chosen over "keep data in ml/ and just clean labels"? | [ ] |
| 0.4 | **The hard boundary** | What does "sentinel-ml never touches a raw contract" mean in practice? | [ ] |
| 0.5 | **One-way dependency** | Why is the dependency `sentinel-ml → sentinel-data` only, never reverse? | [ ] |
| 0.6 | **Module location** | Why is the new package rooted at the existing `~/projects/sentinel/Data/` (not a new top-level dir)? | [ ] |
| 0.7 | **Schema v9 (not v8)** | The proposal said v8, but the live schema is v9. Why does it matter? | [ ] |
| 0.8 | **MLflow backend** | Why `sqlite:///` and not `file:///`? | [ ] |
| 0.9 | **BCCC deferred** | Why is BCCC in `deferred_sources:` and not in regular `sources:`? | [ ] |
| 0.10 | **8 already-fixed bugs** | Name 3 of the 8 bugs already fixed in `ml/src/` and what file:line they're at. | [ ] |
| 0.11 | **3 still-open bugs** | Name the 3 bugs that Stage 7 must close. | [ ] |
| 0.12 | **22 sources, correctly tiered** | Why 8 T1 gold + 2 T2 clean + 3 T2 silver + 3 T3 structural + 2 T3 bronze + 2 T4 bronze + 2 v1 extras? Why this distribution? | [ ] |
| 0.13 | **Stub vs real code** | Why does Stage 0 ship stubs (not real code)? | [ ] |
| 0.14 | **27 tests** | What does the test suite guarantee? What's the 36-issue audit regression test? | [ ] |
| 0.15 | **Stage 0 exit criteria** | What are the 15 exit criteria, and which 5 are "testable" vs "design-only"? | [ ] |

---

## Stage 1 — Ingestion + Preprocessing

| # | Concept | Test yourself | Tick |
|---|---|---|---|
| 1.1 | **5-step pipeline order** | Why is the order flatten → compile → dedup → normalize → segment+bucket? What breaks if you reorder? | [ ] |
| 1.2 | **Two-pass compile** | What does Pass 1 try, what does Pass 2 try, and which Phase 5 contracts does this recover? | [ ] |
| 1.3 | **Pragma tolerance** | Why does the compiler strip whitespace from the pragma before regex? What's `^ 0.4 .9` vs `^0.4.9`? | [ ] |
| 1.4 | **3-level dedup** | What does each level catch (exact SHA / address / AST), and why is the AST level stubbed for Stage 2? | [ ] |
| 1.5 | **`ast_similarity_threshold=0.85`** | Why not 0.92? What's the BCCC near-dup range, and what's the friend-research sweet spot? | [ ] |
| 1.6 | **Drop-not-fix for compile failures** | Why is a compile-failed file dropped (not passed through with a warning)? What's the BCCC precedent? | [ ] |
| 1.7 | **Sidecar `meta.json`** | Name 5 of the 18 fields in the `ContractMeta` schema. Which downstream stages read which fields? | [ ] |
| 1.8 | **Ingestion manifest + SHA-256** | What does the manifest record, and what does `verify_manifest` do? Why is "silent file change" the failure mode this prevents? | [ ] |
| 1.9 | **One connector per family** | Why are there 5 connector classes (Git, HF, Zenodo, Etherscan, Manual) and not 17 (one per source)? | [ ] |
| 1.10 | **Pinned versions** | Why is the pin currently empty for the 5 critical-path sources in `config.yaml`? What's the stage prerequisite for filling it in? | [ ] |
| 1.11 | **A9 regression test** | What bug does the `now` keyword survival test prevent? Where was the fix, and what was the failure rate before the fix? | [ ] |
| 1.12 | **`freshness` subcommand** | What 2 things does it check? Why is the report informational, not blocking? | [ ] |
| 1.13 | **`version_bucket` + `has_unchecked_block`** | Why is the 0.6–0.7 era called "transitional"? Why does `has_unchecked_block` matter for Stage 4's IntegerUO checker? | [ ] |
| 1.14 | **Friend-review impact on Stage 1** | What 3 friend-review changes affected Stage 1? (Critical-path corpus, ReentrancyStudy drop, DIVE bad_randomness) | [ ] |
| 1.15 | **Stage 1 partial vs pass** | Which 4 of the 14 exit criteria are PARTIAL (not fully PASS), and why are they deferred? | [ ] |

---

## Stage 2 — Representation (port from ml/)

(Will be filled in when we start Stage 2)

---

## Stage 3 — Labeling (parsers + crosswalks + 99% co-occurrence)

(Will be filled in when we start Stage 3)

---

## Stage 4 — Verification (BCCC-failure catcher)

(Will be filled in when we start Stage 4)

---

## Stage 5 — Splitting + Registry

(Will be filled in when we start Stage 5)

---

## Stage 6 — Analysis (complexity_proxy_risk)

(Will be filled in when we start Stage 6)

---

## Stage 7 — Export + Seam Swap (predictor fix + EMITS fix)

(Will be filled in when we start Stage 7)

---

## Stage 8 — Run 11 launch (12-condition checklist)

(Will be filled in when we start Stage 8)

---

## Mastery milestone

When all 8 stages are ticked off, you can:

1. Explain the v2 build end-to-end from raw .sol to trained model
2. Diagnose any future data issue (BCCC-class failure) using the verification stage + co-occurrence matrices
3. Add a new data source to the v2 build (crosswalk + parser + connector)
4. Defend every design decision against the alternatives
5. Onboard the next person to the build in <1 day

**This is the goal. We get there incrementally, one stage at a time, with comprehension before code.**
