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

(Will be filled in when we start Stage 1)

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
