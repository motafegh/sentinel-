# 03 — Merger Sensitivity Table

The current merger consumes already-binarized per-source records. Therefore it cannot recover semantic distinctions that parsers already erased.

## Implemented behavior

| Situation | Current implementation | Semantic effect |
|---|---|---|
| one source for a contract | passes all class entries through, adding source metadata | parser-produced zeros survive unchanged |
| multiple sources, any positive for class | filters to positives and chooses the highest-confidence positive | **any positive defeats every zero**, even if the zero came from a nominally higher-confidence source |
| multiple sources, no positive for class | chooses the entry with best tier rank | parser negatives normally have `tier=None`, so this is not meaningful negative-confidence arbitration |
| same-tier positive vs negative | positive wins | consistent with implementation comment |
| low-tier positive vs higher-tier zero | positive still wins | source-tier precedence does not cause an explicit higher-tier zero to defeat a positive |
| DoS+Reentrancy suspect condition | adds `dos_reentrancy_cooccur_suspect` flag only | label cells are not changed |
| verification after merge | produces class verdicts / hard-fail status | current verification source does not mutate merged per-contract targets |

## Important implementation observations

### 1. Positive precedence is stronger than tier precedence

`_merge_class_entries()` first removes every zero entry and considers only positives. It then chooses the best positive tier. Only when there are no positives does it select among zero entries.

Therefore the effective rule is:

```text
any positive
  > every zero
then, among positives:
  T0 > T1 > T2 > T3 > T4
```

This differs from a strict interpretation of “T0 > T1 > T2 > T3 > T4 conflict precedence.”

### 2. Negative provenance is weak

Per-source parsers normally emit zero entries with `tier=None`. When every source is zero, `_tier_rank(None)` is the same for all entries. The selected `source` then follows iteration/input ordering rather than a demonstrated negative-authority rule.

`_SOURCE_PRECEDENCE` is defined in `merger.py` but is not used by `_merge_class_entries()`.

Therefore a merged zero's attached source must **not** be interpreted as “this source proved the class negative.”

### 3. Historical multi-source sensitivity was low in the early active corpus

The June 13 pipeline audit reported zero multi-source SHA overlaps in the 22,356-row DIVE+SolidiFI corpus. Thus most historical target corruption occurred **before** the merger, inside source semantics/crosswalk/default handling, rather than through cross-source conflict resolution.

Later SmartBugs inclusion can introduce additional source rows, but no tracked remote evidence establishes a material multi-source overlap population for the protected Run12 bundle.

### 4. Verification is currently a gate, not a rewrite layer

Current `verification/gate.py` computes `VERIFIED`, `PROVISIONAL`, `BEST-EFFORT`, or `FAIL` per class and can block by return status. It does not rewrite `data/labels/merged/*.labels.json`.

The old June audit describes a historical process in which verification/patching affected target artifacts. That is historical evidence, not current executable behavior. Phase 2 therefore separates:

- **current verification behavior:** report/gate only;
- **historical direct mutations:** e.g. the 2,655-row DoS parquet patch.

## Sensitivity scenarios

| Input evidence for one class | Output | Correct Phase-2 semantic interpretation |
|---|---:|---|
| DIVE `1`, SolidiFI absent | 1 | DIVE historical positive |
| SolidiFI `1`, DIVE `0` | 1 | T0 positive; DIVE zero does not contradict it because DIVE zero lacks negative authority |
| DIVE `1`, hypothetical T0 zero | 1 | implementation still chooses DIVE positive; this is not strict confidence precedence |
| DIVE `0`, SolidiFI `0` | 0 | unresolved/absence-derived zero unless independent negative authority exists |
| only DIVE all-zero record | all zero | passes through and later becomes NonVulnerable |
| DoS+Reentrancy flagged | both remain 1 in merger | flag is metadata only; historical DoS zeroing occurred elsewhere/out-of-band |

## Phase-2 conclusion

The merger is **not the primary creator of historical zeros**. Its main semantic defect is that it receives already-collapsed binary inputs and has no representation for `UNKNOWN`, `UNSUPPORTED`, `DROPPED`, or `CONFLICTING_EVIDENCE`. Consequently it preserves or arbitrates zeros as if they shared one meaning.
