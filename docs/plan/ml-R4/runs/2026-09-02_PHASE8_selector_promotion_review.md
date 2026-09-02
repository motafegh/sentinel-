# Phase-8 target-aware guarded selector promotion review

Date: 2026-09-02
Decision: R4-D-012 / `PROMOTED_FOR_NEW_VERSIONED_CANDIDATE_ONLY`
Training: NOT AUTHORIZED

## Outcome

The source-first review supports promoting `target_aware_guarded_v1` only as
the token-selection policy for a new versioned physical candidate. It does not
mutate or supersede the physical acceptance of R4-D-011 until that new candidate
is independently generated, bound, and accepted.

## Verified evidence

- The committed logical-V3 snapshot passes all 11 checksums and 60 recomputed
  coherence checks.
- Selector/span source is unchanged from the hardened evidence commit.
- CPU coverage: 1,018/1,018 records, 737 over-cap, 476 improved, 261 equal via
  control fallback, zero regressions, zero failures.
- Over-cap median target coverage: 0.630063 control versus 0.879447 guarded.
- Over-cap median overall retention: 0.601010 control versus 0.577922 guarded;
  this explicit tradeoff prevents an overclaim of universally better context.
- CUDA: identical initialization, four of four worst-case forward probes, valid
  `[1,10]` logits, no checkpoint, no Run12 weights, no training authority.
- R4-D-011 control equivalence: 22,540/22,540 tensor and selected-index matches,
  zero failures or mismatches.

## Bounded next construction

Create a fresh versioned candidate that reuses the accepted V2.6 graph and
sidecar semantics but dynamically produces guarded token payloads. Bind a new
digest and prove that differences from R4-D-011 are restricted to declared
selector/token fields. Do not accept or train from it until a separate physical
review passes.
