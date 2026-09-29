# ADR-R4-013 — Guarded-token physical representation acceptance and training hold

Date: 2026-09-29  
Status: ACCEPTED  
Decision ID: R4-D-013  
Scope: R4 guarded-token physical lineage and Phase-8 ML handoff boundary

## Context

R4-D-011 accepted the exact V10 V2.6 graph/control-token physical lineage but
did not authorize selector promotion or training. R4-D-012 later promoted
`target_aware_guarded_v1` only for construction/evaluation of a fresh versioned
token successor, with `historical_linspace_v1` retained as explicit control and
fallback.

The technical-completion DATA workstream implemented that successor without
mutating R4-D-011. D4 passed the bounded protected-local tranche. D5 then
generated the full 22,540-identity guarded candidate from source commit
`733f0c73eb76ab107751c30345d9a169a0429fdd`, producing 14,751 guarded
selections and 7,789 historical fallbacks with binding digest
`9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`.

Independent D6 validation at tracked-clean source commit
`405b58e19210e6ff796aa943f8699b788d5229c1` checked all 22,540 identities,
reconstructed the binding digest, verified exact R4-D-011 graph-parent bytes and
source identities, validated V10 graph/token/runtime/selector contracts, and
regenerated the nine required D4 probes twice byte-for-byte. D6 passed without
an unexplained regression.

## Decision

Accept the exact protected-local root

`data_module/data/r4-guarded-d5-full-2026-09-29-a/representations-r4-v10-v26-guarded-v1-candidate`

as the immutable guarded-token physical successor under representation lineage
`r4-v10-v26-guarded-tokens-v1` and binding digest
`9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`.

The accepted selector policy is `target_aware_guarded_v1`; exact
`historical_linspace_v1` fallback is part of that accepted guarded policy, not
a defect. The frozen token shape remains `[4,512]`, graph schema remains
`v10`, and extractor remains
`v2.6-r4-call-semantics-deterministic-cfg-mutators`.

R4-D-011 remains immutable as the exact graph parent and historical
control/rollback lineage. R4-D-013 does not rewrite R4-D-011.

This acceptance grants physical DATA/representation authority only. G8 remains
open and full training remains unauthorized.

## Consequences

- R4-B006 is closed for the exact guarded root/digest above.
- The DATA/representation technical-completion workstream is complete and may
  hand off the accepted physical lineage to the ML training/evaluation
  completion workstream.
- Future repaired ML integration must bind to the exact R4-D-013 digest,
  selector identity, R4-D-011 graph parent, logical V3 roles/publication, and
  later accepted objective/evaluation policy.
- No confirmed-negative, discrimination, threshold, calibration, untouched
  acceptance, checkpoint-quality, production, or on-chain claim follows from
  this decision.
- No full training run is authorized by this ADR.
- Any later token-selector semantic change requires another fresh versioned
  lineage and acceptance decision.

## Rollback

Use immutable R4-D-011 with `historical_linspace_v1` for controlled historical
reproduction. Never reverse-edit R4-D-013 or patch R4-D-011 to imitate it.

## Evidence

- `runs/2026-09-29_PHASE8_guarded_token_physical_acceptance_and_no_launch.md`;
- `evidence/2026-09-29_guarded_token_physical_acceptance/acceptance.json`;
- D5 generation report SHA-256
  `e3202b77b4cb2557bcc47d19287a7c72ce248e1119f8b7d49521bce90d82f589`;
- D6 validation report SHA-256
  `8468a2b840131363444532ac4264ca048a2a39247b35d6552eea28f695b17289`;
- candidate manifest SHA-256
  `5aa48ce6eb218b742af3c99823db333272d65599039b6ec550d56f713b839dca`;
- accepted binding digest
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- R4-D-011 parent digest
  `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.
