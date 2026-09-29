# Phase-8 guarded-token physical acceptance and no-launch decision

Date: 2026-09-29  
Decision ID: `R4-D-013`  
Physical decision: `ACCEPTED_IMMUTABLE_LOCAL_GUARDED_TOKEN_REPRESENTATION`  
Training decision: `NOT_AUTHORIZED`  
Gate: G8 remains open

## Outcome

Accept the exact protected-local guarded-token representation root

`data_module/data/r4-guarded-d5-full-2026-09-29-a/representations-r4-v10-v26-guarded-v1-candidate`

as the immutable R4-D-012 successor token/representation lineage for controlled
Phase-8 research and possible later training eligibility.

The accepted full-population binding digest is

`9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`.

The accepted token lineage is `r4-v10-v26-guarded-tokens-v1`, using
`target_aware_guarded_v1` with explicit `historical_linspace_v1` guarded
fallback. The accepted population contains 22,540 identities: 14,751 use the
guarded selector and 7,789 use the exact historical fallback.

R4-D-011 is not replaced or mutated as graph/control history. Its exact V10 V2.6
graph root and digest remain the immutable parent and rollback/control authority.
R4-D-013 accepts the new token successor bound to those unchanged graph bytes.

This decision does **not** authorize full training, pass G8, create confirmed
negatives, approve a PU or supervised-negative objective, fit thresholds or
calibration, create untouched acceptance support, promote a checkpoint, or
change model architecture.

## Protected-local D5 generation review

D5 ran from generation source commit
`733f0c73eb76ab107751c30345d9a169a0429fdd` and produced:

- 22,540 requested / 22,540 written identities;
- 14,751 `target_aware_guarded_v1` identities;
- 7,789 `historical_linspace_v1` guarded fallbacks;
- graph schema `v10`;
- extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- frozen token shape `[4,512]`;
- Transformers `4.46.3`;
- candidate manifest SHA-256
  `5aa48ce6eb218b742af3c99823db333272d65599039b6ec550d56f713b839dca`;
- binding digest
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`.

The protected-local D5 generation report SHA-256 is
`e3202b77b4cb2557bcc47d19287a7c72ce248e1119f8b7d49521bce90d82f589`.

## Independent D6 acceptance review

D6 ran with tracked source clean at
`405b58e19210e6ff796aa943f8699b788d5229c1`. Pre-existing untracked local
files were preserved and were not acceptance evidence.

The independent D6 validator:

- re-inventoried all 22,540 R4-D-011 parent and guarded-candidate identities;
- verified exact population equality with no missing, extra or unexpected files;
- verified every repaired source byte hash against its contract identity;
- re-hashed every graph/token/sidecar artifact;
- independently reconstructed the D5 binding digest exactly;
- proved every candidate graph is byte-identical to the immutable R4-D-011 graph
  parent and verified parent graph/token/sidecar hashes recorded in each sidecar;
- validated V10 graph payload/schema/extractor invariants for all identities;
- validated frozen `[4,512]` int64 token tensors for all identities;
- validated full-population guarded-selector semantics, including strict target
  improvement or exact historical-control fallback;
- reproduced the selector distribution exactly as 14,751 guarded / 7,789
  historical fallback;
- reproduced the required runtime partition exactly as 22,539 Slither 0.10.0
  primary identities plus one Slither 0.11.5 identity-bound exception;
- regenerated the nine required D4 stress/evidence identities twice and required
  graph/token/sidecar bytes to equal both each other and the full D5 candidate.

The D6 validator status was `PASS_D6_REVIEW_REQUIRED`. The protected-local D6
report SHA-256 is
`8468a2b840131363444532ac4264ca048a2a39247b35d6552eea28f695b17289`.

The D6 report deliberately retained `physical_acceptance=false` and
`acceptance_decision=PENDING_EXPLICIT_REVIEW`; diagnostic tooling cannot grant
governance authority. This R4-D-013 decision supplies that explicit authority.

## Acceptance meaning and immutability

Physical acceptance means the exact guarded root/digest above is now the
accepted token/representation successor for later repaired ML integration. It
may be consumed only under later ML objective/evaluation and launch decisions.

The following remain immutable:

- R4-D-011 physical graph/control root and binding digest
  `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- the accepted D5 guarded root and R4-D-013 digest
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- historical token-control evidence and prior R4 decisions.

Never regenerate, patch, rename in place, or silently rebind either accepted
physical lineage.

## Why G8 and training remain open

R4-B006 / selector physical-lineage acceptance is closed by this decision, but
other ML gates remain independent:

1. confirmed-negative evaluation support remains zero;
2. candidate #2 still requires genuinely independent adjudication;
3. objective/evaluation semantics remain unresolved;
4. threshold fitting, calibration fitting and untouched acceptance remain
   unsupported/empty;
5. no explicit full-training authorization exists.

The next technical-completion workstream is therefore the ML
training/evaluation plan, beginning from its current M0/M1 boundary rather than
launching the 100-epoch run.

## Rollback

Rollback means selecting immutable R4-D-011 historical-control tokens and graph
parent for controlled reproduction. It never means editing R4-D-013 in place or
reinterpreting R4-D-011 as containing guarded tokens.

## Evidence bindings

| Evidence | SHA-256 / identity |
|---|---|
| R4-D-011 acceptance manifest | `5fc83eff39d4a28db9a5b6b5255a95ad64ee75ca88a948ba99dadb2bc03ee165` |
| D5 generation report | `e3202b77b4cb2557bcc47d19287a7c72ce248e1119f8b7d49521bce90d82f589` |
| D5 candidate manifest | `5aa48ce6eb218b742af3c99823db333272d65599039b6ec550d56f713b839dca` |
| D5/D6 binding digest | `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881` |
| D6 validation report | `8468a2b840131363444532ac4264ca048a2a39247b35d6552eea28f695b17289` |
| Generation source commit | `733f0c73eb76ab107751c30345d9a169a0429fdd` |
| Acceptance-review source commit | `405b58e19210e6ff796aa943f8699b788d5229c1` |

The machine-readable decision is
`evidence/2026-09-29_guarded_token_physical_acceptance/acceptance.json`.
