# 15 — Change playbooks

**Read this when:** you are about to change DATA semantics, representations, retraining behavior, proxy/circuit artifacts, V3 protocol/contracts, or add a node/tool/class/source.

**Skip this if:** you are only reading or operating unchanged code.

**Estimated reading time:** 13 minutes.

## 30-second summary

SENTINEL changes have two permanent compatibility disciplines. **DATA/ML changes** must preserve historical v1/Run12/G7 evidence while evolving the current R4 semantic, logical-role, physical-representation, and training boundaries in versioned steps. **Chain/proof changes** must preserve V1/V2 history while evolving the V3 context-attested protocol. Start with the semantic invariant, enumerate producers/consumers/artifacts/evidence, add mismatch tests, then migrate in dependency order. Never patch a downstream consumer to reinterpret an old artifact silently.

Current DATA/ML authority is not “finish G7 then retrain.” G0–G7 are already historical/immutable; Phase 8 is in progress. The live boundary is D-009 logical V3 + D-011 accepted V10 V2.6 physical representation + D-012 fresh guarded-selector successor pending separate physical acceptance, with full repaired training still unauthorized.

## Just-enough mental model

```text
source/evidence policy
→ repaired semantic state
→ logical V3 grouping / roles
→ accepted D-011 physical graph lineage
→ fresh D-012 guarded-token successor
→ separate physical acceptance
→ later repaired teacher only if authorized
→ proxy/circuit if selected teacher changes
→ V3 verifier/context protocol
→ AGENTS observation/report/eval
→ operations/docs
```

Historical v1/G7/Run12/V1-V2 registry artifacts remain reproducibility roots; new work is additive/versioned unless an approved migration explicitly says otherwise.

## Actual runtime/source walkthrough

### DATA label/evidence policy change

1. identify the controlling R4 decision/ADR/source/class authority;
2. preserve source-native claim and historical artifact hashes;
3. version policy/schema if meaning changes;
4. regenerate explicit contract×class semantic artifacts;
5. reassess logical V3 grouping/role eligibility when required;
6. validate that unknown/no-target states cannot become negative;
7. update downstream dataset/training/evaluation compatibility;
8. preserve historical v1/G7 exports for reproduction.

Never fix a semantic issue by hand-editing binary/Parquet target cells.

### Leakage grouping / role change

Logical grouping and role authority is versioned independently from physical representation bytes. If grouping authority changes:

1. identify the defect/evidence that invalidates current logical authority;
2. preserve D-009/V3 artifacts as history;
3. build a new grouping version from explicitly allowed identity sources;
4. prove no prohibited authority (such as arbitrary address coincidence) creates edges;
5. regenerate role assignments with leakage/exposure constraints;
6. verify downstream dataset/evaluation role semantics;
7. promote only through a new explicit logical decision.

Do not edit `r4-vnext-roles-v3` in place.

### Representation / token-selector change

Representation changes must distinguish graph semantics from token-window selection.

Current example:

- D-010 withdrew v9 from eligibility for the new full run;
- D-011 accepted an exact V10 V2.6 graph/control-token root;
- D-012 promotes `target_aware_guarded_v1` only for a fresh token successor.

For a new representation/selector change:

1. preserve the accepted rollback/control root;
2. version extractor/schema/selector identities separately;
3. regenerate only from declared source/runtime/toolchain authority;
4. bind exact artifacts/digests;
5. explain every structural/semantic drift class rather than weakening equivalence checks;
6. require a separate physical acceptance decision;
7. only then allow downstream trainer/evaluation work to consume the successor.

### Repaired teacher training without architecture-shape change

1. verify current R4 status/D-009/D-011/D-012 authority;
2. require the exact accepted successor representation/token lineage for the intended run;
3. bind target/strength/mask/role/group/config/seed/source/runtime identities;
4. use repaired Phase-8 mechanics with the existing four-eye architecture;
5. use only optimizer-authorized cells/roles for gradients;
6. use model-selection evidence only for claims it can support;
7. do not reuse Run12 optimizer/threshold/calibration state as repaired truth;
8. report unsupported threshold/calibration/acceptance claims explicitly;
9. require explicit full-run authorization before launching a full repaired run;
10. select/promote a teacher before redistilling ZKML.

Same tensor shape does not make a new checkpoint semantically compatible with old decision artifacts.

### Threshold/calibration/acceptance evidence change

The current repaired path has no authorized threshold-fit, calibration-fit, or untouched-acceptance population. To add one:

1. acquire/recover class-specific trustworthy outcome evidence;
2. prove exposure/leakage independence;
3. create a new evidence/role decision and versioned manifest;
4. keep it out of optimization/model-selection as required;
5. only then fit thresholds/calibration or claim untouched acceptance.

Do not repurpose unknown, exposed, quickstart, BCCC/tool-silent, historical test, or queue-reservation data merely to fill a role.

### Proxy/circuit regeneration after a repaired teacher

Pin the selected teacher/data/config identity; generate distillation evidence; retrain proxy; measure agreement; export ONNX; regenerate settings/compiled/key/verifier artifacts as required; prove/verify 138-signal behavior; bind a new proxy-bundle identity; update V3 verifier/config/deployment through controlled tests.

Do not regenerate the retained proxy solely because R4 DATA artifacts changed. Trigger the migration when the actually selected teacher/fusion seam changes.

### V3 protocol / UUPS change

Preserve V1/V2/V3 storage order and historical reads. If any signed field/type/domain meaning changes, treat it as a protocol migration affecting:

- EIP-712 typehash/digest;
- `policy_signer.py` request builder;
- contract input validation/storage/events;
- replay/expiry/signature tests;
- submitting-service schema;
- read-only versioned observation;
- deployment/rotation/rollback.

Do not mutate V2 semantics and call it V3 compatibility.

### V3 signer/broadcaster implementation

A real submission service is a new security domain, not an audit-MCP helper:

1. define isolated KMS/HSM/key custody and policy authority;
2. consume exact validated V3 request identities;
3. construct/broadcast transactions separately from the analysis MCP;
4. implement idempotency, confirmations/reorgs, retries, receipt persistence, and rotation;
5. add end-to-end Anvil/testnet evidence;
6. keep the audit MCP read-only unless an explicit new architecture decision changes that trust boundary.

### New LangGraph node / MCP tool

Classify evidence determinism and mutation authority first. Add state/reducer/status schema, resource limits, tests, routing, and provenance. A mutating MCP tool requires a new trust/security decision; do not smuggle signing or ground-truth mutation into a generic service.

### New vulnerability class

This is a full-system migration: class vocabulary/policy, DATA semantic/representation compatibility, teacher output head/metrics, proxy output/circuit signal layout, V3 class-schema identity/fixed arrays, AGENTS mapping/evaluation, and every consumer. Prefer a new versioned contract/protocol rather than mutating the locked ten-class meaning in place.

### New DATA/RAG/feedback source

Pin provenance/license/identity; determine what outcome claim the source can actually establish; separate unknown from negative; deduplicate/contamination-test; define allowed dataset/feedback roles; preserve removal/rebuild/versioning. A chain/report/RAG record is not automatically a ground-truth label.

## Interfaces, data shapes, and configuration

Every significant change record should contain:

- motivation and measured baseline;
- semantic/version migration type;
- affected source symbols and machine-readable policies;
- old/new artifact hashes;
- logical role/group and exposure effects;
- graph/token/model/proxy/V3 identity effects;
- verification matrix: static, unit/module, artifact, local/live;
- trust/security changes;
- rollback bundle;
- current-status/documentation updates.

## Failure modes and current limitations

- Same shape with changed meaning is silent corruption.
- Reusing historical zeros with repaired semantics is semantic rollback.
- Treating G6/G7 role authority as current after D-009 reintroduces superseded logical state.
- Treating v9 as new-full-training eligible violates D-010.
- Mutating D-011 in place to apply D-012 destroys accepted rollback identity.
- Reusing Run12 thresholds/calibration after repaired training is invalid.
- Reusing old proxy agreement after teacher changes is invalid.
- Reusing V2 submission assumptions in V3 loses context binding.
- Adding signer/broadcast logic to the audit MCP violates current least-privilege architecture.
- Calling exposed/noisy data untouched acceptance manufactures evidence.
- Updating implementation without the R4/ADR/current-status record makes later ownership unreliable.

## Common change recipe

Universal recipe:

1. bind current commit/artifacts/policy/role evidence;
2. state the semantic/trust invariant;
3. classify historical compatibility, current accepted authority, pending successor, and new version;
4. add failure/mismatch tests;
5. implement producer → consumer without default/fallback reinterpretation;
6. regenerate/version artifacts deterministically and hash-bind them;
7. run static → focused/module → artifact → relevant local/live checks;
8. record unsupported evidence instead of weakening gates;
9. update R4/ADR/handbook/current status;
10. verify rollback selects an older compatible bundle rather than mutating history.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
git diff --check
git diff --name-only
```

Then run every affected subsystem/artifact/local gate required by the change.

## Optional deep references

- [Architecture](01_architecture.md)
- [Cross-module contracts](11_cross_module_contracts.md)
- [Security and trust](12_security_and_trust.md)
- [Evaluation](13_evaluation.md)
- [Current status](16_current_status.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)

## Technical mastery layer

### Prerequisite knowledge

Know dependency graphs, migrations, partial-label semantics, artifact lineage, representation versioning, EIP-712, staged rollout, rollback, and evidence-role separation.

### Source map and reading order

Start with current status and the governing R4/ADR record. Then inspect the authoritative producer source, cross-module consumers, focused tests, artifact manifests, and operational/security boundaries. Supplementary technical guides/labs may help with mechanics but do not override current canonical decisions.

### Execution trace and worked example

Applying a new token selector is not a local tokenizer tweak: preserve D-011, generate a new successor under the selector decision, bind/accept its bytes, then allow the training consumer to use that exact identity. A later selected repaired teacher may then require proxy/circuit/V3 identity updates. Each transition is versioned and rollback-safe.

### Implementation practice

Characterize with a failing semantic/compatibility test, edit the smallest authoritative producer, update all consumers explicitly, regenerate hash-bound artifacts, run role/trust validation, then document promotion/rollback.

### Review and ownership check

Can you state the historical compatibility roots, current accepted logical/physical authorities, pending successor, every consumer that must migrate, the evidence required for promotion, and the exact rollback bundle before editing?