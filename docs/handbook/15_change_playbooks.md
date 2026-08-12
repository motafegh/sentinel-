# 15 — Change playbooks

**Read this when:** you are about to change DATA semantics, retrain, regenerate a circuit, change V3 protocol/contracts, or add a node/tool/class/source.

**Skip this if:** you are only reading or operating unchanged code.

**Estimated reading time:** 13 minutes.

## 30-second summary

SENTINEL changes now have two permanent compatibility disciplines. **DATA/ML semantic changes** must preserve historical v1/Run12 artifacts while evolving the explicit DATA vNext policy/schema/roles. **Chain/proof changes** must preserve V1/V2 history while evolving the V3 context-attested protocol. Start with the semantic invariant, enumerate producers/consumers/artifacts/evidence, add mismatch tests, then migrate in versioned dependency order. Never patch a downstream consumer to reinterpret an old artifact silently.

## Just-enough mental model

```text
source/evidence policy
→ DATA vNext semantic artifacts + roles
→ teacher retrain
→ proxy/circuit
→ V3 verifier/context protocol
→ AGENTS observation/report/eval
→ operations/docs
```

Historical v1/Run12/V1-V2 registry artifacts remain reproducibility roots; new work is additive/versioned unless an approved migration explicitly says otherwise.

## Actual runtime/source walkthrough

### DATA label/evidence policy change

1. identify affected R4 decision/ADR/source/class authority;
2. preserve source-native claim and existing historical artifact hashes;
3. version policy/schema if meaning changes;
4. regenerate contract×class semantic artifacts;
5. reassess leakage-group role eligibility and re-freeze roles if necessary;
6. validate that unknown/no-target states cannot become negative;
7. update downstream trainer/evaluation compatibility;
8. preserve old v1 export for Run12 reproduction.

Never fix a semantic issue by hand-editing binary Parquet cells.

### Representation/schema/class-order change

A graph-feature or class-order change has wider blast radius than a label-policy change. Version the representation/class contract, regenerate dependent graph/token artifacts, rebuild vNext bindings/roles as required, retrain teacher, redistill proxy, regenerate circuit/verifier, and update V3 class-schema/proxy/data identities. Do not pretend a label-policy version bump is the same thing as graph schema v9 changing.

### Retrain teacher without architecture-shape change

1. finish/verify DATA vNext G7;
2. implement target/strength/mask/role-aware consumer changes;
3. bind config/seed/initialization/DATA identities;
4. train the existing architecture;
5. use only authorized role evidence for optimization/checkpoint selection;
6. do not reuse Run12 thresholds/calibration;
7. report unsupported threshold/calibration/acceptance claims explicitly;
8. select a candidate before redistilling ZKML.

Same tensor shape does not make a new checkpoint semantically compatible with old decision artifacts.

### Threshold/calibration/acceptance evidence change

The current first repaired baseline has no authorized threshold-fit, calibration-fit, or untouched-acceptance population. To add one:

1. acquire/recover class-specific trustworthy outcome evidence;
2. prove exposure/leakage independence;
3. create a new evidence/role decision and manifest version;
4. keep it out of training/model-selection as required;
5. only then fit thresholds/calibration or claim untouched acceptance.

Do not repurpose unknown, exposed, quickstart, BCCC/tool-silent, or historical test data to fill the role.

### Proxy/circuit regeneration after repaired teacher

Pin the selected teacher/data/config identity; generate distillation evidence; retrain proxy; measure agreement; export ONNX; regenerate settings/compiled/key/verifier artifacts as needed; prove/verify 138-signal behavior; bind a new proxy-bundle identity; update V3 verifier/config/deployment through controlled tests.

### V3 protocol / UUPS change

Preserve V1/V2/V3 storage order and historical reads. If any signed field/type/domain meaning changes, treat it as a protocol migration affecting:

- EIP-712 typehash/digest;
- `policy_signer.py` request builder;
- contract input validation/storage/events;
- replay/expiry/signature tests;
- submitting service schema;
- read-only versioned observation;
- deployment/rotation/rollback.

Do not mutate V2 semantics and call it V3 compatibility.

### V3 signer/broadcaster implementation

A real submission service is a new security domain, not an audit-MCP helper:

1. define isolated KMS/HSM/key custody and policy authority;
2. consume exact validated V3 request identities;
3. construct/broadcast transaction separately from analysis MCP;
4. implement idempotency, confirmations/reorgs, retries, receipt persistence, and rotation;
5. add end-to-end Anvil/testnet evidence;
6. keep read-only audit MCP read-only unless a new approved architecture explicitly changes it.

### New LangGraph node / MCP tool

Classify evidence determinism and mutation authority first. Add state/reducer/status schema, resource limits, tests, routing, and provenance. A mutating MCP tool requires a new trust/security decision; do not smuggle signing or ground-truth mutation into a generic service.

### New vulnerability class

This remains a full-system migration: class vocabulary/policy, DATA state/export, model head/metrics, proxy output/circuit signal layout, V3 class-schema identity and fixed arrays, AGENTS mapping/eval, and every consumer. Prefer a new versioned contract/protocol rather than mutating the locked ten-class meaning in place.

### New DATA/RAG/feedback source

Pin provenance/license/identity; determine what outcome claim the source can actually establish; separate unknown from negative; deduplicate and contamination-test; define allowed dataset/feedback roles; preserve removal/rebuild/versioning. A chain/report/RAG record is not automatically a ground-truth label.

## Interfaces, data shapes, and configuration

Every significant change record should contain:

- motivation and measured baseline;
- semantic/version migration type;
- affected source symbols and machine-readable policies;
- old/new artifact hashes;
- dataset role/exposure effects;
- model/proxy/V3 identity effects;
- verification matrix: static, unit/module, artifact, local/live;
- trust/security changes;
- rollback bundle;
- current-status/doc updates.

## Failure modes and current limitations

- Same shape with changed meaning is silent corruption.
- Reusing v1 zeros with vNext is semantic rollback.
- Reusing Run12 thresholds/calibration after retraining is invalid.
- Reusing old proxy agreement after teacher changes is invalid.
- Reusing V2 submission assumptions in V3 loses context binding.
- Adding signer/broadcast logic to audit MCP violates current least-privilege architecture.
- Calling exposed/noisy data untouched acceptance manufactures evidence.
- Updating implementation without the R4/ADR/current-status record makes later ownership unreliable.

## Common change recipe

Universal recipe:

1. bind current commit/artifacts/policy/role evidence;
2. state the semantic/trust invariant;
3. classify historical compatibility versus new version;
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

- [Cross-module contracts](11_cross_module_contracts.md)
- [Security and trust](12_security_and_trust.md)
- [Evaluation](13_evaluation.md)
- [Current status](16_current_status.md)
- [R4 decision register](../plan/ml-R4/DECISION_REGISTER.md)

## Technical mastery layer

### Prerequisite knowledge

Know dependency graphs, migrations, partial-label semantics, artifact lineage, EIP-712, staged rollout, rollback, and evidence-role separation.

### Source map and reading order

Start with current status and the governing R4/ADR record. Then inspect the producer source, cross-module consumers, focused tests, artifact manifests, and operational/security boundaries. Supplementary technical guides/labs may help with mechanics but do not override current canonical decisions.

### Execution trace and worked example

A future new class would require a class-policy/schema version, new DATA semantic/representation compatibility, retrained teacher, new proxy/circuit output layout, new V3 class-schema/protocol compatibility, updated AGENTS mapping/evaluation, and migration/rollback evidence. It is not a one-line enum edit.

### Implementation practice

Characterize with a failing semantic/compatibility test, edit the smallest authoritative producer, update all consumers explicitly, regenerate hash-bound artifacts, run role/trust validation, then document promotion/rollback.

### Review and ownership check

Can you state the current historical compatibility roots, the new semantic/protocol version, every consumer that must migrate, the evidence required for promotion, and the exact rollback bundle before editing?
