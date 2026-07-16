# SENTINEL ML R4 — Trustworthy-Label Recovery, DATA vNext, and Existing-Model Retraining Master Plan

**Status:** APPROVED FOR PHASED EXECUTION  
**Primary defect:** materially untrustworthy contract-vulnerability labels  
**Model architecture:** frozen through the first DATA vNext baseline  
**Program location:** `docs/plan/ml-R4/`

---

## 1. Executive decision

R4 is not an open-ended model-improvement project and not a repeat investigation into whether the labels are unreliable.

The project starts from substantial prior evidence that the historical vulnerability labels are semantically misaligned, misleading, or otherwise unsuitable as trustworthy ground truth. The principal technical task is therefore to repair the data and evidence foundation.

The R4 program will:

1. freeze the current artifact state;
2. recover previous evidence without repeating it;
3. reconstruct how source-native claims became historical binary targets;
4. represent uncertainty and class-specific evidence explicitly;
5. fill only critical evidence gaps;
6. build a versioned DATA vNext;
7. retrain the existing architecture;
8. evaluate it using independent evidence-qualified populations;
9. promote or restrict the resulting model with tested rollback.

---

## 2. Why data repair comes before retraining

A supervised model learns the target it is given. If labels encode source coverage, tool votes, folder presence, dropped categories, or merger defaults rather than real vulnerability outcomes, a better optimizer or architecture can become better at reproducing the wrong target.

R4 therefore treats the chain below as one validity system:

```text
source-native assertion
→ source acquisition and pin
→ parser
→ crosswalk
→ missing/absence default
→ merger
→ all-zero interpretation
→ split and dedup behavior
→ export target and mask
→ training loss
→ model selection
→ threshold fitting
→ calibration
→ acceptance evaluation
→ inference policy language
```

The repaired model is trustworthy only if this chain is explicit and evidence-backed.

---

## 3. Established prior evidence

Prior investigations are inputs to R4, not invitations to start over.

A retained DIVE investigation, for example, concluded that DIVE ExternalBug and Reentrancy labels should be dropped from supervised training, found that the parser/folder mechanics were largely faithful, and attributed the principal defect to automated source labels rather than parser implementation. It also explicitly left several DIVE classes and other sources for later work.

R4 must recover the underlying artifacts, reconcile internal count or wording differences, and import the usable evidence. It must not repeat the same broad DIVE reviews unless a registered gap makes additional review necessary.

The same reuse-first rule applies to BCCC and every other prior source or contract analysis.

---

## 4. Program goal

Create a trustworthy, versioned DATA vNext and use it to retrain and evaluate the existing SENTINEL ML architecture.

### 4.1 DATA vNext must provide

- source-native claims;
- mapping provenance;
- historical target preservation;
- contract-class outcome states;
- class masks;
- evidence identifiers;
- conflicts and uncertainty;
- duplicate/project-family groups;
- dataset-role eligibility;
- immutable partition manifests;
- schema and artifact hashes.

### 4.2 The retrained ML bundle must provide

- checkpoint bound to DATA vNext and code commit;
- unchanged architecture unless an approved compatibility modification is required;
- mask-aware training;
- raw probability outputs;
- independently assessed calibration;
- independently selected thresholds;
- abstention or limitation for unsupported classes;
- documented acceptance evidence;
- rollback to the pre-R4 bundle.

---

## 5. Non-goals

R4 does not normally include:

- a new GNN or Transformer design;
- broad architecture search;
- re-review of every contract;
- a fixed-size universal gold set;
- deletion or overwriting of historical exports;
- treating tool consensus as ground truth;
- treating all-zero as confirmed safe;
- proving that every source is unusable;
- improving historical metrics against corrupted targets;
- silently changing API verdict meanings.

---

## 6. Governing principles

### P1 — Reuse previous evidence first

Search, preserve, hash, register, and import previous evidence before authorizing new review.

### P2 — Review only registered gaps

Every new contract review requires an approved gap ID and a specific decision it will resolve.

### P3 — Preserve history

Historical targets, exports, reports, and checkpoints remain immutable. Repairs create new versions.

### P4 — Use class-specific states

Truth and uncertainty are contract-class properties. One contract-level tier is insufficient.

### P5 — Zero is not negative by default

A zero may mean explicit negative, absence, unsupported class, dropped source category, unreviewed, conflict, or transformation default.

### P6 — Evidence is not automatically independent

Multiple tools may share detectors, ASTs, heuristics, or source assertions. Independence groups must be recorded.

### P7 — Dataset use is a separate decision

Evidence strength does not automatically authorize training, calibration, or acceptance use.

### P8 — Freeze the architecture

The first repaired-data baseline uses the existing architecture.

### P9 — Separate data roles

Training, model selection, threshold fitting, calibration fitting, internal audit, and untouched acceptance must be isolated by leakage group.

### P10 — Raw probability is distinct from policy

Keep logits/probabilities, calibration, thresholds, tiers, and human-facing verdict strings separate.

### P11 — Unsupported is a valid conclusion

A class may be training-only, provisional, disabled, or unsupported for outcome claims.

### P12 — Every promoted claim needs artifact identity

No conclusion is authoritative without population definition, code commit, artifact hash, and method.

---

## 7. Program phases and gates

### Phase 0 — Baseline and evidence location

Freeze the current local repository and DATA/ML bundle. Locate prior evidence. No contract review.

**Gate G0:** exact baseline, protected artifacts, availability inventory, and evidence locations are explicit.

### Phase 1 — Previous evidence recovery

Recover DIVE, BCCC, other source investigations, manual review artifacts, and model-run lineage. Deduplicate prior work.

**Gate G1:** each major previous claim is linked to retained evidence, conclusion-only status, or unavailable status.

### Phase 2 — Label-corruption reconstruction

Trace source claims through parser, crosswalk, defaults, merger, split, export, and ML loading. Quantify corruption mechanisms.

**Gate G2:** every historical target category has a named origin and transformation path.

### Phase 3 — Evidence ledger

Build a sidecar contract-class ledger preserving historical labels and structured evidence states.

**Gate G3:** schema supports uncertainty, conflict, provenance, historical/new separation, and role eligibility.

### Phase 4 — Targeted evidence-gap adjudication

Use prior evidence first. Review only approved gaps needed for DATA vNext decisions.

**Gate G4:** critical source/class/role decisions are supported; unresolved areas are masked or excluded.

### Phase 5 — DATA vNext policy and design

Approve source/class retention, masks, weak/strong roles, crosswalk corrections, merger behavior, and schema migration.

**Gate G5:** DATA vNext specification and ADRs are complete.

### Phase 6 — Leakage-safe partitions

Freeze train, model-selection, threshold, calibration, internal-audit, and untouched-acceptance roles by leakage group.

**Gate G6:** no incompatible role leakage and support is adequate or explicitly limited.

### Phase 7 — DATA vNext implementation

Implement versioned registry, labels/ledger export, masks, partitions, manifests, and compatibility loaders.

**Gate G7:** DATA vNext is reproducible, validated, and historical artifacts remain intact.

### Phase 8 — Existing-model retraining

Add only required mask/metadata compatibility changes and retrain the existing architecture.

**Gate G8:** reproducible checkpoint is bound to DATA vNext and no acceptance data was used.

### Phase 9 — Evidence-qualified evaluation and policy

Evaluate discrimination, calibration, thresholds, uncertainty, workflow utility, and verdict language on independent roles.

**Gate G9:** every class receives a supported status and policy.

### Phase 10 — Acceptance, promotion, and rollback

Use untouched acceptance data, compatibility tests, and rollback rehearsal.

**Gate G10:** promote, restrict, partially promote, or reject the repaired bundle.

---

## 8. Stop conditions

Stop and report rather than improvise when:

- active artifacts cannot be bound;
- historical evidence identity is unclear;
- a proposed review lacks a gap ID;
- class definitions are not frozen;
- acceptance groups have been exposed;
- DATA vNext schema changes lack an ADR;
- retraining would use unresolved zeros as negatives;
- architecture changes are proposed before the repaired-data baseline;
- a protected historical artifact changed;
- counts cannot be reconciled.

---

## 9. Human and AI responsibilities

### AI agent

- inventory and hash artifacts;
- recover evidence;
- build traceable manifests;
- implement deterministic scripts;
- propose gaps and ADRs;
- enforce masks and partition rules;
- run reproducible training/evaluation;
- report contradictions and limitations.

### Human owner

- approve class definitions and policy costs;
- approve evidence-gap reviews;
- resolve high-impact adjudication disagreements;
- approve DATA vNext ADRs;
- approve promotion, restrictions, and public claim language.

The AI must not silently turn an uncertain technical judgment into a policy decision.

---

## 10. Commit discipline

- Work on isolated branches/worktrees.
- Prefer one commit per work package or gate.
- Do not mix historical artifact edits with R4 outputs.
- Generated outputs include manifests and hashes.
- Scripts must be rerunnable and deterministic where possible.
- Record commands, environment, and seeds.
- Do not commit large raw artifacts unless repository policy explicitly permits it; commit manifests and documented storage locations instead.

---

## 11. Definition of done

R4 is complete when:

1. prior work has been recovered and not needlessly repeated;
2. label corruption mechanisms are explicit and quantified;
3. historical labels are preserved;
4. DATA vNext represents uncertainty and class-specific provenance;
5. only justified gap reviews were added;
6. partitions are leakage-safe;
7. the existing model architecture has been retrained reproducibly;
8. calibration and thresholds are independently assessed;
9. every class has a bounded claim status;
10. deployment migration and rollback are tested.

Historical metric improvement is not a definition-of-done requirement.
