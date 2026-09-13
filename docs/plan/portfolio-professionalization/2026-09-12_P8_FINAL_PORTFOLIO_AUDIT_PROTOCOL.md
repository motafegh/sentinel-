# P8 Final Portfolio / Interviewer Audit Protocol

**Date:** 2026-09-12  
**Branch:** `portfolio/professionalization-2026-09-02`  
**PR:** #72  
**Status:** READY FOR EXECUTION AFTER P7 OWNER/EXTERNAL DECISIONS

## Objective

Perform the final public-portfolio review from three external perspectives before PR #72 is merged and the first portfolio snapshot is released:

1. recruiter / hiring-manager skim;
2. senior engineer / technical reviewer audit;
3. adversarial credibility / overclaim audit.

The purpose is not to make Sentinel look maximally impressive. The purpose is to make the repository easy to evaluate, technically defensible, and difficult to misinterpret.

## Entry conditions

P8 final execution should begin only after:

- P6 case-study package content is complete;
- P7 repository-name decision is resolved;
- P7 license posture is resolved;
- description/topics are finalized or explicitly deferred;
- historical provider-credential revocation/rotation disposition is known if applicable;
- temporary no-work refs are removed or explicitly accepted as a pre-merge blocker;
- current-head validation can be run.

A preliminary audit can be performed earlier, but it is not the final P8 gate.

## Pass A — Recruiter / hiring-manager skim

Assume the reviewer gives the repository 2–5 minutes.

Check:

- repository name and description communicate the domain immediately;
- README first screen explains what Sentinel is and is not;
- contribution-era disclosure is visible enough to prevent authorship ambiguity without overwhelming the project story;
- architecture is understandable without reading R4 internals;
- strongest engineering decisions are discoverable quickly;
- `SHOWCASE.md` offers a credible low-friction way to inspect the repository;
- no stale status or contradictory “production-ready” language appears;
- technology lists support the story instead of becoming keyword stuffing;
- current limitations strengthen credibility rather than reading as hidden footnotes.

**Recruiter pass condition:** a technically literate non-specialist can state the project purpose, major engineering themes, current maturity, and Ali's contribution model without needing historical project context.

## Pass B — Senior engineer / technical reviewer

Assume the reviewer follows claims into evidence.

Check:

- README current-state claims agree with `docs/handbook/16_current_status.md` and R4 authority;
- architecture views agree with executable source boundaries;
- DATA/ML claims preserve unknown ≠ negative and current evaluation limitations;
- Run12 historical runtime is not confused with a repaired R4 teacher;
- D-011 physical acceptance is not presented as model-quality evidence;
- D-012 selector-policy promotion is not presented as physical successor acceptance;
- case-study numerical claims can be traced to primary evidence;
- AGENTS tool-status semantics match state/orchestration source;
- ZK proof, V3 context attestation, and transaction authority remain separate;
- development/validation instructions distinguish dependency-light, module-specific, heavy/local, and production-like responsibilities;
- CI descriptions state what checks do not prove;
- submodule/data/model/artifact licensing boundaries are not obscured by a root license choice.

**Engineer pass condition:** following any high-value public claim leads to a coherent primary authority rather than a stale or marketing-only source.

## Pass C — Adversarial credibility audit

Act as a skeptical interviewer trying to find exaggerated claims.

Attempt to falsify statements such as:

- “I built the entire current system myself”;
- “the repaired model is trained and better”;
- “the project has confirmed negative evaluation data”;
- “V10 physical acceptance proves vulnerability-classification quality”;
- “the guarded selector is already the accepted physical runtime lineage”;
- “the ZK proof proves the Solidity audit / teacher / agent verdict”;
- “the analysis service signs and submits production transactions”;
- “all external tools were run in the showcase”;
- “a fresh clone contains all historical data/models/proving artifacts”;
- “green unit tests establish security/model quality.”

Search README, module READMEs, handbook current-facing chapters, showcase/validation docs, case studies, PR/release text, and repository metadata for wording that could accidentally imply any of these.

**Adversarial pass condition:** no prominent public surface materially overstates authorship, evidence strength, production readiness, model quality, or cryptographic assurance.

## Pass D — Repository hygiene and release integrity

Check:

- only intentional long-lived branches remain;
- PR #72 is mergeable against current `main`;
- description/topics/name/license match P7 decisions;
- no placeholder release text remains;
- no release is created from an unmerged or unvalidated intermediate SHA;
- final release/tag points to the exact intended merged commit;
- secret-scanning boundary remains green and the historical provider finding has its required external disposition;
- README clone/setup links remain valid after any repository rename;
- no obsolete portfolio plan claims itself as newer authority than `CURRENT_STATUS.md`.

## Pass E — Final automated validation

Run all applicable current lightweight/current-PR checks against the exact final candidate SHA. At minimum, expect the same current surfaces established by P5:

- Handbook;
- Portfolio showcase;
- DATA reproducibility;
- SENTINEL system alignment;
- Security hygiene.

Where a check cannot run, record `NOT_RUN` with the reason. Never replace missing validation with an assumed pass.

## Final audit output

Create a dated P8 closeout recording:

- exact audited commit SHA;
- repository public identity values;
- recruiter findings and disposition;
- senior-engineer findings and disposition;
- adversarial findings and disposition;
- hygiene/release findings and disposition;
- exact automated check results;
- unresolved external limitations, if any;
- final recommendation: `READY_TO_MERGE_AND_RELEASE` or `HOLD`.

## CV/interview wording boundary

Only after P8 passes should concise CV/interviewer language be derived from the repository.

CV wording must distinguish:

- original hands-on/AI-assisted learning and building experience;
- later AI-led R4 research performed under Ali's direction;
- architecture/evidence decisions Ali can explain and defend;
- repository capabilities versus personal implementation ownership;
- established results versus research directions that remain blocked.

Do not reverse the process by writing ambitious CV claims first and then trying to make the repository justify them.
