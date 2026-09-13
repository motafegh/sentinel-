# P7 GitHub Identity / Release Decision Resolution

**Date:** 2026-09-12  
**Branch:** `portfolio/professionalization-2026-09-02`  
**PR:** #72  
**Status:** DECISIONS RESOLVED / EXTERNAL OPERATIONS REMAIN

## Decisions

### Repository name — retain `sentinel-` for this professionalization cycle

The earlier preparation record considered `sentinel-smart-contract-security`. The final decision is to **retain the existing repository name** for this cycle.

Reasoning:

- renaming is presentation polish rather than a technical or credibility requirement;
- the existing repository URL is already the durable project/history anchor;
- a rename would create migration/link/update work with little engineering value;
- the improved README makes the domain immediately clear even though the slug is imperfect;
- a future rename remains possible if there is a stronger branding reason.

The trailing hyphen is accepted as minor naming debt, not a release blocker.

### License — MIT with explicit scope/notices

The repository now contains:

- root `LICENSE` — MIT;
- root `THIRD_PARTY_NOTICES.md` — explicit scope, attribution, generated-material, dependency, data/model/proof-artifact, and third-party-rights boundaries;
- README license wording aligned to those files.

The root MIT license is intended for Sentinel's original project code and documentation. It does not assert rights the repository owner does not possess over third-party source, data, models, proof artifacts, generated material, trademarks, or external content.

Relevant observed license context:

- Sentinel Solidity source files use MIT SPDX identifiers;
- the generated `ZKMLVerifier.sol` carries MIT SPDX;
- OpenZeppelin dependencies are MIT upstream;
- `forge-std` is Apache-2.0 upstream and remains a Git submodule rather than copied Sentinel-owned source.

### GitHub description/topics/homepage

Recommended description/topics remain recorded in `2026-09-12_P7_GITHUB_IDENTITY_AND_RELEASE_PREPARATION.md`.

The current ChatGPT GitHub connection does not expose repository-settings mutation for description/topics/homepage, so these values are **operationally deferred**, not analytically unresolved.

Homepage remains intentionally unset unless a real durable destination exists.

### First release semantics

If/when the release gate becomes satisfied:

- release title: `Sentinel Portfolio Research Snapshot — 2026-09`;
- tag: `portfolio-2026-09`;
- release semantics: research/portfolio snapshot, **not** product-maturity `v1.0.0`;
- use the prepared `2026-09-12_P7_RELEASE_NOTE_TEMPLATE.md`;
- bind the release only to the final merged/validated accepted commit.

## External / capability-bound operations still open

### Historical provider credential

Repository evidence proves only that the historical provider-RPC credential-shaped value is known/baselined and absent from current tracked content. It **cannot prove external revocation/rotation**.

No attempt is made to reuse, probe, or validate the exposed historical credential. External revocation/rotation confirmation remains a hard release prerequisite.

### Accidental no-work refs

The refs:

- `tmp-should-not-create`;
- `noop`;
- `ignore-me`

contain no intended portfolio work. The available GitHub connection exposes ref creation/update but not deletion. They therefore remain explicit branch-hygiene debt to delete with local Git or another GitHub administration surface.

They must not be mistaken for active development branches.

## P7 disposition

**Repository-content decisions are resolved.**

P7 is `COMPLETE_AT_REPOSITORY_CONTENT_SCOPE / EXTERNAL_OPERATIONS_OPEN`.

Remaining external operations do not justify inventing repository state:

1. remove the three accidental refs;
2. optionally set description/topics using GitHub settings;
3. confirm/revoke/rotate the historical provider credential before release;
4. publish the prepared snapshot only after the final release gate is satisfied.
