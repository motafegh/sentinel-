# P7 GitHub Identity and Release Preparation

**Date:** 2026-09-12  
**Branch:** `portfolio/professionalization-2026-09-02`  
**PR:** #72  
**Status:** PREPARED — owner/external decisions remain

## Purpose

Prepare Sentinel's public GitHub identity and first portfolio release without confusing repository capability with production readiness, individual authorship, or accepted R4 authority.

This file separates routine repository presentation work from decisions that should not be made silently on the owner's behalf.

## Current public GitHub surface

Verified on 2026-09-12:

- repository: `motafegh/sentinel-`;
- visibility: public;
- default branch: `main`;
- repository description: **unset**;
- topics: **empty**;
- homepage: **unset**;
- repository license: **none**;
- Git tags: **none**;
- GitHub releases: **none**;
- open portfolio PR: #72, still draft;
- current portfolio branch is synchronized with `main` and carries the P0–P6 work;
- three accidental no-work refs remain: `tmp-should-not-create`, `noop`, `ignore-me`.

## Recommended public identity

### Repository name

**Recommended if renaming is desired:** `sentinel-smart-contract-security`

Reasons:

- removes the accidental-looking trailing hyphen;
- keeps the established Sentinel identity;
- immediately communicates domain to recruiters and technical reviewers;
- is more distinguishable than a generic `sentinel` name;
- no repository currently exists at `motafegh/sentinel-smart-contract-security` as of this review.

**Reasonable alternative:** keep `sentinel-` to avoid any migration ceremony. The current name is functional, but it is weaker as a portfolio identifier.

This is an owner-facing identity choice, not a technical default. Do not rename automatically.

### Description

Recommended description:

> Evidence-aware smart-contract security research spanning ML/data repair, agentic analysis, ZKML, and on-chain provenance with explicit claim boundaries.

This describes the repository without claiming production readiness or repaired-model quality.

### Topics

Recommended initial topic set:

- `smart-contract-security`
- `solidity`
- `ethereum`
- `machine-learning`
- `ai-security`
- `langgraph`
- `zkml`
- `pytorch`
- `security-research`
- `reproducible-research`

Avoid generic topic stuffing. These topics map directly to visible repository responsibilities.

### Homepage

Recommendation: leave unset unless a real durable destination exists. Do not invent a placeholder website merely to fill the field.

## License decision

Current state: no repository license.

The bounded license-sensitive content inventory is recorded in [`2026-09-12_P7_LICENSE_CONTENT_INVENTORY.md`](2026-09-12_P7_LICENSE_CONTENT_INVENTORY.md).

That review changed the preliminary recommendation. **MIT is now the preferred repository-level candidate for original Sentinel code/documentation**, because:

- every current local Solidity source file in `contracts/src/` declares MIT;
- the retained/generated ZKML verifier declares MIT;
- OpenZeppelin Contracts and its upgradeable variant are MIT;
- `forge-std` is separately Apache-2.0 as a Git submodule and should retain its upstream license rather than drive Sentinel's root license;
- MIT is simple and well matched to a public portfolio/research repository.

Tracked retained model/proof artifacts and externally sourced DATA/provenance material mean a root license should be paired with an explicit third-party boundary rather than described as relicensing all upstream material.

Therefore the recommended posture, if the owner wants open reuse, is:

**MIT + `THIRD_PARTY_NOTICES.md`**, preserving per-file SPDX notices, submodule licenses, and applicable upstream terms for datasets/models/tools/artifacts.

The alternatives remain Apache-2.0 plus the same third-party boundary, or intentionally remaining unlicensed. Do not silently add a license before the owner chooses.

## Release strategy

There are currently no tags or releases. The first public release should represent a **portfolio/research snapshot**, not a claim of production software maturity.

Recommended release semantics:

- release title: `Sentinel Portfolio Research Snapshot — 2026-09`;
- preferred tag style: `portfolio-2026-09`;
- clearly state: research/portfolio snapshot, not production-ready security software;
- bind the release to the final merged professionalization commit after P8 and final green validation;
- summarize the P0–P8 public-facing improvements rather than restating the whole project history;
- link the case-study package, architecture, showcase, validation guide, and current R4 status;
- preserve the contribution-era disclosure from README.

A calendar-style portfolio tag is preferred here over pretending the monorepo has reached conventional semantic-version `v1.0.0` product maturity.

## Release blockers / prerequisites

Do not publish the first release until all of the following are satisfied:

1. current-head applicable validation is green;
2. accidental no-work refs are removed;
3. repository-name decision is explicit;
4. license decision is explicit; the bounded inventory is now complete for decision support;
5. description/topics are set or intentionally deferred;
6. the historical provider-RPC credential-shaped finding has an externally confirmed revocation/rotation disposition if it represented a live credential;
7. P8 recruiter/engineer/adversarial credibility audit is complete;
8. PR #72 is merged at the final accepted boundary.

## Contribution and claim boundary for release text

Any P7 release text must preserve these facts:

- original Sentinel: Ali's long-running AI-assisted learning/building work;
- later R4 continuation: substantially AI-led research under Ali's direction;
- repository capability does not imply independent authorship/mastery of every subsystem;
- G8 remains open and no repaired R4 teacher has been trained/promoted;
- Run12 remains the historical operational ML baseline;
- confirmed negatives remain zero;
- current threshold/calibration/untouched-acceptance support is unavailable;
- the retained ZK proof is proxy-only;
- the live audit MCP is read-only and there is no claimed production signer/broadcaster.

## Actions completed without owner identity decisions

- P7 public-surface inventory;
- recommended description/topics prepared;
- no-tag/no-release state confirmed;
- candidate repository rename checked in the owner's namespace;
- bounded license-sensitive content inventory completed;
- release semantics and blockers defined.

## Actions still possible before owner decisions

- prepare the release-note template;
- prepare P8 audit inputs;
- obtain final validation evidence when execution becomes available;
- clean accidental refs when a delete-ref capability or local Git access is available.

## Owner/external decisions that remain

1. **Repository name:** keep `sentinel-` or rename (recommended candidate: `sentinel-smart-contract-security`).
2. **License posture:** MIT + third-party notices (recommended), Apache-2.0 + notices, or intentionally remain unlicensed.
3. **Historical provider credential:** confirm whether the identified endpoint/credential has already been revoked or rotated externally.

These decisions should be resolved before the first release rather than guessed by automation.
