# Phase 8 Local DATA Gate Re-audit and Corrections

**Date:** 2026-08-15
**Repository state inspected:** `main` at `433c5cd021b608d37929578102e0a4d2fa445fdb` before the corrections in this report
**Decision:** keep the 100-epoch run on hold; correct the local rebuild gates, then perform a fresh full repaired-v2 rebuild.

## Outcome first

The protected local Solidity bytes are usable: all 22,823 ingestion-manifest
records match their declared size and SHA-256. The repository-only repair was
not yet sufficient to start the physical rebuild, however. A local re-audit
found six fail-open or incomplete bindings that could have allowed a partial or
semantically different candidate to look acceptable.

The corrected code now:

1. accepts the intentional DIVE/SmartBugs `repo` symlinks while rejecting
   absolute paths, `..`, and resolved targets outside each source's explicit
   raw/`repo` roots;
2. verifies every raw byte before creating preprocessing output;
3. records and enforces full-manifest reconciliation, so `--limit` output is
   explicitly incomplete and cannot feed claims, grouping, or representations;
4. represents file-level labels with all unrelated inheritance leaves instead
   of guessing one contract;
5. deserializes and validates graph/token payloads rather than hashing opaque
   files only;
6. requires the publication to consume and hash-bind the materialized evidence
   ledger, asserts enabled-class strong coverage in all three strong roles, and
   binds acceptance/GPU smoke to the exact publication and representation
   digest.

No repaired production root and no training run was created during this
correction tranche. Historical v1/G7 artifacts remain untouched.

## Real raw-byte evidence

The corrected verifier checked the complete manifests:

| Source | Records | Bytes | Manifest SHA-256 | Result |
|---|---:|---:|---|---|
| DIVE | 22,330 | 499,607,903 | `d33dab2df2f149b19d2b978596d765d949facc554320341ab4d2ee5398f00a84` | PASS |
| SmartBugs Curated | 143 | 504,194 | `e7eae8d3d336b16012e1b0d16e2637eb4ee23bf0cca0314b756bf612a7544336` | PASS |
| SolidiFI | 350 | 4,924,791 | `54399c115f77b53f461d7accbe352a91c8c5c464d1faa9eafb633b0ed76e4bb3` | PASS |

This establishes agreement with the local ingestion manifests. It does not
prove upstream reacquisition or that every source label is correct.

## Why the original graph rule was not scalable

The first repaired selector required one unambiguous application contract per
file. Applied to the raw corpus, even a unique-inheritance-leaf improvement
still left 4,241 files ambiguous and 19 DIVE files with no application-contract
declaration:

| Source | Unique target | Ambiguous | No application contract |
|---|---:|---:|---:|
| DIVE | 18,187 | 4,124 | 19 |
| SmartBugs Curated | 110 | 33 | 0 |
| SolidiFI | 266 | 84 | 0 |

Those are file-level labelled samples; selecting one unrelated leaf would move
the file label onto a guessed contract. The corrected policy therefore builds
one disconnected PyG graph per file from every application inheritance leaf.
Inheritance parents are included by Slither through their leaf. Library-only
files retain their executable libraries; interfaces alone are not treated as
implementations.

Full lexical census under this rule:

| Source | Files resolved | Graph components | Multi-component files | Maximum components |
|---|---:|---:|---:|---:|
| DIVE | 22,330 | 28,931 | 4,139 | 28 |
| SmartBugs Curated | 143 | 184 | 33 | 7 |
| SolidiFI | 350 | 441 | 84 | 3 |
| **Total** | **22,823** | **29,556** | **4,256** | **28** |

A real two-file DIVE smoke compiled and tokenized both a single-component file
and a five-component file. Both graph/token/sidecar triples passed the new
physical validators. The five-component graph contained 1,595 nodes and 2,485
edges. This validates mechanics, not full-corpus cost or model quality. The full
binding report now records component/node/edge quantiles so graph inflation and
GPU feasibility are reviewed before training.

Repository validation after the latest corrections: repaired focused suite `98
passed`; corrected verifier PASS over all three full manifests; frozen G6
validator PASS; handbook validator `11 passed`; `git diff --check` PASS.

### First full-preprocessing attempt found a compiler-selection defect

The first full DIVE attempt reconciled all 22,330 records but reported 81
compile drops: 57 sources without a pragma and 24 sources failing every selected
version. Drop inspection showed the compiler helper treated a missing pragma as
an immediate failure and misread upper-bound-only or adjacent constraints. For
example, `<0.6.0` was incorrectly interpreted as both a floor and ceiling, so it
attempted no compiler.

The selector now evaluates exact, comparator, adjacent comparator, caret, tilde,
and `||` clauses and deterministically tries installed versions for no-pragma
sources. Real rechecks recovered all three inspected cases: a no-pragma source
with solc 0.4.26, `<0.6.0` with solc 0.5.17, and the flattened adjacent clause
`<0.8.0=0.6.12>=0.6.0>=0.6.2` with solc 0.6.12. The first DIVE output is a
failed attempt, not candidate evidence, and must be archived before rebuilding
the source from a fresh directory.

## What is and is not known about expected model benefit

A repaired run is worth testing because it removes known training-input defects:
address-only deletion, old-solc flag rejection, post-compile mutation,
source-category conflation, leakage-group fragmentation, and wrong single graph
targets. Those changes improve evidence fidelity and should reduce avoidable
label/representation noise.

They do **not** prove the next model will be better. The actual recovered
population, strong/weak class support after grouping, representation success,
long-contract coverage, graph-size distribution, and bounded GPU behavior are
unknown until the local rebuild finishes. Policy v1 still has no confirmed
negative population, two disabled classes, positive-only model selection, and
no threshold/calibration/untouched-acceptance set. Therefore even a successful
100-epoch optimization run could not by itself establish production security
accuracy or an operating threshold.

## Correct next boundary

After these corrections are committed and the tracked worktree is clean:

1. rerun raw verification and prerequisites;
2. preprocess all three sources without `--limit`;
3. inspect drop/reconciliation manifests;
4. build claims and grouping (both now reject partial preprocessing);
5. build all representations and inspect failures plus graph/token distributions;
6. materialize the ledger, publish from that exact ledger, bind physical payloads,
   and persist the exact-hash acceptance report;
7. stop for evidence review;
8. only then run the bounded window experiment and bounded GPU smoke.

Full training remains unauthorized until a separate decision records why the
accepted physical evidence is sufficient.
