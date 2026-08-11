#!/usr/bin/env python3
"""Idempotently register R4 Phase-4 artifacts and append the G4 execution log."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("docs/plan/ml-R4")
ARTIFACT_INDEX = ROOT / "ARTIFACT_INDEX.md"
EXECUTION_LOG = ROOT / "EXECUTION_LOG.md"
AUTHORIZATION_REL = "authorizations/2026-08-11_R4-GAP-002_authorization.md"
LEGACY_AUTHORIZATION_REL = "findings/05_gap002_authorization.md"

ARTIFACT_MARKER = "## Phase 4 artifacts — targeted gap adjudication (G4 PASS)"
LOG_MARKER = "### R4-LOG-20260811-007 — Phase 4 Targeted DIVE Gap Adjudication and G4 Closure"

ARTIFACT_BLOCK = r'''## Phase 4 artifacts — targeted gap adjudication (G4 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P4-PLN-001 | 4 | execution_plan | runs/2026-08-11_PHASE4_gap_authorization_and_adjudication_plan.md | — | d8b138b1 | New | AVAILABLE | NO | Scope-minimal Phase-4 authorization/adjudication plan |
| R4-P4-AUT-001 | 4 | authorization | authorizations/2026-08-11_R4-GAP-002_authorization.md | — | 0613aeee | New | AVAILABLE | NO | Delegated approval of R4-GAP-002; five mapped DIVE strata only |
| R4-P4-MAN-001 | 4 | population_manifest | manifests/p4_gap002_population_manifest.json | — | 4e5ff9be | New | AVAILABLE_VERIFIED | NO | Phase-3-ledger-bound DIVE positive population counts and group-aware eligibility |
| R4-P4-MAN-002 | 4 | frozen_sample | manifests/p4_gap002_initial_sample.jsonl | 2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9 | 757c368d | New | AVAILABLE_VERIFIED | NO | 100 TRAIN-only contracts; 20 per stratum; no review-group reuse; groups touching val/test excluded |
| R4-P4-BND-001 | 4 | blind_source_bundle | review_bundles/r4_gap002_blind_review_bundle_v1.zip | 2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38 | 02f254249 | New | AVAILABLE_VERIFIED | NO | Checksum-bound normalized/flattened Solidity for the exact frozen 100-contract sample |
| R4-P4-REV-001 | 4 | semantic_review | reviews/R4-GAP-002/p4_gap002_blind_semantic_review_v1.jsonl | 7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473 | c8f283f5 | New | AVAILABLE_VERIFIED | NO | 100 source-only blind AI primary semantic review rows; no confirmed negatives created |
| R4-P4-FND-001 | 4 | review_report | findings/06_gap002_blind_semantic_review_report.json | — | c8f283f5 | New | AVAILABLE_VERIFIED | NO | Exact per-stratum review counts, descriptive Wilson intervals, and bounded role recommendations |
| R4-P4-FND-002 | 4 | adjudication | findings/06_gap002_blind_semantic_review.md | — | 3f3b6123 | New | AVAILABLE | NO | Source-role interpretation: four DIVE strata masked/excluded; TOD limited to TRAIN_WEAK |

'''

LOG_BLOCK = r'''

---

### R4-LOG-20260811-007 — Phase 4 Targeted DIVE Gap Adjudication and G4 Closure

- **Phase:** 4
- **Gap ID, if review work:** `R4-GAP-002`
- **Operator:** ChatGPT / GPT-5.6 Sol primary semantic reviewer + user local protected-source bundle materialization; routine technical/governance approval delegated by the human owner
- **Date/timezone:** 2026-08-11 Europe/Berlin (execution crossed into 2026-08-12 in the user's local timezone)
- **Repository branch/commit:** `r4/phase4-targeted-gap-adjudication`; frozen source bundle commit `02f254249f16b2f940dca0c9a9309e6b38bade12`; machine-readable review publication commit `c8f283f5961f2955c7738409bf8298dc41c599bd`
- **Worktree status before:** Phase-3 G3-passed canonical ledger on `main`; protected local DIVE preprocessed source available without modifying protected historical artifacts
- **Input artifact IDs/hashes:** R4-P3-LED-001 `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`; DIVE crosswalk/evidence R4-P0-XWK-001 and R4-P0-EVD-001..003; frozen sample SHA-256 `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`; blind source bundle SHA-256 `2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38`
- **Command(s):** `p4_freeze_gap002_sample.py`; `p4_build_gap002_review_bundle.py`; checksum verification and safe-unpack CI; source-only blind semantic review; `p4_publish_gap002_blind_review.py`; review/sample binding CI
- **Environment and seed(s):** deterministic SHA-ranked group-aware sample from committed Phase-3 ledger; TRAIN-only; groups touching val/test excluded; no stochastic review seed; blind source-only semantic pass with model/tool/merger/non-target-label evidence hidden
- **Expected outputs:** authorization record; frozen population/sample; checksum-bound blind source bundle; 100 semantic verdicts; role recommendation and uncertainty report; explicit gap/G4 disposition
- **Actual outputs/hashes:** 100 unique contracts/review groups, 20 per stratum; review rows SHA-256 `7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473`. Blind results: DenialOfService 0 support / 20 not-support; IntegerUO 3 / 16 / 1 unclear; Timestamp 4 / 15 / 1 unclear; TransactionOrderDependence 12 / 5 / 3 class-boundary conflicts; UnusedReturn 9 / 11. CI regenerated and bound all review identities successfully.
- **Result:** PASS
- **Historical evidence reused:** Phase-1 DIVE EB/RE manual reviews and correlated Slither/Aderyn findings; Phase-2 source/crosswalk semantics; Phase-3 ledger. Historical/tool evidence was revealed only after the blind semantic verdicts were locked.
- **New evidence created:** single-AI primary source-only semantic review of 100 checksum-bound DIVE contracts; explicit source/stratum reliability evidence. This is not human/inter-rater or untouched-acceptance evidence.
- **Protected artifacts changed:** NO
- **Register updates:** R4-GAP-002 -> RESOLVED; Web3Bugs and provisional inactive BCCC first-baseline populations -> MASK_OR_EXCLUDE/deferred; Phase 4 -> PASSED; Phase 5 -> READY; Phase-4 artifacts registered in ARTIFACT_INDEX
- **Gate effect:** **G4 PASS.** DIVE DoS/Arithmetic/Time manipulation/Unchecked Return Values source assertions are masked/excluded for the first baseline; DIVE Front Running/TOD is limited to `TRAIN_WEAK` and barred from outcome metrics, model selection, threshold/calibration fitting, and untouched acceptance. `DOES_NOT_SUPPORT_POSITIVE` does not create a confirmed negative.
- **Next permitted action:** Begin Phase 5 — DATA vNext Policy and Design. Encode the Phase-0–4 source/class/state/role decisions in versioned ADRs/specification before any implementation makes semantic choices.
'''


def main() -> int:
    original_artifact_text = ARTIFACT_INDEX.read_text(encoding="utf-8")
    artifact_text = original_artifact_text.replace(LEGACY_AUTHORIZATION_REL, AUTHORIZATION_REL)
    if ARTIFACT_MARKER not in artifact_text:
        anchor = "## Availability\n"
        if anchor not in artifact_text:
            raise RuntimeError("ARTIFACT_INDEX availability anchor missing")
        artifact_text = artifact_text.replace(anchor, ARTIFACT_BLOCK + anchor, 1)
    if artifact_text != original_artifact_text:
        ARTIFACT_INDEX.write_text(artifact_text, encoding="utf-8")

    log_text = EXECUTION_LOG.read_text(encoding="utf-8")
    if LOG_MARKER not in log_text:
        if not log_text.endswith("\n"):
            log_text += "\n"
        EXECUTION_LOG.write_text(log_text.rstrip() + LOG_BLOCK + "\n", encoding="utf-8")

    final_index = ARTIFACT_INDEX.read_text(encoding="utf-8")
    final_log = EXECUTION_LOG.read_text(encoding="utf-8")
    if final_index.count(ARTIFACT_MARKER) != 1:
        raise RuntimeError("Phase-4 artifact block must appear exactly once")
    if final_log.count(LOG_MARKER) != 1:
        raise RuntimeError("Phase-4 execution-log entry must appear exactly once")
    if LEGACY_AUTHORIZATION_REL in final_index:
        raise RuntimeError("stale Phase-4 authorization path remains in artifact index")
    if AUTHORIZATION_REL not in final_index:
        raise RuntimeError("canonical Phase-4 authorization path missing from artifact index")
    if not (ROOT / AUTHORIZATION_REL).is_file():
        raise RuntimeError("indexed Phase-4 authorization artifact does not exist")
    print("PASS: Phase-4 G4 closeout records and indexed authorization path are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
