# Phase-8 V10 structural-drift probe handoff

Date: 2026-08-23
Status: LOCAL EVIDENCE REQUIRED; PHYSICAL ACCEPTANCE REMAINS BLOCKED
Scope: R4-B008 structural blocker only; no label, selector, objective, threshold,
checkpoint, training, or model-quality authority
Starting repository commit: `be8a656cfdb08ba025c45de6b441680ee318c9b8`

## Authority and inherited boundary

This tranche continues the v2.4 checkpoint in
`2026-08-23_PHASE8_v10_parse_only_resolution_working_plan.md` and the machine
record `reviews/R4-GAP-008/v10_transition_audit_v2.json`.

Do not restart the completed 26-contract parse-only repair. The protected v2.4
candidate already has 22,540 identities, exact accepted-V9 token bytes, zero
parse-only outputs, zero unclassified call IR, and the required runtime split.
The remaining physical blocker is the 20 previously full-analysis identities
whose unchanged structure differs from the frozen passing Slither-0.10 V10
reference. Training remains unauthorized.

## Unexpected identities under investigation

All 20 are DIVE identities and had `v9_parse_only=false` in transition audit v2:

1. `047f9d7c2db3d6ba43b62e9c1b35adb1ed5a6bd36d68da46e0f877b0974b73e4`
2. `1d9ce79b93c3a1bd7597a76204ae65027fc0471517b2a247c1e536a260c296fd`
3. `42489184f712d85f392a47db45110b4406436bc8b648524300a18319111ab350`
4. `48beaa23f916dfd3acbc86a799b0859709b29defa3796450693bab13f8e6e777`
5. `5a626b8baef72b243f1812118862af26ea796462c38e900f2f595ba73b55495e`
6. `6376d572b974fb2ba2c074bf7d43972b241a1000731563b44ee09ef72eeaca3e`
7. `73cbc254caad8a7a6b8674125c029a530973459c294c6897dca01d219307c669`
8. `7e9bfccd7d3ed5076b7ea61fe444f33b50deb78c27582bcc413b7303422dc551`
9. `83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d`
10. `85c5c0d173dbaed126f4bc5165c7453262b6ff50c89c52a458034f322a06a714`
11. `8b1792cb3c0a40a4ebeec72ffe69d00920c80203213f24ec3e2d5a867eeae3d5`
12. `95f7d52dff443cc825e20477a62de371cc4bbc31b6ba5aae653ff51caaaf974c`
13. `a4068383ed30b56a39771e1dcbe835726242c164d125908207b4e616030aaa8c`
14. `a7faec46ab38dbf5b87b1e1ef0e56fc5da743ac450535b1cc09f12922c86f46c`
15. `af947a2b1a6d7c6fa500f5604bc7b3d3e8bbab6711c30b54f601b6db5db19464`
16. `beaa4d742f0b52b301fc2f143072b57ef8170540cdaaa096cdd2f51b047ab1ca`
17. `c159c57b830cb77686cb5a2a7b40f1452cae516ec688b55331adb61b1669064d`
18. `c1d21cda50fb1f0c1194392080a2c7a21b3baed5edbe140caaec8c3f257f756b`
19. `dcf66533d7ee72d2a59ab07fd4117f03ea035489f53ac9184e1dcb4d82c6d823`
20. `f8cd4abd34d1aa1ca5d3293ee286bfedfa39d0cab550a547ba22e798d5439b35`

The raw v2 audit flags show eight identities with exact feature/metadata rows but
raw endpoint drift, while twelve show feature/metadata drift with raw unchanged
edge topology. The preceding bounded trace recorded a deeper 10/10 split after
inspecting indistinguishable-node permutations. Neither observation is itself
acceptance evidence. Repeat generation plus an identity-aware exact comparison
is required.

## Repository-side diagnostic added

`docs/plan/ml-R4/scripts/p8_probe_v10_structural_drift.py` is a fail-closed
structural probe. It does not alter `p8_audit_v10_transition.py` acceptance
semantics.

For edges through unchanged type 10 it compares a labelled directed multigraph
where every node label contains the complete persisted `node_metadata` record
and exact feature row. It first constrains candidate mappings with incoming and
outgoing neighbour-label signatures, then performs exact backtracking. Edge
direction, edge type, and multiplicity are all preserved. Search-limit
exhaustion is `INCONCLUSIVE_FAIL_CLOSED`, never semantic equivalence.

Candidate and repeat sidecars must prove exact `slither-analyzer=0.10.0`,
`runtime_role=primary`, and `required_for_physical_acceptance=0.10.0`. The frozen
v2.3 reference predates the v2.4 runtime sidecar contract and is therefore bound
by its existing candidate binding digest instead.

The focused audit tests now prove that the diagnostic:

- accepts an exact graph isomorphism when only indistinguishable node indices are
  permuted;
- rejects `CFG_NODE_OTHER -> CFG_NODE_WRITE` feature/classification drift even
  when raw unchanged-edge topology is identical;
- rejects a genuine unchanged-edge topology change.

No extractor ordering or feature-classification behavior was changed remotely.
That would be speculative before the exact 20-contract repeat evidence identifies
the responsible seam.

## Required protected-local repeat procedure

Use the exact primary DATA/ML environment containing Slither 0.10.0. Generate
only these 20 identities; do not include any of the completed historical
parse-only identities.

```bash
cd ~/projects/sentinel

git switch main
git pull --ff-only origin main

IDS=(
047f9d7c2db3d6ba43b62e9c1b35adb1ed5a6bd36d68da46e0f877b0974b73e4
1d9ce79b93c3a1bd7597a76204ae65027fc0471517b2a247c1e536a260c296fd
42489184f712d85f392a47db45110b4406436bc8b648524300a18319111ab350
48beaa23f916dfd3acbc86a799b0859709b29defa3796450693bab13f8e6e777
5a626b8baef72b243f1812118862af26ea796462c38e900f2f595ba73b55495e
6376d572b974fb2ba2c074bf7d43972b241a1000731563b44ee09ef72eeaca3e
73cbc254caad8a7a6b8674125c029a530973459c294c6897dca01d219307c669
7e9bfccd7d3ed5076b7ea61fe444f33b50deb78c27582bcc413b7303422dc551
83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d
85c5c0d173dbaed126f4bc5165c7453262b6ff50c89c52a458034f322a06a714
8b1792cb3c0a40a4ebeec72ffe69d00920c80203213f24ec3e2d5a867eeae3d5
95f7d52dff443cc825e20477a62de371cc4bbc31b6ba5aae653ff51caaaf974c
a4068383ed30b56a39771e1dcbe835726242c164d125908207b4e616030aaa8c
a7faec46ab38dbf5b87b1e1ef0e56fc5da743ac450535b1cc09f12922c86f46c
af947a2b1a6d7c6fa500f5604bc7b3d3e8bbab6711c30b54f601b6db5db19464
beaa4d742f0b52b301fc2f143072b57ef8170540cdaaa096cdd2f51b047ab1ca
c159c57b830cb77686cb5a2a7b40f1452cae516ec688b55331adb61b1669064d
c1d21cda50fb1f0c1194392080a2c7a21b3baed5edbe140caaec8c3f257f756b
dcf66533d7ee72d2a59ab07fd4117f03ea035489f53ac9184e1dcb4d82c6d823
f8cd4abd34d1aa1ca5d3293ee286bfedfa39d0cab550a547ba22e798d5439b35
)

ARGS=()
for id in "${IDS[@]}"; do ARGS+=(--contract-id "$id"); done

for n in 1 2 3; do
  root="/tmp/sentinel-v10-structural-repeat-${n}/representations-r4-v3-candidate"
  rm -rf "$(dirname "$root")"
  PYTHONPATH=data_module data_module/.venv/bin/python \
    docs/plan/ml-R4/scripts/p8_generate_v10_candidate.py \
    --mode regression \
    --output-root "$root" \
    "${ARGS[@]}" \
    --report "$(dirname "$root")/repeat-${n}.json"
done
```

The generator itself fails unless the installed runtime is the exact primary
Slither 0.10.0 for all 20 identities. The probe independently rechecks the
runtime sidecars.

Then compare the frozen reference, canonical v2.4 candidate, and all repeat
roots:

```bash
PYTHONPATH=data_module data_module/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_probe_v10_structural_drift.py \
  --reference-root <FROZEN_V23_SLITHER_010_V10_REFERENCE_ROOT> \
  --candidate-root data_module/data/representations-r4-v3-candidate \
  --repeat-root /tmp/sentinel-v10-structural-repeat-1/representations-r4-v3-candidate \
  --repeat-root /tmp/sentinel-v10-structural-repeat-2/representations-r4-v3-candidate \
  --repeat-root /tmp/sentinel-v10-structural-repeat-3/representations-r4-v3-candidate \
  --output docs/plan/ml-R4/reviews/R4-GAP-008/v10_structural_drift_repeat_probe_v1.json
```

`<FROZEN_V23_SLITHER_010_V10_REFERENCE_ROOT>` must be the exact protected root
whose binding digest is
`6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`.
Do not reconstruct it and do not overwrite it.

## Evidence interpretation and deterministic-fix decision

For each identity:

- exact labelled isomorphism with only raw index/endpoint differences proves
  node-order/index equivalence; the audit must not be weakened merely from
  counts or visually similar metadata;
- repeat outputs alternating between reference-like and candidate-like feature
  states proves Slither feature-classification nondeterminism but remains a
  blocker until the extractor deterministically derives the intended semantic
  class from stable evidence;
- repeat outputs consistently reproducing candidate semantic differences from
  the frozen reference require source/IR review before deciding whether the
  candidate is a justified correction or a regression;
- any third state, search-limit exhaustion, missing runtime binding, missing
  artifact, or unmatched topology remains unexplained and therefore blocking.

Only after the exact local repeat report identifies the seam should the smallest
deterministic extractor fix be implemented. Add a real regression for the exact
observed failure mode, regenerate only the affected identities again, and rerun
the complete transition audit against the same frozen reference.

Physical acceptance requires zero unexplained structural drift and an explicit
review/decision record. Do not convert this diagnostic file itself into an
acceptance waiver. Training and model-quality claims remain unauthorized even
after a future physical V10 acceptance.

## Current handoff boundary

This GitHub-connected session can modify and validate repository source but does
not have access to the protected ignored representation roots or the primary WSL
Slither toolchain. Therefore the 20 identities have not been truthfully claimed
as repeatedly regenerated or finally classified here.

The next authorized action is exactly the bounded local procedure above. It is
not a rerun of the completed 26-contract parse-only repair, not a full-population
regeneration, and not training.
