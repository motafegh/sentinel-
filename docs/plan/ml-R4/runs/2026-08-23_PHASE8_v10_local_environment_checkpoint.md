# Phase-8 V10 local environment checkpoint

Date: 2026-08-23
Status: LOCAL RUNTIME VERIFICATION REQUIRED; PHYSICAL ACCEPTANCE REMAINS BLOCKED
Scope: R4-B008 structural-drift investigation only; no training or model-quality authority

## Reusable protected-local paths now resolved

The exact frozen Slither-0.10 V10 v2.3 structural reference is present locally at:

`data_module/data/representations-r4-v3-candidate-v2.3-structural-reference-6087dc6d`

Its `v10_candidate_binding_report.json` records:

- `binding_digest_sha256 = 6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`
- `extractor_version = v2.3-r4-call-semantics`
- `passed = true`
- `physical_acceptance = false`

The current protected v2.4 candidate remains:

`data_module/data/representations-r4-v3-candidate`

and reports extractor identity `v2.4-r4-call-semantics-compat` with diagnostic binding pass but no physical acceptance.

A retained pre-ICFG-fix v2.4 root also exists locally at:

`data_module/data/representations-r4-v3-candidate-v2.4-pre-icfg-fix-ba010d47`

It is historical evidence only and is not the structural reference for the current repeat probe.

## Local launcher issue discovered

Direct execution of:

`data_module/.venv/bin/slither --version`

currently fails because the generated console-script shebang points to the obsolete interpreter path:

`/home/motafeq/projects/sentinel/Data/.venv/bin/python`

The present repository path is `data_module`, so this launcher cannot be used as evidence that the DATA Python environment itself is invalid. The next step is to verify the actual `data_module/.venv/bin/python` interpreter and query the installed `slither-analyzer` package version through Python. Do not recreate or mutate the environment until that bounded verification is complete.

## Investigation boundary

Do not restart the completed 26-contract parse-only repair. Do not launch training. Once the underlying DATA Python environment and exact Slither 0.10.0 package are confirmed, regenerate only the 20 unexpected structural-drift identities and compare them against the frozen reference above with `p8_probe_v10_structural_drift.py`.
