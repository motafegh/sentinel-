# Phase-8 V10 local environment checkpoint

Date: 2026-08-23
Status: PRIMARY ML RUNTIME VERIFICATION REQUIRED; PHYSICAL ACCEPTANCE REMAINS BLOCKED
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

## Local environment split confirmed

The repository working plan already distinguishes the two relevant Slither environments:

- the locked DATA environment uses Slither 0.11.5;
- the ML environment / structurally stable V9/V10 baseline uses Slither 0.10.0 and is the required primary runtime for the 20-contract repeat investigation.

The local bounded check confirmed that `data_module/.venv/bin/python` is Python 3.12.3 and currently contains:

- `slither-analyzer = 0.11.5`
- `crytic-compile = 0.3.11`
- importable `slither` at `data_module/.venv/lib/python3.12/site-packages/slither/__init__.py`.

Therefore `data_module/.venv` must not be downgraded or repurposed for the primary 0.10.0 repeat generation.

## DATA launcher issue confirmed

Direct execution of:

`data_module/.venv/bin/slither --version`

fails because the generated console-script shebang points to the obsolete interpreter path:

`/home/motafeq/projects/sentinel/Data/.venv/bin/python`

while the current repository path is `data_module`. This launcher defect is real but secondary: the underlying DATA Python environment is healthy and intentionally carries Slither 0.11.5. It is not the primary runtime required by this structural-drift tranche.

## Next bounded runtime check

The next step is to inspect the known project ML environment at `ml/.venv/bin/python` without mutating any environment. Confirm its Python executable, `slither-analyzer`, `crytic-compile`, and Slither import path. If it proves exact Slither 0.10.0, use that interpreter for the bounded 20-contract repeated generation while retaining `data_module` on `PYTHONPATH`.

Do not install, downgrade, rebuild, or repair either environment until that check is complete.

## Investigation boundary

Do not restart the completed 26-contract parse-only repair. Do not launch training. Once the exact primary Slither 0.10.0 interpreter is confirmed, regenerate only the 20 unexpected structural-drift identities and compare them against the frozen reference above with `p8_probe_v10_structural_drift.py`.
