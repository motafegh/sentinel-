# R0.0 acceptance record

Candidate `7d10da27fc6f2ad158b39d5d37b17fee6f640c2d` is accepted against the approved R0.0 exit gate under Ali Motafegh's standing local-R0 authorization. It is ready for integration.

## Outcome first

R0.0 now provides a strict runtime-profile boundary and a reproducible evidence system that cannot close a global R0 row without comparable before/after measurements and explicit accepted review. Strengthened baseline series 2 binds exact lockfile hashes, interpreter/package inventories, runtime bindings, frozen probe bytes, and sanitized probe-environment policy. The clean D2 baseline reproduces all eight unsafe states, and the coverage report correctly keeps all eight rows open because their owning remediation packages have not produced after evidence.

## Measured before and after

| R0.0 concern | Before at `1256d9aab` | After at `7d10da27` |
|---|---|---|
| Evidence owner | Absent | Versioned schemas, eight stable rows, command/environment/baseline manifests |
| Baseline reproduction | Ad hoc D2 observations | 8/8 executable expected failures, clean commit, fixture- and environment-bound commands |
| False closure protection | Absent | 23/23 harness tests pass; incomplete/pending/incomparable/malformed evidence stays open |
| Measurement comparability | OS/Python class only in series 1 | Series 2 candidate and baseline share fingerprint `c21fac2037ea…`; runtime identity is remeasured, never cached |
| Runtime profile | Absent | 13/13 strict boundary tests pass |
| Existing config behavior | 13 tests available | 13/13 continue to pass |
| AGENTS regression | Affected baseline subset: 12 pass, 9 fail | Canonical full candidate run from `agents/`: 638 pass, 9 fail; identical failure set, zero new failures |
| Global R0 closure | 0/8 | 0/8, intentionally unchanged until R0.1–R0.5 after evidence exists |

## Retained limitations

- Six RAG tests fail because seed corpora are absent in both isolated worktrees.
- Three static-analysis/smoke tests fail because `solc` is absent in both baseline and candidate environments.
- Mypy could not run because the existing tool environment has incompatible `mypy` and `mypy_extensions` installations. Black, isort, flake8, JSON, compile, and diff checks pass.

These limitations do not alter the R0.0 focused exit gate, but they remain explicit and cannot be converted into passes. R0.6 must revisit every unavailable prerequisite during full-wave closure.

## Review boundary

R0.0 acceptance authorizes integration of this package into the R0 containment branch and progression to R0.1. It does not accept any of the eight global R0 security invariants, authorize deployment, start LM Studio, move keys, send transactions, change policy numbers without measurement, promote models, or delete artifacts.
