# R0.0 acceptance handoff

Candidate `caabefcbc6f2d422247c94fd9f5cb69ce84fdd3c` is ready for review against the approved R0.0 exit gate. It is not yet accepted or integrated.

## Outcome first

R0.0 now provides a strict runtime-profile boundary and a reproducible evidence system that cannot close a global R0 row without comparable before/after measurements and explicit accepted review. The clean D2 baseline reproduces all eight unsafe states, and the coverage report correctly keeps all eight rows open because their owning remediation packages have not produced after evidence.

## Measured before and after

| R0.0 concern | Before at `1256d9aab` | After at `caabefcbc` |
|---|---|---|
| Evidence owner | Absent | Versioned schemas, eight stable rows, command/environment/baseline manifests |
| Baseline reproduction | Ad hoc D2 observations | 8/8 executable expected failures, clean commit, fixture-bound commands |
| False closure protection | Absent | 18/18 harness tests pass; incomplete/pending/incomparable/malformed evidence stays open |
| Runtime profile | Absent | 13/13 strict boundary tests pass |
| Existing config behavior | 13 tests available | 13/13 continue to pass |
| AGENTS regression | Affected baseline subset: 12 pass, 9 fail | Full candidate: 638 pass, 9 fail; identical failure set, zero new failures |
| Global R0 closure | 0/8 | 0/8, intentionally unchanged until R0.1–R0.5 after evidence exists |

## Retained limitations

- Six RAG tests fail because seed corpora are absent in both isolated worktrees.
- Three static-analysis/smoke tests fail because `solc` is absent in both baseline and candidate environments.
- Mypy could not run because the existing tool environment has incompatible `mypy` and `mypy_extensions` installations. Black, isort, flake8, JSON, compile, and diff checks pass.

These limitations do not alter the R0.0 focused exit gate, but they remain explicit and cannot be converted into passes. R0.6 must revisit every unavailable prerequisite during full-wave closure.

## Review boundary

Accepting R0.0 would authorize integration of this package into the R0 containment branch and progression to R0.1. It would not accept any of the eight global R0 security invariants, authorize deployment, start LM Studio, move keys, send transactions, change policy numbers, promote models, or delete artifacts.
