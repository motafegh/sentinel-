# R0 main-integration closeout plan

## Objective

Integrate the formally closed R0 branch into `main` without losing or absorbing unrelated active
work, and leave a concise maintainer explanation beside the R0 closure evidence.

## Steps

1. Confirm the R0 validator remains `complete=true` and both worktree states are known.
2. Add a bounded educational section to the existing R0 closure record; do not extend the numbered
   learning curriculum or create another architecture subsystem.
3. Commit closeout documentation on `codex/r0-containment`.
4. Merge that branch into `main` with an explicit merge commit while preserving unrelated dirty
   files in place; stop if Git reports overlap or conflict.
5. Re-run the R0 validator and focused post-merge regression groups from `main`.
6. Record the exact merge identity and preserved dirty-state inventory.

## Acceptance

- R0 validator remains `complete=true` with 8/8 rows closed.
- Post-merge focused suites pass with zero failures.
- `main` contains the R0 executable, evidence, closure, and integration documentation.
- Pre-existing uncommitted ML/R4/audit work remains present and is not included in the merge commit.
- No push, deployment, live-chain write, key movement, or model promotion occurs.
