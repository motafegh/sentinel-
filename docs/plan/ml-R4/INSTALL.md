# Installation — historical bootstrap package

> **Historical notice (2026-08-27):** this file documents how the original 2026-07-16 R4 bootstrap archive was installed. It is **not** a current update/restart procedure. Do not replace the live `docs/plan/ml-R4` tree from that archive. `PACKAGE_MANIFEST.json` is likewise the immutable manifest of the original bootstrap package and its hashes intentionally do not describe the evolved current tree.

For current project work, use the checked-out repository and read:

1. applicable repository/module `CLAUDE.md` files;
2. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
3. `docs/plan/ml-R4/runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`;
4. the current phase/run records referenced from that checkpoint.

Do **not** use the old package archive or `PACKAGE_MANIFEST.json` as current integrity authority. Current artifact identity comes from the committed repository, current machine-readable governance/evidence bindings, and the versioned hashes/digests recorded by the active R4 gates.

## Original bootstrap installation procedure — retained for provenance only

The historical archive contained a top-level `ml-R4/` directory.

The original installation procedure was:

```bash
rm -rf /tmp/sentinel-ml-r4-package
mkdir -p /tmp/sentinel-ml-r4-package
unzip /path/to/sentinel_ml_R4_label_recovery_package.zip -d /tmp/sentinel-ml-r4-package

mkdir -p docs/plan
rm -rf docs/plan/ml-R4
cp -a /tmp/sentinel-ml-r4-package/ml-R4 docs/plan/ml-R4

find docs/plan/ml-R4 -maxdepth 2 -type f | sort
```

**Do not execute those replacement commands against the current project.** They would overwrite months of accepted R4 decisions, evidence, implementation support, and current Phase-8 state.

The original package then directed the agent to:

`docs/plan/ml-R4/START_HERE_AGENT.md`

That file is still retained but now contains a current redirect before its historical Phase-0 bootstrap instructions.
