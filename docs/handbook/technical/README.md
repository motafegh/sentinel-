# Technical guides — supplementary learning layer

These guides are retained because they contain useful source-reading detail and exercises. They are **not the current-state authority**.

Some guides were authored against the pre-R4 DATA/ML baseline or pre-V3 chain/runtime design. Before using a guide for implementation or operational decisions, read:

1. [`../16_current_status.md`](../16_current_status.md)
2. the owning canonical handbook chapter
3. current executable source
4. `docs/plan/ml-R4/` when DATA/ML semantics or evaluation roles are involved

Current canonical corrections that override older guide examples include:

- historical binary DATA v1 labels are not DATA vNext truth;
- Run12 is the historical operational teacher, not a repaired-vNext retrain;
- threshold/calibration/untouched-acceptance roles are unsupported for the first repaired baseline;
- live audit MCP is read-only;
- V3 is the current registry submission protocol; V1/V2 are historical compatibility;
- the retained EZKL proof proves only the proxy computation; V3 context attestation is separate.

A guide may still be completely useful for a local mechanism even when its surrounding end-to-end example is historical. Treat canonical chapters/current source as the final authority.
