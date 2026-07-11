# Agents Module Ownership — 03: Evidence, Fusion, and Reliability

## Ownership Target

Understand how the module represents claims from independent channels and how `fuse()` produces the report and ZK-boundary verdicts.

## Source Reading Order

1. `agents/src/orchestration/verdict/evidence.py`
2. `agents/src/orchestration/verdict/emit.py`
3. `agents/src/orchestration/verdict/verdict.py`
4. `agents/src/orchestration/verdict/fuse.py`
5. `agents/src/orchestration/verdict/reliability.py`
6. `agents/src/orchestration/nodes/consensus_engine.py`
7. `agents/src/orchestration/nodes/synthesizer.py`

## Items to Own

- `Evidence` fields, polarity, kind, determinism, strength, and reliability.
- Which nodes emit evidence and why a node should not directly own the final verdict.
- Witness-family de-correlation in `fuse()`.
- Signed evidence mass and verdict-band selection.
- The strong-support rule that prevents a strong supporting signal from being silently cleared to `SAFE`.
- The difference between `verdict_provable` and `verdict_full`.
- How reliability resolves from a fitted config, L1 prior, and hard fallback.
- Why decision numbers belong in versioned configuration and require measurement before change.

## Trace Exercise

Choose one vulnerability class and list every evidence item produced for it on a deep-path audit. For each item, record source, family, polarity, deterministic flag, strength, and reliability. Then identify which items can affect each dual verdict.

## Verification

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest \
  tests/test_verdict_evidence.py \
  tests/test_verdict_fuse.py \
  tests/test_verdict_reliability.py \
  tests/test_p2_evidence_integration.py -q
```

## Completion Check

- Why are Slither and Aderyn not independent witness families?
- What can enter `verdict_full` but not `verdict_provable`?
- Where would a new source's reliability and family be defined?
- Why is changing a fusion band a policy change rather than a syntax edit?

## Intentionally Out of Scope

- Internal behavior of each evidence-producing tool.
- ZK proof generation and on-chain submission.
- Gateway and MCP transport.
