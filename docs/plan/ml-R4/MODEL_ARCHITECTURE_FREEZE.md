# Model Architecture Freeze

## Default R4 decision

The existing model architecture and high-level ML design are frozen during label recovery and the first DATA vNext retraining cycle.

The initial R4 ML experiment is:

```text
existing architecture
+ DATA vNext
+ explicit class masks
+ leakage-safe partitions
+ correct calibration/threshold roles
```

## Prohibited normal-path work

Without an approved decision record, do not:

- replace the GNN;
- replace the Transformer/token encoder;
- add architecture search;
- change embedding dimensions to chase metrics;
- introduce a new fusion architecture;
- add complexity to compensate for label noise;
- compare many architectures on the acceptance set;
- attribute weak historical performance to architecture before DATA vNext evaluation.

## Permitted compatibility changes

Implementation changes are allowed when strictly required to consume DATA vNext, including:

- class masks;
- evidence-role metadata;
- grouped partition identifiers;
- loss masking;
- sample weighting approved by policy;
- new export schema loading;
- improved artifact binding and validation;
- logging raw probabilities;
- correct calibration and threshold sidecar loading.

These are interface and correctness changes, not architecture redesign.

## Unfreezing criterion

Architecture work may be proposed only after:

1. DATA vNext is accepted for training;
2. the existing architecture is retrained reproducibly;
3. evaluation uses evidence-qualified, leakage-safe populations;
4. a documented failure analysis identifies a representational/model limitation rather than target or evaluation defects;
5. an ADR defines the architecture hypothesis, development population, acceptance isolation, and rollback.

Architecture unfreezing is not automatic if metrics are lower than historical metrics; historical metrics may have measured corrupted targets.
