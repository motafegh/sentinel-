# `sentinel_data.analysis` — Understanding Your Data Before Training

## What This Module Does

The analysis module is Stage 8 of the SENTINEL data pipeline. It runs 6 read-only exploratory tools that surface dataset properties **before and after** pipeline runs. The headline output is the `complexity_proxy_risk.md` report — the single most important analysis artifact that catches the Run 9 failure mode (model learning complexity as a proxy for vulnerability) before training.

## Why This Matters

Run 9 trained for 9 runs before discovering that the model was learning complexity (node count, edge count) as a proxy for vulnerability, not the actual vulnerability patterns. The analysis module would have caught this **before the first epoch** by computing per-class feature distributions and flagging any class-pair where the complexity features differ by > 1.5σ.

The L4 interpretability finding was: "complexity dominates all 10 classes at 34-36%." The data-side `complexity_proxy_risk.md` is the **same diagnosis from the data side** — run before training, not after.

## The 6 Analysis Tools

| Tool | Output | What It Catches |
|------|--------|-----------------|
| `balance_viz` | Per-class/source/tier counts + bar plot | Class imbalance |
| `feature_dist` | Per-class feature distributions + `complexity_proxy_risk.md` | **Run 9 failure mode** |
| `cooccurrence` | Directed + conditional co-occurrence matrices | 99% DoS↔Reentrancy pattern |
| `overlap_detector` | Pairwise Jaccard similarity between sources | Source redundancy |
| `drift_monitor` | Per-feature KS test between dataset versions | Label/feature drift |
| `probe_dataset` | Re-export from verification | Model interpretability input |

## The `complexity_proxy_risk.md` Report

This is the headline output. It computes, for each pair of classes, the difference in mean (and std) of:
- Node count
- Edge count
- Cyclomatic complexity
- Call depth
- Function count
- LOC

If any pair differs by > 1.5σ, the pair is flagged as **HIGH-RISK** for the model to learn a complexity proxy instead of a class-specific pattern.

Example output:
```markdown
## Complexity Proxy Risk Report

### HIGH-RISK Pairs (σ-difference > 1.5)

| Class A | Class B | Feature | σ-difference | Risk |
|---------|---------|---------|--------------|------|
| Reentrancy | Timestamp | node_count | 2.3 | HIGH |
| IntegerUO | ExternalBug | edge_count | 1.8 | HIGH |

### Recommendation
Reentrancy positives are 2.3σ more complex than Timestamp positives.
Consider stratified sampling or class-weight adjustment in the loss.
```

## The Co-occurrence Matrix

The `cooccurrence` tool produces **two matrices** (directed + conditional):

- **Directed:** X→Y means "if class X is positive, class Y is also positive with probability p"
- **Conditional:** P(Y=1 | X=1) — the conditional probability

The BCCC 99% DoS→Reentrancy co-occurrence is visible as a very high entry in both matrices. The conditional matrix is what the multi-label loss design consumes.

## How to Use

```bash
# Run all 6 analysis tools
sentinel-data analyze

# Run a specific tool
sentinel-data analyze --only feature_dist

# Analyze a specific dataset version
sentinel-data analyze --corpus sentinel-v2-dryrun-2026-08

# Compare against a baseline
sentinel-data analyze --baseline-version v1.4-bccc
```

## Pipeline Position

```
Stage 7: Registry (catalog + lineage)
    ↓
Stage 8: Analysis ← YOU ARE HERE (6 exploratory tools)
    ↓
Stage 9: Export + Seam Swap (sharded output for ML)
```

## Design Decisions

1. **Read-only** — analysis tools don't modify any input artifacts
2. **DVC-tracked outputs** — analysis results are part of the reproducible pipeline
3. **Complexity proxy as headline** — the Run 9 failure mode is the primary concern
4. **Directed + conditional co-occurrence** — catches the 99% DoS↔Reentrancy pattern
5. **Probe dataset re-export** — single source of truth for model interpretability input
