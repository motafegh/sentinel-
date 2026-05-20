# 2026-05-20 — v8.0-AB Training Analysis + tqdm Log Fix

## Summary

Full diagnostic analysis of the v8.0-AB training run through epoch 26.
Root cause identified for logs appearing "missing" — tqdm carriage-return pollution.
Fix applied to `trainer.py`. Kill condition defined for v8-AB.

---

## Root cause: tqdm polluting log files

The training was launched with `2>&1 | tee`, merging stderr (tqdm progress bars) and stdout (loguru) into the same file. tqdm writes `\r` before each update, so in the log file each loguru step line is physically on the same byte-line as the preceding tqdm bar string. Normal `grep` returns fused lines; the terminal renders only the suffix (after the last `\r`), which looks like tqdm output eating the loguru lines.

**Fix** (`ml/src/training/trainer.py`):
- Line 504 (training loop): `tqdm(..., disable=not sys.stdout.isatty())`
- Line 411 (eval loop): `tqdm(..., disable=not sys.stdout.isatty())`

When stdout is piped or redirected to a file, tqdm is silenced entirely. Future runs via `nohup ... | tee` produce clean loguru-only logs.

---

## Key findings

### Eye scores (step 400, epoch-end proxy)

| | Ep 1 | Ep 26 | Interpretation |
|--|------|-------|----------------|
| GNN eye loss | 0.747 | 0.449 | −40%; graph encodes real signal beyond CodeBERT |
| TF eye loss | 0.689 | 0.497 | −28%; CodeBERT is harder to fine-tune for this task |
| Fused eye loss | 0.700 | 0.429 | −39%; cross-modal fusion is the most discriminative head |

All three plateau around epoch 10 — the ceiling is structural (schema), not optimizer-related.

### Fusion grad norms are the breakthrough signal

Every significant F1 jump (ep6: +0.047, ep20: +0.009) coincides with a fused grad spike (0.067–0.073). The fusion layer is the learning bottleneck; improving GNN input quality (better cross-function edges) is what will unlock the next plateau escape.

### JK attention — v8 ICFG/DEF_USE edges contribute

```
v7 final (ep33):  P1=0.050  P2=0.182  P3=0.768
v8-AB (ep26):     P1=0.072  P2=0.263  P3=0.665
```

Phase 2 carries 45% more weight in v8. The CALL_ENTRY/RETURN_TO/DEF_USE edges are not being ignored. Phase 3 (REVERSE_CONTAINS) remains dominant but less extreme, partially displaced by the cross-function signal.

Phase 2 weight declining from peak 0.362 (ep3) to 0.263 (ep26): model initially finds the new edges useful, then increasingly trusts the more stable containment hierarchy as training progresses. This suggests the current DEF_USE extraction (intra-function only) has limited reach — PLAN-3A/3B will isolate which edge type is driving the contribution.

### F1 plateau

- Zone: 0.236–0.259, epochs 10–26 (16 epochs of oscillation)
- Best: 0.2593 at epoch 22
- Near-miss: 0.2588 at epoch 25
- v7 best was 0.2651 at epoch 23 — v8 tracking 0.006 below despite richer schema
- Structural ceiling, not optimizer plateau — same onset (ep10) as eye loss plateau

---

## Decision: run to patience=10/30, then kill

- Patience=4/30 at epoch 26 — not technically stuck yet
- Ep25 near-miss suggests ~30% chance of new best in next 6–8 epochs
- Kill manually around epoch 32 if no new best (patience=10/30)
- Beyond that, probability of breakthrough <10%; PLAN-3A info is more valuable

---

## Files changed

| File | Change |
|------|--------|
| `ml/src/training/trainer.py` | tqdm `disable=not sys.stdout.isatty()` on both training and eval loops |
| `docs/ml/v8-AB-training-analysis.md` | New — full per-epoch table + 5 findings + ablation decision matrix |
| `docs/STATUS.md` | Updated — v8-AB current state, v7 marked complete, Agents Phase 0 complete |
| `docs/changes/2026-05-20-v8-AB-training-analysis.md` | This file |
