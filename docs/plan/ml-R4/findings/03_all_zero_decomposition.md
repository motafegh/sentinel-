# 03 — All-Zero and Historical-Zero Decomposition

## Core finding

Historical numeric `0` is not one semantic state. The current pipeline serializes multiple distinct origins into the same value and ML has no target mask to distinguish them.

## Zero-origin taxonomy reconstructed from source

| Zero origin | Where introduced | Example | Historical serialization | Correct semantic reading |
|---|---|---|---|---|
| explicit source zero | source-native assertion where recoverable | DIVE CSV cell `0` | no symlink → parser `0` | source-native zero only; strength depends on source semantics |
| source-native unknown | DIVE CSV folderization | empty CSV cell | no symlink → parser `0` | `UNKNOWN`, not negative |
| class unsupported by source | parser expands to all ten classes | DIVE→GasException; SolidiFI→DoS | `0` | `NOT_APPLICABLE` / unsupported |
| dropped source-native category | crosswalk | DIVE Bad Randomness | no mapped positive; possibly all-zero | dropped taxonomy information |
| mapped-to-NonVulnerable | crosswalk | SmartBugs `short_addresses`, `other` | all ten `0` | out-of-taxonomy / mapped safe bucket, not ten independent negatives |
| ordinary non-target class | single-category parser | SolidiFI injected Reentrancy produces other 9 zeros | `0` | source absence for those class assertions |
| parser missing-key default | split/export projection | `entry.get("value",0)`, `classes.get(name,0)` | `0` | missing value erased into zero |
| all-zero row classification | split | no positive cell | `primary_class="NonVulnerable"` | synthetic pipeline role, not proof of safety |
| historical post-export suppression | manual/out-of-band patch | DoS+Reentrancy patch | DoS `1→0` | suppressed historical positive |
| missing representation | export/ML loader | label row has no graph shard | label retained in parquet but row omitted from ML | population exclusion, not class negative |

## DIVE all-zero mechanisms

### A. Source row with no positive mapped folder

The DIVE parser uses:

```text
filename → mapped folder memberships
no mapped membership → frozenset()
for every canonical class: value = 0
```

Recovered DIVE metadata reports **2,686** files in `__source__` with no vulnerability-folder membership before accounting for Bad-Randomness crosswalk loss.

### B. Unknown cell collapse

Folderization explicitly recognizes empty CSV cells as unknown, but creates symlinks only for positive values. After folderization, the parser cannot distinguish:

```text
source cell = 0
source cell = empty/unknown
class never represented by this source
```

All are observed as absence of a folder and serialized as `0`.

### C. Bad-Randomness-only collapse

DIVE reports **634** files in the Bad Randomness folder. That category is intentionally absent from `class_map`. If a contract has no other mapped positive, it becomes all-zero. The exact exclusive Bad-Randomness-only count is not preserved in tracked remote evidence, so Phase 2 leaves that sub-count unresolved rather than inventing it.

### D. Unsupported canonical classes

DIVE has no direct mapping for CallToUnknown, GasException or MishandledException. Across the recovered 22,073 labeled DIVE records, this produces at least **66,219 unsupported-class zero cells** (`22,073 × 3`) before any multi-source override.

## SolidiFI zero mechanisms

Each of **283** recovered SolidiFI rows has exactly one injected positive and nine zeros.

- total emitted zero cells: **2,547** (`283 × 9`);
- canonical classes wholly unsupported by SolidiFI: DenialOfService, GasException, UnusedReturn;
- wholly-unsupported cells: **849** (`283 × 3`).

The remaining non-target zeros still lack explicit negative-control authority: an injection benchmark proves the injected bug exists, not that every other bug class is absent.

## SmartBugs Curated zero mechanisms

The recovered corpus has **143** hand-labeled contracts and Phase-1 evidence reports **4** NonVulnerable examples. The parser emits one positive for a mapped vulnerability row and zero for every other class; an all-zero row is emitted when the mapped canonical category is NonVulnerable.

Given the parser's single-category representation:

- 139 vulnerability rows × 9 zero cells = 1,251;
- 4 NonVulnerable rows × 10 zero cells = 40;
- total serialized zero cells = **1,291**.

This count describes serialization, not 1,291 independent negative adjudications.

Crosswalk categories `short_addresses` and `other` explicitly map to NonVulnerable, so some all-zero rows can mean “outside the locked ten-class taxonomy” rather than “reviewed clean across all ten classes.”

## Merger and split amplification

The merger cannot preserve zero origin because every per-source class entry is already binary. Single-source all-zero rows pass through unchanged.

During split construction:

- `entry.get("value", 0)` defaults missing values to zero;
- first positive becomes `primary_class`; if none exists, `primary_class="NonVulnerable"`;
- an all-zero record receives default tier `T0` in the `Contract` object because no positive tier replaces the initialization;
- `apply_nonvulnerable_cap()` then actively samples these rows as the NonVulnerable population.

Therefore a semantic unknown can become not only a numeric zero, but an **actively selected negative training example**.

## Export and ML amplification

`label_writer.py` again projects missing classes with `classes.get(name, 0)`. `labels.parquet` contains no class mask or zero-origin field.

`SentinelDataset` returns only a ten-float target plus one row-level confidence tier. `sentinel_collate_fn` stacks the targets with no class mask. `AsymmetricLoss` computes negative loss as:

```text
-(1 - labels) * focal_neg * log_neg
```

Thus every historical zero that reaches ML contributes as a negative class target, regardless of whether its origin was unknown, unsupported, dropped, mapped-to-NonVulnerable, or explicitly negative.

## Historical DoS suppression

A separate June 13 patch directly changed **2,655** `DenialOfService=1` cells to `0` for DoS+Reentrancy co-occurrence records, reducing the reported DoS count from 3,750 to 1,095. These zeros mean **historically suppressed positive**, not source-confirmed negative.

## Decomposition conclusion

At category/mechanism level, historical zero origins are now explained as:

1. source-native explicit zero where present;
2. source-native unknown erased by folderization;
3. class unsupported by source;
4. non-target class expanded to zero by a one-category parser;
5. dropped category;
6. mapped-to-NonVulnerable category;
7. parser/projection missing-key default;
8. merger-preserved zero;
9. split-time synthetic NonVulnerable classification;
10. out-of-band positive→zero patch;
11. row exclusion from ML due missing representation.

No one of these categories may be promoted to `CONFIRMED_NEGATIVE` merely because the historical serialized value is `0`.
