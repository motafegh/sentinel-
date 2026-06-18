# Method 1 — DIVE Dataset Origin and Labeling Methodology

**Status:** COMPLETE
**Started:** 2026-06-18
**Completed:** 2026-06-18

**Provenance:**
- Paper: Alsunaidi, Aljamaan, Hammoudeh. "DIVE: A Multi-Label Smart Contract Vulnerability Dataset." *Scientific Data* 13, Article 664 (2026). Published 12 Mar 2026. DOI: `10.1038/s41597-026-07025-5`
- URL: `https://www.nature.com/articles/s41597-026-07025-5`
- Dataset: Zenodo `https://doi.org/10.5281/zenodo.18519253`
- Framework: Zenodo `https://doi.org/10.5281/zenodo.18779606`
- No local README/methodology files found in raw DIVE directories. Ingestion manifest has `url: ""`.

---

## 1. How DIVE labels were produced — FULLY AUTOMATED, no manual review

DIVE labels come from a **two-stage automated pipeline**:

### Stage 1 — Power-based voting across 6 analysis tools

Six tools analyzed each contract:
| Tool | Type |
|---|---|
| MAIAN | Symbolic execution (bug finder) |
| Mythril | Symbolic execution (security analysis) |
| Semgrep | Static pattern matching |
| Slither | Static analysis framework |
| Solhint | Solidity linter |
| VeriSmart | Symbolic verification |

The **MultiTagging** framework (by the same authors) parsed each tool's output and mapped tool-specific findings to DASP Top 10 classes. Final labels were assigned using **Power-based voting** which "determines the role of each tool and selects the optimal voting strategy for each vulnerability based on prior analytical evaluation."

### Stage 2 — Post-hoc rule-based validation

A rule-based validator reassesses positive findings from voting against vulnerability-specific code-evidence criteria. Findings not meeting category-specific criteria are reclassified as negative. The authors report this corrected **14.3% false positives in DoS** and **24.9% false positives in Time Manipulation**.

### What this means

**Labels are entirely tool-derived. No human expert validated a single contract.** The paper states explicitly (Limitations section):

> "Due to the scale of the dataset, manual audits and validation against external ground-truth annotations or audit reports were not conducted."

The authors themselves describe the labels as:

> "systematically derived, high-confidence annotations rather than manually verified ground truth."

---

## 2. Multi-label is by design, not an artifact

The dataset uses a "multi-label scheme, as a single SC may exhibit multiple vulnerability types." All 8 DASP classes are assigned independently. The authors explicitly support multi-label, multi-class, binary, and multi-task classification use cases. The 69% multi-label rate (15,423/22,330) is a feature, not a bug.

---

## 3. No confidence scores in the released data

The final labels are **binary 0/1** per vulnerability category. There is no probability or confidence column. The paper mentions "confidence scores" only once — referring to internal scores used by the MultiTagging framework during post-hoc validation to decide which findings to keep. The released CSV and dataset contain only binary labels.

Additionally, the dataset provides **per-tool binary labels** (which tool flagged which class for each contract) — but this information was NOT captured in our ingestion (only the final merged binary labels were imported via the CSV).

---

## 4. Data source and scope

- **Sources:** Contract addresses from ScrawlD dataset + Etherscan + Smart Contract Sanctuary (GitHub)
- **Initial pool:** 32,766 unique addresses → filtered to 22,330 final contracts
- **Deployment years:** 2016–2024
- **Compiler versions:** v0.4.x through v0.8.x (v0.5.x, v0.6.x, v0.7.x underrepresented)
- **Only verified contracts** on Etherscan — a systematic bias against bytecode-only, obfuscated, short-lived, or adversarial contracts

---

## 5. Limitations disclosed by the authors

1. **No manual verification.** Labels are from automated tools only. "Static and symbolic tools may produce false positives and false negatives, particularly for vulnerabilities requiring execution context, environmental assumptions, cross-contract interactions, or complex state evolution."
2. **No precision/recall estimates** for the labeling process.
3. **Bias toward verified contracts.** Underrepresents bytecode-only, intentionally obfuscated, or adversarial contracts.
4. **Class imbalance.** Access Control (16,723) and Reentrancy (11,400) dominate; Bad Randomness (634) and Front Running (606) are infrequent.
5. **Snapshot in time.** Does not reflect subsequent contract updates or patches.

---

## 6. What this means for the Phase 1 investigation (EB/RE)

### The prior ~4-5% TP rate is expected, not evidence of data corruption

If DIVE labels are produced by automated tools voting on code patterns, a ~5% TP rate (when judged against stricter human criteria) means the *tools* have a ~5% precision on Access Control/Reentrancy — not that the data is "wrong." The authors never claimed expert-level ground truth. A low TP rate against frozen human-criteria is the expected outcome of tool-derived labels, and the authors' own post-hoc validation corrected 14-25% false positives.

### Implications for Method 0 (freeze TP criteria)

- Setting an "exploitable" bar would produce near-zero TP rates — the tools don't assess exploitability, only code patterns. A "pattern-present" bar aligns better with what the label source actually represents.
- The BORDERLINE bucket is especially important here, because tool-derived labels may flag patterns that are security-relevant but not clearly vulnerable.
- The control arm (§6) is critical: the same frozen criteria applied to known-clean contracts calibrates whether the tools/criteria are too strict or the labels are genuinely noisy.

### Implications for the keep/drop decision

The authors' own caveat — "systematically derived, high-confidence annotations rather than manually verified ground truth" — means a KEEP decision for DIVE labels is a decision to train on **tool-consensus signal**, not ground truth. That may still be useful if the signal carries information the model can exploit. But it should not be confused with "these contracts are actually vulnerable."

### What was NOT covered

- Did NOT retrieve the per-tool binary labels (the paper provides them; our ingestion did not capture them). Could be used for cross-tool agreement analysis (Method 6).
- Did NOT read the MultiTagging framework paper (ref 4) for deeper voting-algorithm details.
- Did NOT verify whether the DIVE_Labels.csv we have matches the Zenodo release (hash comparison).
