# Stage 3 — Labeling (parsers + crosswalks + merger)

**Date:** 2026-06-30
**Status:** NOT STARTED. Reading required before Stage 4.
**Reading time:** 25-35 minutes.
**Goal:** After this doc, you can answer all 8 items in `LEARNING_CHECKLIST.md` §"Stage 3" from memory.

---

## 1️⃣ The Problem

### What Stage 3 has to deliver

Stages 1–2 produced preprocessed `.sol` files and PyG graph tensors. But the model doesn't know *which contracts are vulnerable* — it needs **labels**. Stage 3 attaches a per-class vulnerability label to every contract, with provenance and confidence.

The BCCC failure was a labeling failure. The 89% Reentrancy FP rate existed because BCCC used **folder names as labels** — `BCCC-SCsVul-2024/Source Codes/Reentrancy/foo.sol` means "foo.sol is labeled Reentrancy." But many of those files weren't actually reentrancy vulnerabilities.

Stage 3 prevents this by using **source-specific crosswalk YAMLs** (human-reviewed mappings from each source's native taxonomy to the canonical 10 classes) and **source-specific parsers** (which read the source's actual metadata, not folder names).

### The canonical 10-class taxonomy (D-3.1)

The taxonomy is LOCKED to the v1 checkpoint's class order:

| ID | Class | What it means |
|---|---|---|
| 0 | Reentrancy | External call + state change AFTER call (CEI violation) |
| 1 | CallToUnknown | `.call{}` / `.delegatecall{}` / `.send()` / `.transfer()` to unknown target |
| 2 | Timestamp | `block.timestamp` / `now` in a conditional |
| 3 | ExternalBug | Cross-contract call to non-interface target |
| 4 | GasException | Unchecked `send()` / `transfer()` / low-level call |
| 5 | DenialOfService | Loop with external call or unbounded iteration |
| 6 | IntegerUO | Arithmetic op in pre-0.8 or `unchecked{}` block in 0.8+ |
| 7 | UnusedReturn | Internal function call with unused return value |
| 8 | MishandledException | Call with unused return value (cross-contract) |
| 9 | NonVulnerable | Clean contract (negative class) |

**Why the order is locked:** the `class_<i>` columns in the exported parquet are positional. The model's classifier head reads them in order. Changing the order means re-training from scratch.

### The 5 critical-path sources

| Source | Difficulty | Crosswalk complexity | Why critical |
|---|---|---|---|
| **DeFiHackLabs** | LOW (1 day) | Exploit PoC files; folder name = exploit type → direct mapping. T0 confidence. | Highest-confidence Tier-1 source |
| **SolidiFI** | MEDIUM (1-2 days) | 9,369 injected bugs, 100% ground-truth certainty (mathematically guaranteed). 7 types map to our 10. | Bug-injection = perfect labels |
| **DIVE** | MEDIUM (1-2 days) | 22,330 contracts, 8 DASP classes, multi-label. "bad_randomness" dropped (no 10-class equivalent). | Largest real-world source |
| **SmartBugs Curated** | LOW (0.5 day) | DASP categories → 10 classes direct. 143 hand-labeled contracts. | Ground-truth probe for Stage 4 recall test |
| **Web3Bugs** | MEDIUM-HIGH (1-2 days) | `bugs.csv` + `contests.csv` + report text. O/L/S severity: only O and L map to positive. | Largest Tier-1 source (~3,500 contests) |

### DISL as NonVulnerable source (no crosswalk needed)

DISL provides 514,506 unlabeled Solidity files. They're used as the **NonVulnerable** class (label=9). No crosswalk needed — they're negative by definition. But they need the **NonVulnerable 3:1 cap** (Stage 5) to prevent 514K:1 imbalance.

---

## 2️⃣ The Solution

### Crosswalk YAML format (D-3.2)

Each source has a `sentinel_data/labeling/crosswalks/<source>.yaml`. Example structure:

```yaml
source: dive
version: "1.0"
author: Ali
reviewed: 2026-07-01
notes:
  bad_randomness: DROPPED — no 10-class equivalent (v2.1 migration note)
mapping:
  reentrancy:
    target_class: Reentrancy
    confidence: T1
    notes: "Direct DASP mapping"
  access_control:
    target_class: ExternalBug
    confidence: T1
  arithmetic:
    target_class: IntegerUO
    confidence: T1
  # ... etc
```

The crosswalk is the **human decision**. It is never auto-generated. LLM-assist may draft the initial mapping, but every entry is reviewed by Ali before commit.

### Source-specific parsers (D-3.5)

Each parser reads the source's actual metadata format and produces per-contract `.labels.json`:

```json
{
  "contract_id": "<sha256>",
  "source": "dive",
  "classes": {
    "Reentrancy": {"value": 1, "confidence": 0.95, "tier": "T1", "evidence": "..."},
    "CallToUnknown": {"value": 0, "confidence": 0.0, "tier": null, "evidence": null}
  },
  "primary_class": "Reentrancy",
  "n_pos": 1
}
```

The parser joins with Stage 1's manifest (via `contract_id = meta.sha256`) to know which file came from which source.

### Multi-source merger (D-3.3)

When a contract appears in multiple sources, the merger combines labels with precedence:
1. DeFiHackLabs T0 (exploit-verified) overrides everything
2. Expert-audited sources (Bastet, ScaBench, Web3Bugs — T1)
3. Curated sources (SmartBugs Curated, Ethernaut, OZ — T2)
4. Tool-generated sources (Slither-Audited, Messi-Q — T3)
5. Heuristic / derived (T4)

Within a tier, positive wins over negative (false negatives are worse than false positives for the v2 baseline).

### The 99% DoS↔Reentrancy co-occurrence prevention

The merger explicitly handles the BCCC failure pattern: when a single source labels the same contract with both DoS and Reentrancy, that's near-certain noise. The merger de-duplicates co-occurring labels from the same source unless:
- (a) there's independent evidence (a different source also labels both)
- (b) the crosswalk explicitly marks the co-occurrence as legitimate

This is the data-side defense against the 99% co-occurrence that plagued BCCC.

### CallToUnknown merge rule (friend review)

If CallToUnknown verified count < 300, the merger **pauses and asks a human** to merge CallToUnknown into ExternalBug. The rule does NOT auto-merge — Ali's explicit approval is required. The decision is recorded in the catalog.

### Go/No-Go minimum-viable-corpus gate (friend review)

At the end of Stage 3, the gate validates that the corpus meets minimum thresholds:

| Criterion | Threshold | If below |
|---|---|---|
| Total contracts | ≥ 4,000 | Defer Run 11 to v2.1 |
| Reentrancy, DoS, IntegerUO positives | ≥ 300 each | Defer Run 11 |
| Other 7 classes positives | ≥ 100 each | Defer OR apply CallToUnknown merge |
| CallToUnknown verified | ≥ 300 | Apply merge rule (not a defer trigger) |
| SmartBugs Curated recall | ≥ 90% | Defer (semantic_checker broken) |
| FORGE agreement (if added) | ≥ 85% | Defer FORGE to v2.2 |

---

## 3️⃣ The Broader Context

### What Stage 3 enables downstream

- **Stage 4 (verification)** reads `.labels.json` and runs semantic checks against the AST
- **Stage 5 (splitting)** uses labels for stratified splitting
- **Stage 6 (analysis)** computes per-class co-occurrence matrices from labels
- **Stage 7 (export)** writes labels to `labels.parquet` for the ML module

### What breaks if Stage 3 is wrong

- Wrong crosswalk → wrong labels → model trains on noise → F1 ceiling again
- Missing co-occurrence prevention → 99% DoS↔Reentrancy re-appears → same BCCC failure
- Missing Go/No-Go gate → Run 11 launches on insufficient data → wasted compute

---

## 4️⃣ Verification — Stage 3 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | `taxonomy.yaml` exists with 10 classes in v1 order | ⏳ |
| 2 | 5 critical-path crosswalks exist | ⏳ |
| 3 | 5 critical-path parsers exist and run | ⏳ |
| 4 | Merger combines multi-source labels correctly | ⏳ |
| 5 | CallToUnknown < 300 merge rule pauses (not auto-merges) | ⏳ |
| 6 | 99% DoS↔Reentrancy co-occurrence regression test passes | ⏳ |
| 7 | Go/No-Go gate runs cleanly | ⏳ |
| 8 | FORGE 50-entry agreement test (or deferral documented) | ⏳ |

---

## 5️⃣ The "got it" checklist

1. **What is a crosswalk YAML?** A human-reviewed mapping from a source's native labels to the canonical 10 classes. One per source. Never auto-generated.

2. **Why is the taxonomy class order LOCKED?** The model's classifier head reads class columns positionally. Changing order = re-training from scratch.

3. **What does the 99% DoS↔Reentrancy co-occurrence prevention do?** When a single source labels the same contract with both DoS and Reentrancy, the merger de-duplicates unless there's independent evidence. This prevents the BCCC failure pattern.

4. **What's the CallToUnknown merge rule?** If verified CallToUnknown count < 300, the merger pauses and asks Ali to merge into ExternalBug. Never auto-merges.

5. **What's the Go/No-Go gate?** A 6-criterion check at the end of Stage 3. If the corpus doesn't meet minimums, Run 11 is deferred to v2.1.

6. **Why DIVE drops "bad_randomness"?** No equivalent in the 10-class taxonomy. The crosswalk documents the drop with a v2.1 migration note.

7. **Why is SmartBugs Curated important?** It's the ground-truth probe for Stage 4's semantic_checker recall test (≥90% threshold).

8. **What's the NonVulnerable 3:1 cap?** DISL has 514K unlabeled contracts. Without capping, the ratio is 514K:1 (same BCCC failure at larger scale). The cap is `positive_ratio_max = 3.0`.

If you can answer all 8, Stage 3 is mastered.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 3"
- **04_stage_3_labeling.md** — the design + intent document
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §3.4 (labeling), §6 (sources)
- **Reference:** `Data/sentinel_data/labeling/` (currently empty `__init__.py`)

When you're ready, say **"Stage 3 is mastered — let's move to Stage 4."**
