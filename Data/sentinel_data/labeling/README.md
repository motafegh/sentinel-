# `sentinel_data.labeling` — Assigning Meaning to Code

## What This Module Does

The labeling module is Stage 4 of the SENTINEL data pipeline. It takes preprocessed contracts and assigns them vulnerability class labels from a canonical 10-class taxonomy. This is the stage where "source code on disk" becomes "source code with semantic meaning."

The module implements three layers:

1. **Crosswalk YAMLs** — human-reviewed mapping tables that translate each source's idiosyncratic taxonomy into the canonical 10 classes
2. **Parsers** — source-specific code that reads each dataset's native format and applies the crosswalk
3. **Merger** — combines labels from multiple sources with conflict resolution and deduplication

## Why This Matters

The BCCC dataset (the predecessor) assigned labels based on folder names — if a contract was in the `reentrancy/` folder, it was labeled Reentrancy. This produced an **89% false positive rate for Reentrancy** and **86.9% for CallToUnknown** because:

- Contracts were copied across folders (38.8% duplication)
- The folder name didn't reflect the actual vulnerability
- There was no human review of the mapping

The labeling module fixes this by making every label mapping **explicit, human-reviewed, and version-controlled** in crosswalk YAMLs.

## Architecture Overview

```
Preprocessed .sol + meta.json
        │
        ▼
┌─────────────────────────────────────────┐
│         Per-source Parser                │
│  Reads native format → applies crosswalk │
│  → produces .labels.json per contract    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│           Merger                        │
│  Combines multi-source labels           │
│  Resolves conflicts by tier precedence  │
│  De-duplicates co-occurring labels      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Confidence Assigner             │
│  T0-T4 tiers per (contract, class)      │
│  Based on source reliability            │
└─────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `schema/taxonomy.yaml` | Canonical 10-class taxonomy (LOCKED — class order matches v1 checkpoint) |
| `schema/__init__.py` | Taxonomy loader with LRU cache |
| `crosswalks/defihacklabs.yaml` | DeFiHackLabs exploit folder → 10-class mapping |
| `crosswalks/smartbugs_curated.yaml` | SmartBugs Curated DASP categories → 10-class mapping |
| `crosswalks/solidifi.yaml` | SolidiFI bug-injection types → 10-class mapping |
| `crosswalks/dive.yaml` | DIVE 8-class DASP → 10-class mapping (bad_randomness DROPPED) |
| `crosswalks/web3bugs.yaml` | Web3Bugs contest findings → 10-class mapping |

## The Canonical 10-Class Taxonomy

The taxonomy is defined in `schema/taxonomy.yaml` and is **LOCKED** — the class order matches the v1 checkpoint's classifier head. Changing the order would break every existing checkpoint.

```yaml
classes:
  - id: 0
    name: CallToUnknown
    description: "Call to unknown address via .call{} / .delegatecall{}"
    severity: high
    dasp: DASP-7
  - id: 1
    name: DenialOfService
    description: "Gas griefing or unbounded iteration"
    severity: high
    dasp: DASP-5
  # ... (8 more classes)
```

## How Crosswalks Work

Each crosswalk YAML maps a source's native labels to the canonical 10 classes. Example from `crosswalks/smartbugs_curated.yaml`:

```yaml
source: smartbugs_curated
mappings:
  reentrancy: Reentrancy
  arithmetic: IntegerUO
  denial_of_service: DenialOfService
  time_manipulation: Timestamp
  unchecked_low_level_calls: CallToUnknown
  access_control: ExternalBug
  bad_randomness: Timestamp  # mapped to closest equivalent
  front_running: Timestamp
  short_addresses: NonVulnerable
  other: NonVulnerable
```

**Key design decisions:**
- Crosswalks are **human-reviewed, never auto-generated** — LLM-assist is allowed for drafting, but every mapping requires human approval
- The DIVE crosswalk **drops "bad_randomness"** (no 10-class equivalent) with a documented v2.1 migration note
- The Web3Bugs crosswalk **filters by severity** — only O (Optimistic) and L (Low) map to positive; S (Speculative) defaults to negative

## The Label Output Format

Every labeled contract has a `.labels.json` file:

```json
{
  "contract_id": "abc123...",
  "source": "defihacklabs",
  "classes": {
    "Reentrancy": {"value": 1, "confidence": 0.95, "tier": "T0", "evidence": "exploit PoC"},
    "CallToUnknown": {"value": 0, "confidence": 0.0, "tier": null, "evidence": null},
    "DenialOfService": {"value": 0, "confidence": 0.0, "tier": null, "evidence": null}
  },
  "primary_class": "Reentrancy",
  "n_pos": 1
}
```

## Confidence Tiers (T0–T4)

The confidence tier is a property of the **evidence**, not the contract:

| Tier | Source Type | Example |
|------|------------|---------|
| T0 | Exploit-verified | DeFiHackLabs (exploit PoC confirms the vulnerability) |
| T1 | Expert-audited | ScaBench, Web3Bugs, Bastet (human auditors) |
| T2 | Curated | SmartBugs Curated, OpenZeppelin, Ethernaut |
| T3 | Tool-generated | Slither-Audited, Messi-Q, Zenodo |
| T4 | Heuristic/derived | DISL (unlabeled), SmartBugs Wild (97% FP rate) |

A contract can have different tiers for different classes — it might be T0 for Reentrancy (exploit-verified) but T3 for Timestamp (only flagged by Slither).

## The Merger

When a contract appears in multiple sources, the merger combines the labels with explicit precedence rules:

1. **T0 (exploit-verified)** overrides everything
2. **T1 (expert-audited)** overrides T2–T4
3. **T2 (curated)** overrides T3–T4
4. **T3 (tool-generated)** overrides T4
5. **Within a tier**, positive wins over negative (false negatives are worse than false positives)

### The 99% Co-occurrence De-duplication Rule

The BCCC failure had a 99% co-occurrence between DoS and Reentrancy labels on the same contracts. The merger prevents this:

```python
def check_cooccurrence(labels, source):
    """If the same source labels a contract with both DoS and Reentrancy,
    that's near-certain noise — de-duplicate unless independent evidence exists."""
    if labels["DenialOfService"].value == 1 and labels["Reentrancy"].value == 1:
        if not independent_evidence_from_other_source:
            # Flag for human review or auto-deduplicate
            ...
```

### The CallToUnknown < 300 Merge Rule

If fewer than 300 contracts are verified as CallToUnknown across all sources, the merger pauses and asks a human whether to merge CallToUnknown into ExternalBug. This is a safety valve — the threshold is in `config.yaml` and requires explicit human approval.

## How to Use

```bash
# Label a single source
sentinel-data label --source defihacklabs

# Label all enabled sources
sentinel-data label

# Dry-run
sentinel-data label --source scabench --dry-run
```

## Pipeline Position

```
Stage 3: Representation (graph + token extraction)
    ↓
Stage 4: Labeling ← YOU ARE HERE (crosswalks + parsers + merger)
    ↓
Stage 5: Verification (are these labels correct?)
```

## What This Module Does NOT Do

- It does not verify label correctness (that's Stage 5: verification)
- It does not modify the graph representation (labels are stored separately)
- It does not split the data into train/val/test (that's Stage 6: splitting)

The labeling module is a **pure annotation layer**. Its output is the `.labels.json` file that every downstream stage reads.

## Design Decisions

1. **Crosswalks are human-reviewed** — prevents the BCCC "folder = label" fiction
2. **Labels are separate from representations** — the same graph can be re-labeled without re-extraction
3. **Confidence tiers per (contract, class)** — a contract can be T0 for one class and T3 for another
4. **Co-occurrence de-duplication** — prevents the 99% DoS↔Reentrancy pattern
5. **CallToUnknown merge rule pauses, not auto-merges** — requires human approval
