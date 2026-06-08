# Sentinel v2 Dataset Proposal

## Purpose

This document summarizes the **valid and usable dataset options** for Sentinel v2 after excluding BCCC-style sources as primary ground truth. The goal is to identify datasets that provide reliable Solidity source code, useful labels, practical scale, and a defensible role in a high-quality smart contract vulnerability detection pipeline.

The focus is on datasets that are actually usable in practice: downloadable, parseable, and relevant to AST/CFG/GNN/Transformer-based learning.

## Selection principles

A dataset is considered valid for this proposal if it satisfies most of the following:

- Solidity `.sol` code is directly available or reproducibly downloadable.
- Labels are expert-audited, exploit-verified, or at least clearly structured.
- The dataset is recent enough to matter for modern Solidity and DeFi patterns.
- The structure is compatible with building training, validation, and analysis pipelines.
- The dataset fills a meaningful role: gold positives, clean negatives, curated structural bugs, or volume/pretraining.

---

## Recommended dataset list

| Dataset | URL | Structure | Quality | Quantity | Why it is good for Sentinel v2 |
|---|---|---|---|---|---|
| **Bastet** | https://github.com/OneSavieLabs/Bastet | `dataset/repos/` for project source trees, `dataset/reports/` for audit reports, `dataset.csv` for finding-level labels and mappings. Full dataset downloaded via dataset ZIP linked from README. | **Very high** — expert-labeled, modern DeFi, audit-derived, two-annotator consensus. | 4,402 findings from 394 Code4rena reports; 849 fully annotated findings. | Best current gold dataset for real DeFi vulnerabilities. Modern Solidity, fine-grained taxonomy, audit-grounded semantics, and source-code mapping make it ideal as a Tier-1 supervised core. |
| **SC-Bench** | https://github.com/charlesxsh/scbench | `dataset/audited/` contains real audited contracts; `dataset/err-inj/origin/` and `dataset/err-inj/injected/` contain original and injected `.sol` files with matching JSON metadata. | **High** for audited subset; **medium** for injected subset because injected errors are synthetic. | 30 manually inspected ERC-20 contracts with 145 real issues; 5,347 original contracts and 15,836 injected errors. | Useful because it provides both real audited issues and controlled before/after error pairs. Strong for supervised detection experiments, mutation studies, and localization tasks. |
| **ScaBench** | https://github.com/scabench-org/scabench | Curated JSON dataset contains project metadata, repo URLs, exact commits, and vulnerability descriptions; `.sol` files are fetched using `checkout_sources.py`. | **High** — benchmark built from real audit ecosystems (Code4rena/Cantina/Sherlock). | 31 projects; 555 vulnerabilities; 114 high/critical. | Excellent benchmark-quality source for modern real-world vulnerabilities. Good for evaluation and for building audited-project positives plus realistic “no known issue” regions. |
| **Web3Bugs** | https://github.com/ZhangZhuoSJTU/Web3Bugs | `contracts/` contains contest-time source code, `reports/` contains Code4rena reports, `results/bugs.csv` and `results/contests.csv` provide labels and metadata. | **High** — based on accepted audit findings from real contests. | Repository spans many contest findings; often described as ~3,500 bugs across many projects. | Strong source of exploitable, contest-verified issues. Especially useful for semantic vs simple bug separation and for mining both positives and realistic hard negatives. |
| **DeFiHackLabs** | https://github.com/SunWeb3Sec/DeFiHackLabs | Foundry-style exploit repository, mostly under `src/test/` with one `_exp.sol` exploit reproduction per incident/year folder. | **Very high** for positive exploit ground truth. | 400+ incidents tracked in practice; growing exploit corpus. | Best source of true real-world exploit-positive examples. These are not hypothetical vulnerabilities; they were actually exploited, making them ideal for high-confidence positive supervision. |
| **solidity-defi-vulnerabilities (HF)** | https://huggingface.co/datasets/seyyedaliayati/solidity-defi-vulnerabilities | HuggingFace rows contain `testcase` Solidity code, `interfaces`, `github_path`, `contract_path`, explanation fields, `attack_title`, and `is_real`. | **High** for real examples; **medium-high** overall because it mixes real and synthetic educational cases. | 270 examples. | Good curated bridge between raw DeFiHackLabs incidents and ML-ready structured metadata. Useful for attack-family normalization, exploit explanation alignment, and compact gold exploit experiments. |
| **SmartBugs Curated** | https://github.com/smartbugs/smartbugs-curated | Vulnerable contracts organized by class folders; line-level annotations embedded in `.sol` comments; index in `vulnerabilities.json`. | **Medium-high** — classic benchmark, carefully curated, but older Solidity era and smaller scale. | 143 annotated contracts. | Best compact dataset for classical structural vulnerabilities with line-level supervision. Very useful for sanity checks, explainability experiments, and targeted structural-bug training. |
| **SmartBugs Wild** | https://github.com/smartbugs/smartbugs-wild | `contracts/<address>.sol` for raw verified contracts, plus metadata archive and collection scripts. | **Medium** as raw code only; label quality depends on external tool outputs. | 47,398 contracts; ~9.7M lines of code. | Great for unlabeled pretraining, representation learning, or broad candidate mining. Not good as a primary labeled source by itself, but excellent for volume. |
| **slither-audited-smart-contracts** | https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts | HF rows include `address`, flattened `source_code`, `bytecode`, and `slither` labels or detector JSON. | **Medium** — very large and convenient, but labels are tool-derived rather than expert-verified. | 467,216 contracts; ~1.75 GB. | Best large-scale auxiliary dataset for volume, weak supervision, or semi-supervised training. Useful only if down-weighted relative to gold sources. |
| **Messi-Q Smart-Contract-Dataset** | https://github.com/Messi-Q/Smart-Contract-Dataset | Multiple folders for raw Ethereum contracts, preprocessing artifacts, and labeled subsets for several vulnerability studies. | **Medium** — useful and widely reused, but older and not at Bastet/ScaBench quality. | 40K+ raw contracts; labeled subsets around 12K for vulnerability tasks. | A practical backup dataset for additional coverage and volume. Better treated as secondary training data than primary truth. |
| **OpenZeppelin Contracts** | https://github.com/OpenZeppelin/openzeppelin-contracts | Production-grade Solidity library with `contracts/` modules and historical `audits/` folder. | **Very high** as clean negative source. | Hundreds of production contracts across modules and versions. | One of the strongest available sources for high-confidence negative examples. Extensively audited, heavily used, modern Solidity, and ideal for “clean code” class construction. |
| **OpenZeppelin Ethernaut** | https://github.com/OpenZeppelin/ethernaut | Level-based repository of intentionally vulnerable challenge contracts with one exploit concept per level. | **High** for pedagogical positives; small scale. | 30+ levels/challenges. | Excellent for small, perfectly interpretable positive examples. Ideal for debugging class definitions, benchmarking exploitability concepts, and creating clean challenge-style seeds. |

---

## Supporting but lower-priority options

| Dataset | URL | Structure | Quality | Quantity | Use case |
|---|---|---|---|---|---|
| **DIVE** | https://www.nature.com/articles/s41597-026-07025-5 | Primarily feature matrices and labels; raw `.sol` availability is not the main distribution mode. | **Medium** | 22,330 contracts; 8 vulnerability types. | Useful as a benchmark/feature dataset, but less attractive than source-first datasets for Sentinel’s architecture. |
| **smart-contract-vulndb** | https://github.com/tintinweb/smart-contract-vulndb | `vulns.json` aggregation of public issues from audit reports; no direct source-code corpus. | **Medium-high** as issue metadata | N/A as contract corpus | Good as a metadata layer or audit-finding index, but not a stand-alone training dataset. |
| **Kaggle mirrors / repackaged datasets** | Varies | Usually CSV/feature mirrors of existing datasets such as Messi-Q or BCCC-like corpora. | **Low to medium** | Varies | Generally inferior to pulling from original GitHub/HuggingFace sources directly. |

---

## Recommended role assignment in Sentinel v2

### Tier 1 — Gold supervised positives

Use these as the highest-trust vulnerability labels:

- **Bastet**
- **ScaBench**
- **Web3Bugs**
- **DeFiHackLabs**
- **solidity-defi-vulnerabilities**

These datasets are the closest match to Sentinel’s needs because they contain real vulnerabilities from audits or actual exploits, modern Solidity/DeFi code, and enough semantic richness to justify a serious classifier.

### Tier 2 — High-confidence clean negatives

Use these as the strongest available negative pools:

- **OpenZeppelin Contracts**
- **OpenZeppelin upgradeable contracts** (same ecosystem, if added later)
- Audited code regions / unaffected modules from **ScaBench**, **Web3Bugs**, and **Bastet** projects

These are not mathematically “proven safe,” but they are the most defensible practical approximation available.

### Tier 3 — Curated structural bug datasets

Use these for classic vulnerability patterns, line-level experiments, and diagnostic evaluation:

- **SmartBugs Curated**
- **OpenZeppelin Ethernaut**
- Selected subsets of **SC-Bench**

These are especially useful for debugging model behavior and ensuring the system can detect well-understood bug classes before tackling semantic DeFi logic issues.

### Tier 4 — Volume and weak supervision

Use these only as auxiliary sources, not as core truth:

- **slither-audited-smart-contracts**
- **SmartBugs Wild**
- **Messi-Q dataset**

These are valuable for scale, representation learning, weak labels, contrastive pretraining, or broad negative mining, but they should be weighted lower than Tier-1 and Tier-2 sources.
**Zenodo record 16910242** is the dataset published by **Yizhou Chen** (Peking University) to accompany the paper *"Improving Smart Contract Security with Contrastive Learning-based Vulnerability Detection"* (CLEAR, published at ICSE 2025). [arxiv](https://arxiv.org/abs/2404.17839)

**Download:** [https://zenodo.org/records/16910242](https://zenodo.org/records/16910242) — single file `contracts.zip`, 88.3 MB, open access, no login required.

***

## Structure

Based on the CLEAR paper and Yizhou Chen's prior dataset releases: [github](https://github.com/Messi-Q/Smart-Contract-Dataset)

```
contracts.zip
└── contracts/
    ├── <contract_address>.sol     ← raw Solidity source files
    └── labels.csv / metadata      ← vulnerability labels per contract
```

The paper evaluated on a **large-scale real-world dataset of over 40,000 smart contracts** with labels across multiple vulnerability types.  The Zenodo release at 88.3 MB likely represents a cleaned or filtered subset of that larger corpus — comparable to the Messi-Q family (which Chen's group also contributed to). [github](https://github.com/Messi-Q/Smart-Contract-Dataset)

**Vulnerability types covered** align with Chen's group's prior work: Reentrancy, Timestamp Dependency, Integer Overflow/Underflow, Dangerous DelegateCall, and related structural classes. [arxiv](https://arxiv.org/abs/2404.17839)

---

## Final recommendation

If Sentinel v2 is built around **quality first**, the best composite stack is:

1. **Bastet** as the main semantic vulnerability anchor.
2. **ScaBench + Web3Bugs** as real audited vulnerability expansion.
3. **DeFiHackLabs + solidity-defi-vulnerabilities** as exploit-ground-truth positives.
4. **OpenZeppelin Contracts** as the strongest practical negative class.
5. **SmartBugs Curated + Ethernaut** as interpretable structural-bug calibration sets.
6. **SC-Bench, SmartBugs Wild, Slither-Audited, and Messi-Q** as secondary volume layers.
7. Zenodo record 16910242

This combination gives Sentinel v2:

- modern Solidity and DeFi coverage,
- expert and exploit-grounded labels,
- a clean negative pool,
- small interpretable debugging sets,
- and enough total volume to support pretraining and robust experimentation.

## Download links (plain list)

**Original 12 sources:**
- Bastet — https://github.com/OneSavieLabs/Bastet
- SC-Bench — https://github.com/charlesxsh/scbench
- ScaBench — https://github.com/scabench-org/scabench
- Web3Bugs — https://github.com/ZhangZhuoSJTU/Web3Bugs
- DeFiHackLabs — https://github.com/SunWeb3Sec/DeFiHackLabs
- solidity-defi-vulnerabilities — https://huggingface.co/datasets/seyyedaliayati/solidity-defi-vulnerabilities
- SmartBugs Curated — https://github.com/smartbugs/smartbugs-curated
- SmartBugs Wild — https://github.com/smartbugs/smartbugs-wild
- slither-audited-smart-contracts — https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts
- Messi-Q Smart-Contract-Dataset — https://github.com/Messi-Q/Smart-Contract-Dataset
- OpenZeppelin Contracts — https://github.com/OpenZeppelin/openzeppelin-contracts
- OpenZeppelin Ethernaut — https://github.com/OpenZeppelin/ethernaut

**Added from friend's research (2026-06-08, see `datasources_suggestions.md`):**

Tier 1 (gold) — added with priority:
- DIVE (Nature Scientific Data 2025) — https://www.nature.com/articles/s41597-026-07025-5 — **22,330 contracts, 8 DASP classes, multi-label, peer-reviewed**
- FORGE (ICSE 2026, arXiv:2506.18795) — https://github.com/shenyimings/FORGE-Artifacts — **LLM-driven extraction from real audit reports, CWE classification**
- SolidiFI Benchmark (ISSTA 2020) — https://github.com/DependableSystemsLab/SolidiFI-benchmark — **9,369 injection-ground-truth bugs, 100% certainty**
  *(Note: SolidiFI was already in the original "supporting" list — moved up to Tier 1 per friend)*

Tier 2 (silver/gold) — added with notes:
- ScrawlD (MSR 2022, arXiv:2202.11409) — https://github.com/sujeetc/ScrawlD — **6,780 mainnet contracts, 5-tool majority voting**
- Code4rena Audit Reports — https://code4rena.com/reports — **6,454 contracts, 1,361 high-risk findings, 499 confirmed high-severity vulns**
- DeFi Hacks REKT Database — https://deepwiki.com/helenmand/DeFi-Hacks-and-GPTScan-Top200-Dataset — **3,216 hacked incidents + 181 curated high-impact**

Tier 4 (bronze, for pretraining only) — added as volume sources:
- DISL — 514,506 unique Solidity files, unlabeled
- ReentrancyStudy-Data — 230,548 Etherscan contracts, reentrancy-labeled (single class, tool-based)
- EVMbench — 120 curated vulns from 40 Code4rena audits (subset of Code4rena; may not need separate connector)
- DeFiVulnLabs — 48 vulnerability types, Foundry-style exploits

**Friend's 3-tool ensemble warning (Part 1, datasources_suggestions.md):** Conkas + Slither + Smartcheck only detects 76.78% of actual vulnerabilities; Slither reentrancy precision is 51.97%. **Implication:** tool agreement is corroborative, NOT authoritative. The Stage 4 `tool_validator` design (D-4.3 in the integration proposal) is validated by this number.

**Friend's 97% SmartBugs Wild FP rate warning (Part 4):** SmartBugs Wild as labeled data is a "97% FP" trap. The existing Tier 3 (T3 tool-generated) design for `smartbugs_wild` in the merger is correct; the friend confirms this. **No change needed.**

***

